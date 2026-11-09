import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    N = 20
    FULL_MASK = (1 << N) - 1

    # Bitmask-based K3,3-free check
    # adj[i] is a bitmask of neighbors of vertex i
    # A graph is K3,3-free iff for every triple (a,b,c), |N(a) & N(b) & N(c)| <= 2

    def is_valid_addition_bitmask(adj, u, v):
        """
        Check if adding edge (u,v) keeps graph K3,3-free.
        adj is a list of bitmasks. Assumes (u,v) not currently an edge.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # Adding (u,v): new Nu' = Nu | (1<<v), new Nv' = Nv | (1<<u)
        
        # Check triples {u, x, y} for x,y != u,v
        # |Nu' & Nx & Ny| <= 2
        # Nu' & Nx & Ny = (Nu & Nx & Ny) | ((1<<v) & Nx & Ny)
        # Since v not in Nu (no edge u-v currently), these are disjoint
        # So size = popcount(Nu & Nx & Ny) + (1 if (Nx & Ny) has bit v set else 0)
        
        for x in range(N):
            if x == u or x == v:
                continue
            Nx = adj[x]
            for y in range(x + 1, N):
                if y == u or y == v:
                    continue
                Ny = adj[y]
                
                inter = Nu & Nx & Ny
                pc = inter.bit_count()
                
                if (Nx >> v) & 1 and (Ny >> v) & 1:
                    if pc > 1:
                        return False
                else:
                    if pc > 2:
                        return False
        
        # Triples {u,v,x}: intersection is Nu & Nv & Nx (unchanged from before)
        # Triples not containing u or v: unchanged
        
        return True

    def is_valid_removal_bitmask(adj, u, v):
        """
        Removing an edge always preserves K3,3-freeness (removing edges can't create new K3,3).
        """
        return True

    def popcount(x):
        return x.bit_count()

    def greedy_init_bitmask():
        """
        Greedy construction using bitmask representation.
        """
        adj = [0] * N
        edge_set = set()
        possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(possible)
        
        for u, v in possible:
            if is_valid_addition_bitmask(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        count = len(edge_set)
        return adj, edge_set, count

    def simulated_annealing(adj, edge_set, count, t_start, t_end, t_min=0.01):
        """
        Simulated annealing with moves: add edge, remove edge, or swap (remove one, add another).
        adj: list of bitmasks
        edge_set: set of (u,v) tuples
        count: number of edges
        """
        current_adj = list(adj)
        current_edges = set(edge_set)
        current_count = count
        
        best_adj = list(adj)
        best_edges = set(edge_set)
        best_count = count
        
        n_steps = 0
        t = t_start
        
        while time.time() < deadline:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random non-edge
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in current_edges:
                    continue
                
                if is_valid_addition_bitmask(current_adj, u, v):
                    # Accept (improvement)
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = list(current_adj)
                        best_edges = set(current_edges)
                else:
                    # Reject (would violate K3,3)
                    pass
                    
            elif move_type < 0.8:
                # Try to remove a random edge
                if not current_edges:
                    continue
                e = random.choice(list(current_edges))
                u, v = e
                
                # Removal is always valid
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges.remove(e)
                current_count -= 1
                
            else:
                # Swap: remove one edge, try to add another
                if not current_edges:
                    continue
                e1 = random.choice(list(current_edges))
                u1, v1 = e1
                
                # Try to add a random non-edge
                u2 = random.randint(0, N - 1)
                v2 = random.randint(0, N - 1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                if (u2, v2) in current_edges:
                    continue
                
                # Temporarily remove e1
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                
                # Try to add e2
                if is_valid_addition_bitmask(current_adj, u2, v2):
                    # Accept swap
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    current_edges.remove(e1)
                    current_edges.add((u2, v2))
                    # count stays same
                else:
                    # Reject: put back e1
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            
            n_steps += 1
            # Cool down
            if n_steps % 100 == 0:
                t *= 0.995
                if t < t_min:
                    t = t_min
            
            # Check for best update
            if current_count > best_count:
                best_count = current_count
                best_adj = list(current_adj)
                best_edges = set(current_edges)
        
        return best_adj, best_edges, best_count

    # Main: run multiple restarts with greedy init + simulated annealing
    best_count = 0
    best_edges = set()
    best_adj = [0] * N
    
    # Phase 1: Quick greedy inits to find a good starting point
    n_greedy = 30
    for i in range(n_greedy):
        if time.time() > deadline:
            break
        adj, edges, count = greedy_init_bitmask()
        if count > best_count:
            best_count = count
            best_edges = edges
            best_adj = adj
    
    # Phase 2: Simulated annealing on the best greedy result
    if time.time() < deadline:
        # Allocate remaining time for SA
        remaining = deadline - time.time()
        t_start = 1.0
        t_end = 0.05
        
        best_adj, best_edges, best_count = simulated_annealing(
            best_adj, best_edges, best_count, t_start, t_end
        )
    
    # Convert to list of edges
    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    # Test
    edges = construct()
    print(len(edges))
