import time
import random

def construct():
    start = time.time()
    deadline = start + 115.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    edge_set = set(all_edges)
    
    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2
    
    def is_valid(adj):
        """Check if graph is K2,3-free"""
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                common &= ~(1 << u) & ~(1 << v)
                if common.bit_count() >= 3:
                    return False
        return True
    
    def get_common_count(adj, u, v):
        """Number of common neighbors of u and v"""
        common = adj[u] & adj[v]
        common &= ~(1 << u) & ~(1 << v)
        return common.bit_count()
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free"""
        if (adj[u] >> v) & 1:
            return True
        # Adding (u,v) affects common neighbor counts
        # For any w in N(u) \ {v}: u and w now share v as common neighbor
        # For any w in N(v) \ {u}: v and w now share u as common neighbor
        # u and v now share all common neighbors of N(u)∩N(v) plus each other's other neighbors
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # u and v: common neighbors = N(u) ∩ N(v) (excluding u,v)
        # After adding, u and v have |N(u) ∩ N(v)| common neighbors
        if get_common_count(adj, u, v) >= 3:
            return False
        
        # For each w in N(u) \ {v}: check if u and w would have >= 3 common neighbors
        # u and w currently share some neighbors. Adding (u,v) means v becomes a common neighbor of u and w
        # So we need: current_common(u,w) + 1 <= 2, i.e., current_common(u,w) <= 1
        wmask = nu
        while wmask:
            w = (wmask & -wmask).bit_length() - 1
            wmask &= wmask - 1
            if get_common_count(adj, u, w) >= 2:
                return False
        
        # For each w in N(v) \ {u}: check if v and w would have >= 3 common neighbors
        wmask = nv
        while wmask:
            w = (wmask & -wmask).bit_length() - 1
            wmask &= wmask - 1
            if get_common_count(adj, v, w) >= 2:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def local_search(adj, max_iter=1000):
        """Greedy local search: try to add any valid edge"""
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for (u, v) in all_edges:
                if time.time() > deadline - 5.0:
                    return adj
                if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        return adj
    
    def swap_search(adj, max_swaps=200):
        """Try edge swaps: remove one edge, add another"""
        current = adj[:]
        best = adj[:]
        best_count = count_edges(adj)
        
        for _ in range(max_swaps):
            if time.time() > deadline - 10.0:
                break
            # Pick a random edge to remove
            edges_in_graph = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
            if not edges_in_graph:
                break
            ru, rv = random.choice(edges_in_graph)
            remove_edge(current, ru, rv)
            
            # Try to add a random non-edge
            non_edges = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
            if not non_edges:
                add_edge(current, ru, rv)
                continue
            au, av = random.choice(non_edges)
            if can_add_edge(current, au, av):
                add_edge(current, au, av)
                cnt = count_edges(current)
                if cnt > best_count:
                    best_count = cnt
                    best = current[:]
                else:
                    # Revert
                    remove_edge(current, au, av)
                    add_edge(current, ru, rv)
            else:
                add_edge(current, ru, rv)
        
        return best, best_count
    
    def simulated_annealing(adj, steps=5000):
        """Simulated annealing with edge additions and swaps"""
        current = adj[:]
        best = adj[:]
        best_count = count_edges(adj)
        temp = 2.0
        cooling = 0.995
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
            temp *= cooling
            
            # Choose move: add edge, remove edge, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random non-edge
                non_edges = [(u, v) for (u, v) in all_edges if not ((current[u] >> v) & 1)]
                if not non_edges:
                    continue
                u, v = random.choice(non_edges)
                if can_add_edge(current, u, v):
                    add_edge(current, u, v)
                    delta = 1
                else:
                    continue
            elif move_type < 0.5:
                # Try to remove a random edge
                edges_in_graph = [(u, v) for (u, v) in all_edges if (current[u] >> v) & 1]
                if not edges_in_graph:
                    continue
                u, v = random.choice(edges_in_graph)
                remove_edge(current, u, v)
                delta = -1
            else:
                # Swap: remove one edge, add another
                edges_in_graph = [(u, v) for (u, v) in all_edges if (current[u] >> v) & 1]
                non_edges = [(u, v) for (u, v) in all_edges if not ((current[u] >> v) & 1)]
                if not edges_in_graph or not non_edges:
                    continue
                ru, rv = random.choice(edges_in_graph)
                au, av = random.choice(non_edges)
                remove_edge(current, ru, rv)
                if can_add_edge(current, au, av):
                    add_edge(current, au, av)
                    delta = 0
                else:
                    add_edge(current, ru, rv)
                    continue
            
            cnt = count_edges(current)
            if cnt > best_count:
                best_count = cnt
                best = current[:]
            elif delta < 0 or random.random() < 0.1:
                # Accept worse moves with some probability
                pass
            else:
                # Revert
                if move_type < 0.3:
                    remove_edge(current, u, v)
                elif move_type < 0.5:
                    add_edge(current, u, v)
                else:
                    # Already reverted
                    pass
        
        return best, best_count
    
    def structured_seed(rng):
        """Create a structured initial graph"""
        adj = [0] * n
        mode = rng.random()
        
        if mode < 0.2:
            # 5 disjoint K4s
            for k in range(5):
                for a in range(4):
                    for b in range(a+1, 4):
                        u = 4*k+a; v = 4*k+b
                        add_edge(adj, u, v)
        elif mode < 0.4:
            # Two K5s and two K3s and one K2... let's do: K5, K5, K5, K5 minus some edges
            # Actually let's do a different structure: complete bipartite-like
            # Split into 4 groups of 5, connect within groups partially
            for g in range(4):
                for a in range(5):
                    for b in range(a+1, 5):
                        if rng.random() < 0.7:
                            u = 5*g+a; v = 5*g+b
                            add_edge(adj, u, v)
        elif mode < 0.6:
            # Random sparse graph
            for (u, v) in all_edges:
                if rng.random() < 0.3:
                    add_edge(adj, u, v)
        elif mode < 0.8:
            # Cycle-based: start with a 20-cycle, add chords
            for i in range(n):
                j = (i+1) % n
                add_edge(adj, i, j)
            for _ in range(10):
                u = rng.randint(0, n-1)
                v = rng.randint(0, n-1)
                if u != v and not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
        else:
            # Random dense attempt
            for (u, v) in all_edges:
                if rng.random() < 0.4:
                    add_edge(adj, u, v)
        
        return adj
    
    best_adj = None
    best_count = -1
    rng = random.Random(42)
    
    # Multiple restarts with different strategies
    for attempt in range(200):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize
        adj = structured_seed(rng)
        
        # Local search
        adj = local_search(adj, max_iter=200)
        
        # Swap search
        adj, cnt = swap_search(adj, max_swaps=100)
        
        # Simulated annealing
        adj, cnt = simulated_annealing(adj, steps=2000)
        
        # Final local search
        adj = local_search(adj, max_iter=500)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Occasionally do more intensive search on best
        if best_adj is not None and attempt % 5 == 0 and time.time() < deadline - 30.0:
            adj2 = best_adj[:]
            adj2, cnt2 = simulated_annealing(adj2, steps=3000)
            adj2 = local_search(adj2, max_iter=500)
            cnt2 = count_edges(adj2)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2[:]
    
    if best_adj is None:
        # Fallback: 5 disjoint K4s
        adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    u = 4*k+a; v = 4*k+b
                    add_edge(adj, u, v)
        best_adj = adj
    
    # Final polish: one more local search
    best_adj = local_search(best_adj, max_iter=1000)
    
    # Build result
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                res.append([u, v])
    
    return res
