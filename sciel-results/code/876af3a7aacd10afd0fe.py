import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill

    # Bitmask adjacency representation
    # adj[i] is a bitmask where bit j is set iff edge (i,j) exists
    
    def build_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_degree(adj, i):
        return bin(adj[i]).count('1')

    def compute_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid_adj(adj, degrees=None):
        if degrees is None:
            degrees = compute_degrees(adj)
        
        # For each edge (u, v), check all paths s-u-v-t where s != t
        for u in range(n):
            au = adj[u]
            for v in range(n):
                if not (au & (1 << v)):
                    continue
                if u > v:
                    continue
                # Edge (u, v) exists
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj[v] & ~(1 << u)
                
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    
                    t_mask = av_minus_u & ~(1 << s)
                    while t_mask:
                        t_bit = t_mask & -t_mask
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        
                        if degrees[s] == degrees[t]:
                            return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Strategy: Start with a complete graph and remove edges to satisfy constraints
    # This is a "reverse" approach - start dense and prune
    
    # First, let's try a constructive approach:
    # Build graph by adding edges one at a time, preferring to add edges that maintain validity
    
    # Initialize with empty graph
    current_adj = [0] * n
    current_deg = [0] * n
    current_count = 0
    
    best_adj = [0] * n
    best_deg = [0] * n
    best_count = 0

    # Try adding all possible edges in a greedy fashion
    # For each pair (u,v), try to add it if valid
    # Use multiple random orderings to escape local optima
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    random.seed(42)
    
    # Phase 1: Greedy construction with multiple random orderings
    for trial in range(50):
        trial_start = time.time()
        if trial_start - start_time > deadline - 5:
            break
            
        trial_adj = [0] * n
        trial_deg = [0] * n
        trial_count = 0
        
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Try to add edge (u, v)
            new_adj = trial_adj[:]
            new_deg = trial_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_adj(new_adj, new_deg):
                trial_adj = new_adj
                trial_deg = new_deg
                trial_count += 1
        
        if trial_count > best_count:
            best_count = trial_count
            best_adj = trial_adj[:]
            best_deg = trial_deg[:]
            
        if best_count >= 25:  # Good enough threshold
            break
    
    # Phase 2: Local search from best solution
    current_adj = best_adj[:]
    current_deg = best_deg[:]
    current_count = best_count
    
    # Tabu search with edge swaps
    tabu_list = []
    tabu_tenure = 20
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline:
        no_improve += 1
        
        # Try to add a random non-edge
        added = False
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in tabu_list:
                continue
            if current_adj[u] & (1 << v):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_adj(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                    best_deg = current_deg[:]
                added = True
                no_improve = 0
                break
        
        if added:
            continue
            
        # Try swap: remove one edge, add another
        swapped = False
        # Pick random edge to remove
        edge_to_remove = None
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                edge_to_remove = (u, v)
                break
        
        if edge_to_remove is None:
            break
            
        u1, v1 = edge_to_remove
        
        # Pick random non-edge to add
        for _ in range(30):
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) == (u1, v1):
                continue
            if (u2, v2) in tabu_list:
                continue
            if current_adj[u2] & (1 << v2):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            if is_valid_adj(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                # Count stays same
                tabu_list.append((u1, v1))
                tabu_list.append((u2, v2))
                while len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
                swapped = True
                no_improve = 0
                break
        
        if not swapped:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart from best
            current_adj = best_adj[:]
            current_deg = best_deg[:]
            current_count = best_count
            no_improve = 0
            tabu_list = []
            max_no_improve *= 1.3
    
    # Convert best_adj back to edge list
    final_edges = adj_to_edges(best_adj)
    
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    
    return final_edges
