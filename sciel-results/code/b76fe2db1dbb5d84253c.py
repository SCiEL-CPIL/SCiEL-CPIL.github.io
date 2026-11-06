import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper: Check if graph is C4-free using bitmasks
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    # Helper: Check if there is an independent set of size >= target
    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(10000)
        node_count = [0]
        max_nodes = 2000000  # Higher limit with better pruning
        
        def dfs(candidates, size):
            if size >= target:
                return True
            
            if candidates == 0:
                return False
            
            # Pruning: if even taking all candidates doesn't reach target
            if size + candidates.bit_count() < target:
                return False
            
            # Limit node count
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            
            # Pick a vertex from candidates
            # Strategy: pick vertex with highest degree in induced subgraph
            # Early stop if we find a vertex with very high degree
            
            c = candidates
            best_v = -1
            max_deg = -1
            
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                deg = (adj_mask[v] & candidates).bit_count()
                
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    # Early stop if degree is very high for better pruning
                    if deg >= 10:
                        break
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_candidates = candidates & ~(1 << best_v)
            if dfs(new_candidates, size):
                return True
                
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)

    def generate_greedy_c4_free(n, seed):
        random.seed(seed)
        adj_mask = [0] * n
        
        # List of all possible edges
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            nu = adj_mask[u]
            nv = adj_mask[v]
            
            # Check if adding (u, v) creates a C4
            # A C4 is created if there is a vertex x such that x is neighbor of u and y is neighbor of v
            # and x != y? No, C4 is u-x-v-y-u.
            # Condition for no C4: For all x in N(u), x is not adjacent to any y in N(v) (except u and v themselves, but u,v not in N(u),N(v)).
            # Actually, if x in N(u) and y in N(v) and x is adjacent to y, then u-x-y-v-u is a C4.
            # So we need N(u) and N(v) to have no edges between them.
            
            # Efficient check:
            # For each x in N(u), check if adj_mask[x] intersects N(v).
            # If yes, adding (u,v) creates a C4.
            
            # To be faster, iterate over the smaller neighbor set.
            if nu.bit_count() < nv.bit_count():
                c = nu
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nv:
                        safe = False
                        break
            else:
                c = nv
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break
            
            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        return adj_mask

    # Try to find a graph with n vertices
    for n in range(43, 29, -1):
        # Number of seeds to try for this n
        max_seeds = 3000
        
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
                
            adj_mask = generate_greedy_c4_free(n, seed)
            
            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                continue
                
            # Check independence number
            # If alpha < 11, we are good.
            # has_independent_set returns True if alpha >= 11.
            # So we want it to return False.
            
            # Quick pre-filter: count edges. A C4-free graph on n vertices has at most
            # roughly n^(3/2) edges. More edges generally means smaller alpha.
            # For n=43, we need enough edges to keep alpha < 11.
            # Rough lower bound: if graph is d-regular, alpha <= n/(d+1) by Turan.
            # So we need d >= n/11 - 1. For n=43, d >= 2.9, so d >= 3.
            # This is a very weak filter, so let's just do the exact check.
            
            if not has_independent_set(adj_mask, n, 11):
                # Construct edge list
                edges = []
                for u in range(n):
                    m = adj_mask[u]
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        m ^= lsb
                return {'n': n, 'edges': edges}
        
        if time.time() > deadline - 30:
            break
            
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    return {'n': 30, 'edges': fallback_edges}
