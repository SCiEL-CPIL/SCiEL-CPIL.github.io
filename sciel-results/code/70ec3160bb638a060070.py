"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138
    
    n = 18
    best_score = 12
    best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) 
                  for a in [i*6+k for k in range(6)] 
                  for b in [j*6+k for k in range(6)]]
    
    def _max_tf_size_fast(adj):
        """Fast upper bound: greedy triangle-free set construction."""
        # Try multiple random orderings
        best = 0
        for _ in range(50):
            order = list(range(n))
            random.shuffle(order)
            in_set = []
            for v in order:
                # Check if v can be added without creating a triangle
                ok = True
                for i, u in enumerate(in_set):
                    if adj[v] & (1 << u):
                        for w in in_set[:i]:
                            if (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    in_set.append(v)
            if len(in_set) > best:
                best = len(in_set)
        return best
    
    def _exact_max_tf_size(adj, timeout=2.0):
        """Branch and bound with timeout."""
        t0 = time.time()
        best = [0]
        
        def dfs(candidates_mask, size, in_set_mask):
            if size > best[0]:
                best[0] = size
            if time.time() - t0 > timeout:
                return
            if candidates_mask == 0:
                return
            # Upper bound
            if size + bin(candidates_mask).count('1') <= best[0]:
                return
            
            # Pick vertex with most constraints
            # Try each candidate
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            rest_mask = candidates_mask & ~(1 << v)
            
            # Exclude v
            dfs(rest_mask, size, in_set_mask)
            
            # Include v if no triangle formed
            # v forms triangle with u,w in in_set if adj[v]&u, adj[u]&w, adj[v]&w
            can_include = True
            in_set_list = [i for i in range(n) if in_set_mask & (1 << i)]
            for u in in_set_list:
                if adj[v] & (1 << u):
                    common = adj[v] & adj[u]
                    for w in in_set_list:
                        if w != u and (common & (1 << w)):
                            can_include = False
                            break
                if not can_include:
                    break
            
            if can_include:
                # Remove from candidates any w that would form triangle with v and some u in in_set
                new_candidates = rest_mask
                for u in in_set_list:
                    if adj[v] & (1 << u):
                        common = adj[v] & adj[u]
                        # Remove all w in rest_mask that are in common
                        new_candidates &= ~common
                dfs(new_candidates, size + 1, in_set_mask | (1 << v))
        
        dfs((1 << n) - 1, 0, 0)
        return best[0]
    
    random.seed(42)
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        edge_set = set()
        
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.38:
                    common = adj[i] & adj[j]
                    creates_k4 = False
                    for k in range(18):
                        if common & (1 << k):
                            for l in range(k+1, 18):
                                if (common & (1 << l)) and (adj[k] & (1 << l)):
                                    creates_k4 = True
                                    break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edge_set.add((i, j))
        
        # Fast check first
        fast_score = _max_tf_size_fast(adj)
        if fast_score < best_score:
            # Verify with exact (with timeout)
            exact_score = _exact_max_tf_size(adj, timeout=3.0)
            if exact_score < best_score:
                best_score = exact_score
                best_edges = [[u, v] for u, v in edge_set]
        
        if best_score <= 9:
            break
    
    return best_edges
# EVOLVE-BLOCK-END
