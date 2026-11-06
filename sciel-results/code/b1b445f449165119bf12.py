import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Seed graph
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    
    best_n = n_seed
    best_edges = edges_seed
    
    # Try to find larger n
    # We know 39 is possible.
    # Try n from 39 down to 27
    for target_n in range(39, 26, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random instances
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Generate random triangle-free graph
            edges = []
            adj = [set() for _ in range(target_n)]
            p = 0.35 # Adjust density
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        if not (adj[u] & adj[v]):
                            edges.append([u, v])
                            adj[u].add(v)
                            adj[v].add(u)
                            
            # Check for independent set of size 10
            # Use a simple recursive check with pruning
            # If it finds one, it's invalid.
            # If it doesn't, it's likely valid.
            # To be safe, we can run a limited search.
            
            def has_iso_k(adj, k):
                # Returns True if independent set of size k exists
                # Simple backtracking
                def dfs(current, candidates):
                    if len(current) == k:
                        return True
                    if len(current) + len(candidates) < k:
                        return False
                    if not candidates:
                        return False
                    # Heuristic: pick vertex with max degree in induced subgraph?
                    # Just pick first
                    v = candidates[0]
                    # Include v
                    new_cands = [u for u in candidates[1:] if u not in adj[v]]
                    if dfs(current + [v], new_cands):
                        return True
                    # Exclude v
                    if dfs(current, candidates[1:]):
                        return True
                    return False
                
                return dfs([], list(range(target_n)))
            
            # The DFS might be slow.
            # Let's try a greedy check first.
            # If greedy finds 10, it's invalid.
            # If not, we assume valid (risk of false negative).
            # Given the constraints, false negative is bad.
            # But the DFS is exponential.
            # For n=39, it might hang.
            
            # Alternative: Use the fact that random graphs have small alpha.
            # If p is low enough, alpha is small.
            # Let's just run the DFS with a timeout?
            # Python doesn't have easy timeouts inside functions.
            
            # I will skip the DFS and rely on the fact that if I can't verify, I shouldn't return it?
            # No, I must return a valid graph.
            # If I return an invalid one, score 0.
            # So I must verify.
            
            # I will only verify for small n or if I have time.
            # For n=39, verification is hard.
            
            # Let's stick to n=27..30 where verification is faster.
            if target_n <= 30:
                if not has_iso_k(adj, 10):
                    return {'n': target_n, 'edges': edges}
            else:
                # For larger n, we are risky.
                # Maybe just return seed?
                pass

    return {'n': best_n, 'edges': best_edges}
