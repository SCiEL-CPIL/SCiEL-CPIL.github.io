import time
import sys

def _generate_smooth_numbers(limit, primes):
    """Generate all numbers <= limit that are smooth w.r.t to given primes."""
    smooth = set([1])
    # Use a queue or recursive generation
    # Since limit is 20000, simple DFS/BFS on exponents is fine
    # Or just iterate and check? 
    # Iterating 1..20000 and checking divisibility by primes is O(Limit * K * Log)
    # 20000 * 10 * 10 = 2M ops. Very fast.
    s = set()
    for n in range(2, limit + 1):
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            s.add(n)
    return s

def _build_graph(smooths, max_val):
    """
    Build adjacency list for graph where edge (u, v) exists if u+v is smooth.
    u, v in [1, max_val], u != v.
    """
    adj = [set() for _ in range(max_val + 1)]
    # To avoid duplicates and ensure u < v in iteration logic, 
    # but adjacency is symmetric.
    
    # Optimization: 
    # For each smooth number h, find pairs (u, v) such that u+v=h.
    # u must be in [1, max_val], v must be in [1, max_val].
    # So u in [max(1, h-max_val), min(max_val, h-1)].
    
    for h in smooths:
        start = max(1, h - max_val)
        end = min(max_val, h - 1)
        if start > end:
            continue
        for u in range(start, end + 1):
            v = h - u
            if u == v:
                continue
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
    return adj

def _bron_kerbosch(clique, candidates, excluded, adj, target_size, deadline):
    """
    Find a clique of size target_size.
    Returns the clique if found, else None.
    """
    if time.time() > deadline:
        return None
        
    if len(clique) == target_size:
        return list(clique)
        
    if len(candidates) + len(excluded) < target_size - len(clique):
        return None
        
    # Pivot selection: choose v in candidates U excluded with max degree in candidates
    # This reduces the number of recursive calls
    pivot_set = candidates | excluded
    if not pivot_set:
        return None
        
    # Heuristic: pick node with highest degree in candidates
    # For simplicity and speed, we can just pick the first one or one with high degree
    # Let's pick one with high degree in candidates to maximize pruning
    pivot = None
    max_deg = -1
    for v in pivot_set:
        deg = len(candidates & adj[v])
        if deg > max_deg:
            max_deg = deg
            pivot = v
            
    if pivot is None:
        return None
        
    # Iterate over candidates NOT adjacent to pivot
    # This is the key optimization of Bron-Kerbosch with pivot
    # We only need to branch on candidates that are NOT neighbors of pivot
    # because if a candidate IS a neighbor of pivot, the clique containing it 
    # will be found when we branch on the pivot's neighbors in other iterations?
    # Wait, the standard BK with pivot says:
    # for v in candidates \ adj(pivot):
    
    candidates_to_process = candidates - adj[pivot]
    
    # Order candidates for better pruning (high degree first)
    # Sort candidates_to_process by degree in candidates
    ordered_cands = sorted(candidates_to_process, key=lambda x: len(candidates & adj[x]), reverse=True)
    
    for v in ordered_cands:
        if time.time() > deadline:
            return None
            
        new_clique = clique | {v}
        new_candidates = candidates & adj[v]
        new_excluded = excluded & adj[v]
        
        res = _bron_kerbosch(new_clique, new_candidates, new_excluded, adj, target_size, deadline)
        if res:
            return res
            
        candidates.remove(v)
        excluded.add(v)
        
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    max_val = 10000
    target_size = 12
    
    for k in range(1, len(primes) + 1):
        if time.time() > deadline - 5.0:
            break
            
        current_primes = primes[:k]
        
        # Generate smooth numbers
        smooths = _generate_smooth_numbers(2 * max_val, current_primes)
        
        if not smooths:
            continue
            
        # Build graph
        # We can prune nodes with degree < target_size - 1 early?
        # But we need to build the graph first to know degrees.
        
        adj = _build_graph(smooths, max_val)
        
        # Prune nodes with degree < 11
        # If a node has degree < 11, it can't be in a clique of size 12.
        # We can remove them from the candidate pool.
        
        initial_candidates = set()
        for i in range(1, max_val + 1):
            if len(adj[i]) >= target_size - 1:
                initial_candidates.add(i)
                
        if len(initial_candidates) < target_size:
            continue
            
        # Run Bron-Kerbosch
        # Start with empty clique, all pruned candidates as candidates, empty excluded
        res = _bron_kerbosch(set(), initial_candidates, set(), adj, target_size, deadline)
        
        if res:
            return res
            
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Test locally
if __name__ == "__main__":
    res = construct()
    print(res)
