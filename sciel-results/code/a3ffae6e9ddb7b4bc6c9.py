import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    # Target primes sets to try, in order of increasing size (lower value is better)
    prime_sets = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    max_val = 10000
    k = 12
    
    # Helper to check if a number is P-smooth
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
    
    # Generate P-smooth numbers <= max_val
    def gen_smooth(primes):
        smooths = [1]
        # Simple BFS/DFS generation
        # Or just iterate 1..max_val and check
        # Iterating 1..10000 and checking smoothness is O(10000 * len(primes) * log(max_val))
        # This is very fast.
        return [i for i in range(1, max_val + 1) if is_smooth(i, primes)]
    
    # Find a clique of size k in the graph where nodes are smooths
    # Edge exists if u+v is smooth
    def find_clique(smooths, primes, k, time_limit):
        n = len(smooths)
        if n < k:
            return None
            
        # Precompute smoothness of sums
        # smooth_sum[i][j] is True if smooths[i] + smooths[j] is smooth
        # To save memory, we can just check on the fly or use a bitset
        # With n ~ 700, n^2 is 500k checks. Fast.
        
        # Adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            ui = smooths[i]
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    return None
                if is_smooth(ui + smooths[j], primes):
                    adj[i].append(j)
                    adj[j].append(i)
                    
        # Backtracking clique search
        # We want a clique of size k.
        # Use a simple recursive approach with pruning.
        # Order nodes by degree (descending) to find large cliques faster.
        degrees = [len(adj[i]) for i in range(n)]
        nodes = list(range(n))
        nodes.sort(key=lambda x: degrees[x], reverse=True)
        
        # Map new indices to original
        # Actually, let's just use the sorted list of node indices
        # We need to maintain the mapping to original smooths
        
        best_clique = []
        
        def backtrack(start, current_clique, candidates):
            nonlocal best_clique
            if time.time() > time_limit:
                return
            if len(current_clique) == k:
                best_clique = current_clique[:]
                return
            if len(candidates) == 0:
                return
            # Pruning: if len(current_clique) + len(candidates) < k, return
            if len(current_clique) + len(candidates) < k:
                return
                
            for idx, node in enumerate(candidates):
                # Try adding node to clique
                new_clique = current_clique + [node]
                # New candidates are intersection of current candidates (after idx) and neighbors of node
                # Since candidates are already filtered to be neighbors of all nodes in current_clique?
                # Standard Bron-Kerbosch style:
                # candidates passed in are nodes that can be added to current_clique.
                
                # Filter candidates to those that are neighbors of node
                # And index > idx to avoid duplicates
                new_candidates = []
                for next_idx, next_node in enumerate(candidates[idx+1:]):
                    if next_node in adj[node]: # This check is O(n) if adj is list. Use set for adj?
                        new_candidates.append(next_node)
                        
                backtrack(0, new_clique, new_candidates)
                
                if best_clique:
                    return
                    
        # Initial candidates: all nodes
        # But we need to ensure they form a clique.
        # The recursive function above assumes 'candidates' are valid neighbors of ALL nodes in 'current_clique'.
        # So for the first call, current_clique is empty, so candidates can be all nodes.
        # But we should pick the first node to anchor.
        
        # Optimized Backtracking:
        # Pick a node, add to clique, filter neighbors.
        
        def search_clique(clique, candidates):
            nonlocal best_clique
            if time.time() > time_limit:
                return
            if len(clique) == k:
                best_clique = clique[:]
                return
            if len(candidates) == 0:
                return
            if len(clique) + len(candidates) < k:
                return
            
            for i in range(len(candidates)):
                node = candidates[i]
                # New candidates are neighbors of node from candidates[i+1:]
                new_candidates = [c for c in candidates[i+1:] if c in adj_set[node]]
                search_clique(clique + [node], new_candidates)
                if best_clique:
                    return
                    
        # Convert adj to sets for O(1) lookup
        adj_set = [set(nei) for nei in adj]
        
        # Start search with each node as the first element?
        # Or just pass all nodes as candidates and let the loop pick the first one?
        # The loop in search_clique iterates through candidates.
        # So we can just call search_clique([], nodes)
        
        search_clique([], nodes)
        
        if best_clique:
            return [smooths[i] for i in best_clique]
        return None

    for primes in prime_sets:
        if time.time() > deadline:
            break
        smooths = gen_smooth(primes)
        if len(smooths) < k:
            continue
        clique = find_clique(smooths, primes, k, deadline)
        if clique:
            # Verify
            return clique

    # Fallback: Return a set that minimizes primes, even if not perfect clique
    # e.g. powers of 2?
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Sums: 1+2=3 (3), 1+4=5 (5), 1+8=9 (3), 1+16=17 (17)...
    # This introduces many primes.
    
    # Better fallback: Small integers
    return list(range(1, 13))
