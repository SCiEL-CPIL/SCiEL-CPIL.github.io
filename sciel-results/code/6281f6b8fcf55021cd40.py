"""Maximize pairwise-noncoprime subset of [1,255] containing 255."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # We need a subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    # Maximize cardinality.
    
    # Key insight: if we take all multiples of a prime p, they form a clique.
    # Multiples of 3: floor(255/3) = 85 numbers.
    # Multiples of 5: floor(255/5) = 51 numbers.
    # Multiples of 17: floor(255/17) = 15 numbers.
    
    # But we can do better by mixing numbers that share different prime factors,
    # as long as every pair shares at least one common prime factor.
    
    # The candidate set should be numbers in [1,255] that share a factor with 255,
    # i.e., numbers divisible by 3, 5, or 17.
    # Actually, we must include 255. Any number in our set must share a factor with 255
    # (since gcd(a, 255) > 1 is required for every a in the set).
    # So candidates = {x in [1,255] : gcd(x, 255) > 1}
    
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    
    # Precompute adjacency: two candidates are adjacent if gcd > 1
    # Use bitmasks or sets for efficiency
    # n is at most ~127, so sets work fine
    
    # Build adjacency as list of sets
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find maximum clique using greedy + local search
    # Start with the largest single-prime clique (multiples of 3)
    # Then try to add more vertices
    
    # First, find a good initial clique
    # Strategy: start with all multiples of 3 (largest prime factor group)
    clique = set()
    for i, v in enumerate(candidates):
        if v % 3 == 0:
            clique.add(i)
    
    # Verify 255 is in the clique
    idx_255 = candidates.index(255)
    if idx_255 not in clique:
        clique.add(idx_255)
    
    # Now try to add more vertices greedily
    # A vertex can be added if it's adjacent to all current clique members
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in clique:
                continue
            # Check if i is adjacent to all clique members
            if all(j in adj[i] for j in clique):
                clique.add(i)
                changed = True
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Now try local search: swap operations to improve
    # Try removing one vertex and adding one or more others
    best_clique = set(clique)
    best_size = len(clique)
    
    # Iterative improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            if i in best_clique:
                continue
            # Try adding i
            if all(j in adj[i] for j in best_clique):
                best_clique.add(i)
                improved = True
                best_size = len(best_clique)
                break
        # Try swap: remove one, add another
        if not improved:
            for remove_idx in list(best_clique):
                if time.time() > deadline - 5:
                    break
                temp_clique = set(best_clique)
                temp_clique.discard(remove_idx)
                for add_idx in range(n):
                    if add_idx in temp_clique:
                        continue
                    if all(j in adj[add_idx] for j in temp_clique):
                        # This swap adds one vertex
                        if len(temp_clique) + 1 > best_size:
                            best_clique = temp_clique | {add_idx}
                            best_size = len(best_clique)
                            improved = True
                            break
                if improved:
                    break
    
    # Try a more systematic approach: greedy coloring-based upper bound and branch
    # For n~127, we can try a simple branch and bound
    # But first, let's try a better greedy: sort by degree and pick
    
    # Alternative: use the fact that the graph is dense
    # Try multiple greedy starts
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy: shuffle candidates, pick highest degree compatible vertex
        import random
        random.seed(trial)
        
        remaining = set(range(n))
        clique_trial = set()
        
        # Always include 255
        clique_trial.add(idx_255)
        remaining.discard(idx_255)
        
        # Greedy: repeatedly add the vertex with highest degree in remaining
        # that is compatible with all current clique members
        while remaining:
            best_vertex = None
            best_deg = -1
            for v in remaining:
                # Check compatibility
                if not all(u in adj[v] for u in clique_trial):
                    continue
                deg = len(adj[v] & clique_trial)  # number of neighbors in clique
                # Actually we want to maximize future potential
                # Use degree in remaining
                deg_rem = len(adj[v] & remaining)
                if deg_rem > best_deg:
                    best_deg = deg_rem
                    best_vertex = v
            
            if best_vertex is None:
                break
            
            clique_trial.add(best_vertex)
            remaining.discard(best_vertex)
        
        if len(clique_trial) > best_size:
            best_clique = set(clique_trial)
            best_size = len(clique_trial)
    
    # Final local search on best_clique
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        # Try adding any compatible vertex
        for i in range(n):
            if i in best_clique:
                continue
            if all(j in adj[i] for j in best_clique):
                best_clique.add(i)
                improved = True
                best_size = len(best_clique)
        
        # Try swaps
        if not improved:
            for remove_idx in list(best_clique):
                temp_clique = set(best_clique)
                temp_clique.discard(remove_idx)
                for add_idx in range(n):
                    if add_idx in temp_clique:
                        continue
                    if all(j in adj[add_idx] for j in temp_clique):
                        new_size = len(temp_clique) + 1
                        if new_size > best_size:
                            best_clique = temp_clique | {add_idx}
                            best_size = new_size
                            improved = True
                            break
                if improved:
                    break
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is present
    if 255 not in result:
        result.append(255)
    
    return sorted(result)
# EVOLVE-BLOCK-END
