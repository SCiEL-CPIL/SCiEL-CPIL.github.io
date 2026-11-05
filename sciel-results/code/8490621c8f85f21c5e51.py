import time
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We aim for n=20.
    # A valid graph on n vertices certifies G(6) > n.
    # Maximize n.
    
    # Known construction for n=20:
    # The graph is the complement of the Petersen graph? No, Petersen has 10 vertices.
    # The graph is the Clebsch graph? No, 16 vertices.
    # The graph is the Schläfli graph? No, 27 vertices.
    
    # Let's use a known 20-vertex graph that avoids regular induced subgraphs of size 6.
    # One such graph is the 20-vertex graph from the OEIS or known extremal graph theory results.
    # Actually, there is a specific construction.
    # Let's try to find one via local search with a fast verifier, but starting from a smart guess.
    
    # Smart guess: The graph is the 20-vertex graph formed by the "20-vertex graph" in the literature.
    # Alternatively, we can use the fact that random graphs with p=0.5 might work if we search enough.
    # But the current program with random search is scoring 2.8, which means it's finding n=20?
    # Wait, the score is value/seed. If seed is 14 (maybe n=14 is the baseline?), and we get 2.8, then value is 39.2?
    # No, the task says "A valid graph on n vertices certifies G(6) > n".
    # So the value is n.
    # If the score is 2.8, and the baseline seed is maybe 14 (since 2.8 * 10 = 28? No, seed is a program).
    # The seed program returns some n. If the seed program returns n=14, and we return n=20, the score is 20/14 ≈ 1.43.
    # But the score is 2.8. This implies the seed program returns n=7? 20/7 ≈ 2.85.
    # Or maybe the seed program returns n=5? 20/5 = 4. No.
    # Maybe the seed program returns n=10? 20/10 = 2. No.
    # Maybe the score is not n/seed_n.
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # "1.0 equals the seed program and above 1.0 is an improvement".
    # So if the seed program returns n_seed, and we return n, the score is n/n_seed.
    # If score is 2.8, and we return n=20, then n_seed = 20/2.8 ≈ 7.14. So seed returns 7?
    # If seed returns 7, and we return 20, score is 2.857.
    # If we return 21, score is 3.0.
    # If we return 22, score is 3.14.
    
    # So the current program is finding n=20.
    # The goal is to find n=21 or n=22.
    # "Graphs with 20 vertices are known; a 21-vertex graph would show G(6) >= 22."
    
    # So we need to find a 21-vertex graph.
    # The current program only searches for n=20 down to 6.
    # We need to search for n=21 and n=22.
    
    # Let's modify the search to try n=22, 21, 20...
    # And use a faster verifier to allow more restarts.
    
    def verify_fast(n, adj_masks):
        """
        Verify if graph with adj_masks has no regular induced subgraph of size >= 6.
        Returns True if valid.
        """
        # Check size 6
        for subset in itertools.combinations(range(n), 6):
            mask_subset = 0
            for v in subset:
                mask_subset |= (1 << v)
            
            degs = []
            for v in subset:
                degs.append((adj_masks[v] & mask_subset).bit_count())
            
            if len(set(degs)) == 1:
                return False
        
        # Check larger sizes
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask_subset).bit_count())
                
                if len(set(degs)) == 1:
                    return False
        
        return True

    best_n = 5
    best_edges = []
    
    import random
    
    # Try n=22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline:
            break
        
        # We use multiple random restarts with different seeds
        found = False
        for seed in range(1000):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            
            adj_masks = [0] * target_n
            # Use p=0.5 for random graph
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Fast check for size 6 only first
            valid_size_6 = True
            for subset in itertools.combinations(range(target_n), 6):
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask_subset).bit_count())
                
                if len(set(degs)) == 1:
                    valid_size_6 = False
                    break
            
            if not valid_size_6:
                continue
            
            # If size 6 is clean, check larger sizes
            if verify_fast(target_n, adj_masks):
                # Found a valid graph!
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
    
    # If we failed to find n=22, 21, or 20, return n=5 (trivially valid)
    # But we should return the best valid graph we found.
    # Since we didn't find any, return n=5.
    
    return {'n': best_n, 'edges': best_edges}
