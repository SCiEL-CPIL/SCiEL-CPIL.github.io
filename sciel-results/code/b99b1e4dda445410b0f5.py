import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n_points = 16
    all_points = list(range(n_points))
    all_pairs = list(combinations(all_points, 2))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    total_pairs = len(all_pairs)
    
    # Pre-compute all triples and their pair indices
    all_triples = list(combinations(all_points, 3))
    triple_to_pairs = {}
    for t in all_triples:
        a, b, c = t
        p1 = (a, b)
        p2 = (a, c)
        p3 = (b, c)
        triple_to_pairs[t] = (pair_to_idx[p1], pair_to_idx[p2], pair_to_idx[p3])
        
    # Map pair index to list of triples containing that pair
    pair_to_triples = defaultdict(list)
    for t, pairs in triple_to_pairs.items():
        for p_idx in pairs:
            pair_to_triples[p_idx].append(t)

    # Exact Independence Number Calculation using Bitmask Superset Marking
    # This is O(2^n * T) if naive, but we can optimize.
    # Better: For a given system of T triples, we want the max size of a set S 
    # such that no triple is a subset of S.
    # Equivalent: Find the max size of a set S that does NOT contain any triple from the system.
    # Let's use a DP or simple iteration over masks for n=16.
    # 2^16 = 65536. This is small.
    
    def calc_alpha_exact(system):
        if not system:
            return n_points
        
        # Create a set of triple masks
        triple_masks = set()
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.add(mask)
            
        # We want to find the maximum k such that there exists a mask of size k
        # that does not contain any triple_mask as a subset.
        # A mask M contains triple T if (M & T) == T.
        
        # Optimization: 
        # If we iterate masks by size descending, the first one that is "independent" is the answer.
        # Generating masks of size k is expensive if done repeatedly.
        # Instead, iterate all masks once, track max size of independent mask.
        # 65536 iterations * T checks. If T is ~20, that's 1.3M ops. Very fast.
        
        max_size = 0
        # Pre-compute popcount for all masks? Or use bin.count.
        # bin.count is fast enough in Python for 65k items.
        
        for mask in range(1 << n_points):
            # Check if this mask is independent
            is_indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            if is_indep:
                size = mask.bit_count()
                if size > max_size:
                    max_size = size
                    if max_size == n_points: # Can't get better
                        break
        return max_size

    # Greedy Initialization
    def greedy_init(seed):
        random.seed(seed)
        used_pairs = [False] * total_pairs
        current_triples = []
        shuffled_triples = all_triples[:]
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            p1, p2, p3 = triple_to_pairs[t]
            if not used_pairs[p1] and not used_pairs[p2] and not used_pairs[p3]:
                current_triples.append(t)
                used_pairs[p1] = True
                used_pairs[p2] = True
                used_pairs[p3] = True
        return current_triples

    # Local Search with 1-Swap and 2-Swap
    def local_search(system, max_iters=500):
        current_alpha = calc_alpha_exact(system)
        best_alpha = current_alpha
        best_system = system[:]
        
        used_pairs = [False] * total_pairs
        for a, b, c in system:
            p1, p2, p3 = triple_to_pairs[(a,b,c)]
            used_pairs[p1] = True
            used_pairs[p2] = True
            used_pairs[p3] = True
            
        # Precompute valid triples for current state? 
        # It's better to check on the fly for swaps.
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try 1-Swap moves
            # For each triple in system, try removing it and adding a new one
            # To be efficient, we only check a subset of candidates or random samples
            
            # Strategy: Randomly select a triple to remove, then try to add a random valid triple
            # This is stochastic local search.
            
            # Let's do a more systematic 1-swap: 
            # For each triple t_in in system:
            #   Remove t_in
            #   For each triple t_out not in system:
            #      If t_out is valid with t_in removed:
            #         Calculate alpha
            #         If better, accept
            # This is O(T^2 * AlphaCalc). T~20. 400 * AlphaCalc. 
            # AlphaCalc is ~1ms? 400ms per iteration. 500 iters = 200s. Too slow.
            
            # We need a faster alpha or fewer checks.
            # Let's use a heuristic alpha for intermediate steps, exact for best.
            
            def calc_alpha_heuristic(system):
                # Greedy Vertex Cover approximation?
                # Or just count triples? More triples usually means smaller alpha.
                # Let's return -len(system) as a proxy for ordering, 
                # but we need to track actual alpha for the final result.
                # Actually, let's just do exact alpha but limit the search space.
                pass

            # Given the time constraint, we will use a **Stochastic Local Search** approach.
            # At each step, pick a random triple to remove.
            # Pick a random triple to add (if valid).
            # Calculate exact alpha.
            # If alpha < current_alpha, accept.
            # If alpha == current_alpha, accept with some probability (plateau).
            # If alpha > current_alpha, reject.
            
            # To speed up, we will cache alpha calculations? No, systems change.
            # We will limit the number of alpha calculations per second.
            
            for _ in range(100): # 100 random moves per "iteration"
                if time.time() > deadline - 5.0:
                    return best_system, best_alpha
                
                if not system:
                    break
                    
                # Remove a random triple
                idx_remove = random.randint(0, len(system) - 1)
                t_remove = system[idx_remove]
                p1, p2, p3 = triple_to_pairs[t_remove]
                
                # Temporarily remove
                used_pairs[p1] = False
                used_pairs[p2] = False
                used_pairs[p3] = False
                system.pop(idx_remove)
                
                # Find a valid triple to add
                # Sample random triples until we find a valid one
                t_add = None
                for _ in range(20):
                    candidate = random.choice(all_triples)
                    c1, c2, c3 = triple_to_pairs[candidate]
                    if not used_pairs[c1] and not used_pairs[c2] and not used_pairs[c3]:
                        t_add = candidate
                        break
                
                if t_add is None:
                    # Revert removal
                    system.insert(idx_remove, t_remove)
                    used_pairs[p1] = True
                    used_pairs[p2] = True
                    used_pairs[p3] = True
                    continue
                
                # Add the new triple
                system.append(t_add)
                c1, c2, c3 = triple_to_pairs[t_add]
                used_pairs[c1] = True
                used_pairs[c2] = True
                used_pairs[c3] = True
                
                # Evaluate
                new_alpha = calc_alpha_exact(system)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_system = system[:]
                    improved = True
                elif new_alpha < current_alpha:
                    # Accept improvement
                    current_alpha = new_alpha
                    improved = True
                else:
                    # Revert to previous state (before this move)
                    # Note: We modified 'system' in place. We need to revert to the state BEFORE the move.
                    # This is tricky with in-place modification.
                    # Let's restructure: Keep a copy? No, too slow.
                    # Let's just revert the specific change.
                    system.pop() # Remove t_add
                    used_pairs[c1] = False
                    used_pairs[c2] = False
                    used_pairs[c3] = False
                    system.insert(idx_remove, t_remove)
                    used_pairs[p1] = True
                    used_pairs[p2] = True
                    used_pairs[p3] = True
                    
                    # If we rejected, we don't update current_alpha.
                    # But we broke the "current" state? No, we reverted it.
                    # So current_alpha is still valid for the reverted state.
                    
        return best_system, best_alpha

    # Main Loop: Iterated Local Search
    best_system_overall = None
    best_alpha_overall = 16
    
    seed = 42
    while time.time() < deadline - 10.0:
        # 1. Generate a new starting point (Greedy)
        # Occasionally, we can start from the best found so far with a perturbation
        if best_system_overall is not None and random.random() < 0.5:
            # Perturb the best system
            start_system = best_system_overall[:]
            # Remove k random triples
            k = random.randint(1, 3)
            for _ in range(k):
                if start_system:
                    idx = random.randint(0, len(start_system) - 1)
                    start_system.pop(idx)
            # Re-greedy fill
            used_pairs = [False] * total_pairs
            for a, b, c in start_system:
                p1, p2, p3 = triple_to_pairs[(a,b,c)]
                used_pairs[p1] = True
                used_pairs[p2] = True
                used_pairs[p3] = True
            
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for t in shuffled:
                p1, p2, p3 = triple_to_pairs[t]
                if not used_pairs[p1] and not used_pairs[p2] and not used_pairs[p3]:
                    start_system.append(t)
                    used_pairs[p1] = True
                    used_pairs[p2] = True
                    used_pairs[p3] = True
        else:
            start_system = greedy_init(seed)
            seed += 1
            
        # 2. Local Search
        local_best, local_alpha = local_search(start_system, max_iters=200)
        
        # 3. Update Global Best
        if local_alpha < best_alpha_overall:
            best_alpha_overall = local_alpha
            best_system_overall = local_best[:]
            
        if best_alpha_overall <= 5:
            break

    if best_system_overall is None:
        # Fallback
        return greedy_init(42)
        
    return best_system_overall

# Quick test
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Triples: {len(res)}")
    # Verify linearity
    pairs = set()
    for a,b,c in res:
        p1, p2, p3 = (a,b), (a,c), (b,c)
        for p in [p1, p2, p3]:
            if p in pairs:
                print("Invalid: Duplicate pair", p)
            pairs.add(p)
    # Calculate alpha
    print(f"Alpha: {calc_alpha_exact(res)}")
