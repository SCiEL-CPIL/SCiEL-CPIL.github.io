import time
import numpy as np
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_dissociated_subset_float(S, recips=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals using floats.
    Fast, but susceptible to FP errors. Used for search.
    """
    if not S:
        return True

    if recips is None:
        reciprocals = [1.0 / s for s in S]
    else:
        reciprocals = [recips[s] for s in S]

    sums = {0.0}
    for r in reciprocals:
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums

    return True

def _is_dissociated_subset_exact(S):
    """
    Checks if a set of integers S is dissociated in the reciprocals using exact fractions.
    Slower, but correct. Used for final verification.
    """
    if not S:
        return True
    
    reciprocals = [Fraction(1, s) for s in S]
    sums = {Fraction(0)}
    for r in reciprocals:
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _local_search_advanced(S, recips, deadline, max_iters=10000):
    """
    Advanced local search with kicks.
    """
    S = list(S)
    best = S[:]
    used = set(S)
    universe = list(range(1, 23))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        
        # Phase 1: Try adding any unused element
        for x in universe:
            if x in used:
                continue
            new_set = S + [x]
            if _is_dissociated_subset_float(new_set, recips):
                S = new_set
                used.add(x)
                improved = True
                if len(S) > len(best):
                    best = S[:]
                break  # restart scan after improvement

        if improved:
            continue

        # Phase 2: Try swap moves (remove one, add one)
        for i, x in enumerate(S):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i+1:]
            used.discard(x)
            for y in universe:
                if y in used:
                    continue
                new_set = S_without + [y]
                if _is_dissociated_subset_float(new_set, recips):
                    S = new_set
                    used.discard(x)
                    used.add(y)
                    improved = True
                    if len(S) > len(best):
                        best = S[:]
                    break
            if improved:
                break
            # restore x
            S = S_without + [x]
            used.add(x)

        if not improved:
            # Phase 3: Kick - remove 2 random elements and try to re-add
            if len(S) > 4:
                # Pick 2 random elements to remove
                kick_indices = random.sample(range(len(S)), 2)
                kick_indices.sort(reverse=True)
                for idx in kick_indices:
                    x = S.pop(idx)
                    used.discard(x)
                
                # Try to re-add elements greedily
                for x in universe:
                    if x in used:
                        continue
                    new_set = S + [x]
                    if _is_dissociated_subset_float(new_set, recips):
                        S = new_set
                        used.add(x)
                
                if len(S) > len(best):
                    best = S[:]
            else:
                # If set is small, just break or random restart
                break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Greedy with various orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    max_greedy_randoms = 50
    for _ in range(max_greedy_randoms):
        if time.time() > deadline - 30:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
        
    for order in orders:
        if time.time() > deadline - 30:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            new_set = current_set + [x]
            if _is_dissociated_subset_float(new_set, recips):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)

    # Strategy 2: Local Search from best greedy result
    if time.time() < deadline - 10:
        # Run local search on the best set found so far
        search_set = _local_search_advanced(best_set, recips, deadline - 10)
        if len(search_set) > best_size:
            best_set = search_set
            best_size = len(best_set)
            
        # Run local search on a few random good seeds
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Create a random seed of size ~10
            seed_size = min(10, best_size)
            seed = random.sample(universe, seed_size)
            # Quick check if seed is valid, if not, try to fix it or just use it as a start for local search?
            # Local search can handle invalid starts? No, it assumes valid.
            # So we need a valid start. Let's use greedy to make a valid start of size ~10.
            current_set = []
            used = set()
            for x in universe:
                if len(current_set) >= 10:
                    break
                new_set = current_set + [x]
                if _is_dissociated_subset_float(new_set, recips):
                    current_set = new_set
                    used.add(x)
            
            if len(current_set) < 8:
                continue
                
            search_set = _local_search_advanced(current_set, recips, deadline - 5)
            if len(search_set) > best_size:
                best_set = search_set
                best_size = len(best_set)

    # Strategy 3: Random Sampling for larger sizes
    # If we haven't found a very large set, try random sampling
    if best_size < 18:
        max_target = min(22, best_size + 2)
        for target_size in range(max_target, best_size, -1):
            if time.time() > deadline - 2:
                break
            
            attempts = 0
            max_attempts = 5000 if target_size <= 15 else 1000
            
            while attempts < max_attempts:
                if time.time() > deadline - 2:
                    break
                
                candidate = random.sample(universe, target_size)
                if _is_dissociated_subset_float(candidate, recips):
                    # Verify with exact arithmetic
                    if _is_dissociated_subset_exact(candidate):
                        best_set = candidate
                        best_size = target_size
                        break
                attempts += 1
            
            if len(best_set) == target_size:
                break

    # Final Verification
    if best_set and not _is_dissociated_subset_exact(best_set):
        # Fallback to a known good set if the best found is invalid due to FP issues
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
