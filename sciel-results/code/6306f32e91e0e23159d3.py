"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
from fractions import Fraction

# Precompute all subsets for each cardinality, verify all 2^n subset sums (as Fractions) are distinct.
# n=22, max subsets per n is C(22,11)=705432. For each subset, checking 2^k sums.
# We'll do a BFS/DFS by cardinality from high to low, verifying candidates.

# To speed up verification: represent subset sums using integer arithmetic with a common denominator.
# The LCM of 1..22 is 232792560. Using this as denominator, each 1/i becomes LCM//i.
# Then subset sums are integers, and we just check for collisions in a set.

_LCM = 232792560

def _verify_recip_dissociated(subset):
    """Return True if all 2^len(subset) reciprocal subset sums are distinct."""
    k = len(subset)
    if k == 0:
        return True
    # Map each element to its scaled reciprocal
    scaled = [_LCM // x for x in subset]
    # Generate all subset sums via bitmasks
    n = 1 << k
    sums = [0] * n
    # Fast subset sum enumeration using DP-like approach
    for mask in range(1, n):
        # lowest set bit
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    # Check for duplicates
    return len(set(sums)) == n

def _find_collision(subset):
    """Find two different subsets with the same sum, or None if dissociated."""
    k = len(subset)
    if k == 0:
        return None
    scaled = [_LCM // x for x in subset]
    n = 1 << k
    # Use DP to compute all subset sums efficiently
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    # Check for duplicates
    seen = {}
    for mask in range(n):
        s = sums[mask]
        if s in seen:
            return mask, seen[s]
        seen[s] = mask
    return None

def _greedy_remove_initial(current, deadline, rng):
    """Greedy removal approach starting from a given set."""
    current = list(current)
    while True:
        if time.time() > deadline - 5:
            break
        col = _find_collision(current)
        if col is None:
            break
        mask1, mask2 = col
        # Find elements in symmetric difference
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        # Heuristic: remove a random element from sym_diff to escape local minima
        to_remove_idx = rng.choice(sym_diff)
        current.pop(to_remove_idx)
    return current

def _random_greedy_add(deadline, rng):
    """Randomized greedy addition approach."""
    universe = list(range(1, 23))
    rng.shuffle(universe)
    current = []
    for x in universe:
        if time.time() > deadline - 5:
            break
        candidate = current + [x]
        if _verify_recip_dissociated(candidate):
            current = candidate
    return current

def _polish_set(best, deadline, rng):
    """Attempt to improve the set by adding elements or swapping."""
    best = list(best)
    in_set = set(best)
    not_in_set = [x for x in range(1, 23) if x not in in_set]
    
    # Phase 1: Add any possible element
    improved = True
    while improved:
        improved = False
        for x in not_in_set[:]:
            if time.time() > deadline - 5:
                return best
            candidate = best + [x]
            if _verify_recip_dissociated(candidate):
                best = candidate
                in_set = set(best)
                not_in_set = [x for x in range(1, 23) if x not in in_set]
                improved = True
                break # Restart loop with new state
    
    # Phase 2: Swap and Add (1-swap + 1-add)
    if time.time() > deadline - 10:
        return best
        
    found_swap = True
    while found_swap:
        found_swap = False
        if time.time() > deadline - 5:
            return best
        in_set = set(best)
        not_in_set = [x for x in range(1, 23) if x not in in_set]
        
        for i_idx, i_val in enumerate(best):
            if time.time() > deadline - 5:
                return best
            for j_val in not_in_set:
                # Try swap
                candidate_swap = best[:]
                candidate_swap[i_idx] = j_val
                # Check if swap itself is valid
                if not _verify_recip_dissociated(candidate_swap):
                    continue
                
                # If swap is valid, try to add any element from (not_in_set - {j_val})
                potential_adds = [x for x in not_in_set if x != j_val]
                
                for k_val in potential_adds:
                    candidate_add = candidate_swap + [k_val]
                    if _verify_recip_dissociated(candidate_add):
                        # Accept swap and add
                        best = candidate_add
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
    # Phase 3: 2-Swap moves to explore landscape
    # Try to swap two elements in best with two elements in not_in_set
    # This maintains size but changes the set, potentially allowing future additions.
    if time.time() > deadline - 5:
        return best
        
    # Limit 2-swaps to avoid excessive time
    max_2swaps = 50
    for _ in range(max_2swaps):
        if time.time() > deadline - 5:
            break
        in_set = set(best)
        not_in_set = [x for x in range(1, 23) if x not in in_set]
        if len(not_in_set) < 2 or len(best) < 2:
            break
        # Pick 2 random from best, 2 random from not_in_set
        i1, i2 = rng.sample(range(len(best)), 2)
        j1, j2 = rng.sample(not_in_set, 2)
        
        candidate = best[:]
        candidate[i1] = j1
        candidate[i2] = j2
        
        if _verify_recip_dissociated(candidate):
            # Accept if it allows an addition?
            in_set_c = set(candidate)
            not_in_set_c = [x for x in range(1, 23) if x not in in_set_c]
            for k_val in not_in_set_c:
                if _verify_recip_dissociated(candidate + [k_val]):
                    best = candidate + [k_val]
                    break
            else:
                # No immediate addition, but maybe this state is better for future?
                # We'll only accept if we can add, to ensure progress.
                pass

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    universe = list(range(1, 23))
    rng = random.Random(42)
    
    # Strategy 1: Greedy Remove from Full (Randomized)
    best = _greedy_remove_initial(list(universe), deadline, rng)
    
    # Strategy 2: Random Greedy Add (Multiple Restarts)
    # Run several times to find a better local optimum
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        candidate = _random_greedy_add(deadline, rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 3: Polish the best result
    best = _polish_set(best, deadline, rng)
    
    # Final safety check
    if not _verify_recip_dissociated(best):
        # Fallback to a known small set if something went wrong
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best
