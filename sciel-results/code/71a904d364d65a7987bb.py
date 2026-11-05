"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys
import numpy as np
from collections import defaultdict

def _triple_sums_involving_x(x, S):
    """Return the set of triple sums that involve x, given x is NOT in S."""
    n = len(S)
    if n == 0:
        return {3 * x}
    S_arr = np.array(S, dtype=np.int32)
    # 3x, 2x+y, x+2y for all y in S, x+S[i]+S[j] for i<j
    parts = [np.array([3 * x], dtype=np.int32)]
    parts.append(2 * x + S_arr)
    parts.append(x + 2 * S_arr)
    # Pairwise sums
    for i in range(n):
        parts.append(x + S[i] + S_arr[i+1:])
    new_sums = np.concatenate(parts)
    return set(new_sums.tolist())

def _check_candidate_vectorized(x, S_arr, existing_bitset, max_sum):
    """Fast vectorized check if x can be added to S. Returns (valid, new_sums_arr)."""
    n = len(S_arr)
    if n == 0:
        s = 3 * x
        if s > max_sum:
            return False, None
        if existing_bitset[s]:
            return False, None
        return True, np.array([s], dtype=np.int32)
    
    # Build all new sums vectorized
    parts = [np.array([3 * x], dtype=np.int32)]
    parts.append(2 * x + S_arr)
    parts.append(x + 2 * S_arr)
    # Pairwise: x + S[i] + S[j] for i < j
    # Use broadcasting: S_arr[:, None] + S_arr[None, :], then take upper triangle
    if n >= 2:
        pair_sums = x + S_arr[:, None] + S_arr[None, :]
        # Take upper triangle indices
        iu, ju = np.triu_indices(n, k=1)
        pair_sums_flat = pair_sums[iu, ju]
        parts.append(pair_sums_flat)
    
    new_sums_arr = np.concatenate(parts)
    
    # Check range
    if np.any(new_sums_arr < 0) or np.any(new_sums_arr > max_sum):
        return False, None
    
    # Check internal collisions
    if len(new_sums_arr) != len(np.unique(new_sums_arr)):
        return False, None
    
    # Check against existing
    if np.any(existing_bitset[new_sums_arr] == 1):
        return False, None
    
    return True, new_sums_arr

def _greedy_build_bitset(deadline, seed=0, order=None):
    """Greedy build using a numpy bitset for fast collision checks."""
    max_val = 4096
    max_sum = 3 * max_val
    existing = np.zeros(max_sum + 1, dtype=np.uint8)
    S = []
    
    if order is None:
        order = list(range(1, max_val + 1))
    
    S_arr = np.empty(max_val, dtype=np.int32)
    n = 0
    
    for x in order:
        if time.time() > deadline:
            break
        
        if n == 0:
            s = 3 * x
            if s > max_sum:
                continue
            S_arr[0] = x
            existing[s] = 1
            S.append(x)
            n = 1
            continue
        
        valid, new_sums_arr = _check_candidate_vectorized(x, S_arr[:n], existing, max_sum)
        if not valid:
            continue
        
        # Accept x
        existing[new_sums_arr] = 1
        S_arr[n] = x
        S.append(x)
        n += 1
    
    return S

def _compute_all_triple_sums(S):
    """Compute all triple sums for set S. Returns a dict mapping sum -> list of triples."""
    sums_dict = defaultdict(list)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                sums_dict[s].append((i, j, k))
    return sums_dict

def _local_search_fast(S, deadline, seed=0):
    """Fast hill-climbing using vectorized collision checks."""
    rng = random.Random(seed)
    best = S[:]
    best_size = len(S)
    max_val = 4096
    max_sum = 3 * max_val
    
    # Build bitset of current sums
    existing = np.zeros(max_sum + 1, dtype=np.uint8)
    S_arr = np.array(best, dtype=np.int32)
    n = len(best)
    
    if n == 0:
        return best
    
    # Mark all current triple sums
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = best[i] + best[j] + best[k]
                if s <= max_sum:
                    existing[s] = 1
    
    # Try adding elements
    added = False
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
        if x in best:
            continue
        
        valid, new_sums_arr = _check_candidate_vectorized(x, S_arr, existing, max_sum)
        if valid:
            existing[new_sums_arr] = 1
            best.append(x)
            S_arr = np.append(S_arr, x)
            n += 1
            best_size = n
            added = True
    
    # Try swaps: remove one, add another
    for _ in range(50):
        if time.time() > deadline:
            break
        
        if n == 0:
            break
        
        idx = rng.randint(0, n - 1)
        removed = best[idx]
        
        # Remove element and rebuild sums
        S_without = best[:idx] + best[idx+1:]
        n_w = len(S_without)
        existing_w = np.zeros(max_sum + 1, dtype=np.uint8)
        
        if n_w > 0:
            Sw_arr = np.array(S_without, dtype=np.int32)
            for i in range(n_w):
                for j in range(i, n_w):
                    for k in range(j, n_w):
                        s = S_without[i] + S_without[j] + S_without[k]
                        if s <= max_sum:
                            existing_w[s] = 1
            
            # Try to find a replacement
            for x in range(1, max_val + 1):
                if time.time() > deadline:
                    break
                if x in S_without:
                    continue
                
                valid, new_sums_arr = _check_candidate_vectorized(x, Sw_arr, existing_w, max_sum)
                if valid:
                    existing_w[new_sums_arr] = 1
                    best = S_without + [x]
                    best_size = n_w + 1
                    S_arr = np.array(best, dtype=np.int32)
                    n = best_size
                    # Rebuild full existing bitset
                    existing = np.zeros(max_sum + 1, dtype=np.uint8)
                    for i in range(n):
                        for j in range(i, n):
                            for k in range(j, n):
                                s = best[i] + best[j] + best[k]
                                if s <= max_sum:
                                    existing[s] = 1
                    break
    
    return best

def _greedy_with_randomization(deadline, seed=0):
    """Greedy with randomized ordering using bitset for speed."""
    rng = random.Random(seed)
    best = []
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        strategy = attempt % 6
        if strategy == 0:
            order = list(range(1, 4097))
            rng.shuffle(order)
        elif strategy == 1:
            order = list(range(1, 4097))
            small = order[:30]
            large = order[30:]
            rng.shuffle(large)
            order = small + large
        elif strategy == 2:
            start = rng.randint(1, 300)
            order = list(range(start, 4097))
            remaining = list(range(1, start))
            rng.shuffle(remaining)
            order = remaining + order
        elif strategy == 3:
            # Exponential spacing bias
            order = list(range(1, 4097))
            rng.shuffle(order)
            # Sort by a random weight to create non-uniform density
            weights = [rng.expovariate(1.0) for _ in order]
            order = [x for _, x in sorted(zip(weights, order))]
        elif strategy == 4:
            # Two-phase: small dense, then sparse large
            order = list(range(1, 4097))
            small = order[:100]
            rng.shuffle(small)
            large = order[100:]
            rng.shuffle(large)
            # Interleave
            new_order = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if si < len(small) and (li >= len(large) or rng.random() < 0.6):
                    new_order.append(small[si])
                    si += 1
                else:
                    new_order.append(large[li])
                    li += 1
            order = new_order
        else:
            # Geometric progression bias
            order = list(range(1, 4097))
            # Create groups with geometric spacing
            groups = []
            base = 1
            while base <= 4096:
                group = [x for x in order if base <= x < base * 2]
                if group:
                    rng.shuffle(group)
                    groups.append(group)
                base *= 2
            order = []
            for g in groups:
                order.extend(g)
        
        S = _greedy_build_bitset(deadline, seed=seed + attempt, order=order)
        
        if len(S) > len(best):
            best = S[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8 second safety margin
    
    best = []
    
    # Strategy 1: Greedy in increasing order (bitset version)
    S1 = _greedy_build_bitset(deadline, seed=42)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Randomized greedy with diverse orderings
    if time.time() < deadline:
        S2 = _greedy_with_randomization(deadline, seed=456)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Fast local search to improve
    if time.time() < deadline:
        S3 = _local_search_fast(best, deadline, seed=123)
        if len(S3) > len(best):
            best = S3[:]
    
    # Strategy 4: More randomized greedy attempts with time remaining
    remaining_time = deadline - time.time()
    if remaining_time > 8:
        S4 = _greedy_with_randomization(deadline, seed=789)
        if len(S4) > len(best):
            best = S4[:]
    
    # Final local search on best
    if time.time() < deadline:
        S5 = _local_search_fast(best, deadline, seed=321)
        if len(S5) > len(best):
            best = S5[:]
    
    return best
