"""Optimized program for sumset/difference-set exponent lower bound."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the lower bound score for a given set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute U+U and U-U
    U_arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairwise sums
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    # Difference set: all pairwise differences
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0 or size_diff == 0:
        return 0.0
    
    # Score: 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0
    
    log_ratio = np.log(ratio)
    log_denom = np.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
    
    score = 1.0 + log_ratio / log_denom
    return float(score)


def _greedy_construct(target_size=50, time_limit=100.0):
    """
    Greedily construct a set U starting with {0} to maximize the score.
    Uses local search to improve.
    """
    start_time = time.time()
    
    # Start with a small seed
    U = [0]
    
    # Try to build up the set greedily
    # At each step, try adding each candidate element and keep the one that improves score most
    candidates_max = 500  # max element to consider
    
    for step in range(1, target_size):
        if time.time() - start_time > time_limit - 10:
            break
        
        best_elem = None
        best_score = _score_set(U)
        
        # Try candidates in a smart order
        # Prefer elements that create good structure
        # Try multiples and combinations
        candidates = set()
        
        # Add some structured candidates
        if len(U) >= 2:
            max_u = max(U)
            # Try max_u + 1, 2*max_u, etc.
            candidates.add(max_u + 1)
            candidates.add(2 * max_u)
            candidates.add(2 * max_u + 1)
            # Try elements between existing ones
            for u in U:
                candidates.add(u + 1)
                candidates.add(u * 2)
                candidates.add(u * 2 + 1)
        
        # Also try random candidates
        for _ in range(20):
            candidates.add(int(np.random.randint(1, candidates_max + 1)))
        
        candidates.discard(0)
        candidates = [c for c in candidates if c not in U and c <= candidates_max]
        
        if not candidates:
            candidates_max *= 2
            continue
        
        # Try each candidate
        for c in candidates:
            if time.time() - start_time > time_limit - 10:
                break
            new_U = U + [c]
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_elem = c
        
        if best_elem is not None:
            U.append(best_elem)
        else:
            # Try expanding the range
            candidates_max *= 2
            if candidates_max > 10000:
                break
    
    # Local search: try swapping elements
    if time.time() - start_time < time_limit - 20:
        U = _local_search(U, time_limit - 20 - (time.time() - start_time))
    
    return sorted(U)


def _local_search(U, time_budget=50.0):
    """Local search to improve the set by swapping or adding/removing elements."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_set(U)
    
    max_iterations = 1000
    iteration = 0
    
    while time.time() - start_time < time_budget and iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Try adding a new element
        max_u = max(U) if U else 0
        candidates = set()
        
        # Structured candidates
        for u in U:
            candidates.add(u + 1)
            candidates.add(u * 2)
            candidates.add(u * 2 + 1)
            if u > 0:
                candidates.add(u - 1)
        
        candidates.add(max_u + 1)
        candidates.add(2 * max_u)
        candidates.add(2 * max_u + 1)
        
        # Random candidates
        for _ in range(10):
            candidates.add(int(np.random.randint(1, max_u * 2 + 10)))
        
        candidates.discard(0)
        candidates = [c for c in candidates if c not in U]
        
        # Try additions
        best_add = None
        best_add_score = current_score
        for c in candidates:
            if time.time() - start_time > time_budget:
                break
            new_U = U + [c]
            s = _score_set(new_U)
            if s > best_add_score + 1e-12:
                best_add_score = s
                best_add = c
        
        if best_add is not None:
            U.append(best_add)
            U = sorted(U)
            current_score = best_add_score
            improved = True
        
        # Try removals (if set is large)
        if len(U) > 5 and time.time() - start_time < time_budget - 5:
            best_remove = None
            best_remove_score = current_score
            for i, u in enumerate(U):
                if u == 0:  # Can't remove 0
                    continue
                if time.time() - start_time > time_budget:
                    break
                new_U = U[:i] + U[i+1:]
                s = _score_set(new_U)
                if s > best_remove_score + 1e-12:
                    best_remove_score = s
                    best_remove = i
            
            if best_remove is not None:
                U = U[:best_remove] + U[best_remove+1:]
                current_score = best_remove_score
                improved = True
        
        # Try swaps
        if len(U) > 3 and time.time() - start_time < time_budget - 5:
            best_swap = None
            best_swap_score = current_score
            # Only try a few swaps
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                for j in range(i+1, len(U)):
                    if U[j] == 0:
                        continue
                    if time.time() - start_time > time_budget:
                        break
                    # Try swapping U[i] and U[j] with nearby values
                    for delta in [-2, -1, 1, 2]:
                        new_val = U[i] + delta
                        if new_val <= 0 or new_val in U:
                            continue
                        new_U = U[:]
                        new_U[i] = new_val
                        new_U = sorted(set(new_U))
                        s = _score_set(new_U)
                        if s > best_swap_score + 1e-12:
                            best_swap_score = s
                            best_swap = (i, new_val)
                    if time.time() - start_time > time_budget:
                        break
                if time.time() - start_time > time_budget:
                    break
            
            if best_swap is not None:
                i, new_val = best_swap
                U[i] = new_val
                U = sorted(set(U))
                current_score = best_swap_score
                improved = True
        
        if not improved:
            # Try random restarts within local neighborhood
            if time.time() - start_time < time_budget - 5 and len(U) > 3:
                # Perturb randomly
                U_perturbed = U[:]
                idx = int(np.random.randint(1, len(U_perturbed)))
                U_perturbed[idx] = max(1, U_perturbed[idx] + int(np.random.randint(-5, 6)))
                U_perturbed = sorted(set(U_perturbed))
                if 0 in U_perturbed:
                    s = _score_set(U_perturbed)
                    if s > current_score + 1e-12:
                        U = U_perturbed
                        current_score = s
            else:
                break
    
    return sorted(U)


def _structured_construct():
    """
    Try a structured construction based on known results.
    The paper mentions sizes 2003 and 54265.
    Try a geometric progression or similar structure.
    """
    # Try a set based on powers of 2 with some structure
    # U = {0} ∪ {2^i + j : i in range, j in range}
    # This creates a large difference set relative to sumset
    
    # Strategy: use a "thin" set that has many differences but few sums
    # A set like {0, 1, 3, 7, 15, ...} (2^i - 1) has good properties
    
    # Let's try: U = {2^i : i = 0, 1, ..., k} ∪ {2^i - 1 : i = 1, 2, ..., k}
    # This gives a set where differences are diverse but sums are constrained
    
    best_U = [0]
    best_score = 0.0
    
    # Try different structured families
    for k in range(1, 20):
        if time.time() - _construct_start_time > 120:
            break
        
        # Family 1: Powers of 2
        U1 = [0] + [2**i for i in range(k)]
        s1 = _score_set(U1)
        if s1 > best_score:
            best_score = s1
            best_U = U1[:]
        
        # Family 2: 2^i - 1
        U2 = [0] + [2**i - 1 for i in range(1, k+1)]
        s2 = _score_set(U2)
        if s2 > best_score:
            best_score = s2
            best_U = U2[:]
        
        # Family 3: Mixed
        U3 = [0] + [2**i for i in range(k)] + [2**i - 1 for i in range(1, k+1)]
        U3 = sorted(set(U3))
        s3 = _score_set(U3)
        if s3 > best_score:
            best_score = s3
            best_U = U3[:]
        
        # Family 4: Geometric with ratio 3
        U4 = [0] + [3**i for i in range(k)]
        s4 = _score_set(U4)
        if s4 > best_score:
            best_score = s4
            best_U = U4[:]
    
    return best_U


_construct_start_time = time.time()

def construct():
    """
    Main entry point. Returns a list of nonnegative integers containing 0.
    """
    global _construct_start_time
    _construct_start_time = time.time()
    deadline = _construct_start_time + 120.0
    
    # Start with structured construction
    U = _structured_construct()
    current_score = _score_set(U)
    
    # Check time
    elapsed = time.time() - _construct_start_time
    if elapsed > deadline - 30:
        return U
    
    # Try greedy construction with remaining time
    time_for_greedy = min(30.0, deadline - time.time() - 10)
    if time_for_greedy > 5:
        U_greedy = _greedy_construct(target_size=30, time_limit=time_for_greedy)
        s_greedy = _score_set(U_greedy)
        if s_greedy > current_score:
            U = U_greedy
            current_score = s_greedy
    
    # Final local search with remaining time
    remaining = deadline - time.time() - 5
    if remaining > 5:
        U = _local_search(U, time_budget=remaining)
    
    # Ensure 0 is in the set
    if 0 not in U:
        U.append(0)
    
    return sorted(set(U))

# EVOLVE-BLOCK-END
