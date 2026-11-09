import time
import random
import sys
import numpy as np

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts_np(A_set):
    """
    Returns a list of representation counts for sums in [_LO, _HI] using NumPy.
    Returns None if A is empty.
    """
    if not A_set:
        return None
    
    # Create indicator vector
    v = np.zeros(_N + 1, dtype=np.int32)
    for a in A_set:
        v[a] = 1
        
    # Ordered pair counts
    conv = np.convolve(v, v, mode='full')
    
    # Unordered pair counts for k in [0, 256]
    # N_total(k) = (conv[k] + diag(k)) / 2
    # diag(k) is 1 if k is even and k//2 is in A, else 0
    
    counts = np.zeros(_RANGE, dtype=np.int32)
    
    # We only need [128, 256]
    # diag(k) check
    for i in range(_RANGE):
        k = _LO + i
        c_ord = conv[k]
        # Check if k/2 is in A
        if k % 2 == 0:
            mid = k // 2
            if 0 <= mid <= _N and v[mid] == 1:
                c_diag = 1
            else:
                c_diag = 0
        else:
            c_diag = 0
            
        counts[i] = (c_ord + c_diag) // 2
        
    return counts

def _is_valid(counts):
    if counts is None:
        return False
    return np.all(counts >= 1)

def _get_peak(counts):
    if counts is None:
        return float('inf')
    return np.max(counts)

def _greedy_construct(rng, time_limit):
    """
    Construct a valid set using greedy addition.
    """
    A = {0, 128}
    counts = _get_counts_np(A)
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        if counts is not None and _is_valid(counts):
            # Try to peel
            break
            
        # Find best element to add
        best_elem = None
        best_peak = float('inf')
        best_counts = None
        
        # To speed up, we can't try all 129 elements with full count every time if it's too slow.
        # But 129 * (cost of count) is manageable.
        # Cost of count is ~129^2 = 16k ops. 129 * 16k = 2M ops. Very fast.
        
        candidates = [x for x in range(_N + 1) if x not in A]
        if not candidates:
            break
            
        # Randomize order to avoid bias
        rng.shuffle(candidates)
        
        for x in candidates:
            new_A = A | {x}
            new_counts = _get_counts_np(new_A)
            if new_counts is None:
                continue
            # We want to minimize peak, but also ensure we are making progress towards validity.
            # If not valid, we prioritize covering missing sums.
            # Let's define a score: 
            # If valid: minimize peak.
            # If invalid: minimize number of uncovered sums, then minimize peak.
            
            if _is_valid(new_counts):
                p = _get_peak(new_counts)
                # Prefer valid sets with lower peak
                if p < best_peak:
                    best_peak = p
                    best_elem = x
                    best_counts = new_counts
            else:
                # Count uncovered
                uncovered = np.sum(new_counts == 0)
                # We want to minimize uncovered. 
                # If we find one that covers more, we might take it even if peak is high?
                # For now, let's just look for valid ones first. If none, pick one that minimizes uncovered.
                # This logic is complex. Let's stick to: 
                # 1. If any candidate yields a valid set, pick the one with min peak.
                # 2. If no candidate yields a valid set, pick the one that minimizes (uncovered, peak).
                
                # We'll store the best invalid candidate separately
                pass
        
        if best_elem is not None:
            A.add(best_elem)
            counts = best_counts
            continue
            
        # If no valid candidate found, pick the one that minimizes uncovered count
        best_invalid_elem = None
        best_uncovered = float('inf')
        best_invalid_peak = float('inf')
        best_invalid_counts = None
        
        for x in candidates:
            new_A = A | {x}
            new_counts = _get_counts_np(new_A)
            if new_counts is None:
                continue
            uncovered = np.sum(new_counts == 0)
            peak = _get_peak(new_counts)
            
            if uncovered < best_uncovered or (uncovered == best_uncovered and peak < best_invalid_peak):
                best_uncovered = uncovered
                best_invalid_peak = peak
                best_invalid_elem = x
                best_invalid_counts = new_counts
                
        if best_invalid_elem is not None:
            A.add(best_invalid_elem)
            counts = best_invalid_counts
        else:
            break
            
    return A, counts

def _greedy_peel(A_init, time_limit, rng):
    """
    Greedily remove elements from A_init to minimize peak while maintaining validity.
    """
    A = set(A_init)
    counts = _get_counts_np(A)
    
    if not _is_valid(counts):
        return A, counts
        
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        current_peak = _get_peak(counts)
        
        # Try removing each element
        # Randomize order
        elems = list(A)
        rng.shuffle(elems)
        
        for x in elems:
            if time.time() - start_time > time_limit:
                break
            if x not in A:
                continue
                
            new_A = A - {x}
            if not new_A:
                continue
                
            new_counts = _get_counts_np(new_A)
            if new_counts is None:
                continue
                
            if _is_valid(new_counts):
                new_peak = _get_peak(new_counts)
                # Accept if peak decreases or (peak same and size decreases - implicit)
                if new_peak < current_peak:
                    A = new_A
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    # Restart loop as A changed
                    break
                    
    return A, counts

def _simulated_annealing(A_init, time_limit, rng):
    """
    Simulated annealing to refine the set.
    """
    A = set(A_init)
    counts = _get_counts_np(A)
    
    if not _is_valid(counts):
        return A, counts
        
    current_peak = _get_peak(counts)
    best_A = A.copy()
    best_counts = counts.copy()
    best_peak = current_peak
    
    start_time = time.time()
    T = 5.0 # Initial temperature
    
    while time.time() - start_time < time_limit:
        # Cooldown
        T *= 0.99
        
        # Generate move
        move_type = rng.choice(['swap', 'add', 'remove'])
        
        in_A = list(A)
        not_in_A = [x for x in range(_N + 1) if x not in A]
        
        new_A = A.copy()
        
        if move_type == 'swap':
            if not in_A or not not_in_A:
                continue
            x = in_A[rng.randint(0, len(in_A) - 1)]
            y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
            new_A.discard(x)
            new_A.add(y)
        elif move_type == 'add':
            if not not_in_A:
                continue
            y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
            new_A.add(y)
        elif move_type == 'remove':
            if not in_A:
                continue
            x = in_A[rng.randint(0, len(in_A) - 1)]
            new_A.discard(x)
            
        if not new_A:
            continue
            
        new_counts = _get_counts_np(new_A)
        if new_counts is None:
            continue
            
        if not _is_valid(new_counts):
            # Reject invalid moves in SA? Or penalize?
            # Let's reject invalid moves to keep search space manageable.
            continue
            
        new_peak = _get_peak(new_counts)
        
        delta = new_peak - current_peak
        
        if delta <= 0:
            A = new_A
            counts = new_counts
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = A.copy()
                best_counts = counts.copy()
        else:
            # Accept with probability
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    A = new_A
                    counts = new_counts
                    current_peak = new_peak
                    
    return best_A, best_counts

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0 # 5s margin
    
    rng = random.Random(42)
    
    best_A_global = None
    best_peak_global = float('inf')
    
    # 1. Greedy Construction
    A_greedy, counts_greedy = _greedy_construct(rng, 5.0)
    if counts_greedy is not None and _is_valid(counts_greedy):
        peak = _get_peak(counts_greedy)
        if peak < best_peak_global:
            best_peak_global = peak
            best_A_global = A_greedy.copy()
            
    # 2. Greedy Peeling on the constructed set
    if best_A_global is not None:
        A_peeled, counts_peeled = _greedy_peel(best_A_global, 5.0, rng)
        if counts_peeled is not None and _is_valid(counts_peeled):
            peak = _get_peak(counts_peeled)
            if peak < best_peak_global:
                best_peak_global = peak
                best_A_global = A_peeled.copy()
                
    # 3. Simulated Annealing from best found so far
    if best_A_global is not None:
        remaining = DEADLINE - time.time()
        if remaining > 1.0:
            A_sa, counts_sa = _simulated_annealing(best_A_global, remaining - 1.0, rng)
            if counts_sa is not None and _is_valid(counts_sa):
                peak = _get_peak(counts_sa)
                if peak < best_peak_global:
                    best_peak_global = peak
                    best_A_global = A_sa.copy()
                    
    # Fallback if nothing worked (should not happen)
    if best_A_global is None:
        return list(range(64, 129))
        
    return sorted(best_A_global)
