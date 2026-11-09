import math
import time
import random
import numpy as np

def _score(U):
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Calculate sumset and difference set
    plus_set = set()
    minus_set = set()
    for a in U:
        for b in U:
            plus_set.add(a + b)
            minus_set.add(a - b)
            
    n_plus = len(plus_set)
    n_minus = len(minus_set)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _score_fast(U, max_val):
    """
    Fast score calculation using FFT-based convolution.
    U is a sorted list of unique non-negative integers containing 0.
    max_val is an upper bound on the maximum element in U.
    """
    U = sorted(list(set(U)))
    if 0 not in U or len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    n = len(U)
    # Size of FFT needed for sumset: max sum is 2*mx, so need size > 2*mx
    # For difference set: min diff is -mx, max diff is mx, so range is 2*mx+1
    # We can compute sumset via convolution of indicator vector with itself
    # And difference set via convolution of indicator vector with reversed indicator vector
    
    # Build indicator vector
    # For sumset: a + b where a, b in U. Max value is 2*mx.
    # For diffset: a - b where a, b in U. Range is [-mx, mx].
    
    # Use FFT size that is a power of 2 for efficiency
    size_sum = 1
    while size_sum <= 2 * mx:
        size_sum *= 2
    
    # Indicator vector for U
    ind = np.zeros(size_sum, dtype=np.float64)
    for x in U:
        if x < size_sum:
            ind[x] = 1.0
    
    # Sumset: convolution of ind with ind
    # The result at index k gives the number of pairs (a,b) with a+b=k
    # We just need to know if it's > 0
    if size_sum < 2 * mx + 1:
        size_sum *= 2
    
    ind = np.zeros(size_sum, dtype=np.float64)
    for x in U:
        if x < size_sum:
            ind[x] = 1.0
    
    # FFT convolution for sumset
    ind_fft = np.fft.rfft(ind)
    sum_fft = ind_fft * ind_fft
    sum_conv = np.fft.irfft(sum_fft, n=size_sum)
    
    # Sumset size: count of indices where sum_conv > 0.5 (threshold for floating point)
    n_plus = np.sum(sum_conv > 0.5)
    
    # Difference set: a - b
    # This is equivalent to convolution of ind with reversed ind
    # If we reverse ind (i.e., ind_rev[k] = ind[size_sum-1-k]), then
    # the convolution gives us counts of a - b + (size_sum-1)
    # Actually, let's think more carefully.
    # a - b = k means a = b + k.
    # We want the set of all k = a - b.
    # This is the same as the set of all k such that there exists b with b in U and b+k in U.
    # This is equivalent to the autocorrelation of the indicator function.
    
    # Autocorrelation: c[k] = sum_x ind[x] * ind[x+k]
    # This gives the number of pairs (x, x+k) both in U, i.e., the number of ways to write k as a-b.
    # So the difference set size is the number of k where c[k] > 0.
    
    # To compute autocorrelation via FFT:
    # c[k] = sum_x ind[x] * ind[x+k]
    # This is the convolution of ind with ind_reversed, where ind_reversed[x] = ind[-x]
    # But since we're dealing with non-negative integers, let's use the standard approach:
    # c[k] for k in [-mx, mx]
    # c[k] = sum_x ind[x] * ind[x+k]
    # = sum_x ind[x] * ind_rev[-x-k] where ind_rev[y] = ind[-y]
    # This is getting complicated. Let's use a different approach.
    
    # Actually, the difference set U-U = {a-b : a,b in U}
    # The size of this set is what we need.
    # We can compute it by noting that |U-U| = |U+(-U)| where -U = {-x : x in U}
    # So we need the sumset of U and -U.
    # -U contains values in [-mx, 0].
    # So U + (-U) contains values in [-mx, mx].
    
    # Let's shift: let V = U + mx, so V contains values in [mx, 2*mx]
    # And W = -U + mx, so W contains values in [0, mx]
    # Then U + (-U) = (V - mx) + (W - mx) = V + W - 2*mx
    # So |U-U| = |V + W|
    
    # V = {x + mx : x in U}, W = {mx - x : x in U}
    # Both V and W are subsets of [0, 2*mx]
    
    v_ind = np.zeros(size_sum, dtype=np.float64)
    w_ind = np.zeros(size_sum, dtype=np.float64)
    for x in U:
        if x + mx < size_sum:
            v_ind[x + mx] = 1.0
        if mx - x >= 0 and mx - x < size_sum:
            w_ind[mx - x] = 1.0
    
    v_fft = np.fft.rfft(v_ind)
    w_fft = np.fft.rfft(w_ind)
    diff_fft = v_fft * w_fft
    diff_conv = np.fft.irfft(diff_fft, n=size_sum)
    
    n_minus = np.sum(diff_conv > 0.5)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _is_sidon(U):
    """Check if U is a Sidon set (all sums a_i+a_j with i<=j are distinct)."""
    sums = set()
    n = len(U)
    for i in range(n):
        for j in range(i, n):
            s = U[i] + U[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _simulated_annealing(U_init, max_iter=5000, deadline=None, T0=1.0, T_end=0.01, cooling=0.999):
    """
    Simulated annealing to improve the set U.
    Moves: add a new element, remove an element (not 0), shift an element to a nearby value.
    """
    U = sorted(list(set(U_init)))
    if 0 not in U:
        U.append(0)
        U.sort()
    
    current_score = _score(U)
    best_U = list(U)
    best_score = current_score
    
    T = T0
    n = len(U)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3 and n < 20:
            # Add a new element
            max_val = max(U) if U else 0
            # Try adding a value in a reasonable range
            new_val = random.randint(1, max_val + max_val // 2 + 10)
            if new_val not in U:
                new_U = U + [new_val]
                new_U.sort()
                new_score = _score(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(U)
                else:
                    # Accept with probability exp((new_score - current_score) / T)
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        U = new_U
                        current_score = new_score
        elif move_type < 0.5 and n > 2:
            # Remove an element (not 0)
            removable = [x for x in U if x != 0]
            if removable:
                idx = random.choice(removable)
                new_U = [x for x in U if x != idx]
                if len(new_U) >= 2:
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_U = list(U)
                    else:
                        delta = new_score - current_score
                        if random.random() < math.exp(delta / T):
                            U = new_U
                            current_score = new_score
        else:
            # Shift an element to a nearby value
            if n > 1:
                idx = random.randint(1, n - 1)  # Don't shift 0
                old_val = U[idx]
                # Try shifting to a nearby value
                shift_range = max(5, old_val // 5)
                new_val = random.randint(1, old_val + shift_range)
                if new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_U = list(U)
                    else:
                        delta = new_score - current_score
                        if random.random() < math.exp(delta / T):
                            U = new_U
                            current_score = new_score
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
    
    return best_U, best_score

def _local_search(U, max_iter=1000, deadline=None):
    """
    Try to improve the set U by moving one element to a smaller value
    while maintaining the Sidon property (or just improving score if relaxed).
    We prioritize reducing max(U).
    """
    current_U = sorted(U)
    current_score = _score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    n = len(current_U)
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        for i in range(1, n):
            original_val = current_U[i]
            existing = set(current_U)
            existing.remove(original_val)
            
            if original_val != max(current_U):
                continue
            
            for x in range(0, original_val):
                if x in existing:
                    continue
                
                new_U = list(current_U)
                new_U[i] = x
                new_U.sort()
                
                if not _is_sidon(new_U):
                    continue
                
                new_score = _score(new_U)
                if new_score > best_score + 1e-9:
                    best_score = new_score
                    best_U = new_U
                    current_U = new_U
                    improved = True
                    break
        
        if not improved:
            break
            
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # List of known good Golomb rulers / Sidon sets
    candidates = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 24, 28, 34],
        [0, 1, 4, 10, 18, 24, 28, 34, 43],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77, 88],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77, 88, 100],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77, 88, 100, 112],
    ]
    
    best_U = candidates[0]
    best_score = 0.0
    
    for U in candidates:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        
        if time.time() < deadline - 10:
            improved_U, improved_score = _local_search(U, max_iter=100, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Simulated annealing from the best candidate found
    if time.time() < deadline - 20:
        sa_U, sa_score = _simulated_annealing(best_U, max_iter=3000, deadline=deadline, T0=0.5, T_end=0.01, cooling=0.9995)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
    
    # Try additional random starts with simulated annealing
    if time.time() < deadline - 30:
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            # Generate a random sparse set
            n_elems = random.randint(5, 15)
            U = [0]
            last = 0
            for _ in range(n_elems - 1):
                # Add a value with a large gap
                gap = random.randint(5, 30)
                new_val = last + gap
                if new_val not in U:
                    U.append(new_val)
                    last = new_val
            
            if len(U) >= 2:
                sa_U, sa_score = _simulated_annealing(U, max_iter=1000, deadline=deadline, T0=0.3, T_end=0.01, cooling=0.9995)
                if sa_score > best_score:
                    best_score = sa_score
                    best_U = sa_U
    
    # Final local search polish
    if time.time() < deadline - 5:
        polished_U, polished_score = _local_search(best_U, max_iter=200, deadline=deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U
    
    return best_U
