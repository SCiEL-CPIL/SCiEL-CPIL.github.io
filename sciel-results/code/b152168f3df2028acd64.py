import time
import random
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 20.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _score_of(U):
    """
    Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    U is a list of nonnegative integers containing 0.
    """
    if not U or len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute sum set and difference set
    # Using outer operations creates |U|x|U| arrays
    # For |U| up to 2000, this is 4M elements, which is manageable.
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    # Count unique elements
    # Using set on flattened numpy array is fast enough for moderate sizes
    # For larger sizes, we might need unique() which sorts, O(N log N)
    # set() is O(N) average
    n_sum = len(set(sums.flatten()))
    n_diff = len(set(diffs.flatten()))
    
    if n_sum == 0:
        return 0.0
        
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    
    if log_max == 0:
        return 0.0
    
    score = 1.0 + log_ratio / log_max
    return score

def _local_search(U, time_budget=None):
    """
    Local search to improve the score of U.
    """
    U = list(set(U))
    U.sort()
    if 0 not in U:
        U.append(0)
        U.sort()
    
    current_score = _score_of(U)
    best_U = list(U)
    best_score = current_score
    
    max_u = max(U) if U else 0
    # Range for new elements
    # We can expand the range if we add large elements
    max_range = max_u * 2 + 100
    
    if time_budget is None:
        time_budget = _time_left() / 2.0
    
    start = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start < time_budget and _time_left() > _TIME_MARGIN:
        if no_improve_count > max_no_improve:
            break
            
        U_copy = list(U)
        action = random.random()
        
        if action < 0.4:
            # Add a random element
            if max_range > 0:
                new_elem = random.randint(1, max_range)
                if new_elem not in U_copy:
                    U_copy.append(new_elem)
                    U_copy.sort()
        elif action < 0.7 and len(U_copy) > 1:
            # Remove a random element (not 0)
            idx = random.randint(1, len(U_copy) - 1)
            U_copy.pop(idx)
        else:
            # Swap an element
            if len(U_copy) > 1:
                idx = random.randint(1, len(U_copy) - 1)
                new_elem = random.randint(1, max_range)
                if new_elem not in U_copy:
                    U_copy[idx] = new_elem
                    U_copy.sort()
        
        new_score = _score_of(U_copy)
        
        if new_score > best_score:
            best_score = new_score
            best_U = U_copy
            U = U_copy
            no_improve_count = 0
            max_u = max(U)
            max_range = max_u * 2 + 100
        else:
            no_improve_count += 1
            
            # If stuck, try a more aggressive move: random restart of a few elements
            if no_improve_count % 100 == 0 and len(U) > 3:
                # Reset some elements to random values
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(1, len(U) - 1)
                    U[idx] = random.randint(1, max_range)
                U.sort()
                U = list(set(U))
                if 0 not in U:
                    U.append(0)
                    U.sort()
                current_score = _score_of(U)
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(U)
                    no_improve_count = 0
                else:
                    U = list(best_U)
    
    return best_U, best_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Strategy: Try multiple initial structures and run local search on each.
    
    # 1. Superincreasing sequence (powers of 2 minus 1)
    # This tends to have distinct sums, keeping |U+U| small relative to |U-U|?
    # Actually, for superincreasing, |U-U| = n^2, |U+U| = n(n+1)/2.
    # Ratio ~ 2.
    for n in range(3, 20):
        U = [0]
        val = 1
        for _ in range(n-1):
            U.append(val)
            val = val * 2 + 1
        if _time_left() < _TIME_MARGIN:
            break
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        
        # Local search from this start
        if _time_left() > _TIME_MARGIN + 10:
            U_new, s_new = _local_search(U, time_budget=_time_left() * 0.1)
            if s_new > best_score:
                best_score = s_new
                best_U = U_new

    # 2. Random sparse sets
    for _ in range(10):
        if _time_left() < _TIME_MARGIN + 10:
            break
        n = random.randint(5, 30)
        # Randomly generate a sparse set
        U = [0]
        current = 0
        for _ in range(n-1):
            # Jump by a random amount
            jump = random.randint(1, 10)
            current += jump
            U.append(current)
        U.sort()
        
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        
        if _time_left() > _TIME_MARGIN + 10:
            U_new, s_new = _local_search(U, time_budget=_time_left() * 0.1)
            if s_new > best_score:
                best_score = s_new
                best_U = U_new

    # 3. Final polish on the best found so far
    if _time_left() > _TIME_MARGIN + 5:
        U_final, s_final = _local_search(best_U, time_budget=_time_left() * 0.5)
        if s_final > best_score:
            best_score = s_final
            best_U = U_final

    return best_U
