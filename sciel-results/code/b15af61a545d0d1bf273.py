import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = 115.0  # safety margin before 120s kill

def _score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    n = len(U)
    if n < 2:
        return 1.0
    maxv = max(U)
    # Build sumset and difference set using numpy convolution (FFT)
    # For sumset: indicator array of length maxv+1, convolve with itself
    arr = np.zeros(maxv + 1, dtype=np.int64)
    for u in U:
        arr[u] = 1
    # Sumset: convolve arr with arr (full)
    # Using FFT for speed
    size = 2 * (maxv + 1)
    fa = np.fft.rfft(arr, n=size)
    sumset_arr = np.fft.irfft(fa * fa, n=size)[:2 * maxv + 1]
    sumset_count = int(np.sum(sumset_arr > 0))
    # Difference set: correlate arr with arr
    # U-U = {a-b : a,b in U}, range [-maxv, maxv], size 2*maxv+1
    # Correlation: fft(arr) * conj(fft(arr))
    diffset_arr = np.fft.irfft(fa * np.conj(fa), n=size)
    # The correlation is centered; shift to get [0, 2*maxv] representing [-maxv, maxv]
    diffset_full = np.concatenate([diffset_arr[-maxv:], diffset_arr[:maxv + 1]])
    diffset_count = int(np.sum(diffset_full > 0))
    
    if sumset_count == 0 or diffset_count == 0:
        return 1.0
    log_denom = math.log(2 * maxv + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + math.log(diffset_count / sumset_count) / log_denom

def _score_fast(U, maxv):
    """Faster score computation using preallocated arrays when maxv is known."""
    n = len(U)
    if n < 2:
        return 1.0
    arr = np.zeros(maxv + 1, dtype=np.int64)
    for u in U:
        arr[u] = 1
    size = 2 * (maxv + 1)
    fa = np.fft.rfft(arr, n=size)
    sumset_arr = np.fft.irfft(fa * fa, n=size)[:2 * maxv + 1]
    sumset_count = int(np.sum(sumset_arr > 0))
    diffset_arr = np.fft.irfft(fa * np.conj(fa), n=size)
    diffset_full = np.concatenate([diffset_arr[-maxv:], diffset_arr[:maxv + 1]])
    diffset_count = int(np.sum(diffset_full > 0))
    if sumset_count == 0 or diffset_count == 0:
        return 1.0
    log_denom = math.log(2 * maxv + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + math.log(diffset_count / sumset_count) / log_denom

def _golomb_ruler_init(n):
    """Generate a Golomb-ruler-like set of n nonnegative integers containing 0."""
    if n <= 1:
        return [0]
    # Use a greedy construction: start with 0, then add elements that maximize
    # the minimum pairwise difference (Golomb ruler property)
    U = [0]
    # Initial spread
    for i in range(1, n):
        # Try to place at position that maximizes minimum distance to existing
        best_pos = i * i  # quadratic growth as a starting guess
        # Refine: check a range of candidates
        candidates = range(max(1, best_pos - 10), best_pos + 11)
        best_min_dist = 0
        best_cand = best_pos
        for c in candidates:
            if c in U:
                continue
            min_d = min(abs(c - u) for u in U)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_cand = c
        U.append(best_cand)
    U.sort()
    return U

def _local_search(U, max_iterations=5000):
    """Local search with add, remove, and swap operations."""
    U = sorted(U)
    best_score = _score(U)
    best_U = list(U)
    current_score = best_score
    maxv = max(U)
    
    for it in range(max_iterations):
        if time.time() - _START > _DEADLINE:
            break
        # Choose operation
        op = random.random()
        U_new = list(U)
        maxv_new = maxv
        
        if op < 0.35:
            # Add a new element
            # Try adding near gaps or at the end
            candidates = []
            # Near existing elements
            for u in U:
                for d in [1, 2, 3, 5, 8, 13, 21]:
                    c = u + d
                    if c not in U:
                        candidates.append(c)
                    c2 = u - d
                    if c2 > 0 and c2 not in U:
                        candidates.append(c2)
            # At the end
            candidates.append(maxv + random.randint(1, max(1, maxv // 10)))
            candidates.append(maxv + random.randint(1, max(1, maxv // 5)))
            
            if not candidates:
                continue
            c = random.choice(candidates)
            if c in U:
                continue
            U_new.append(c)
            maxv_new = max(maxv, c)
        elif op < 0.55:
            # Remove an element (not 0)
            if len(U) <= 2:
                continue
            removable = [u for u in U if u != 0]
            if not removable:
                continue
            c = random.choice(removable)
            U_new.remove(c)
            maxv_new = max(U_new) if U_new else 0
        else:
            # Swap: replace one element with another
            if len(U) < 2:
                continue
            removable = [u for u in U if u != 0]
            if not removable:
                continue
            old = random.choice(removable)
            # New position: near other elements or random
            candidates = []
            for u in U:
                if u == old:
                    continue
                for d in [1, 2, 3, 5, 8, 13]:
                    c = u + d
                    if c not in U and c != old:
                        candidates.append(c)
                    c2 = u - d
                    if c2 > 0 and c2 not in U and c2 != old:
                        candidates.append(c2)
            candidates.append(maxv + random.randint(1, max(1, maxv // 10)))
            candidates.append(random.randint(1, maxv + maxv // 5))
            
            if not candidates:
                continue
            new_val = random.choice(candidates)
            if new_val == old or new_val in U:
                continue
            U_new.remove(old)
            U_new.append(new_val)
            U_new.sort()
            maxv_new = max(U_new)
        
        # Evaluate
        if not U_new or 0 not in U_new:
            continue
        s = _score_fast(U_new, maxv_new)
        
        if s > current_score:
            current_score = s
            U = U_new
            maxv = maxv_new
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    return best_U, best_score

def _simulated_annealing(U, max_iterations=3000):
    """Simulated annealing to escape local minima."""
    U = sorted(U)
    best_score = _score(U)
    best_U = list(U)
    current_score = best_score
    maxv = max(U)
    
    T = 0.01  # initial temperature
    T_min = 0.0001
    cooling = 0.995
    
    for it in range(max_iterations):
        if time.time() - _START > _DEADLINE:
            break
        if T < T_min:
            break
        
        op = random.random()
        U_new = list(U)
        maxv_new = maxv
        
        if op < 0.3:
            # Add
            candidates = []
            for u in U:
                for d in [1, 2, 3, 5, 8, 13, 21, 34]:
                    c = u + d
                    if c not in U:
                        candidates.append(c)
            candidates.append(maxv + random.randint(1, max(1, maxv // 5)))
            if not candidates:
                continue
            c = random.choice(candidates)
            if c in U:
                continue
            U_new.append(c)
            maxv_new = max(maxv, c)
        elif op < 0.5:
            # Remove
            if len(U) <= 2:
                continue
            removable = [u for u in U if u != 0]
            if not removable:
                continue
            c = random.choice(removable)
            U_new.remove(c)
            maxv_new = max(U_new) if U_new else 0
        else:
            # Swap
            if len(U) < 2:
                continue
            removable = [u for u in U if u != 0]
            if not removable:
                continue
            old = random.choice(removable)
            candidates = []
            for u in U:
                if u == old:
                    continue
                for d in [1, 2, 3, 5, 8, 13, 21]:
                    c = u + d
                    if c not in U and c != old:
                        candidates.append(c)
                    c2 = u - d
                    if c2 > 0 and c2 not in U and c2 != old:
                        candidates.append(c2)
            candidates.append(maxv + random.randint(1, max(1, maxv // 5)))
            if not candidates:
                continue
            new_val = random.choice(candidates)
            if new_val == old or new_val in U:
                continue
            U_new.remove(old)
            U_new.append(new_val)
            U_new.sort()
            maxv_new = max(U_new)
        
        if not U_new or 0 not in U_new:
            continue
        
        s = _score_fast(U_new, maxv_new)
        delta = s - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = s
            U = U_new
            maxv = maxv_new
            if s > best_score:
                best_score = s
                best_U = list(U)
        
        T *= cooling
    
    return best_U, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound."""
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 115.0
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Phase 1: Try different initial sizes with Golomb-ruler-like construction
    for target_n in [5, 8, 12, 18, 25, 35, 50, 70, 100]:
        if time.time() - _START > _DEADLINE - 20:
            break
        
        U = _golomb_ruler_init(target_n)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        
        # Local search refinement
        if time.time() - _START > _DEADLINE - 15:
            break
        U_ls, s_ls = _local_search(U, max_iterations=3000)
        if s_ls > best_score:
            best_score = s_ls
            best_U = U_ls
    
    # Phase 2: Simulated annealing from the best found so far
    if time.time() - _START < _DEADLINE - 10:
        U_sa, s_sa = _simulated_annealing(best_U, max_iterations=4000)
        if s_sa > best_score:
            best_score = s_sa
            best_U = U_sa
    
    # Phase 3: Final local search polish
    if time.time() - _START < _DEADLINE - 5:
        U_final, s_final = _local_search(best_U, max_iterations=5000)
        if s_final > best_score:
            best_score = s_final
            best_U = U_final
    
    # Ensure 0 is in the set and all values are nonnegative integers
    best_U = sorted(set(int(u) for u in best_U if u >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

# EVOLVE-BLOCK-END
