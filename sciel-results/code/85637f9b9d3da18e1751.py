import time
import math
import numpy as np
from numpy.fft import fft, ifft

# EVOLVE-BLOCK-START

def _set_sizes_from_indicator(indicator):
    """Given a boolean indicator array where indicator[i]=1 means i in U,
    return (|U+U|, |U-U|, max_element, |U|)."""
    n = len(indicator)
    # Sum set: convolution of indicator with itself
    # U+U = {a+b : a,b in U}, range [0, 2*(n-1)]
    conv_sum = np.convolve(indicator, indicator, mode='full')
    sum_set_size = int(np.sum(conv_sum > 0))
    
    # Difference set: U-U = {a-b : a,b in U}, range [-(n-1), n-1]
    # Use: ifft(fft(indicator) * fft(flip(indicator)))
    # Actually, difference set size = number of k where sum over i of ind[i]*ind[i+k] > 0 (for k>=0) 
    # plus symmetric. Better: autocorrelation.
    # U-U has elements d where exists a,b in U with a-b=d.
    # |U-U| = 1 + 2 * sum_{d=1}^{n-1} [exists a in U: a-d in U]
    # Use autocorrelation: r[d] = sum_i ind[i]*ind[i+d]
    # |U-U| = 1 + 2 * sum_{d=1}^{max} (r[d] > 0)
    
    # Compute autocorrelation via FFT
    # r[d] = sum_i ind[i] * ind[i+d] for d >= 0
    # This is the autocorrelation.
    f = fft(indicator, n=2*n)
    r = ifft(f * np.conj(f)).real
    # r[0] = |U|, r[d] for d=1..n-1 gives sum_i ind[i]*ind[i+d]
    # For negative differences, by symmetry, same count
    positive_diffs = int(np.sum(r[1:n] > 0.5))  # d from 1 to n-1
    diff_set_size = 1 + 2 * positive_diffs
    
    # Find max element and |U|
    indices = np.where(indicator > 0.5)[0]
    if len(indices) == 0:
        return 0, 0, 0, 0
    max_el = int(indices[-1])
    u_size = len(indices)
    
    return sum_set_size, diff_set_size, max_el, u_size


def _score_from_sizes(sum_size, diff_size, max_el, u_size):
    """Compute the score from set sizes."""
    if u_size < 2 or max_el == 0 or sum_size == 0:
        return 0.0
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_el + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _score_of_set(U):
    """Compute score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    max_el = max(U)
    if max_el == 0:
        return 0.0
    indicator = np.zeros(max_el + 1, dtype=np.float64)
    for x in U:
        indicator[x] = 1.0
    sum_size, diff_size, max_el2, u_size = _set_sizes_from_indicator(indicator)
    return _score_from_sizes(sum_size, diff_size, max_el2, u_size)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Start with a good initial set
    # Powers of 2 give decent ratio but large max
    # Let's try a mix: small structured set
    best_set = [0, 1, 3]
    best_score = _score_of_set(best_set)
    
    # Try several initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 9],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 9, 27, 81],
    ]
    
    for cand in candidates:
        s = _score_of_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Local search: try adding, removing, swapping elements
    # Work with a set representation
    current_set = set(best_set)
    current_score = best_score
    
    # Determine a reasonable max range to search
    max_range = max(200, 2 * max(best_set) + 10)
    
    # Local search loop
    while time.time() < deadline:
        # Try random perturbations
        improved = False
        
        # Try adding a random element
        if len(current_set) < 30:
            candidates_to_add = [x for x in range(1, max_range + 1) if x not in current_set]
            if candidates_to_add:
                for _ in range(20):
                    x = int(np.random.randint(0, len(candidates_to_add)))
                    new_el = candidates_to_add[x]
                    new_set = current_set | {new_el}
                    s = _score_of_set(list(new_set))
                    if s > current_score + 1e-9:
                        current_set = new_set
                        current_score = s
                        improved = True
                        max_range = max(max_range, new_el + 10)
                        break
        
        # Try removing a random element (not 0)
        removable = [x for x in current_set if x != 0]
        if removable and len(current_set) > 2:
            for _ in range(10):
                idx = int(np.random.randint(0, len(removable)))
                x = removable[idx]
                new_set = current_set - {x}
                s = _score_of_set(list(new_set))
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
        
        # Try swapping: remove one, add another
        if removable and len(current_set) > 2:
            candidates_to_add = [x for x in range(1, max_range + 1) if x not in current_set]
            if candidates_to_add:
                for _ in range(10):
                    idx_rm = int(np.random.randint(0, len(removable)))
                    x_rm = removable[idx_rm]
                    idx_add = int(np.random.randint(0, len(candidates_to_add)))
                    x_add = candidates_to_add[idx_add]
                    new_set = (current_set - {x_rm}) | {x_add}
                    s = _score_of_set(list(new_set))
                    if s > current_score + 1e-9:
                        current_set = new_set
                        current_score = s
                        improved = True
                        max_range = max(max_range, x_add + 10)
                        break
        
        if not improved:
            # No improvement, try a bigger jump or restart
            # Occasionally restart with a different structure
            if np.random.random() < 0.1:
                # Try a geometric progression
                base = int(np.random.randint(2, 5))
                n_terms = int(np.random.randint(3, 8))
                new_set = {base**i for i in range(n_terms)}
                new_set.add(0)
                s = _score_of_set(list(new_set))
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    max_range = max(max_range, max(new_set) + 10)
            else:
                # Small random perturbation
                if removable:
                    idx = int(np.random.randint(0, len(removable)))
                    x = removable[idx]
                    # Replace with nearby value
                    new_x = x + int(np.random.randint(-3, 4))
                    if new_x > 0 and new_x not in current_set:
                        new_set = (current_set - {x}) | {new_x}
                        s = _score_of_set(list(new_set))
                        if s > current_score + 1e-9:
                            current_set = new_set
                            current_score = s
                            max_range = max(max_range, new_x + 10)
    
    # Update best
    if current_score > best_score:
        best_set = list(current_set)
        best_score = current_score
    
    # Ensure 0 is in the set and return sorted
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
    
    return best_set

# EVOLVE-BLOCK-END
