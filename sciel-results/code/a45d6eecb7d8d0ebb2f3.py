import math
import time
import numpy as np
import random

_start_time = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return max(0, _DEADLINE - _elapsed())

def _score_fft(u):
    """Compute score using FFT for efficient sumset/diffset size calculation."""
    if len(u) < 2:
        return 1.0
    
    max_u = max(u)
    m = 2 * max_u + 1
    
    if m <= 1:
        return 1.0
    
    # Create indicator array for U
    n = len(u)
    arr = np.zeros(m, dtype=np.float64)
    for x in u:
        arr[x] = 1.0
    
    # Sumset: a+b where a,b in U, range [0, 2*max_u]
    # Use convolution: (arr * arr) gives counts of pairs (a,b) with a+b=k
    # But we need |U+U|, the size of the set, not the count
    # For sumset, use autocorrelation-like approach
    # Actually, for sumset U+U, we can use: if we define f(x) = 1 if x in U
    # then (f*f)(k) = sum_x f(x)f(k-x) counts representations
    # |U+U| = number of k where (f*f)(k) > 0
    
    # For difference set U-U, use autocorrelation:
    # (f * f_flipped)(k) where f_flipped(x) = f(-x)
    # More precisely, |U-U| = number of distinct differences a-b
    
    # FFT approach for sumset:
    # Pad to avoid circular convolution
    size = 2 * m
    f = np.zeros(size, dtype=np.float64)
    for x in u:
        f[x] = 1.0
    
    # Sumset via convolution f * f
    F = np.fft.rfft(f)
    conv_sum = np.fft.irfft(F * F, n=size)
    # Sumset elements are in range [0, 2*max_u], so indices 0 to 2*max_u
    sumset_size = np.sum(conv_sum[:2*max_u+1] > 1e-10)
    
    # Difference set: use autocorrelation
    # U-U: differences a-b where a,b in U, range [-max_u, max_u]
    # Shift to non-negative: consider g(x) = f(x) for x >= 0
    # Autocorrelation: sum_x f(x) f(x+d) for d = a-b
    # This gives counts of pairs with difference d
    # We need |{a-b : a,b in U}|
    
    # For difference set, use: h(d) = sum_x f(x) f(x+d) for d in [-max_u, max_u]
    # This is the autocorrelation of f
    # |U-U| = number of d where h(d) > 0
    
    # Autocorrelation via FFT
    F2 = np.fft.rfft(f)
    autocorr = np.fft.irfft(F2 * np.conj(F2), n=size)
    # autocorr[d] for d in [0, size-1]
    # The difference d corresponds to autocorr[d] for d >= 0
    # For negative differences, by symmetry autocorr[size-d] = autocorr[d]
    # So |U-U| = 1 + 2 * (number of positive d where autocorr[d] > 0)
    # Actually, autocorr[0] = |U| (all a-b=0)
    # For d > 0, autocorr[d] counts pairs (a,b) with a-b=d
    # |U-U| = 1 + 2 * sum_{d=1}^{max_u} [autocorr[d] > 0]
    
    # But autocorr from FFT might have numerical issues
    # Let's use a more robust approach for small sets
    
    if n <= 50:
        # Brute force for small sets
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
    else:
        # FFT-based
        sumset_size = int(np.sum(conv_sum[:2*max_u+1] > 1e-10))
        # For diffset, use the fact that |U-U| = |U+(-U)|
        # Create indicator for -U
        neg_arr = np.zeros(size, dtype=np.float64)
        for x in u:
            neg_arr[(size - x) % size] = 1.0
        F_neg = np.fft.rfft(neg_arr)
        conv_diff = np.fft.irfft(F * F_neg, n=size)
        # Differences range from -max_u to max_u, shifted by size/2
        # In circular convolution, difference d appears at index (d + size//2) % size
        # Actually, let's just count non-zero entries in the relevant range
        # The difference d = a - b ranges from -max_u to max_u
        # In the convolution of f and neg_f, the coefficient at index k 
        # represents sum_x f(x) neg_f(k-x) = sum_x f(x) f(x-k)
        # This counts pairs (x, x-k) both in U, i.e., x - (x-k) = k
        # So coefficient at index k counts pairs with difference k
        # But we need to handle the circular wrap-around
        # For k in [0, max_u], coefficient at k counts diff = k
        # For k in [size-max_u, size-1], coefficient at k counts diff = k - size = negative
        
        diffset_size = 0
        for k in range(max_u + 1):
            if conv_diff[k] > 1e-10:
                diffset_size += 1
            if k > 0 and conv_diff[size - k] > 1e-10:
                diffset_size += 1
        
        r = diffset_size / sumset_size
    
    if m <= 1:
        return 1.0
    else:
        return 1 + math.log(r) / math.log(m)

def _score_small(u):
    """Fast score computation for small sets (brute force)."""
    if len(u) < 2:
        return 1.0
    
    max_u = max(u)
    m = 2 * max_u + 1
    
    if m <= 1:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in u:
        for b in u:
            sumset.add(a + b)
            diffset.add(a - b)
    
    r = len(diffset) / len(sumset)
    return 1 + math.log(r) / math.log(m)

def _is_golomb(u):
    """Check if u is a Golomb ruler (all pairwise differences distinct)."""
    n = len(u)
    if n < 2:
        return True
    diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            d = u[j] - u[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _gen_golomb_optimal(n, max_len=500):
    """Generate a Golomb ruler of n marks using greedy + local search."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    if n == 3:
        return [0, 1, 3]
    if n == 4:
        return [0, 1, 4, 6]
    if n == 5:
        return [0, 1, 4, 9, 11]
    if n == 6:
        return [0, 1, 4, 10, 12, 17]
    
    # Greedy construction
    ruler = [0, 1]
    for _ in range(2, n):
        # Try to add the smallest possible value
        found = False
        for candidate in range(1, max_len):
            test = ruler + [candidate]
            if _is_golomb(test):
                ruler = test
                found = True
                break
        if not found:
            # Should not happen for reasonable n
            break
    
    # Local search to optimize
    best = ruler[:]
    best_score = _score_small(best)
    
    for iteration in range(200):
        if _remaining() < 5:
            break
        i = random.randint(0, len(best) - 1)
        if i == 0:
            continue  # Keep 0 fixed
        old_val = best[i]
        new_val = old_val + random.randint(-10, 10)
        if new_val < 0 or new_val == old_val:
            continue
        # Check if new value maintains Golomb property
        test = best[:]
        test[i] = new_val
        test = sorted(set(test))
        if len(test) != len(best):
            continue
        if not _is_golomb(test):
            continue
        new_score = _score_small(test)
        if new_score > best_score:
            best = test
            best_score = new_score
    
    return best

def _local_search(u, max_iters=500):
    """Local search to improve the score of set u."""
    if len(u) < 2:
        return u
    
    best = sorted(set(u))
    best_score = _score_small(best)
    
    for it in range(max_iters):
        if _remaining() < 5:
            break
        # Random move: add, remove, or modify an element
        move_type = random.choice(['add', 'remove', 'modify'])
        
        if move_type == 'add' and len(best) < 30:
            # Try adding a new element
            max_u = max(best)
            new_val = random.randint(1, max_u + 20)
            if new_val in best:
                continue
            test = sorted(best + [new_val])
            # Check Golomb property
            if not _is_golomb(test):
                continue
            new_score = _score_small(test)
            if new_score > best_score:
                best = test
                best_score = new_score
        elif move_type == 'remove' and len(best) > 3:
            # Try removing an element (not 0)
            if len(best) > 1:
                idx = random.randint(1, len(best) - 1)
                test = best[:idx] + best[idx+1:]
                if _is_golomb(test):
                    new_score = _score_small(test)
                    if new_score > best_score:
                        best = test
                        best_score = new_score
        elif move_type == 'modify':
            # Modify a non-zero element
            idx = random.randint(1, len(best) - 1)
            old_val = best[idx]
            new_val = old_val + random.randint(-5, 5)
            if new_val <= 0 or new_val in best:
                continue
            test = best[:]
            test[idx] = new_val
            test = sorted(set(test))
            if len(test) != len(best):
                continue
            if not _is_golomb(test):
                continue
            new_score = _score_small(test)
            if new_score > best_score:
                best = test
                best_score = new_score
    
    return best

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound."""
    
    # Known optimal Golomb rulers
    known_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
    ]
    
    # Generate additional candidates
    candidates = list(known_rulers)
    
    # Generate rulers for n=7 to n=12
    for n in range(7, 13):
        if _remaining() < 10:
            break
        ruler = _gen_golomb_optimal(n)
        if ruler and _is_golomb(ruler):
            candidates.append(ruler)
    
    # Evaluate all candidates
    best_set = [0, 1]
    best_score = 0.0
    
    for u in candidates:
        if _remaining() < 5:
            break
        if 0 not in u:
            u = [0] + u
        u = sorted(set(u))
        
        score = _score_small(u)
        
        if score > best_score:
            best_score = score
            best_set = u[:]
    
    # Local search on the best candidate
    if _remaining() > 10:
        improved = _local_search(best_set, max_iters=1000)
        if improved:
            improved_score = _score_small(improved)
            if improved_score > best_score:
                best_set = improved
                best_score = improved_score
    
    # Try a few more random initializations if time permits
    for _ in range(5):
        if _remaining() < 10:
            break
        # Random Golomb-like set
        n = random.randint(4, 10)
        ruler = [0]
        for _ in range(n - 1):
            candidate = random.randint(1, 100)
            if candidate in ruler:
                continue
            test = ruler + [candidate]
            if _is_golomb(test):
                ruler = test
            else:
                # Try a few alternatives
                for alt in range(candidate - 3, candidate + 4):
                    if alt <= 0 or alt in ruler:
                        continue
                    test = ruler + [alt]
                    if _is_golomb(test):
                        ruler = test
                        break
        
        if len(ruler) >= 3:
            ruler = sorted(set(ruler))
            score = _score_small(ruler)
            if score > best_score:
                best_score = score
                best_set = ruler[:]
            
            # Local search
            if _remaining() > 5:
                improved = _local_search(best_set, max_iters=200)
                if improved:
                    improved_score = _score_small(improved)
                    if improved_score > best_score:
                        best_set = improved
                        best_score = improved_score
    
    return best_set
