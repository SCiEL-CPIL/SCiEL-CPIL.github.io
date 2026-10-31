<<<<<<< SEARCH
import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 6.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _is_golomb(U):
    """Check if U is a Golomb ruler."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i+1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _score(U):
    """Compute the score."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # For Golomb rulers, we can use the formula directly if we know it's Golomb
    # But to be safe and general, we compute |U-U| and |U+U|
    # However, if we are searching Golomb rulers, we can just use the formula
    # |U-U| = n(n-1)+1
    # |U+U| = n(n+1)/2
    # But only if it IS a Golomb ruler.
    # If it's not, the score might be different.
    # Given the insight, we focus on Golomb rulers.
    
    n = len(U)
    if _is_golomb(U):
        diff_count = n * (n - 1) + 1
        sum_count = n * (n + 1) // 2
    else:
        # Fallback: exact computation
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        sum_count = len(sumset)
        diff_count = len(diffset)
        
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio, 2) / denom

def _optimize_golomb(n, time_budget=5.0):
    """Find a short Golomb ruler with n marks."""
    if n <= 1:
        return [0]
    
    # Initial guess: greedy or known optimal for small n
    # Known optimal lengths for n=1..15
    # 1: 0
    # 2: 1
    # 3: 3
    # 4: 6
    # 5: 11
    # 6: 17
    # 7: 25
    # 8: 34
    # 9: 44
    # 10: 55
    # 11: 72
    # 12: 85
    # 13: 106
    # 14: 129
    # 15: 153
    
    # Start with a random valid Golomb ruler or greedy
    U = [0]
    for i in range(1, n):
        # Try to place next mark
        found = False
        # Search range
        upper = i * i * 4 + 20
        for pos in range(i * i, upper):
            cand = U + [pos]
            if _is_golomb(cand):
                U.append(pos)
                found = True
                break
        if not found:
            # Extend search
            for pos in range(upper, upper + 100):
                cand = U + [pos]
                if _is_golomb(cand):
                    U.append(pos)
                    found = True
                    break
            if not found:
                # Fallback: just add a large number
                U.append(U[-1] + 100)
    
    best_U = U
    best_len = max(U)
    start = time.time()
    rng = random.Random(n)
    
    while time.time() - start < time_budget and _remaining() > 10:
        # Local search: move a mark
        improved = False
        for i in range(1, n):
            orig = U[i]
            # Try moving to nearby positions
            for d in range(-20, 21):
                if d == 0: continue
                nv = orig + d
                if nv < 0: continue
                if nv == 0: continue
                # Check if nv is already in U
                if nv in U: continue
                new_U = U[:i] + [nv] + U[i+1:]
                new_U.sort()
                if _is_golomb(new_U):
                    if max(new_U) < best_len:
                        best_len = max(new_U)
                        U = new_U
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Random restart if stuck for a while?
            # Or just break
            if rng.random() < 0.1:
                # Perturb
                i = rng.randint(1, n-1)
                U[i] = U[i] + rng.randint(-10, 10)
                U.sort()
                # Ensure it's still Golomb? If not, revert
                if not _is_golomb(U):
                    U = U[:i] + [U[i] - rng.randint(-10, 10)] + U[i+1:]
                    U.sort()
                    if not _is_golomb(U):
                        # Revert to previous best
                        pass # Hard to revert easily without history
                        # Let's just break if we lose Golomb property
                        break

    return U

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_s = _score(best_U)
    
    # Try various n
    for n in range(3, 16):
        if _remaining() < 15:
            break
        # Allocate time
        alloc = min(5.0, _remaining() - 10)
        if alloc < 1.0:
            break
        U = _optimize_golomb(n, time_budget=alloc)
        s = _score(U)
        if s > best_s:
            best_s = s
            best_U = U
            
    return best_U
=======
import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 6.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _fft_score(U):
    """Score via FFT convolution. U must be nonnegative ints, 0 in U, len>=2."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    n = len(U)
    
    # Sumset: conv of indicator with itself, support [0, 2M]
    size = 2 * M + 1
    # Next power of 2 for FFT efficiency
    N = 1
    while N < 2 * size:
        N *= 2
    ind = np.zeros(N)
    for x in U:
        ind[x] = 1.0
    conv = np.fft.irfft(np.fft.rfft(ind) ** 2)
    sum_count = int(np.sum(conv[:2 * M + 1] > 0.5))
    
    # Difference set: conv of indicator with reversed indicator
    # diff[i] = sum_j ind[j] * ind[j + (i - M)] for offset i-M
    # Use: conv_diff = irfft(rfft(ind) * rfft(flip(ind)))
    ind_rev = np.zeros(N)
    for x in U:
        ind_rev[M - x + M] = 1.0  # shift so that index M-x+M corresponds to x
    # Actually simpler: use correlation via FFT
    # diffset element d = a-b is in [-(M), M]
    # Number of pairs (a,b) with a-b = d is sum_a ind[a]*ind[a-d]
    # This is the autocorrelation at lag d
    # autocorr[d] = sum_x ind[x] * ind[x-d]
    # FFT: autocorr = irfft(rfft(ind) * conj(rfft(ind)))
    spectrum = np.fft.rfft(ind)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))
    # autocorr[k] for k=0..M/2 gives positive lags, wraparound for negative
    # For lag d in [-M, M]:
    # if d >= 0: count = autocorr[d]
    # if d < 0: count = autocorr[N+d] (due to circular)
    # But we want number of distinct d values where count > 0
    diff_count = 0
    for d in range(-M, M + 1):
        if d >= 0:
            c = autocorr[d]
        else:
            c = autocorr[N + d]
        if c > 0.5:
            diff_count += 1
    
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio, 2) / denom

def _score(U):
    """Compute the score."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    n = len(U)
    
    if M <= 5000:
        return _fft_score(U)
    else:
        # Fallback for very large M
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        sum_count = len(sumset)
        diff_count = len(diffset)
        if sum_count == 0:
            return 0.0
        ratio = diff_count / sum_count
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * M + 1, 2)
        if denom == 0:
            return 0.0
        return 1.0 + math.log(ratio, 2) / denom

def _gen_golomb(n, seed=42):
    """Generate a Golomb ruler-like set with n marks using greedy construction."""
    rng = random.Random(seed)
    U = [0]
    for i in range(1, n):
        # Greedy: place next mark as far as possible while maintaining Golomb property
        # Use a heuristic based on known Golomb ruler constructions
        best_pos = None
        # Try positions in a smart order
        lower = U[-1] + i
        upper = U[-1] + i * i + 20
        for pos in range(upper, lower - 1, -1):
            cand = U + [pos]
            # Quick Golomb check
            diffs = set()
            ok = True
            for j in range(len(cand)):
                for k in range(j + 1, len(cand)):
                    d = cand[k] - cand[j]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if ok:
                best_pos = pos
                break
        if best_pos is None:
            best_pos = U[-1] + i * i + 20
        U.append(best_pos)
    return U

def _gen_powers(n, base=2):
    """Generate a set based on powers of base."""
    U = [0]
    for i in range(1, n):
        U.append(base ** i)
    return U

def _gen_fib(n):
    """Generate a set based on Fibonacci numbers."""
    a, b = 0, 1
    U = [0]
    for i in range(1, n):
        a, b = b, a + b
        U.append(a)
    return U

def _gen_random_structured(n, seed=42):
    """Generate a random structured set."""
    rng = random.Random(seed)
    U = [0]
    for i in range(1, n):
        # Geometric growth with randomness
        growth = rng.uniform(1.5, 3.0)
        U.append(int(U[-1] * growth + rng.randint(0, 5)))
    return U

def _local_search(U, time_budget=5.0, rng=None):
    """Local search to improve the score of a set."""
    if rng is None:
        rng = random.Random()
    U = sorted(U)
    n = len(U)
    best_U = U[:]
    best_s = _score(U)
    start = time.time()
    
    while time.time() - start < time_budget and _remaining() > 10:
        improved = False
        # Try moving each element
        for i in range(1, n):
            orig = U[i]
            # Try several nearby positions
            for d in rng.sample(range(-50, 51), min(20, 101)):
                if d == 0:
                    continue
                nv = orig + d
                if nv < 0 or nv == 0:
                    continue
                if nv in U:
                    continue
                new_U = U[:i] + [nv] + U[i+1:]
                new_U.sort()
                s = _score(new_U)
                if s > best_s:
                    best_s = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding or removing an element
            if rng.random() < 0.3 and n < 30:
                # Add a new element
                pos = rng.randint(1, max(U) + 50)
                if pos not in U:
                    new_U = U + [pos]
                    new_U.sort()
                    s = _score(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U
                        U = new_U
                        n = len(U)
                        improved = True
            elif n > 3:
                # Remove an element
                i = rng.randint(1, n - 1)
                new_U = U[:i] + U[i+1:]
                s = _score(new_U)
                if s > best_s:
                    best_s = s
                    best_U = new_U
                    U = new_U
                    n = len(U)
                    improved = True
            
            if not improved:
                # Perturb
                i = rng.randint(1, n - 1)
                U[i] = U[i] + rng.randint(-20, 20)
                U.sort()
                if len(set(U)) != len(U) or U[0] != 0:
                    U = best_U[:]
                    n = len(U)
    
    return best_U

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    rng = random.Random(42)
    
    best_U = [0, 1, 3]
    best_s = _score(best_U)
    
    # Try various sizes and generation strategies
    generators = [
        _gen_golomb,
        _gen_powers,
        _gen_fib,
        _gen_random_structured,
    ]
    
    for n in range(3, 35):
        if _remaining() < 15:
            break
        for gen in generators:
            if _remaining() < 10:
                break
            # Allocate time for local search
            alloc = min(3.0, _remaining() - 10)
            if alloc < 0.5:
                break
            try:
                if gen == _gen_golomb:
                    U = gen(n, seed=n)
                elif gen == _gen_powers:
                    U = gen(n, base=2)
                elif gen == _gen_fib:
                    U = gen(n)
                else:
                    U = gen(n, seed=n)
                
                # Ensure 0 is in U and all nonnegative
                U = sorted(set(int(x) for x in U if x >= 0))
                if 0 not in U:
                    U.insert(0, 0)
                
                if len(U) < 2:
                    continue
                
                s = _score(U)
                if s > best_s:
                    best_s = s
                    best_U = U[:]
                
                # Local search to improve
                if _remaining() > 12:
                    U_opt = _local_search(U, time_budget=alloc, rng=rng)
                    s_opt = _score(U_opt)
                    if s_opt > best_s:
                        best_s = s_opt
                        best_U = U_opt
            except Exception:
                continue
    
    return best_U
>>>>>>> REPLACE
