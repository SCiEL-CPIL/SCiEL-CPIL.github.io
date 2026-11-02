import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    
    # Use FFT for sumset and diffset sizes
    # Choose FFT size as next power of 2 >= 2*(M+1) for speed
    size = 2 * (M + 1)
    fft_size = 1 << (size - 1).bit_length()
    
    vec = np.zeros(fft_size, dtype=np.float64)
    vec[U] = 1.0
    
    f = np.fft.rfft(vec)
    
    # Sumset U+U: circular convolution, but since U+U max is 2M < fft_size, no wraparound
    s = np.fft.irfft(f * f, n=fft_size)
    sumset = int(np.sum(s[:2*M+1] >= 0.5))
    
    # Diffset U-U: use autocorrelation
    g = np.fft.irfft(f * np.conj(f), n=fft_size)
    # g[k] = sum_u vec[u]*vec[u-k] for k>=0 gives |U ∩ (U+k)|
    # diffset size = 2*sum_{k>0} |U ∩ (U+k)| + n
    # Actually g[0]=n, g[k] for k>0 counts pairs (a,b) with a-b=k, i.e., a=b+k
    # So |U-U| = n + 2*sum_{k=1}^{M} g[k]
    diffset = int(n + 2 * np.sum(g[1:M+1] >= 0.5))
    
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _greedy_construct(max_size=20, max_val_limit=800, time_limit=30.0):
    """Greedy construction: start with {0}, add element that maximizes score."""
    U = [0]
    cur_score = _score_set(U)
    
    for step in range(max_size - 1):
        if _elapsed() > _START + time_limit:
            break
        
        best_candidate = None
        best_s = cur_score
        
        max_current = max(U) if U else 0
        upper = min(2 * max_current + 20, max_val_limit)
        
        # Generate a broader set of candidates
        candidates = set()
        # Dense small steps
        for d in range(1, min(25, upper + 1)):
            c = max_current + d
            if 1 <= c <= upper:
                candidates.add(c)
        # Larger jumps
        for d in [25, 35, 50, 70, 100, 150, 200, 300, 400]:
            c = max_current + d
            if 1 <= c <= upper:
                candidates.add(c)
        # Near existing elements
        for u in U:
            for d in [1, 2, 3, 4, 5, 7, 11]:
                for sign in [-1, 1]:
                    c = u + sign * d
                    if 1 <= c <= upper and c not in U:
                        candidates.add(c)
        # Some structured candidates
        for d in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            c = max_current * d
            if 1 <= c <= upper:
                candidates.add(c)
        
        # Evaluate each candidate
        for c in sorted(candidates):
            if c in U:
                continue
            trial = sorted(U + [c])
            s = _score_set(trial)
            if s > best_s + 1e-12:
                best_s = s
                best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            cur_score = best_s
        else:
            # Try a few random candidates before giving up
            found = False
            for _ in range(10):
                if upper > 1:
                    c = np.random.randint(1, upper + 1)
                    if c not in U:
                        trial = sorted(U + [c])
                        s = _score_set(trial)
                        if s > cur_score + 1e-12:
                            U.append(c)
                            U.sort()
                            cur_score = s
                            found = True
                            break
            if not found:
                break
    
    return U, cur_score

def _golomb_ruler(n):
    """Generate a Golomb ruler with n marks (including 0)."""
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
    if n == 7:
        return [0, 1, 4, 10, 18, 23, 29]
    # For larger n, use a greedy approach
    U = [0]
    for i in range(1, n):
        c = 1
        while True:
            U_test = sorted(U + [c])
            diffs = set()
            valid = True
            for j in range(len(U_test)):
                for k in range(j+1, len(U_test)):
                    d = U_test[k] - U_test[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                U.append(c)
                break
            c += 1
    return U

def construct():
    best_U = [0]
    best_score = 0.0
    rng = np.random.RandomState(42)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 14):
        U = _golomb_ruler(n)
        candidates.append(U)
    
    # Prime-based sets
    def primes_up_to(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = primes_up_to(300)
    for k in range(2, 15):
        if k <= len(primes):
            U = [0] + primes[:k]
            candidates.append(U)
    
    # Structured patterns
    def generate_pattern(steps, k):
        U = [0]
        for i in range(k - 1):
            if i < len(steps):
                U.append(U[-1] + steps[i])
            else:
                U.append(U[-1] + steps[-1])
        return U
    
    for steps in [[1, 3, 2], [1, 2, 3], [2, 3, 1], [1, 4, 2, 3], [1, 3, 5, 2]]:
        for k in range(3, 18):
            candidates.append(generate_pattern(steps, k))
    
    # Powers of 2
    for k in range(2, 14):
        candidates.append([0] + [2**i for i in range(k)])
    
    # Modular hash sets: U = {x : x mod p = r} ∩ [0, M]
    for p in [7, 11, 13, 17, 19, 23, 29, 31]:
        for r in range(p):
            M_val = 200
            U = [x for x in range(0, M_val + 1) if x % p == r]
            if 0 in U and len(U) >= 3:
                candidates.append(U)
    
    # Random sparse sets
    for max_val in [30, 60, 120, 240, 400]:
        for size in range(3, min(15, max_val // 4 + 1)):
            for _ in range(8):
                if _elapsed() > _START + 15:
                    break
                try:
                    cands = rng.choice(range(1, max_val + 1), size=size, replace=False)
                    U = sorted([0] + list(cands))
                    candidates.append(U)
                except:
                    pass
    
    # Evaluate all candidates
    for U in candidates:
        if _elapsed() > _START + 25:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Phase 2: Greedy construction
    greedy_U, greedy_score = _greedy_construct(max_size=20, max_val_limit=800, time_limit=30.0)
    if greedy_score > best_score:
        best_score = greedy_score
        best_U = greedy_U
    
    # Phase 3: Local search with add, remove, and replace operations
    U = list(best_U)
    cur_score = best_score
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and _elapsed() < _START + 120:
        improved = False
        no_improve_count = 0
        max_val = max(U) if U else 1
        
        # Try adding elements
        add_attempts = 0
        while add_attempts < 50 and _elapsed() < _START + 120:
            add_attempts += 1
            c = rng.randint(1, int(max_val * 1.8) + 20)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        
        # Try removing elements
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > _START + 120:
                    break
                trial = U[:idx] + U[idx+1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    break
        
        # Try replacing one element with another
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > _START + 120:
                    break
                old_val = U[idx]
                # Try replacing with nearby values
                for delta in [-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30]:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in U:
                        continue
                    if new_val > 3 * max_val:
                        continue
                    trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        best_U = U
                        best_score = cur_score
                        improved = True
                        break
                if improved:
                    break
        
        # Try pairwise swaps
        if len(U) >= 3 and _elapsed() < _START + 120:
            for i in range(1, len(U)):
                for j in range(i+1, len(U)):
                    if _elapsed() > _START + 120:
                        break
                    # Try swapping U[i] and U[j] with nearby values
                    for di in [-2, -1, 1, 2]:
                        for dj in [-2, -1, 1, 2]:
                            ni = U[i] + di
                            nj = U[j] + dj
                            if ni <= 0 or nj <= 0:
                                continue
                            if ni in U or nj in U:
                                continue
                            if ni == nj:
                                continue
                            if ni > 3 * max_val or nj > 3 * max_val:
                                continue
                            trial = U[:]
                            trial[i] = ni
                            trial[j] = nj
                            trial = sorted(trial)
                            s = _score_set(trial)
                            if s > cur_score + 1e-12:
                                cur_score = s
                                U = trial
                                best_U = U
                                best_score = cur_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    # Phase 4: Final polish - try small perturbations
    if _elapsed() < _START + 125:
        for _ in range(30):
            if _elapsed() > _START + 128:
                break
            if len(U) > 2:
                idx = rng.randint(1, len(U))
                delta = rng.randint(-5, 6)
                new_val = U[idx] + delta
                if new_val > 0 and new_val not in U:
                    trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                    s = _score_set(trial)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = trial
    
    return best_U
