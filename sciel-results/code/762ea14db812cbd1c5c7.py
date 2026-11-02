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
    vec = np.zeros(M + 1, dtype=np.float64)
    vec[U] = 1.0
    
    # Sumset U+U
    f = np.fft.rfft(vec, n=2 * (M + 1))
    s = np.fft.irfft(f * f, n=2 * (M + 1))
    sumset = int(np.sum(s >= 0.5))
    
    # Diffset U-U
    g = np.fft.irfft(f * np.conj(f), n=2 * (M + 1))
    diffset = int(np.sum(g >= 0.5))
    
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _greedy_construct(max_size=15, max_val_limit=500, time_limit=20.0):
    """Greedy construction: start with {0}, add element that maximizes score."""
    U = [0]
    cur_score = _score_set(U)
    
    for step in range(max_size - 1):
        if _elapsed() > _START + time_limit:
            break
        
        best_candidate = None
        best_s = cur_score
        
        # Try adding elements in a structured way
        max_current = max(U) if U else 0
        # Candidate range: from 1 to about 2*max_current or up to max_val_limit
        upper = min(2 * max_current + 10, max_val_limit)
        
        # Generate candidates: small increments and some larger jumps
        candidates = set()
        # Small steps
        for d in range(1, min(15, upper + 1)):
            c = max_current + d
            if 1 <= c <= upper:
                candidates.add(c)
        # Some larger jumps
        for d in [15, 20, 30, 50, 80, 120, 200]:
            c = max_current + d
            if 1 <= c <= upper:
                candidates.add(c)
        # Also try values near existing elements
        for u in U:
            for d in [1, 2, 3, 5, 8]:
                for sign in [-1, 1]:
                    c = u + sign * d
                    if 1 <= c <= upper and c not in U:
                        candidates.add(c)
        
        # Evaluate each candidate
        for c in candidates:
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
            # No improvement found, try a random candidate
            if upper > 1:
                c = np.random.RandomState(step).randint(1, upper + 1)
                if c not in U:
                    trial = sorted(U + [c])
                    s = _score_set(trial)
                    if s > cur_score:
                        U.append(c)
                        U.sort()
                        cur_score = s
                    else:
                        break
            else:
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
    # For larger n, use a greedy approach
    U = [0]
    for i in range(1, n):
        # Find smallest positive integer that doesn't create duplicate differences
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
    for n in range(3, 12):
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
    
    primes = primes_up_to(200)
    for k in range(2, 12):
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
    
    for steps in [[1, 3, 2], [1, 2, 3], [2, 3, 1], [1, 4, 2, 3]]:
        for k in range(3, 15):
            candidates.append(generate_pattern(steps, k))
    
    # Powers of 2
    for k in range(2, 12):
        candidates.append([0] + [2**i for i in range(k)])
    
    # Random sparse sets
    for max_val in [20, 40, 80, 160, 320]:
        for size in range(3, min(12, max_val // 5 + 1)):
            for _ in range(5):
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
        if _elapsed() > _START + 20:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Phase 2: Greedy construction
    greedy_U, greedy_score = _greedy_construct(max_size=15, max_val_limit=500, time_limit=25.0)
    if greedy_score > best_score:
        best_score = greedy_score
        best_U = greedy_U
    
    # Phase 3: Local search with add, remove, and replace operations
    U = list(best_U)
    cur_score = best_score
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and _elapsed() < _START + 115:
        improved = False
        no_improve_count = 0
        max_val = max(U) if U else 1
        
        # Try adding elements
        for _ in range(30):
            if _elapsed() > _START + 118:
                break
            c = rng.randint(1, int(max_val * 1.5) + 10)
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
                if _elapsed() > _START + 118:
                    break
                trial = U[:idx] + U[idx+1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    break  # Restart the loop after a removal
        
        # Try replacing one element with another
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > _START + 118:
                    break
                old_val = U[idx]
                # Try replacing with nearby values
                for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20]:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in U:
                        continue
                    if new_val > 2 * max_val:
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
    
    # Phase 4: Final polish - try small perturbations
    if _elapsed() < _START + 120:
        for _ in range(20):
            if _elapsed() > _START + 122:
                break
            # Random small perturbation
            if len(U) > 2:
                idx = rng.randint(1, len(U))
                delta = rng.randint(-3, 4)
                new_val = U[idx] + delta
                if new_val > 0 and new_val not in U:
                    trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                    s = _score_set(trial)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = trial
    
    return best_U
