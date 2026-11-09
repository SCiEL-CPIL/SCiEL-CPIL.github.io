import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # 5s safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

def _sumset_size(A):
    # A is a list of ints; compute |A+A| with i<=j (equal allowed)
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] * A[j])
    return len(s)

def _objective(A):
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _gen_smooth_candidates(max_val=10000, primes=(2,3,5,7,11,13)):
    # Generate all numbers <= max_val whose prime factors are in primes
    cands = [1]
    for p in primes:
        new = []
        for c in cands:
            v = c
            while v * p <= max_val:
                v *= p
                new.append(v)
        cands += new
    cands = sorted(set(c for c in cands if c >= 2))
    return cands

def _random_initial(cands, n=20):
    # Try to get a good starting point: mix small values (for sums) and smooth (for products)
    # Use a mix: some small integers, some smooth numbers
    # Simple: random sample from cands, but ensure distinct
    while True:
        A = random.sample(cands, n)
        if _is_valid(A):
            return sorted(A)

def _swap_move(A, cands, idx, new_val):
    A = list(A)
    A[idx] = new_val
    return A

def _local_search(A, cands, max_iters=200000):
    A = sorted(A)
    best = _objective(A)
    n = len(A)
    # Precompute for speed: use numpy-free but efficient
    # We'll do swaps: pick two indices, swap with a candidate
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        # Random move: pick an index, replace with a random candidate
        i = random.randrange(n)
        # Choose candidate near A[i] or random
        if random.random() < 0.5:
            # Near current value
            lo = max(0, i - 3)
            hi = min(n, i + 4)
            # Replace with a candidate close to A[i]
            # Find candidate index in cands
            # For simplicity, pick random candidate
            new_val = random.choice(cands)
        else:
            new_val = random.choice(cands)
        
        if new_val in A and A.index(new_val) != i:
            continue
        A_new = list(A)
        A_new[i] = new_val
        if len(set(A_new)) != n:
            continue
        if any(x > 10000 for x in A_new):
            continue
        obj = _objective(A_new)
        if obj < best:
            best = obj
            A = A_new
    return A, best

def _greedy_construct(cands, n=20):
    # Start with small numbers, then add numbers that minimize objective increase
    A = list(range(2, 12))  # 10 small numbers
    # Extend to 20
    for _ in range(n - len(A)):
        best_cand = None
        best_obj = float('inf')
        for c in cands:
            if c in A:
                continue
            A_try = A + [c]
            obj = _objective(A_try)
            if obj < best_obj:
                best_obj = obj
                best_cand = c
        if best_cand is None:
            break
        A.append(best_cand)
    A = sorted(A[:n])
    return A

def construct():
    global _START_TIME
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    random.seed(42)
    
    # Generate smooth candidates
    cands = _gen_smooth_candidates(max_val=10000, primes=(2,3,5,7,11,13))
    
    # Also include some small integers for better sumset
    small_ints = list(range(2, 50))
    cands = sorted(set(cands + small_ints))
    
    # Initial solution: greedy
    A = _greedy_construct(cands, n=20)
    best_obj = _objective(A)
    
    # Local search with multiple restarts
    for restart in range(50):
        if _time_left() < 5.0:
            break
        # New initial
        A_init = _random_initial(cands, n=20)
        A_res, obj_res = _local_search(A_init, cands, max_iters=100000)
        if obj_res < best_obj:
            best_obj = obj_res
            A = A_res
            # Continue local search from best
            A, best_obj = _local_search(A, cands, max_iters=100000)
    
    return sorted(A)

# Quick self-test (not called by evaluator)
if __name__ == "__main__":
    A = construct()
    print("Solution:", A)
    print("|A+A| =", _sumset_size(A))
    print("|A*A| =", _prodset_size(A))
    print("max =", _objective(A))
