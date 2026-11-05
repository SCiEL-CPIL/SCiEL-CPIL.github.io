"""Combined strategy: structured geometric/mixed candidates + fast local search in log-space."""
import time
import random
from math import log, exp

# EVOLVE-BLOCK-START

def _set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _is_valid(A):
    return len(A) == 20 and len(set(A)) == 20 and all(1 <= x <= 10000 for x in A)

def _try_candidate(A, best_set, best_obj):
    if not _is_valid(A):
        return best_set, best_obj
    obj = _objective(A)
    if obj < best_obj:
        best_obj = obj
        best_set = A[:]
    return best_set, best_obj

def _generate_structured_candidates():
    """Generate diverse structured candidate sets."""
    candidates = []
    
    # Family 1: Pure geometric sequences (integer ratio)
    for base in range(2, 100):
        max_start = 10000 // (base ** 19)
        if max_start < 1:
            continue
        for start_val in range(1, min(max_start + 1, 20)):
            A = []
            v = start_val
            valid = True
            for i in range(20):
                A.append(v)
                v *= base
                if v > 10000:
                    valid = False
                    break
            if valid and len(set(A)) == 20:
                candidates.append(A)
    
    # Family 2: Near-geometric with floating point ratio
    for ratio in [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]:
        for start_val in [1, 2, 3, 5, 10, 15, 20]:
            A = []
            v = float(start_val)
            valid = True
            for i in range(20):
                val = int(round(v))
                if val < 1:
                    val = 1
                if val > 10000:
                    valid = False
                    break
                A.append(val)
                v *= ratio
            if valid and len(set(A)) == 20:
                candidates.append(A)
    
    # Family 3: Powers of 2 times small odd multipliers
    for offset in [1, 2, 3, 5, 7]:
        for shift in [0, 1, 2, 3]:
            A = []
            valid = True
            for i in range(20):
                val = (2 ** i) * (1 + (i + shift) * offset)
                if val > 10000:
                    valid = False
                    break
                A.append(val)
            if valid and len(set(A)) == 20:
                candidates.append(A)
    
    # Family 4: Sparse geometric (every k-th power)
    for base in [2, 3, 4, 5]:
        for step in [1, 2, 3, 4]:
            A = [base ** (i * step) for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                candidates.append(A)
    
    # Family 5: Mixed - first half geometric, second half linear extension
    for base in [2, 3, 4, 5]:
        A = [base ** i for i in range(10)]
        last = base ** 9
        for i in range(10):
            A.append(last * (i + 2))
        if max(A) <= 10000 and len(set(A)) == 20:
            candidates.append(A)
    
    # Family 6: Log-uniform spread
    for start_log in [0.0, 0.5, 1.0]:
        for end_log in [9.0, 10.0]:
            A = []
            for i in range(20):
                log_val = start_log + (end_log - start_log) * i / 19.0
                val = int(round(exp(log_val)))
                if 1 <= val <= 10000:
                    A.append(val)
                else:
                    break
            if len(A) == 20 and len(set(A)) == 20:
                candidates.append(A)
    
    # Family 7: Arithmetic progressions
    for start_val in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 5, 10, 20, 50]:
            A = [start_val + i * step for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                candidates.append(A)
    
    return candidates

def _local_search(start_set, deadline, max_iters=20000):
    """Fast local search with log-space and additive perturbations."""
    A = list(start_set)
    cur_obj = _objective(A)
    best_A = A[:]
    best_obj = cur_obj
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        i = random.randrange(20)
        x = A[i]
        
        # Generate candidate neighbors
        cands = set()
        
        # Log-space multiplicative neighbors
        for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
            nv = int(round(x * factor))
            if 1 <= nv <= 10000 and nv not in A:
                cands.add(nv)
        
        # Additive neighbors
        for delta in [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]:
            nv = x + delta
            if 1 <= nv <= 10000 and nv not in A:
                cands.add(nv)
        
        # Random
        cands.add(random.randrange(1, 10001))
        
        cands.discard(x)
        cands = [c for c in cands if c not in A]
        if not cands:
            continue
        
        nv = random.choice(cands)
        old = A[i]
        A[i] = nv
        new_obj = _objective(A)
        
        if new_obj <= cur_obj:
            cur_obj = new_obj
            no_improve = 0
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = A[:]
        else:
            A[i] = old
            no_improve += 1
        
        if no_improve > 3000:
            break
    
    return best_A, best_obj

def construct():
    """Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|)."""
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin under 60s
    
    # Generate structured candidates
    candidates = _generate_structured_candidates()
    
    # Evaluate all candidates
    best_set = None
    best_obj = float('inf')
    
    for A in candidates:
        if time.time() > deadline - 10:
            break
        best_set, best_obj = _try_candidate(A, best_set, best_obj)
    
    # Fallback if no valid candidate found
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = _objective(best_set)
    
    # Local search from best candidate
    if time.time() < deadline - 5:
        best_set, best_obj = _local_search(best_set, deadline, max_iters=15000)
    
    # Try local search from a few other promising starts
    other_starts = []
    for A in candidates[:5]:
        if A != best_set and time.time() < deadline - 5:
            other_starts.append(A)
    
    for sp in other_starts:
        if time.time() > deadline - 3:
            break
        if sp is best_set:
            continue
        candidate_A, candidate_obj = _local_search(sp, deadline, max_iters=5000)
        if candidate_obj < best_obj:
            best_obj = candidate_obj
            best_set = candidate_A
    
    # Verify validity
    if not _is_valid(best_set):
        best_set = list(range(1, 21))
    
    return best_set

# EVOLVE-BLOCK-END
