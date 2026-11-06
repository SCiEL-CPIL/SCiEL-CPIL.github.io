"""
Program for integer sum-product extremal sets.
Approach: Deterministic grid search over truncated geometric progressions with
rational ratio p/q (small integers), combined with structured candidates
(1..N, powers of 2, etc.) and a deterministic local search to refine the best candidate.
"""
import math
import time

def _score_set(A):
    """Compute max(|A+A|, |A*A|) for a set/list of distinct positive integers."""
    if len(set(A)) != len(A):
        return float('inf')
    if any(x < 1 for x in A):
        return float('inf')
    # Sumset
    sset = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            sset.add(ai + A[j])
    # Productset
    pset = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            pset.add(ai * A[j])
    return max(len(sset), len(pset))

def construct():
    n = 20
    max_val = 10000
    best_score = float('inf')
    best_set = None
    
    # 1. Structured candidates
    candidates = []
    
    # 1..20
    candidates.append(list(range(1, n + 1)))
    
    # Powers of 2
    p2 = [2**i for i in range(n)]
    if p2[-1] <= max_val:
        candidates.append(p2)
    else:
        p2_fit = [2**i for i in range(n) if 2**i <= max_val]
        remaining = sorted(set(range(1, max_val + 1)) - set(p2_fit))
        p2_fit += remaining[:n - len(p2_fit)]
        candidates.append(p2_fit[:n])
    
    # Powers of 3
    p3 = [3**i for i in range(n)]
    if p3[-1] <= max_val:
        candidates.append(p3)
    else:
        p3_fit = [3**i for i in range(n) if 3**i <= max_val]
        remaining = sorted(set(range(1, max_val + 1)) - set(p3_fit))
        p3_fit += remaining[:n - len(p3_fit)]
        candidates.append(p3_fit[:n])
    
    # 2..21
    candidates.append(list(range(2, n + 2)))
    
    # 3..22
    candidates.append(list(range(3, n + 3)))
    
    # Evaluate all structured candidates
    for cand in candidates:
        if len(cand) != n:
            continue
        if len(set(cand)) != n:
            continue
        if any(x < 1 or x > max_val for x in cand):
            continue
        s = _score_set(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
    
    # 2. Grid search over rational GPs p/q
    for q in range(1, 10):
        for p in range(q + 1, q + 20):
            if math.gcd(p, q) != 1:
                continue
            q_pow = q ** (n - 1)
            p_pow = p ** (n - 1)
            if p_pow > max_val * 100:
                continue
            max_a = int(max_val * q_pow / p_pow)
            if max_a < 1:
                continue
            max_k = max_val // p_pow
            if max_k < 1:
                continue
            for k in range(1, min(max_k + 1, 50)):
                a = k * q_pow
                A = []
                valid = True
                for i in range(n):
                    term = k * (q ** (n - 1 - i)) * (p ** i)
                    if term > max_val:
                        valid = False
                        break
                    A.append(term)
                if not valid:
                    continue
                if len(set(A)) != n:
                    continue
                s = _score_set(A)
                if s < best_score:
                    best_score = s
                    best_set = A[:]
    
    # 3. Deterministic Local Search
    if best_set is None:
        best_set = list(range(1, n + 1))
    
    best_set.sort()
    current_score = best_score
    
    start_time = time.time()
    time_limit = 55
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            old_val = best_set[i]
            candidates_for_swap = set()
            
            # Neighbors
            for d in range(1, 30):
                candidates_for_swap.add(old_val - d)
                candidates_for_swap.add(old_val + d)
            
            # Multiples/Divisors of other elements
            for j in range(n):
                if j == i: continue
                other = best_set[j]
                for d in range(1, int(math.isqrt(other)) + 1):
                    if other % d == 0:
                        candidates_for_swap.add(d)
                        candidates_for_swap.add(other // d)
                for m in range(2, 6):
                    if other * m <= max_val:
                        candidates_for_swap.add(other * m)
            
            valid_candidates = [c for c in candidates_for_swap if 1 <= c <= max_val and c != old_val and c not in best_set]
            valid_candidates.sort()
            
            best_candidate = None
            best_candidate_score = current_score
            
            for new_val in valid_candidates:
                best_set[i] = new_val
                s = _score_set(best_set)
                if s < best_candidate_score:
                    best_candidate_score = s
                    best_candidate = new_val
            
            if best_candidate is not None:
                best_set[i] = best_candidate
                current_score = best_candidate_score
                improved = True
            else:
                best_set[i] = old_val
                
    return best_set
