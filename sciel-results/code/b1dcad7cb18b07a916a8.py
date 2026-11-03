import math
import time
import random

def _score(U):
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for u in U:
        for v in U:
            sumset.add(u + v)
            diffset.add(u - v)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _geometric_set(base, n):
    """Generate U = {floor(b^k) for k=0..n-1}, ensuring 0 is included."""
    if n <= 0:
        return []
    U = [0]
    if n > 1:
        val = 1
        for i in range(1, n):
            U.append(val)
            val = int(round(base ** i))
    return sorted(set(U))

def _score_geometric(base, n):
    """Compute score for a geometric set with given base and length."""
    U = _geometric_set(base, n)
    if not U or 0 not in U:
        return 0.0
    return _score(U)

def _ternary_search_1d(f, a, b, tol=1e-6, max_iter=100):
    """Ternary search for maximum of f on [a, b]."""
    for _ in range(max_iter):
        if b - a < tol:
            break
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            a = m1
        else:
            b = m2
    x = (a + b) / 2.0
    return x, f(x)

def _local_search(U, max_iterations=500, deadline=None):
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U
    current_U = U
    current_score = best_score
    
    for i in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        max_u = max(current_U)
        if random.random() < 0.5:
            new_val = random.randint(1, max_u * 2)
        else:
            new_val = random.randint(1, max_u + 10)
        
        candidate = sorted(set(current_U + [new_val]))
        s = _score(candidate)
        if s > current_score:
            current_U = candidate
            current_score = s
            if s > best_score:
                best_score = s
                best_U = candidate
        
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            candidate = sorted(set(current_U[:idx] + current_U[idx+1:]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate
        else:
            candidate = [0, 1]
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate

        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            shift = random.randint(-5, 5)
            if shift == 0:
                continue
            new_val = current_U[idx] + shift
            if new_val >= 0:
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) < len(current_U):
                    pass
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Continuous sweep over base b and length m
    # For each m, ternary search over b in [1.5, 10]
    for m in range(3, 12):
        if time.time() > deadline:
            break
        
        def f(b):
            return _score_geometric(b, m)
        
        # Ternary search over b
        best_b, best_f = _ternary_search_1d(f, 1.5, 10.0, tol=1e-5, max_iter=50)
        
        # Also check integer bases near the optimum
        for b_int in [max(2, int(best_b) - 1), int(best_b), int(best_b) + 1, int(best_b) + 2]:
            if b_int < 2:
                continue
            s = _score_geometric(b_int, m)
            if s > best_f:
                best_f = s
                best_b = float(b_int)
        
        if best_f > best_score:
            best_score = best_f
            best_U = _geometric_set(best_b, m)
    
    # Polish the best candidate with local search
    if time.time() < deadline:
        refined = _local_search(best_U, max_iterations=300, deadline=deadline)
        s_r = _score(refined)
        if s_r > best_score:
            best_score = s_r
            best_U = refined
    
    return best_U
