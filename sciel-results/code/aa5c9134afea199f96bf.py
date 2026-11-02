"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)

Strategy:
  Phase 1: Greedy construction using prime-based modular arithmetic to build
           a set where U-U is large relative to U+U.
  Phase 2: Local search / refinement to improve the score.
  We use a deadline guard to ensure we finish within the time limit.
"""
import time
import math
import random

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    maxU = max(U_set)
    if maxU == 0:
        return 0.0
    # Compute |U-U| and |U+U|
    diff_set = set()
    sum_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
            sum_set.add(a + b)
    log_arg = len(diff_set) / len(sum_set)
    if log_arg <= 0:
        return 0.0
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(log_arg) / denom

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _greedy_construct(target_size, deadline):
    """
    Build a set U greedily. We use a modular structure:
    Pick a modulus m, and for each residue class mod m, pick elements.
    The key insight: if U is a union of arithmetic progressions with carefully
    chosen moduli, |U-U| can be much larger than |U+U|.
    
    A better approach: use a product structure. Let U = {a_i * b : i in I, b in B}
    where B is a base set. Then U+U and U-U have specific structures.
    
    Actually, let's use the classical approach: pick U as a subset of [0, M]
    that is "sum-free" in a strong sense, making |U+U| small while |U-U| large.
    
    Key idea: If U is contained in an interval [0, M], then |U+U| <= 2M+1 and
    |U-U| <= 2M+1. But if U is structured (e.g., powers of 2 scaled), 
    |U+U| can be much smaller than |U-U|.
    
    Let's try: U = {sum of a_i * p_i : a_i in {0,1}} where p_i are primes.
    This gives |U| = 2^k, |U+U| <= 2 * sum(p_i), |U-U| up to 2^k * (range).
    
    Actually the best known constructions use:
    U = {a_1 * m_1 + ... + a_k * m_k : a_i in [0, b_i]}
    with m_i growing fast enough that sums are unique.
    
    Let me try a simpler greedy: start with {0}, then at each step add the
    element x that maximizes the score.
    """
    U = [0]
    U_set = {0}
    best_score = _score(U)
    
    # Try adding elements greedily
    # Candidate pool: try numbers up to some bound
    max_candidate = 200000
    
    for step in range(target_size - 1):
        if time.time() - _START_TIME > deadline:
            break
        
        best_x = None
        best_s = best_score
        
        # Sample candidates to keep it fast
        # Try a mix of small and large candidates
        candidates = set()
        # Small candidates
        for x in range(1, min(200, max_candidate + 1)):
            if x not in U_set:
                candidates.add(x)
        # Larger candidates based on current max
        current_max = max(U_set) if U_set else 0
        for multiplier in range(2, 20):
            x = current_max * multiplier
            if x < max_candidate and x not in U_set:
                candidates.add(x)
        # Some random candidates
        for _ in range(50):
            x = random.randint(1, max_candidate)
            if x not in U_set:
                candidates.add(x)
        # Powers of 2 and 3
        p = 2
        while p < max_candidate:
            if p not in U_set:
                candidates.add(p)
            p *= 2
        p = 3
        while p < max_candidate:
            if p not in U_set:
                candidates.add(p)
            p *= 3
        
        for x in candidates:
            if x in U_set:
                continue
            new_U_set = U_set | {x}
            # Quick estimate: compute |U-U| and |U+U| approximately
            # For speed, use the formula directly
            diff_set = set()
            sum_set = set()
            for a in new_U_set:
                for b in new_U_set:
                    diff_set.add(a - b)
                    sum_set.add(a + b)
            maxU = max(new_U_set)
            log_arg = len(diff_set) / len(sum_set)
            if log_arg <= 0:
                continue
            denom = math.log(2 * maxU + 1)
            if denom == 0:
                continue
            s = 1.0 + math.log(log_arg) / denom
            if s > best_s + 1e-12:
                best_s = s
                best_x = x
        
        if best_x is not None:
            U.append(best_x)
            U_set.add(best_x)
            best_score = best_s
        else:
            # No improvement found with current candidates
            break
    
    return U, best_score

def _local_search(U, deadline):
    """
    Local search: try removing elements, swapping elements, or adding new ones
    to improve the score.
    """
    U = list(U)
    U_set = set(U)
    best_score = _score(U)
    
    while time.time() - _START_TIME < deadline:
        improved = False
        
        # Try removing each element
        for i in range(len(U)):
            if time.time() - _START_TIME > deadline:
                break
            new_U = U[:i] + U[i+1:]
            if len(new_U) < 2:
                continue
            s = _score(new_U)
            if s > best_score + 1e-12:
                U = new_U
                U_set = set(U)
                best_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping each element with a candidate
        current_max = max(U_set) if U_set else 0
        candidates = set()
        for x in range(1, min(500, current_max * 3 + 100)):
            if x not in U_set:
                candidates.add(x)
        for _ in range(100):
            x = random.randint(1, current_max * 5)
            if x not in U_set:
                candidates.add(x)
        
        for i in range(len(U)):
            if time.time() - _START_TIME > deadline:
                break
            for x in candidates:
                if x in U_set:
                    continue
                new_U = U[:i] + [x] + U[i+1:]
                s = _score(new_U)
                if s > best_score + 1e-12:
                    U = new_U
                    U_set = set(U)
                    best_score = s
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding a new element
            best_x = None
            best_s = best_score
            for x in candidates:
                if x in U_set:
                    continue
                new_U = U + [x]
                s = _score(new_U)
                if s > best_s + 1e-12:
                    best_s = s
                    best_x = x
            
            if best_x is not None:
                U.append(best_x)
                U_set.add(best_x)
                best_score = best_s
            else:
                break
    
    return U, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start = time.time()
    deadline = start + 110.0  # 110s deadline for safety
    
    # Phase 1: Greedy construction
    # Try different target sizes and keep the best
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Start with a small set and grow
    for target_size in [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 500]:
        if time.time() > deadline:
            break
        
        U, s = _greedy_construct(target_size, deadline)
        if s > best_score:
            best_U = U
            best_score = s
    
    # Phase 2: Local search refinement
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, deadline)
    
    # Ensure 0 is in the set and it's a list of nonnegative integers
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U
