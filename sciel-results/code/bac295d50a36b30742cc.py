"""Cap set construction via randomized greedy with adaptive scoring."""
import time
import random
import sys

# Deadline guard: must finish well before 330s wall-clock
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def _is_cap_set_fast(S, n):
    """
    Check if set S (list of tuples of 0,1,2 of length n) is a cap set.
    A cap set has no 3-term arithmetic progression.
    Three distinct elements a, b, c form an AP if a + c = 2*b (mod 3).
    Equivalently, for any two distinct elements a, c in S, the midpoint
    b = (a+c)/2 mod 3 must not be in S.
    
    Fast check: for each pair (a, c), compute midpoint and check membership.
    """
    S_set = set(S)
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            c = S[j]
            # Compute midpoint mod 3
            mid = tuple((a[k] + c[k]) // 2 % 3 for k in range(n))
            # Check if midpoint is in S and distinct from a and c
            if mid in S_set and mid != a and mid != c:
                return False
    return True

def _greedy_with_priority(priorities, n, max_size=None):
    """
    Greedily add vectors in order of decreasing priority, keeping cap set property.
    priorities: list of (vector, priority_value) sorted descending by priority.
    Returns the cap set found.
    """
    S = []
    S_set = set()
    for vec, _ in priorities:
        # Check if adding vec maintains cap set
        ok = True
        for existing in S:
            # Check if vec and existing form an AP with any other element
            # Actually, we need to check: does adding vec create a 3-AP with any two existing?
            # For each pair (vec, e) in S, check if midpoint is in S or is vec/e
            # More precisely: for any two existing elements a, b, check if midpoint with vec
            # is in S, or if midpoint of (vec, a) is in S, etc.
            
            # A new element v creates a 3-AP if there exist a, b in S such that:
            # v = (a+b)/2 mod 3, or a = (v+b)/2 mod 3, or b = (v+a)/2 mod 3
            # i.e., 2*v = a+b mod 3, or 2*a = v+b mod 3, or 2*b = v+a mod 3
            
            # Check 2*v = a + b (mod 3) for some a, b in S
            # This is equivalent to: for each a in S, check if (2*v - a) mod 3 is in S
            # But we also need a != b and the three to be distinct.
            
            # Simpler: for each existing element e in S:
            #   midpoint of (vec, e) = ((vec[k]+e[k])//2 % 3 for k in range(n))
            #   if midpoint is in S_set and midpoint != vec and midpoint != e: violation
            #   also check if vec is midpoint of (e, some other)
            #   Actually, the full check: adding vec to S creates a 3-AP iff
            #   there exist a, b in S (a != b) such that one of {vec, a, b} is the midpoint
            #   of the other two.
            
            # Let's just do the full check for correctness:
            # After adding vec, check all pairs involving vec
            pass
        
        # Do the check properly
        # For each pair (vec, e) where e in S:
        #   mid = midpoint(vec, e)
        #   if mid in S_set and mid != vec and mid != e: not OK
        # For each pair (e1, e2) where e1, e2 in S, e1 != e2:
        #   mid = midpoint(e1, e2)
        #   if mid == vec: not OK (vec is the midpoint of e1, e2)
        
        # Efficient check:
        # 1. For each e in S: mid = midpoint(vec, e); if mid in S_set and mid != vec and mid != e: fail
        # 2. Check if vec is midpoint of any two in S: for each e in S, other = (2*vec - e) mod 3; if other in S_set and other != e and other != vec: fail
        
        for e in S:
            # Check midpoint of (vec, e)
            mid = tuple((vec[k] + e[k]) // 2 % 3 for k in range(n))
            if mid in S_set and mid != vec and mid != e:
                ok = False
                break
            # Check if vec is midpoint of (e, other)
            # 2*vec = e + other => other = 2*vec - e mod 3
            other = tuple((2 * vec[k] - e[k]) % 3 for k in range(n))
            if other in S_set and other != e and other != vec:
                ok = False
                break
            if not ok:
                break
        
        if ok:
            S.append(vec)
            S_set.add(vec)
            if max_size and len(S) >= max_size:
                break
    
    return S

def _score_vector(vec, n, S):
    """
    Score a vector based on how 'compatible' it is with the current set S.
    Higher score means more likely to be addable without creating APs.
    """
    score = 0.0
    # Count how many existing elements would create conflicts
    for e in S:
        mid = tuple((vec[k] + e[k]) // 2 % 3 for k in range(n))
        # If mid is in S (and not vec or e), that's a conflict
        # We can't easily check membership here without a set, so approximate
        # Instead, use a heuristic: vectors that are "far" from existing ones are better
        # Use Hamming distance as proxy
        dist = sum(1 for k in range(n) if vec[k] != e[k])
        score += dist  # Prefer vectors far from existing ones
    return score

def priority(vector, n):
    """
    Main entry point. The evaluator calls this with vectors of {0,1,2}^n.
    We build a cap set greedily using adaptive priorities.
    
    Strategy: 
    1. Use multiple randomized priority schemes
    2. For each scheme, do greedy insertion
    3. Keep the best cap set found
    """
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 320.0
    
    best_set = []
    best_size = 0
    
    # Generate all vectors
    all_vectors = []
    for i in range(3 ** n):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        all_vectors.append(tuple(v))
    
    random.seed(42)
    
    # Strategy 1: Random priorities, multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if _time_remaining() < 5.0:
            break
        
        # Generate random priorities
        priorities = [(v, random.random()) for v in all_vectors]
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        S = _greedy_with_priority(priorities, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        
        # Also try with a biased priority: favor vectors with more 1s
        if restart % 3 == 0:
            def biased_score(v):
                return sum(1 for x in v if x == 1) + random.random() * 0.1
            priorities2 = [(v, biased_score(v)) for v in all_vectors]
            priorities2.sort(key=lambda x: x[1], reverse=True)
            S2 = _greedy_with_priority(priorities2, n)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = S2
        
        # Try with bias toward 0s
        if restart % 3 == 1:
            def biased_score2(v):
                return sum(1 for x in v if x == 0) + random.random() * 0.1
            priorities3 = [(v, biased_score2(v)) for v in all_vectors]
            priorities3.sort(key=lambda x: x[1], reverse=True)
            S3 = _greedy_with_priority(priorities3, n)
            if len(S3) > best_size:
                best_size = len(S3)
                best_set = S3
        
        # Try with bias toward 2s
        if restart % 3 == 2:
            def biased_score3(v):
                return sum(1 for x in v if x == 2) + random.random() * 0.1
            priorities4 = [(v, biased_score3(v)) for v in all_vectors]
            priorities4.sort(key=lambda x: x[1], reverse=True)
            S4 = _greedy_with_priority(priorities4, n)
            if len(S4) > best_size:
                best_size = len(S4)
                best_set = S4
    
    # Local search: try to swap elements to improve
    if _time_remaining() > 10.0 and best_set:
        # Try adding random vectors not in best_set
        S_set = set(best_set)
        remaining = [v for v in all_vectors if v not in S_set]
        random.shuffle(remaining)
        
        for v in remaining[:1000]:  # Try up to 1000 random additions
            if _time_remaining() < 5.0:
                break
            new_S = best_set + [v]
            if _is_cap_set_fast(new_S, n):
                best_set = new_S
                S_set.add(v)
    
    return float(best_size)

# The actual priority function that the evaluator calls
# The evaluator adds vectors greedily in descending priority
# So priority(vector, n) should return a scalar that determines order
# We need to return a value such that when sorted descending, we get a good cap set

# Since the evaluator does the greedy insertion, we just need to return priorities
# that lead to a large cap set.

# Let's use a smarter approach: return priorities based on a precomputed strategy
# But we don't know n ahead of time...

# Actually, re-reading the task: "vectors of {0,1,2}^n are added greedily in descending priority"
# So the evaluator has all 3^n vectors, sorts them by priority(vector, n) descending,
# and greedily adds them if the set remains a cap set.
# We just need to return a good priority for each vector.

# Let's use a combination of features:
# - Sum of coordinates
# - Number of 1s
# - Random component (but must be deterministic for same vector)
# - Use a hash-based random priority for diversity

def priority(vector, n):
    """
    Return a priority value for the given vector.
    Vectors with higher priority are considered first in greedy insertion.
    """
    # Use a multi-feature scoring
    total = sum(vector)
    num_ones = sum(1 for x in vector if x == 1)
    num_zeros = sum(1 for x in vector if x == 0)
    num_twos = sum(1 for x in vector if x == 2)
    
    # Use a hash-based deterministic random component
    h = hash(vector) % 1000000
    rand_comp = h / 1000000.0
    
    # Combine features
    # Prefer vectors with balanced composition
    balance = -abs(num_ones - num_zeros) - abs(num_ones - num_twos) - abs(num_zeros - num_twos)
    
    score = 10.0 * balance + 5.0 * rand_comp + 0.1 * total
    
    return score
