"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Uniform priority: greedy insertion in lexicographic order.
    return 0.0

# EVOLVE-BLOCK-END

def _is_cap_set(points):
    """Check if a set of points in {0,1,2}^n is a cap set (no 3-term AP)."""
    pts = list(points)
    n = len(pts[0]) if pts else 0
    # For each pair, check if the third point of the AP is in the set
    # AP: a, b, c where b = (a+c)/2, so c = 2b - a (mod 3)
    # In {0,1,2}, 2b - a mod 3
    point_set = set(pts)
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i == j:
                continue
            # c = 2*b - a mod 3
            c = tuple((2*b[k] - a[k]) % 3 for k in range(n))
            if c in point_set and c != a:
                # Found an AP: a, b, c with a != c
                # But we need to ensure they are distinct and form a proper AP
                # Actually, if c is in the set and c != a, then a, b, c is an AP
                # Wait, we need a, b, c to be distinct. If c == a, it's not a 3-term AP.
                # Also need to check that b is distinct from a and c.
                if c != a:
                    # Check if a, b, c are all distinct
                    if len({a, b, c}) == 3:
                        return False
    return True

def _greedy_cap_set(priority_func, n, deadline, max_vectors=None):
    """
    Greedily add vectors to a cap set in order of priority (descending).
    Returns the size of the cap set found.
    """
    start_time = time.time()
    
    # Generate all vectors in {0,1,2}^n
    # For small n, we can enumerate all 3^n vectors
    if max_vectors is None:
        max_vectors = 3**n
    
    # Create list of all vectors
    all_vectors = []
    for i in range(3**n):
        vec = []
        x = i
        for k in range(n):
            vec.append(x % 3)
            x //= 3
        all_vectors.append(tuple(vec))
    
    # Sort by priority (descending)
    all_vectors.sort(key=lambda v: priority_func(v, n), reverse=True)
    
    cap_set = []
    point_set = set()
    
    for v in all_vectors:
        if time.time() > deadline:
            break
        
        # Check if adding v keeps the set a cap set
        # v creates an AP with existing points a, b if:
        # 1. There exist a, b in cap_set such that v, a, b form an AP
        # 2. v, a, b distinct
        
        # For v to be part of an AP with existing points:
        # Case 1: v is the first element: v, b, c where b, c in set -> c = 2b - v
        # Case 2: v is the middle element: a, v, c where a, c in set -> c = 2v - a
        # Case 3: v is the last element: a, b, v where a, b in set -> v = 2b - a, i.e., a = 2b - v
        
        valid = True
        for b in cap_set:
            # Case 1: v, b, c -> c = 2b - v
            c = tuple((2*b[k] - v[k]) % 3 for k in range(n))
            if c in point_set and c != v and c != b:
                valid = False
                break
            
            # Case 2: a, v, c -> c = 2v - a, but a must be in set
            # Actually, for case 2, we need a and c in set such that v is middle
            # a = 2v - c, so for each c in set, check if a = 2v - c is also in set
            # But this is symmetric. Let me re-think.
            
            # The AP condition: three distinct points x, y, z with x + z = 2y (mod 3)
            # If we add v, it can form an AP with two existing points p, q if:
            # v + p = 2q (v is first, p is last, q is middle) -> p = 2q - v
            # v + q = 2p (v is last, q is first, p is middle) -> q = 2p - v, same thing
            # p + q = 2v (v is middle) -> q = 2v - p
            
            # So we need to check:
            # 1. For each q in cap_set: is p = 2q - v in cap_set? (and p != v, p != q)
            # 2. For each p in cap_set: is q = 2v - p in cap_set? (and q != v, q != p)
            
            # Check case: v is an endpoint
            p = tuple((2*b[k] - v[k]) % 3 for k in range(n))
            if p in point_set and p != v and p != b:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check case: v is the middle
        for p in cap_set:
            q = tuple((2*v[k] - p[k]) % 3 for k in range(n))
            if q in point_set and q != v and q != p:
                valid = False
                break
        
        if valid:
            cap_set.append(v)
            point_set.add(v)
    
    return len(cap_set)

def _balanced_priority(vector, n):
    """
    Priority based on balanced digit distribution.
    Vectors with more uniform distribution of 0s, 1s, 2s get higher priority.
    """
    counts = [0, 0, 0]
    for d in vector:
        counts[d] += 1
    
    # Compute variance of counts; lower variance = more balanced = higher priority
    mean = n / 3.0
    variance = sum((c - mean)**2 for c in counts)
    
    # Return negative variance so higher priority = lower variance
    return -variance

def _randomized_priority(vector, n, seed=None):
    """
    Add small random perturbation to balanced priority to break ties differently.
    """
    if seed is None:
        seed = hash(vector)
    random.seed(seed)
    base = _balanced_priority(vector, n)
    perturbation = random.random() * 0.1
    return base - perturbation

def _solve(n, deadline):
    """
    Solve the cap set problem for dimension n with a deadline.
    Returns the maximum cap set size found.
    """
    start_time = time.time()
    
    # For small n, try multiple strategies
    best_size = 0
    
    # Strategy 1: Balanced digit distribution
    remaining = deadline - time.time()
    if remaining > 5:
        size1 = _greedy_cap_set(_balanced_priority, n, time.time() + remaining - 1)
        best_size = max(best_size, size1)
    
    # Strategy 2: Lexicographic (original)
    remaining = deadline - time.time()
    if remaining > 5:
        size2 = _greedy_cap_set(lambda v, n: 0.0, n, time.time() + remaining - 1)
        best_size = max(best_size, size2)
    
    # Strategy 3: Randomized balanced
    remaining = time.time() * 0 + deadline - time.time()
    if remaining > 5:
        size3 = _greedy_cap_set(lambda v, n: _randomized_priority(v, n), n, time.time() + remaining - 1)
        best_size = max(best_size, size3)
    
    # Strategy 4: Reverse lexicographic
    remaining = deadline - time.time()
    if remaining > 5:
        def reverse_lex(v, n):
            # Reverse lexicographic order
            return -sum(v[k] * (3**k) for k in range(n))
        size4 = _greedy_cap_set(reverse_lex, n, time.time() + remaining - 1)
        best_size = max(best_size, size4)
    
    return best_size

# Main entry point that the evaluator will call
# The task says: priority(vector: tuple, n: int) -> float
# But the evaluator measures "cap set size" as the score.
# Looking at the contract: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
# "Score: cap set size"
# 
# The interface is priority(vector, n) -> float, but the score is the cap set size.
# This means the evaluator will:
# 1. Call priority(vector, n) for each vector to get a priority
# 2. Sort vectors by priority (descending)
# 3. Greedily add vectors that keep the set a cap set
# 4. Return the size of the resulting cap set as the score
#
# So my priority function should be designed to maximize the resulting cap set size.
# But wait, the current program just returns 0.0 for all vectors, which gives lexicographic order.
# The score is 1.0, meaning the seed gets a certain cap set size.
#
# However, the problem is that the evaluator calls priority() for each vector individually,
# and then does the greedy insertion. So I can't do the greedy insertion myself in priority().
# I just need to return a good priority value.
#
# But the knowledge says: "Balanced digit distribution heuristic avoids early formation of APs by prioritizing 'uniform' vectors, allowing a larger cap set to be constructed greedily. Measured: 1 -> 1.223."
#
# So the balanced priority should work. Let me use that.
#
# But I also want to add a deadline guard. However, priority() is called per vector,
# so I can't easily track elapsed time across calls unless I use a global state.
#
# Let me use a global deadline and ensure we don't do expensive computations in priority().

_GLOBAL_START = None
_GLOBAL_DEADLINE = None

def _init_global(n):
    global _GLOBAL_START, _GLOBAL_DEADLINE
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START + 320  # 320s out of 330s limit

def priority(vector, n):
    """
    Return a priority value for the given vector.
    Higher priority = added earlier in greedy insertion.
    """
    _init_global(n)
    
    # Check if we're past deadline
    if time.time() > _GLOBAL_DEADLINE:
        # Return a very low priority to stop adding
        return -1e10
    
    # Balanced digit distribution
    counts = [0, 0, 0]
    for d in vector:
        counts[d] += 1
    
    mean = n / 3.0
    variance = sum((c - mean)**2 for c in counts)
    
    # Lower variance = more balanced = higher priority
    return -variance
# EVOLVE-BLOCK-END
