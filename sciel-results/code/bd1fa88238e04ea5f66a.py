"""
Strategy: Online Heuristic Parameter Optimization via Fast Simulation
Instead of learning a static priority map (which is slow to compute and may overfit to random restarts),
we define a parametric family of priority functions. We then use the compute budget to perform
hill-climbing on the parameters of this family. The "fitness" of a parameter set is the size of
the cap set produced by a fast, bitset-accelerated greedy algorithm using that priority ordering.

This approach:
1. Uses a fast bitset-based cap set checker (O(3^n / word_size) per insertion check roughly, or faster with precomputation).
2. Evaluates different "strategies" (parameter vectors) quickly.
3. Returns the priority function defined by the best parameters found.
"""

import time
import random
import sys
from collections import defaultdict

# Global state
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill
_BEST_PARAMS = None
_BEST_SIZE = 0
_PARAMS_CACHE = {}
_ISLAND_STATE = {}

def _get_deadline():
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _int_to_tuple(i, n):
    """Convert integer to tuple in Z_3^n."""
    res = [0] * n
    for k in range(n):
        res[k] = i % 3
        i //= 3
    return tuple(res)

def _tuple_to_int(v):
    """Convert tuple in Z_3^n to integer."""
    res = 0
    for i, x in enumerate(v):
        res += x * (3 ** i)
    return res

def _precompute_vectors(n):
    """Precompute all vectors and their features for fast access."""
    total = 3 ** n
    vectors = [_int_to_tuple(i, n) for i in range(total)]
    ints = list(range(total))
    
    # Features:
    # 1. Hamming weight (number of non-zeros)
    # 2. Sum of coordinates
    # 3. "Energy" (sum of squares, though in Z3 this is just weight for 0/1/2? 0^2=0, 1^2=1, 2^2=4=1. So sum of squares is same as weight for non-zeros if we map 2->1? No, 2^2=4=1 mod 3. So sum of squares is weight of non-zeros.)
    # Let's use:
    # f0: weight (count of non-zeros)
    # f1: sum of coords mod 3
    # f2: variance-like metric? Or just a hash for tie-breaking?
    # f3: "balancedness" - distance from mean?
    
    features = []
    for v in vectors:
        w = sum(1 for x in v if x != 0)
        s = sum(v) % 3
        # A more complex feature: product of coordinates? 
        # Or maybe: count of 1s and 2s separately?
        c1 = sum(1 for x in v if x == 1)
        c2 = sum(1 for x in v if x == 2)
        # Feature vector: [weight, c1, c2, sum_mod_3]
        features.append((w, c1, c2, s))
        
    return vectors, ints, features

def _fast_greedy_cap_set(n, params, vectors, ints, features, time_limit=None):
    """
    Run a fast greedy algorithm using the given params to score vectors.
    params: tuple of weights for features.
    Returns the size of the cap set found.
    """
    if time_limit is None:
        time_limit = 5.0
        
    start = time.time()
    
    # Calculate scores for all vectors
    # Score = sum(params[i] * features[i])
    # We want to sort by descending score.
    # To avoid sorting 3^n elements every time (expensive), we can bucket sort if scores are discrete?
    # Or just sort. For n=8, 6561 elements is small. For n=10, 59049 is okay.
    
    scores = []
    w0, w1, w2, w3 = params
    for i in range(len(vectors)):
        w, c1, c2, s = features[i]
        score = w0 * w + w1 * c1 + w2 * c2 + w3 * s
        scores.append((score, -i)) # -i for deterministic tie-breaker (prefer lower index if tie? or higher? let's use -i so higher index comes later in sort if stable? No, sort is ascending. We want descending score. So -score, i)
        
    # Sort by descending score, then by index for stability
    # Actually, let's just compute a rank.
    # For speed, we can use numpy if available, but standard sort is fine for n<=10.
    
    indices = list(range(len(vectors)))
    # Sort indices by score descending
    # Using key function on the pre-computed scores might be slow.
    # Let's create a list of (score, index) and sort it.
    score_indices = [(scores[i][0], i) for i in range(len(indices))]
    score_indices.sort(key=lambda x: (-x[0], x[1]))
    
    order = [idx for _, idx in score_indices]
    
    # Bitset for cap set check?
    # In Z_3^n, a set S is a cap set if no three distinct elements form an AP.
    # AP condition: a + b + c = 0 mod 3.
    # This is equivalent to: for any a, b in S, c = -(a+b) mod 3 must not be in S.
    # Or: S contains no 3-term AP.
    
    # Fast check:
    # Maintain a boolean array `in_set` of size 3^n.
    # When adding v, we need to check if there exist a, b in S such that a+b+v = 0.
    # This is equivalent to checking if (S + S) intersects with -v.
    # S+S is the set of all sums of pairs in S.
    # Computing S+S from scratch is O(|S|^2).
    
    # Alternative: 
    # For each new v, check all pairs (a,b) in current S.
    # If |S| is small (e.g., < 100), O(|S|^2) is fast.
    # Max cap set size for n=10 is ~ 100-200?
    # 200^2 = 40000 checks per insertion.
    # 60000 insertions * 40000 = 2.4 * 10^9 ops. Too slow in Python.
    
    # We need a faster check.
    # Use the fact that we are in Z_3^n.
    # We can precompute for each vector v, the list of pairs (a,b) that conflict? No, too big.
    
    # Let's stick to the O(|S|^2) check but optimize the inner loop.
    # Or, use a different representation.
    
    # Actually, for the simulation phase, we don't need the *exact* best cap set, we just need a *good* estimate of the quality of the ordering.
    # We can run the greedy for a limited number of steps or use a faster, approximate check?
    # No, the greedy result size is the metric.
    
    # Let's optimize the check:
    # in_set is a set of integers.
    # When adding v, for each a in S, b = -a - v mod 3.
    # Check if b is in S.
    # This is O(|S|) per insertion!
    # Because if a, b, v form an AP, then a+b+v=0 => b = -a-v.
    # So for a fixed v, we just iterate over a in S, compute b, and check membership.
    # If any b is in S, then v conflicts.
    
    # This reduces insertion check to O(|S|).
    # Total complexity: Sum_{i=1}^{K} i = K^2/2.
    # If K=200, 20000 ops.
    # 60000 insertions * 20000 = 1.2 * 10^9. Still might be slow in Python if not careful.
    # But we only run this for a few parameter candidates.
    
    in_set = set()
    count = 0
    
    for v_idx in order:
        if time.time() - start > time_limit:
            break
            
        v = vectors[v_idx]
        v_int = ints[v_idx]
        
        # Check conflict
        # For each a in in_set, check if b = -a - v is in in_set
        # Note: a, b, v must be distinct? 
        # If a == v, then b = -2v = v (since -2=1 in Z3). So a=b=v. This is a trivial AP (v,v,v).
        # Cap sets usually require distinct elements? 
        # "Cap set" usually means no three *distinct* points.
        # If a=b=v, it's not 3 distinct points.
        # If a=b != v, then 2a+v=0 => v = -2a = a. So a=v. Contradiction.
        # So the only case where a or b equals v is if all three are equal.
        # So we just need to check if there exist a, b in S (distinct from v? no, just in S) such that a+b+v=0 and a!=b and a!=v and b!=v?
        # Actually, if a=b, then 2a+v=0 => v=a. So if v is not in S, a cannot be equal to v.
        # So we just iterate a in S. If b is in S, we have a conflict.
        # We must ensure a != b? 
        # If a=b, then v=a. But v is not in S yet. So a cannot be in S if a=v.
        # So a and b are always distinct from v.
        # Are a and b distinct from each other?
        # If a=b, then v=a. Not possible.
        # So a and b are distinct.
        
        conflict = False
        for a_int in in_set:
            # Compute b = -a - v mod 3
            # We need to do this coordinate-wise.
            # This is slow in Python loops.
            
            # Optimization: Precompute "negative sums"?
            # Or use integer representation?
            # Addition in Z_3^n is not simple integer addition.
            
            # Let's use the tuple representation for the check, but it's slow.
            # Can we vectorize?
            # For n <= 10, tuples are short.
            
            a = vectors[a_int]
            # b_k = (-a_k - v_k) % 3
            # b_int = sum(b_k * 3^k)
            
            # This inner loop is the bottleneck.
            # Let's try to optimize by precomputing something?
            # Or just accept it's slow and limit the simulation size?
            
            # Alternative: Use a different check.
            # We can precompute for each pair (a,b) the sum a+b.
            # But S changes.
            
            # Let's just do the loop.
            b_int = 0
            for k in range(n):
                b_k = (-a[k] - v[k]) % 3
                b_int += b_k * (3 ** k)
                
            if b_int in in_set:
                conflict = True
                break
                
        if not conflict:
            in_set.add(v_int)
            count += 1
            
    return count

def _optimize_params(n, vectors, ints, features, time_budget):
    """
    Hill-climb on parameters to maximize cap set size.
    """
    global _BEST_PARAMS, _BEST_SIZE
    
    # Initial parameters: random
    # Feature space: w0 (weight), w1 (count 1s), w2 (count 2s), w3 (sum mod 3)
    # Let's start with a reasonable guess: prefer low weight?
    # Or balanced?
    
    # Random start
    params = [random.uniform(-1, 1) for _ in range(4)]
    
    best_size = 0
    best_params = params
    
    # Evaluate initial
    size = _fast_greedy_cap_set(n, params, vectors, ints, features, time_limit=min(5.0, time_budget * 0.1))
    best_size = size
    
    # Hill climbing
    step_size = 0.5
    while time.time() < _get_deadline() - 10.0:
        # Generate neighbor
        new_params = [p + random.uniform(-step_size, step_size) for p in best_params]
        
        # Evaluate
        size = _fast_greedy_cap_set(n, new_params, vectors, ints, features, time_limit=min(5.0, time_budget * 0.1))
        
        if size > best_size:
            best_size = size
            best_params = new_params
            # Increase step size slightly if improving? Or keep same?
            # step_size *= 1.1
        else:
            # Decrease step size
            step_size *= 0.9
            
        if step_size < 0.01:
            break
            
    _BEST_PARAMS = best_params
    _BEST_SIZE = best_size

def priority(vector, n):
    """
    Returns priority for a vector.
    """
    global _BEST_PARAMS, _ISLAND_STATE
    
    # Check if we have already optimized for this n
    if n not in _ISLAND_STATE:
        # Initialize state for this n
        # Precompute vectors
        vectors, ints, features = _precompute_vectors(n)
        _ISLAND_STATE[n] = {
            'vectors': vectors,
            'ints': ints,
            'features': features,
            'optimized': False
        }
        
        # Start optimization in background? No, we must return something.
        # We will do a quick optimization here.
        # Budget: Use up to 10 seconds for optimization on first call.
        time_budget = 10.0
        if time.time() + time_budget < _get_deadline():
            _optimize_params(n, vectors, ints, features, time_budget)
            _ISLAND_STATE[n]['optimized'] = True
        else:
            # Fallback: default params
            _BEST_PARAMS = [0.1, 0.1, 0.1, 0.0]
            _ISLAND_STATE[n]['optimized'] = True

    # Get best params
    if _BEST_PARAMS is None:
        _BEST_PARAMS = [0.1, 0.1, 0.1, 0.0]
        
    w0, w1, w2, w3 = _BEST_PARAMS
    
    # Calculate features for this vector
    w = sum(1 for x in vector if x != 0)
    c1 = sum(1 for x in vector if x == 1)
    c2 = sum(1 for x in vector if x == 2)
    s = sum(vector) % 3
    
    score = w0 * w + w1 * c1 + w2 * c2 + w3 * s
    
    # Add tiny tie-breaker based on integer representation to ensure determinism
    v_int = _tuple_to_int(vector)
    score += v_int * 1e-9
    
    return score

# EVOLVE-BLOCK-END
