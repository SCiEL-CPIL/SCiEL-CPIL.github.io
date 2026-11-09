import time
import numpy as np
from itertools import product

# Global state to handle the 300s budget and potential multiple calls
_init_done = False
_start_time = 0.0
_best_strategy_name = "spectral_random"
_cache = {}

# Pre-compute roots of unity for Z3
# omega = exp(2*pi*i/3)
# We work with real/imag parts or just use the property that 
# sum_{v in Z3} omega^{k v} = 3 if k%3==0 else 0.
# The Fourier transform of a single vector v (as a delta function) is constant magnitude 1.
# This suggests looking at the vector's "shape" in the frequency domain isn't directly 
# about the vector's own FT (which is flat), but rather its alignment with 
# subgroups or subspaces that support APs.

# A 3-AP in Z3^n is {x, x+d, x+2d}. 
# The "dangerous" directions are those d where many x form APs.
# Structurally, vectors that are "generic" (not lying on low-dimensional subspaces 
# defined by linear equations like sum(v)=const) are safer.

# Strategy: Spectral Randomness
# 1. Calculate the sum of coordinates (S1).
# 2. Calculate the sum of squared coordinates (S2).
# 3. Calculate a hash based on the "phase" of the vector in a complex embedding.
#    Map v in {0,1,2}^n to z in C^n where 0->1, 1->w, 2->w^2.
#    The "spectral norm" or "coherence" can be approximated by how aligned 
#    the vector is with the all-ones vector in this complex space.
#    Alignment = | sum(z_i) |. 
#    High alignment means the vector is "monochromatic" or close to it, 
#    which correlates with being in a large cap set? No, large cap sets are 
#    structured. But *greedy* success often comes from picking "central" vectors first.
    
# Let's try a different spectral angle: 
# Penalize vectors that are "linearly dependent" with the trivial subspace.
# Actually, the current "balanced" strategy is essentially maximizing the DC component 
# of the indicator function of the set. 
# Let's try maximizing the "entropy" of the coordinate distribution combined with 
# a high-frequency hash to break ties in a way that avoids local AP clusters.

def _spectral_priority(vector, n):
    """
    Computes a priority based on:
    1. Balance (distance from center 1.0 per coord) - favors central vectors.
    2. Spectral Randomness: A hash that mimics high-frequency noise to decorrelate 
       from simple geometric patterns.
    """
    # 1. Balance Score: Closer to n is better (sum of coords near n)
    s = sum(vector)
    balance_score = -abs(s - n)
    
    # 2. Variance Score: Prefer uniform distribution of 0,1,2
    #    Variance of the vector itself.
    mean = s / n
    var = sum((v - mean) ** 2 for v in vector) / n
    # Lower variance means more balanced 0,1,2 mix. 
    # Wait, if n is large, a vector like (1,1,...,1) has var 0. 
    # A vector like (0,0,...,2,2) has higher var.
    # Cap sets often contain many vectors with mixed coordinates.
    # Let's penalize high variance slightly, or prefer specific variance?
    # Actually, (1,1,...,1) is a good vector. (0,0,...,0) is also good.
    # The "balanced" heuristic just used sum.
    
    # 3. Spectral Hash: 
    # Use a multiplicative hash to create a pseudo-random "phase" 
    # that is invariant to simple shifts but breaks ties for vectors 
    # with similar sums.
    h = 5381
    for v in vector:
        h = (h * 33 + v) & 0xFFFFFFFF
        # Add a non-linear twist to make it less correlated with sum
        h ^= (h >> 13)
        h = (h * 0x5bd1e995) & 0xFFFFFFFF
    
    # Normalize hash to [0, 1]
    hash_score = (h % 1000000) / 1000000.0
    
    # Combine:
    # Primary: Balance (structural)
    # Secondary: Hash (decorrelation)
    # We want high balance (low abs diff) to be dominant.
    # So: priority = balance_score + small * hash_score
    
    # To make it "spectral", let's interpret the hash as a frequency component.
    # If two vectors have the same sum, the hash decides.
    # We weight the hash very low so it only breaks ties, but we use a 
    # "frequency-aware" hash (xorshift) to avoid linear correlations.
    
    return balance_score + 0.001 * hash_score

def _centered_priority(vector, n):
    """
    Alternative: Purely geometric centering.
    Vectors closer to (1,1,...,1) in L2 norm.
    """
    dist_sq = sum((v - 1) ** 2 for v in vector)
    return -dist_sq

def _random_priority(vector, n):
    """
    Baseline: Random.
    """
    h = 5381
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    return (h % 100000) / 100000.0

def _run_greedy_test(n, priority_func, limit_vectors=None):
    """
    Quick test of a strategy on a subset or full set if n is small.
    """
    if limit_vectors is None:
        vectors = list(product([0, 1, 2], repeat=n))
    else:
        vectors = list(product([0, 1, 2], repeat=n))[:limit_vectors]
        
    # For testing speed, if n is large, we can't run full greedy quickly.
    # But we can run it on n=5 or 6 for tuning.
    
    priorities = [priority_func(v, n) for v in vectors]
    indexed = sorted(range(len(vectors)), key=lambda i: -priorities[i])
    
    S = set()
    for idx in indexed:
        v = vectors[idx]
        v_tuple = tuple(v)
        can_add = True
        for u in S:
            w_candidate = tuple((2*vi - ui) % 3 for vi, ui in zip(v_tuple, u))
            if w_candidate in S:
                can_add = False
                break
        if can_add:
            S.add(v_tuple)
            
    return len(S)

def priority(vector, n):
    global _init_done, _start_time, _best_strategy_name, _cache
    
    if not _init_done:
        _start_time = time.time()
        deadline = _start_time + 15.0  # 15s for tuning
        
        # We need to pick the best strategy.
        # Strategies:
        strategies = {
            "spectral_random": _spectral_priority,
            "centered": _centered_priority,
            "random": _random_priority
        }
        
        # Test on n=5 or n=6 (small enough to be fast, large enough to show difference)
        # n=6 has 729 vectors. Greedy is O(N^2) approx. 729^2 ~ 500k ops. Very fast.
        # n=7 has 2187 vectors. 2187^2 ~ 4.7M ops. Still fast in Python if optimized? 
        # Python loops are slow. 4.7M * inner loop cost. 
        # Inner loop is O(|S|). Avg |S| ~ 20-30. 
        # So 4.7M * 30 ~ 140M ops. Might take ~5-10s in Python.
        # Let's stick to n=6 for tuning to be safe, or run n=7 with a timeout.
        
        test_n = 6
        best_score = -1
        best_name = "spectral_random"
        
        for name, func in strategies.items():
            if time.time() > deadline:
                break
            try:
                score = _run_greedy_test(test_n, func)
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception:
                pass
        
        _best_strategy_name = best_name
        _init_done = True
        
    # Ensure we don't spend too much time in the actual priority call
    # The priority function itself should be O(n) or O(1) ideally.
    # Our spectral_priority is O(n).
    
    if _best_strategy_name == "spectral_random":
        return _spectral_priority(vector, n)
    elif _best_strategy_name == "centered":
        return _centered_priority(vector, n)
    else:
        return _random_priority(vector, n)

# Expose the function
# The evaluator will call priority(vector, n)
