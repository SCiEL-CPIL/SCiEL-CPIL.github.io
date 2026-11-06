"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _max_abs_arith_sum(signs):
    """
    Compute the maximum absolute signed sum over ALL arithmetic progressions
    contained in [1, n] where signs is a list of length n with entries -1 or +1.
    An arithmetic progression is defined by start s (1-indexed), step d >= 1,
    and length L >= 1, such that s, s+d, ..., s+(L-1)*d are all <= n.
    """
    n = len(signs)
    best = 0
    # step d from 1 to n//2 (step > n//2 can only have length 1, which is covered by d=1 or singletons)
    # Actually step can be up to n-1, but for step >= n//2, max length is 1 or 2.
    # Let's iterate d from 1 to n-1, but break early if possible.
    # For efficiency, note that for a given d, the number of APs is O(n^2/d) roughly.
    # Total work is O(n^2 * H_n) which for n=256 is about 256^2 * 6 ~ 393k * 6 ~ 2.3M operations. Fine.
    
    # Singletons: sum = signs[i], abs = 1. So best >= 1.
    best = 1
    
    for d in range(1, n):
        # For this step, iterate over all starting positions
        # start s from 1 to n (1-indexed), index s-1
        # The AP is s, s+d, s+2d, ... <= n
        # For each start, compute the prefix sums of the AP and find max abs.
        # We can do this incrementally.
        
        # For a fixed d, the APs are partitioned by residue class mod d.
        # Residue r (0-indexed: r in 0..d-1) corresponds to positions r+1, r+1+d, ...
        # Actually 1-indexed: position s, s+d, ... where s mod d = r (with appropriate mapping).
        
        # Let's just iterate start from 1 to n, and for each, walk the AP.
        # But that's O(n^2) per d, total O(n^3) = 256^3 = 16M, still fine.
        
        for start in range(1, n + 1):
            # Walk the AP: start, start+d, start+2d, ...
            s = 0
            # We want to track prefix sums and find max abs
            # The AP elements: signs[start-1], signs[start-1+d], ...
            # We can compute all prefix sums and track max abs.
            
            # For efficiency, let's collect the indices and compute.
            # idx = start - 1 + k*d for k=0,1,... while <= n-1
            # Let's just walk.
            
            cur = 0
            idx = start - 1
            while idx < n:
                cur += signs[idx]
                abs_cur = abs(cur)
                if abs_cur > best:
                    best = abs_cur
                # Early termination: if best is already large, we could break, but let's not for simplicity.
                idx += d
            # We could also break the outer start loop if best is already very high, but let's keep it simple.
    
    return best


def construct():
    """
    Construct 256 signs (-1 or +1) to minimize the maximum absolute signed sum
    over all arithmetic progressions.
    """
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    n = 256
    
    # Start with a random initialization
    rng = random.Random(42)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_score = _max_abs_arith_sum(signs)
    best_signs = signs[:]
    
    # Local search: flip one sign at a time, accept if it improves
    # Use a simple hill-climbing with random restarts
    
    # Precompute for each position i, which APs it belongs to, to speed up evaluation?
    # That's complex. Let's just use the full evaluation for now.
    
    # To speed up, we can use a "delta" approach: when flipping sign at position i,
    # only APs containing i change. But enumerating those is also complex.
    
    # Let's do a simple approach: try flipping each position, evaluate, keep best.
    # This is O(n * cost_eval) per iteration. cost_eval is ~2M operations.
    # n * cost_eval = 256 * 2M = 512M per full pass. That's too slow in Python.
    
    # We need a faster approach. Let's use the delta method.
    
    # For each position i, precompute the list of (d, start, k) where i is the k-th element
    # of the AP (start, d). Actually, for each AP containing i, we need to know the sum
    # of the AP and how it changes when we flip i.
    
    # Alternative: use a faster evaluation with numpy.
    
    import numpy as np
    
    def _max_abs_arith_sum_fast(signs_np):
        """Faster version using numpy."""
        n = len(signs_np)
        best = 1
        for d in range(1, n):
            # For each residue class mod d
            for r in range(d):
                # Positions: r, r+d, r+2d, ... (0-indexed)
                # These form an AP with step d in 0-indexed, which corresponds to
                # 1-indexed start = r+1, step d.
                # Get the subarray
                idx = np.arange(r, n, d)
                if len(idx) < 1:
                    continue
                sub = signs_np[idx]
                # Compute prefix sums
                prefix = np.cumsum(sub)
                max_abs = np.max(np.abs(prefix))
                if max_abs > best:
                    best = max_abs
                    # Early exit if best is already very high
                    if best > 20:  # Arbitrary threshold
                        break
            if best > 20:
                break
        return best
    
    signs_np = np.array(signs)
    best_score = _max_abs_arith_sum_fast(signs_np)
    best_signs = signs[:]
    
    # Local search with numpy
    def _try_flip(signs_np, i):
        """Try flipping position i, return new score and whether it's better."""
        old_val = signs_np[i]
        signs_np[i] = -old_val
        new_score = _max_abs_arith_sum_fast(signs_np)
        signs_np[i] = old_val
        return new_score
    
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(n):
            if time.time() > deadline:
                break
            new_score = _try_flip(signs_np, i)
            if new_score < best_score:
                best_score = new_score
                signs_np[i] = -signs_np[i]
                best_signs = signs_np.tolist()
                improved = True
    
    # If we have time, do random restarts
    while time.time() < deadline - 5.0:
        # Random restart
        signs_np = np.array([rng.choice((-1, 1)) for _ in range(n)])
        cur_score = _max_abs_arith_sum_fast(signs_np)
        if cur_score < best_score:
            best_score = cur_score
            best_signs = signs_np.tolist()
        
        # Local search from this restart
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                new_score = _try_flip(signs_np, i)
                if new_score < best_score:
                    best_score = new_score
                    signs_np[i] = -signs_np[i]
                    best_signs = signs_np.tolist()
                    improved = True
    
    return best_signs

# EVOLVE-BLOCK-END
