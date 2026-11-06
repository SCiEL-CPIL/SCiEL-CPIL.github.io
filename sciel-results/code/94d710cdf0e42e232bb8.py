<<<<<<< SEARCH
import time
import random

_N = 256
_DEADLINE = 80.0  # safety margin vs 90s kill
_BIG = 10**9


def _max_ap_abs(s):
    """Return the max absolute signed sum over ANY arithmetic progression in [1,N].

    s: list/array of 256 values in {-1,+1}, 0-indexed (position i is index i).
    For step d, progressions are contiguous sub-arrays within each residue class.
    The max absolute sum over such sub-arrays equals max(prefix)-min(prefix) per class.
    """
    n = len(s)
    worst = 0
    # Steps 1..n//2: each has at least 2 elements in some class -> use prefix trick.
    # Step d: there are d residue classes, class r has positions r, r+d, ...
    # For d > n//2, every class has at most 1 element -> handled by brute force below.
    half = n // 2
    for d in range(1, half + 1):
        # iterate over residue classes
        for r in range(d):
            # prefix sums over this class
            # collect in a tight loop
            # use local vars for speed
            cur = 0
            mn = 0
            mx = 0
            # walk the class
            # positions: r, r+d, r+2d, ...
            # unrolled-ish loop
            i = r
            # first element
            cur = s[i]
            mn = cur if cur < 0 else 0
            mx = cur if cur > 0 else 0
            # Actually we want max-min over prefix sums INCLUDING 0.
            # prefix[0]=0, prefix[k]=sum of first k elements.
            # max-min over {0, p1, p2, ...}
            mn = 0
            mx = 0
            cur = 0
            i = r
            while i < n:
                cur += s[i]
                if cur > mx:
                    mx = cur
                elif cur < mn:
                    mn = cur
                i += d
            span = mx - mn
            if span > worst:
                worst = span
    # Steps d > n//2: each class has at most 2 elements (only when d<=n and r+d<n).
    # For d in (n//2, n): class r has elements r and r+d if r+d<n.
    # So the only progressions of length>=2 with such step are pairs (r, r+d) with r+d<n,
    # i.e. r < n-d. Also singletons (|1|). Pairs have sum s[r]+s[r+d] in {-2,0,2}.
    # So worst over these is 2 if any pair sums to +/-2, else 1 (singletons).
    # We can just check: does there exist r,d with d>n//2 and r+d<n and s[r]==s[r+d]?
    # That's equivalent to: is there any pair (a,b) with b-a>n//2 and a<b<n and s[a]==s[b]?
    # Max b-a is n-1. So we need same sign at distance > n//2.
    # Simplest: for each r in 0..n-1, for d in n//2+1..n-1-r: check.
    # But that's O(n^2/2) ~ 32k, fine.
    # Even simpler: for each pair distance dd in n//2+1..n-1, check any r with s[r]==s[r+dd].
    # We already know singletons give 1. So worst is at least 1.
    # Determine if 2 achievable:
    # We can short-circuit: if worst>=2 already return worst.
    if worst < 2:
        # check for a same-sign pair at distance > n//2
        found = False
        for dd in range(half + 1, n):
            # r from 0 to n-dd-1
            # check s[r]==s[r+dd] for any r
            for r in range(n - dd):
                if s[r] == s[r + dd]:
                    found = True
                    break
            if found:
                break
        if found:
            worst = 2
        else:
            worst = 1  # singletons
    return worst
=======
import time
import random
import numpy as np

_N = 256
_DEADLINE = 50.0  # safety margin vs 60s kill
_BIG = 10**9

# Precompute indices for speed? No, N is small enough that np.take is fast.
# But we can precompute the slice objects or indices for each d, r?
# Let's just use numpy inside the function.

def _max_ap_abs(s):
    """
    Return the max absolute signed sum over ANY arithmetic progression in [1,N].
    s: list/array of 256 values in {-1,+1}, 0-indexed.
    """
    n = len(s)
    # Convert to numpy array once
    s_arr = np.array(s, dtype=np.int8)
    
    worst = 1 # Singletons always give 1
    
    # We only need to check steps d such that there is at least one AP of length >= 2.
    # For step d, max length is ceil(n/d).
    # If ceil(n/d) == 1, i.e., d >= n, no AP of length >= 2.
    # If d > n//2, max length is 2.
    # If d <= n//2, max length >= 3 (for d=1, len=n; for d=n//2, len=2 if n even? 
    #   n=256, d=128 -> indices 0, 128 (len 2); 1, 129 (len 2).
    #   Actually for d > n/2, all classes have length <= 2.
    #   For d <= n/2, some classes have length >= 3?
    #   d=128, n=256. Class 0: 0, 128. Len 2.
    #   So for d > n/2, max AP length is 2.
    #   For d <= n/2, max AP length is at least 2, potentially more.
    
    # Optimization: 
    # If we find worst >= 2, we might still need to check if it's higher.
    # But usually worst is small (2, 3, 4...).
    
    # Check steps d from 1 to n//2.
    # For each d, we have d residue classes.
    # For each class, compute prefix sums, find max-min.
    
    # To speed up, we can vectorize over residue classes for a fixed d.
    # However, the lengths of classes vary.
    # Alternative: Iterate d. For each d, iterate r. 
    # But Python loop over d (1..128) and r (1..d) is ~ 8000 iterations.
    # Inside, we do numpy ops on small arrays. This should be fast enough.
    
    # Even faster: 
    # For d in 1..n//2:
    #   We can extract all elements with step d? No, they are interleaved.
    #   We can use np.arange to get indices for each r?
    
    # Let's try a hybrid: 
    # For small d, the classes are long.
    # For large d (near n/2), classes are short (len 2 or 3).
    
    # Actually, for d > n/2, the max AP length is 2.
    # The sum is s[i] + s[i+d].
    # Max abs sum is max |s[i] + s[i+d]|.
    # Since s[i] in {-1,1}, sum is in {-2,0,2}.
    # So max is 2 if any pair matches, else 1 (but we already have worst>=1 from singletons, 
    # and if no pair matches, the max over pairs is 0? No, sum 0 is possible.
    # Wait, if s[i]=1, s[i+d]=-1, sum=0.
    # If all pairs are opposite, max abs sum from pairs is 0.
    # But singletons give 1. So worst is at least 1.
    # If any pair is same, worst is at least 2.
    
    # So for d > n//2, we just need to check if there is ANY i such that s[i]==s[i+d].
    # This can be done quickly with numpy:
    # For a fixed d, check np.any(s[:n-d] == s[d:]).
    
    # For d <= n//2, we need full prefix sum range.
    
    half = n // 2
    
    # 1. Check large steps (d > half) for value 2
    # If any such d has a matching pair, worst is at least 2.
    # We can check this in bulk? 
    # Or just loop d from half+1 to n-1.
    # There are ~128 such d.
    # For each, np.any(s[:n-d] == s[d:]) is very fast.
    
    has_large_2 = False
    # We can break early if we find one.
    # But we also need to check small steps for potentially higher values (3, 4, etc).
    # So we must check small steps anyway.
    # We can just let the small step check handle everything, and if the max found is 1,
    # we then check large steps to see if it's 2.
    # Or, we can compute the max over small steps, then check large steps.
    
    max_small = 1
    
    for d in range(1, half + 1):
        # For this d, we have d classes.
        # We want max over r of (max_prefix(r) - min_prefix(r)).
        
        # Vectorized approach for fixed d:
        # This is tricky because lengths vary.
        # However, for d <= n/2, the lengths are either floor(n/d) or ceil(n/d).
        # Actually, simpler: just loop r?
        # d=1: 1 class, len 256.
        # d=128: 128 classes, len 2.
        
        # Let's just loop r for now, but use numpy for the prefix sum part?
        # Or pure Python for the inner loop might be okay if d is large (short arrays).
        # For d=1, array is long.
        
        # Let's try to vectorize the whole d-loop?
        # Hard.
        
        # Let's optimize the inner loop with numpy.
        # For a fixed d, we can create an array of all elements?
        # No, they are separate classes.
        
        # Efficient way:
        # For each r, get indices: np.arange(r, n, d)
        # Get values: s_arr[indices]
        # Compute cumsum: np.cumsum(values)
        # Prepend 0: np.concatenate(([0], cumsum))
        # Range: max - min
        
        # Doing this for each r is O(n/d) numpy calls.
        # Total numpy calls for all d: sum(d for d in 1..128) = 128*129/2 = 8256 calls.
        # Each call is on small arrays. This should be < 1s.
        
        # Let's implement this.
        
        for r in range(d):
            idx = np.arange(r, n, d)
            if len(idx) <= 1:
                continue
            vals = s_arr[idx]
            # Compute prefix sums
            # cumsum includes the first element. We need prefix[0]=0.
            # p = [0, v0, v0+v1, ...]
            # np.cumsum(vals) gives [v0, v0+v1, ...]
            # So p = np.concatenate(([0], np.cumsum(vals)))
            # Or simply:
            # c = np.cumsum(vals)
            # mn = min(0, c.min())
            # mx = max(0, c.max())
            # span = mx - mn
            
            c = np.cumsum(vals)
            # c.min() and c.max() are fast
            mn = c.min()
            if mn > 0:
                mn = 0
            mx = c.max()
            if mx < 0:
                mx = 0
            span = mx - mn
            if span > max_small:
                max_small = span
                # Early exit if we find a huge value? Unlikely to be huge.
                # But if max_small >= 4, it's probably good enough? No, we want min.
    
    # Now check large steps for 2
    if max_small < 2:
        # Check if any d > half has a matching pair
        # We can check all d > half in a loop.
        # Or, is it possible that max_small is 1, but a large step gives 2? Yes.
        # Is it possible max_small is 1, and no large step gives 2? Then worst is 1.
        # But worst is at least 1.
        
        # Check large steps
        # For d in half+1 to n-1:
        #   if np.any(s_arr[:n-d] == s_arr[d:]):
        #       return 2
        
        # We can speed this up.
        # If max_small is 1, it means for all d <= half, all APs have sum in {-1,0,1}.
        # This is very restrictive.
        # Let's just check large steps.
        
        found_2 = False
        for d in range(half + 1, n):
            # Check if s[i] == s[i+d] for any i
            # s_arr[:n-d] and s_arr[d:]
            if np.any(s_arr[:n-d] == s_arr[d:]):
                found_2 = True
                break
        if found_2:
            return 2
        else:
            return 1
            
    return max_small
>>>>>>> REPLACE

