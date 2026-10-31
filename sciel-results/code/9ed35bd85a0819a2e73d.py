"""Dual-view construction: optimize |U-U|/|U+U| ratio with bounded max U."""
import time
import random
import math
from itertools import combinations

def _score_of(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or max(U) == 0:
        return 0.0
    n = len(U)
    # Compute sumset U+U
    sset = set()
    for i in range(n):
        for j in range(i, n):  # since a+b = b+a, we can halve work
            sset.add(U[i] + U[j])
    # Compute difference set U-U
    dset = set()
    for i in range(n):
        for j in range(n):
            dset.add(U[i] - U[j])
    
    ratio = len(dset) / len(sset)
    if ratio <= 1:
        return 0.0
    denom = math.log(2 * max(U) + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _compute_ratio(U):
    """Fast ratio computation for hill climbing."""
    n = len(U)
    sset = set()
    for i in range(n):
        for j in range(i, n):
            sset.add(U[i] + U[j])
    dset = set()
    for i in range(n):
        for j in range(n):
            dset.add(U[i] - U[j])
    return len(dset) / len(sset)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit, use 150s budget but be safe
    safety_margin = 10.0  # return 10s before hard limit
    
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Strategy: Try various set sizes and structures
    # Key insight: We want large |U-U|/|U+U| ratio AND small max(U)
    # |U-U| is at most n*(2n-1) for distinct differences (actually n^2 - n + 1 if all positive diffs distinct)
    # |U+U| is at least n(n+1)/2
    # For a Sidon-like set, |U-U| ~ n^2, |U+U| ~ n^2/2, ratio ~ 2
    # But we also need log(2*max+1) in denominator to be small
    
    # Approach: Generate candidate sets with different structures
    # 1. Arithmetic progressions: |U-U| = (n-1)*2+1 = 2n-1, |U+U| = 2n-1, ratio = 1 (bad)
    # 2. Powers of 2: max = 2^(n-1), |U-U| = n^2-n+1, |U+U| = n(n+1)/2, ratio ~ 2 for large n
    #    But max grows exponentially, so log(2*max+1) ~ n*log(2)
    # 3. Geometric with smaller base
    # 4. Random sets with small max
    
    # Let's try systematic search over set sizes
    # For each size n, try to find a set with good ratio and small max
    
    # Method: Start with structured sets, then hill-climb
    
    def try_structured(n, base=None):
        """Generate structured sets of size n."""
        candidates = []
        
        # Powers of small integers
        if base is None:
            bases_to_try = [2, 3, 4, 5, 6, 7, 8]
        else:
            bases_to_try = [base]
        
        for b in bases_to_try:
            # Powers: 0, 1, b, b^2, ..., b^(n-1)
            if n == 1:
                candidates.append([0])
            elif n == 2:
                candidates.append([0, 1])
            else:
                # 0, 1, b, b^2, ..., b^(n-2) gives n elements
                arr = [0]
                for i in range(1, n):
                    arr.append(b ** i)
                candidates.append(arr)
        
        # Also try: 0, 1, 2, 4, 8, ... (powers of 2)
        # And: 0, 1, 3, 9, 27, ... (powers of 3)
        
        # Mixed: 0, 1, then powers
        # Try: 0, 1, k, k^2, ... for various k
        for k in range(2, 20):
            arr = [0, 1]
            val = k
            while len(arr) < n:
                arr.append(val)
                val *= k
            if len(arr) == n:
                candidates.append(arr)
        
        return candidates
    
    # Try small sizes first
    for n in range(2, 30):
        if time.time() > deadline - safety_margin:
            break
        
        # Generate candidates
        cands = try_structured(n)
        
        # Also add some random candidates
        for _ in range(20):
            if time.time() > deadline - safety_margin:
                break
            max_val = n * n  # keep max bounded
            arr = [0]
            for _ in range(n - 1):
                arr.append(random.randint(1, max_val))
            arr = sorted(set(arr))
            if len(arr) == n and 0 in arr:
                cands.append(arr)
        
        for U in cands:
            if time.time() > deadline - safety_margin:
                break
            if len(U) != n or 0 not in U:
                continue
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    # Now hill-climb from best_U
    if time.time() < deadline - safety_margin:
        U = list(best_U)
        n = len(U)
        
        # Try adding/removing elements and perturbing
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > deadline - safety_margin:
                break
            
            improved = False
            n = len(U)
            
            # Try removing a random element (except 0)
            if n > 2:
                idx = random.randint(1, n - 1)
                U_new = U[:idx] + U[idx+1:]
                s = _score_of(U_new)
                if s > best_score:
                    best_score = s
                    best_U = list(U_new)
                    U = list(U_new)
                    improved = True
            
            # Try adding a new element
            if n < 40:
                max_val = max(U) * 2 + 10
                new_val = random.randint(1, max_val)
                if new_val not in U:
                    U_new = sorted(U + [new_val])
                    s = _score_of(U_new)
                    if s > best_score:
                        best_score = s
                        best_U = list(U_new)
                        U = list(U_new)
                        improved = True
            
            # Try perturbing a non-zero element
            if n > 1:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                # Try nearby values
                for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in U:
                        continue
                    U_new = sorted(U[:idx] + [new_val] + U[idx+1:])
                    s = _score_of(U_new)
                    if s > best_score:
                        best_score = s
                        best_U = list(U_new)
                        U = list(U_new)
                        improved = True
                        break
            
            if not improved and it % 100 == 0:
                # Restart with small perturbation
                if random.random() < 0.1 and n > 1:
                    idx = random.randint(1, n - 1)
                    U[idx] = max(1, U[idx] + random.randint(-5, 5))
                    U = sorted(set(U))
                    if 0 not in U:
                        U.insert(0, 0)
    
    # Final: try a few more targeted constructions
    if time.time() < deadline - safety_margin:
        # Try sets of form {0, 1, a, 2a, 3a, ...} with small a
        # These have |U-U| = 2*(n-1)+1 = 2n-1... no that's AP
        # Try {0, 1, b, 2b, 3b, ..., (n-2)b} 
        for b in range(2, 30):
            for n in range(3, 25):
                if time.time() > deadline - safety_margin:
                    break
                U = [0, 1] + [i * b for i in range(2, n)]
                if max(U) > 10000:
                    continue
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
        
        # Try {0, 1, b, b^2, b^3, ...}
        for b in range(2, 15):
            for n in range(3, 20):
                if time.time() > deadline - safety_margin:
                    break
                U = [0, 1]
                val = b
                for _ in range(n - 2):
                    U.append(val)
                    val *= b
                if max(U) > 50000:
                    continue
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return sorted(best_U)
