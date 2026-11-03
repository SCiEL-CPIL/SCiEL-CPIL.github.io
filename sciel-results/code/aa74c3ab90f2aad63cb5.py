import time
import math
import random
import numpy as np
from itertools import product

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    if n > 30:
        sums_set = set()
        diffs_set = set()
        for a in U:
            for b in U:
                sums_set.add(a + b)
                diffs_set.add(a - b)
        ratio = len(diffs_set) / len(sums_set)
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def geometric_score(b, m):
    """Score for geometric progression U = {floor(b^k) : k=0..m}"""
    if b <= 1 or m < 1:
        return -1.0
    U = list(set([int(b**k) for k in range(m+1)]))
    U.sort()
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
    return fast_score(U)

def scaled_geometric_score(c, b, m):
    """Score for U = {floor(c * b^k) : k=0..m}"""
    if b <= 1 or m < 1 or c < 1:
        return -1.0
    U = list(set([int(c * (b**k)) for k in range(m+1)]))
    U.sort()
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
    return fast_score(U)

def construct():
    start = time.time()
    deadline = start + 140
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Systematic exploration of geometric progressions
    # For each m, find optimal b via coarse-to-fine search
    
    for m in range(2, 50):
        if time.time() > deadline - 10:
            return best_set
        
        # Determine reasonable range for b
        # max element ~ b^m, want b^m < 10^6 for reasonable max
        b_max = min(10000.0, (10**6) ** (1.0/m) * 1.5)
        b_min = 1.01
        
        if b_max <= b_min:
            continue
        
        # Coarse grid search
        n_coarse = 80
        best_b_coarse = b_min
        best_score_coarse = -1.0
        
        for i in range(n_coarse + 1):
            b = b_min + (b_max - b_min) * i / n_coarse
            s = geometric_score(b, m)
            if s > best_score_coarse:
                best_score_coarse = s
                best_b_coarse = b
        
        # Golden section search refinement
        gr = (math.sqrt(5) - 1) / 2  # golden ratio conjugate
        lo = max(b_min, best_b_coarse - (b_max - b_min) / n_coarse * 3)
        hi = min(b_max, best_b_coarse + (b_max - b_min) / n_coarse * 3)
        
        for _ in range(40):
            if time.time() > deadline - 5:
                return best_set
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            s1 = geometric_score(x1, m)
            s2 = geometric_score(x2, m)
            if s1 < s2:
                lo = x1
            else:
                hi = x2
        
        b_opt = (lo + hi) / 2
        s_opt = geometric_score(b_opt, m)
        if s_opt > best_score:
            best_score = s_opt
            U = list(set([int(b_opt**k) for k in range(m+1)]))
            U.sort()
            if 0 not in U:
                U.insert(0, 0)
            best_set = sorted(set(U))
    
    # Phase 2: Try scaled geometric progressions c * b^k for small c
    for m in range(2, 30):
        if time.time() > deadline - 5:
            return best_set
        
        for c in [1, 2, 3, 5, 8, 13]:
            b_max = min(10000.0, ((10**6) / c) ** (1.0/m) * 1.5)
            b_min = 1.01
            if b_max <= b_min:
                continue
            
            n_coarse = 40
            best_b_coarse = b_min
            best_score_coarse = -1.0
            
            for i in range(n_coarse + 1):
                b = b_min + (b_max - b_min) * i / n_coarse
                s = scaled_geometric_score(c, b, m)
                if s > best_score_coarse:
                    best_score_coarse = s
                    best_b_coarse = b
            
            gr = (math.sqrt(5) - 1) / 2
            lo = max(b_min, best_b_coarse - (b_max - b_min) / n_coarse * 3)
            hi = min(b_max, best_b_coarse + (b_max - b_min) / n_coarse * 3)
            
            for _ in range(30):
                if time.time() > deadline - 2:
                    break
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                s1 = scaled_geometric_score(c, x1, m)
                s2 = scaled_geometric_score(c, x2, m)
                if s1 < s2:
                    lo = x1
                else:
                    hi = x2
            
            b_opt = (lo + hi) / 2
            s_opt = scaled_geometric_score(c, b_opt, m)
            if s_opt > best_score:
                best_score = s_opt
                U = list(set([int(c * (b_opt**k)) for k in range(m+1)]))
                U.sort()
                if 0 not in U:
                    U.insert(0, 0)
                best_set = sorted(set(U))
    
    # Phase 3: Local search on the best set found
    U = sorted(best_set)
    for iteration in range(200):
        if time.time() > deadline:
            return best_set
        
        improved = False
        n = len(U)
        
        for i in range(1, n):
            if time.time() > deadline:
                return best_set
            old_val = U[i]
            for new_val in range(max(0, old_val - 3), min(max(U) + 4, old_val + 4)):
                if new_val == old_val:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_set
