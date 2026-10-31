"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from itertools import combinations

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    def score(f):
        f = np.array(f, dtype=float)
        if np.any(f < 0):
            return -np.inf
        l1 = np.sum(f)
        if l1 == 0:
            return -np.inf
        l2_sq = np.sum(f**2)
        auto = np.correlate(f, f, mode='full')
        auto_inf = np.max(auto)
        if auto_inf == 0:
            return -np.inf
        return l2_sq / (l1 * auto_inf)
    
    best_f = None
    best_val = -np.inf
    
    def try_candidate(f):
        nonlocal best_f, best_val
        f = np.maximum(f, 0)
        val = score(f)
        if val > best_val:
            best_val = val
            best_f = f.copy()
    
    # Strategy: Use Sidon sets. A Sidon set S has the property that all pairwise sums
    # i+j (i<=j) are distinct. For a vector supported on a Sidon set with equal weights,
    # the autoconvolution off-diagonal peaks are minimized.
    # 
    # For support S of size K with weight w each:
    # L1 = K*w, L2^2 = K*w^2
    # auto(0) = K*w^2
    # auto(k) for k!=0: at most 1 pair contributes (Sidon property), so <= w^2
    # auto_inf = K*w^2
    # ratio = K*w^2 / (K*w * K*w^2) = 1/(K*w)
    #
    # But wait - this gives ratio = 1/K * 1/w. With w=1/K (L1=1), ratio = 1.
    # The key is that for NON-Sidon sets, auto_inf can be larger than K*w^2 due to
    # multiple pairs hitting the same lag. So Sidon sets AVOID this penalty.
    #
    # Actually the real advantage: for a general set, auto_inf could be much larger
    # than the diagonal. For Sidon sets, auto_inf = diagonal = K*w^2.
    # For non-Sidon, auto_inf > K*w^2, making the ratio smaller.
    #
    # So the best strategy is to find the LARGEST Sidon subset of {0,...,49} and
    # put equal weight there. But we can also try varying weights.
    
    # Find large Sidon sets using greedy algorithm
    def is_sidon(s):
        """Check if sorted list s is a Sidon set (all pairwise sums i+j for i<=j are unique)."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def greedy_sidon(n, max_size=None):
        """Greedy construction of a Sidon set from {0, ..., n-1}."""
        s = []
        if max_size is None:
            max_size = n
        for x in range(n):
            candidate = s + [x]
            if is_sidon(candidate):
                s = candidate
                if len(s) >= max_size:
                    break
        return s
    
    # Try greedy Sidon set of various sizes
    for target_k in [10, 12, 14, 15, 16, 18, 20]:
        if time.time() > deadline:
            break
        sidon = greedy_sidon(N, target_k)
        if len(sidon) >= 5:
            f = np.zeros(N)
            w = 1.0 / len(sidon)  # L1 = 1
            for idx in sidon:
                f[idx] = w
            try_candidate(f)
    
    # Also try: for each Sidon set, optimize weights using simple coordinate descent
    # The ratio with L1 normalization is L2^2 / auto_inf
    # = (sum w_i^2) / max_k (sum_i w_i * w_{i+k})
    # For Sidon set, auto(0) = sum w_i^2, and auto(k) for k!=0 is at most max_i w_i^2
    # (actually it's w_i*w_j for the unique pair, so <= max_i w_i * max_j w_j)
    # To maximize sum(w^2)/max(auto), we want weights as unequal as possible?
    # No - we want to minimize the off-diagonal peaks relative to the diagonal.
    
    # Better approach: try random subsets and check if they're "near-Sidon"
    # Then use the subset with minimum autoconvolution peak
    
    best_subset = None
    best_auto_ratio = np.inf
    
    # Try many random subsets of various sizes
    random.seed(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        k = random.randint(8, 25)
        subset = sorted(random.sample(range(N), k))
        f = np.zeros(N)
        w = 1.0 / k
        for idx in subset:
            f[idx] = w
        l2_sq = np.sum(f**2)
        auto = np.correlate(f, f, mode='full')
        auto_inf = np.max(auto)
        ratio = l2_sq / auto_inf  # with L1=1
        if ratio > best_val:
            best_val = ratio
            best_f = f.copy()
            best_subset = subset[:]
    
    # Now try to improve: for the best subset, try varying weights
    # Use a simple approach: make one element large, rest small
    if best_subset is not None:
        k = len(best_subset)
        # Try geometric progression of weights
        for base in [0.5, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            f = np.zeros(N)
            weights = [base**i for i in range(k)]
            total = sum(weights)
            weights = [w / total for w in weights]
            for i, idx in enumerate(best_subset):
                f[idx] = weights[i]
            try_candidate(f)
        
        # Try: one dominant element
        for dominant_pos in range(min(5, k)):
            if time.time() > deadline:
                break
            f = np.zeros(N)
            f[best_subset[dominant_pos]] = 0.8
            remaining = 0.2 / (k - 1)
            for i, idx in enumerate(best_subset):
                if i != dominant_pos:
                    f[idx] = remaining
            try_candidate(f)
    
    # Also try structured patterns: every m-th element
    for m in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        f = np.zeros(N)
        indices = list(range(0, N, m))
        w = 1.0 / len(indices)
        for idx in indices:
            f[idx] = w
        try_candidate(f)
    
    # Try: spread out pairs to minimize collisions
    # Use a "difference set" approach: choose indices so that all differences are distinct
    # This is similar to Sidon but for differences
    def is_b_set(s):
        """Check if all positive differences are distinct."""
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def greedy_b_set(n, max_size=None):
        s = []
        if max_size is None:
            max_size = n
        for x in range(n):
            candidate = s + [x]
            if is_b_set(candidate):
                s = candidate
                if len(s) >= max_size:
                    break
        return s
    
    # B-sets (distinct differences) also minimize autoconvolution peaks
    for target_k in [8, 10, 12, 14, 16]:
        if time.time() > deadline:
            break
        bset = greedy_b_set(N, target_k)
        if len(bset) >= 5:
            f = np.zeros(N)
            w = 1.0 / len(bset)
            for idx in bset:
                f[idx] = w
            try_candidate(f)
    
    # Final: try a few more random sparse patterns with unequal weights
    for trial in range(200):
        if time.time() > deadline:
            break
        k = random.randint(5, 20)
        subset = sorted(random.sample(range(N), k))
        # Random weights, normalized
        weights = np.random.rand(k) + 0.1
        weights /= weights.sum()
        f = np.zeros(N)
        for i, idx in enumerate(subset):
            f[idx] = weights[i]
        try_candidate(f)
    
    if best_f is None:
        return [1.0] * N
    
    best_f = np.maximum(best_f, 0)
    return list(best_f)
# EVOLVE-BLOCK-END
