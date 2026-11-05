"""Optimized additive basis construction for [0,128] covering [128,256]."""
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 128
TARGET_LO = 128
TARGET_HI = 256
TARGETS = list(range(TARGET_LO, TARGET_HI + 1))
NUM_TARGETS = len(TARGETS)

def _compute_peak(A_set):
    """Compute the maximum representation multiplicity for sums in [128,256]."""
    if not A_set:
        return float('inf')
    A = np.array(sorted(A_set), dtype=np.int32)
    # Compute all pair sums
    # For unordered pairs with equal summands permitted:
    # sum_{a,b in A, a<=b} [a+b = t]
    # We can use convolution-like approach but for small sets direct is fine
    # Actually, let's use a more efficient method
    # Count representations: for each target t, count pairs (a,b) with a<=b, a+b=t
    # This equals: for t, count a in A such that t-a in A and a <= t-a
    # i.e., a <= t/2
    
    # Faster: use the fact that number of representations of t as sum of two elements of A (unordered, equal allowed)
    # = (number of ordered pairs (a,b) in A x A with a+b=t + number of pairs where a=b and 2a=t) / 2
    # Actually simpler: let f(t) = |{(a,b) in A x A : a+b=t}| (ordered)
    # Then unordered with equal allowed = (f(t) + g(t)) / 2 where g(t) = 1 if t is even and t/2 in A else 0
    # Wait, that's not quite right either. Let me think again.
    
    # Unordered pairs with equal summands permitted means we count {a,b} where a<=b.
    # For a given t, the number of such pairs is:
    # Sum over a in A with a <= t-a (i.e., a <= t/2) of [t-a in A]
    
    # Let's just compute it directly for each target
    reps = np.zeros(NUM_TARGETS, dtype=np.int32)
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            b = A[j]
            s = a + b
            if TARGET_LO <= s <= TARGET_HI:
                reps[s - TARGET_LO] += 1
    return int(np.max(reps)) if reps.size > 0 else 0

def _is_valid(A_set):
    """Check if every target in [128,256] has at least one representation."""
    if not A_set:
        return False
    A = np.array(sorted(A_set), dtype=np.int32)
    reps = np.zeros(NUM_TARGETS, dtype=np.int32)
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            b = A[j]
            s = a + b
            if TARGET_LO <= s <= TARGET_HI:
                reps[s - TARGET_LO] += 1
    return np.all(reps > 0)

def _objective(x):
    """Objective for differential evolution: minimize peak multiplicity, with penalty for invalid."""
    A_set = set(int(v) for v in x)
    if len(A_set) == 0:
        return 1e6
    A = np.array(sorted(A_set), dtype=np.int32)
    reps = np.zeros(NUM_TARGETS, dtype=np.int32)
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            b = A[j]
            s = a + b
            if TARGET_LO <= s <= TARGET_HI:
                reps[s - TARGET_LO] += 1
    if np.any(reps == 0):
        # Penalty: number of uncovered targets + peak
        uncovered = np.sum(reps == 0)
        return 1000 + uncovered * 10 + int(np.max(reps)) if reps.size > 0 else 1000
    return int(np.max(reps))

def _local_search(A_set, max_iters=5000, deadline=None):
    """Local search to improve the set."""
    if deadline is not None and time.time() > deadline:
        return A_set, _compute_peak(A_set)
    
    A = sorted(A_set)
    current_peak = _compute_peak(A)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try removing elements
        for a in A:
            new_set = A_set - {a}
            if not new_set:
                continue
            if _is_valid(new_set):
                new_peak = _compute_peak(new_set)
                if new_peak < current_peak:
                    A_set = new_set
                    A = sorted(A_set)
                    current_peak = new_peak
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try adding elements
        candidates = [x for x in range(N + 1) if x not in A_set]
        random.shuffle(candidates)
        for a in candidates:
            new_set = A_set | {a}
            if _is_valid(new_set):
                new_peak = _compute_peak(new_set)
                if new_peak < current_peak:
                    A_set = new_set
                    A = sorted(A_set)
                    current_peak = new_peak
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping
        if len(A) >= 2:
            swaps_done = 0
            for i in range(len(A)):
                for j in range(len(A)):
                    if i == j:
                        continue
                    a_remove = A[i]
                    for a_add in range(N + 1):
                        if a_add == a_remove or a_add in A_set:
                            continue
                        new_set = A_set - {a_remove} | {a_add}
                        if _is_valid(new_set):
                            new_peak = _compute_peak(new_set)
                            if new_peak < current_peak:
                                A_set = new_set
                                A = sorted(A_set)
                                current_peak = new_peak
                                improved = True
                                swaps_done += 1
                                if swaps_done >= 10:
                                    break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return A_set, current_peak

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Start with the seed solution
    best_set = set(range(64, 129))
    best_peak = _compute_peak(best_set)
    
    # Try local search from seed
    best_set, best_peak = _local_search(best_set, max_iters=2000, deadline=deadline)
    
    if time.time() > deadline:
        return list(best_set)
    
    # Try differential evolution with a smaller search space
    # We'll encode the set as a binary vector of length 129
    # But DE works on continuous spaces, so we'll use a heuristic approach
    
    # Alternative: try several random initial sets and local search
    random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Generate a random set that's likely to be valid
        # Start with a base and add random elements
        base_size = random.randint(40, 80)
        A_set = set(random.sample(range(N + 1), base_size))
        
        # Ensure basic coverage by adding strategic elements
        # Elements near 64 help cover middle targets
        for x in range(50, 80):
            if random.random() < 0.5:
                A_set.add(x)
        
        # Local search
        A_set, peak = _local_search(A_set, max_iters=1000, deadline=deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_set = A_set
    
    # Try a structured approach: symmetric sets around 64
    if time.time() < deadline:
        # A = {64-k, 64+k for k in range(K)} union {64}
        for K in range(30, 65, 5):
            if time.time() > deadline:
                break
            A_set = {64}
            for k in range(K):
                if 64 - k >= 0:
                    A_set.add(64 - k)
                if 64 + k <= N:
                    A_set.add(64 + k)
            
            A_set, peak = _local_search(A_set, max_iters=500, deadline=deadline)
            
            if peak < best_peak:
                best_peak = peak
                best_set = A_set
    
    # Final local search refinement
    best_set, best_peak = _local_search(best_set, max_iters=1000, deadline=deadline)
    
    return list(best_set)
