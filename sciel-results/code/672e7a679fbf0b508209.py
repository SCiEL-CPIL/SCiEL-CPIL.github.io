"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 50
DEADLINE_MARGIN = 10.0  # seconds to leave before the 120s limit

def _autoconv_peak(h):
    """Return the supremum norm of the circular autoconvolution of h."""
    h_arr = np.asarray(h, dtype=np.float64)
    # Circular convolution: h * h
    conv = np.convolve(h_arr, h_arr, mode='full')
    # For circular convolution of length N:
    # np.convolve gives length 2N-1. We need to wrap around.
    circ_conv = np.zeros(N)
    for k in range(len(conv)):
        circ_conv[k % N] += conv[k]
    return np.max(np.abs(circ_conv))

def _score(h):
    """Compute the raw score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h_arr = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h_arr)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h_arr ** 2)
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def construct():
    start_time = time.time()
    deadline = start_time + (120.0 - DEADLINE_MARGIN)
    
    # Strategy: Search for nonnegative height vectors that maximize
    # ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # 
    # Key insight: The autoconvolution peak ||h*h||_inf is at least ||h||_2^2
    # (by Cauchy-Schwarz, since (h*h)[0] = sum h_i^2 = ||h||_2^2).
    # So the ratio is at most 1/||h||_1.
    # 
    # If we normalize to ||h||_1 = 1, the ratio is at most 1.
    # Equality holds when (h*h)[0] = ||h*h||_inf, i.e., the peak of the
    # autoconvolution is at lag 0. This happens when h is a "Sidon-type"
    # vector where all nonzero differences are distinct, or more generally
    # when the autoconvolution has no peak exceeding the central value.
    #
    # For binary vectors (h_i in {0,1}), ||h||_2^2 = ||h||_1 = k (number of
    # nonzero entries), and ||h*h||_inf >= k. The ratio = k/(k * peak) = 1/peak.
    # To maximize, we want peak = k, which means the autoconvolution is flat
    # at height k. This is achievable with Sidon sets.
    #
    # For a Sidon set of size k in Z_N, the autoconvolution of the indicator
    # function has peak k (at lag 0) and the off-peak values are bounded.
    # If all off-peak values are < k, then peak = k and ratio = 1.
    #
    # So: find the largest Sidon set in {0,...,49} and use its indicator.
    
    # First, try to find a good Sidon set via greedy + local search
    
    def is_sidon_set(S):
        """Check if S is a Sidon set (all pairwise sums are distinct up to symmetry)."""
        S = sorted(S)
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    # Allow the same sum only if it's the same pair
                    if sums[s] != (i, j):
                        return False
                else:
                    sums[s] = (i, j)
        return True

    def greedy_sidon(n):
        """Greedy construction of a Sidon set in {0, ..., n-1}."""
        S = []
        used_sums = set()
        for i in range(n):
            # Check if adding i creates any duplicate sum
            valid = True
            for j in range(len(S)):
                s = S[j] + i
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Also need to check i+i = 2i (for j=i case, but we're adding i)
                # Actually we need to check all pairs (a,b) with a,b in S ∪ {i}
                # Let's be more careful
                pass
        
        # Simpler greedy: add i if it doesn't create a duplicate sum with existing elements
        S = []
        sums = {}
        for i in range(n):
            ok = True
            for j, sj in enumerate(S):
                s = sj + i
                key = (min(sj, i), max(sj, i))
                if s in sums:
                    # Check if the existing pair is the same
                    existing = sums[s]
                    if existing != key:
                        ok = False
                        break
            # Also check 2*i
            s2 = 2 * i
            key2 = (i, i)
            if s2 in sums:
                existing = sums[s2]
                if existing != key2:
                    ok = False
            if ok:
                S.append(i)
                sums[i + i] = (i, i)
                for j in S[:-1]:
                    sums[j + i] = (min(j, i), max(j, i))
        return S

    # Try to find the best Sidon set
    best_h = [1.0] + [0.0] * (N - 1)
    best_score = _score(best_h)
    
    # Try greedy Sidon
    sidon = greedy_sidon(N)
    if len(sidon) > 1:
        h_sidon = [0.0] * N
        for idx in sidon:
            h_sidon[idx] = 1.0
        s_sidon = _score(h_sidon)
        if s_sidon > best_score:
            best_h = h_sidon
            best_score = s_sidon
    
    # Now do local search: try to swap, add, or remove elements
    random.seed(42)
    current = list(best_h)
    current_score = best_score
    no_improve_count = 0
    
    def try_swap(current, i, j):
        """Try swapping the values at positions i and j (0/1 swap)."""
        h_new = list(current)
        h_new[i], h_new[j] = h_new[j], h_new[i]
        return h_new, _score(h_new)

    def try_add(current, pos):
        """Try setting position pos to 1 (if currently 0)."""
        if current[pos] > 0.5:
            return current, current_score
        h_new = list(current)
        h_new[pos] = 1.0
        return h_new, _score(h_new)

    def try_remove(current, pos):
        """Try setting position pos to 0 (if currently 1)."""
        if current[pos] < 0.5:
            return current, current_score
        h_new = list(current)
        h_new[pos] = 0.0
        return h_new, _score(h_new)

    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        
        if i == j:
            continue
        
        if move_type < 0.4:
            # Swap
            h_new, s_new = try_swap(current, i, j)
        elif move_type < 0.7:
            # Add at random position
            h_new, s_new = try_add(current, i)
        else:
            # Remove at random position
            h_new, s_new = try_remove(current, i)
        
        if s_new > current_score:
            current = h_new
            current_score = s_new
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_h = list(current)
        else:
            no_improve_count += 1
            # Occasionally accept worse moves to escape local optima
            if random.random() < 0.1 and no_improve_count > 50:
                current = h_new
                current_score = s_new
    
    # Try a few different random restarts
    for restart in range(20):
        if time.time() >= deadline:
            break
        # Random binary vector with varying density
        density = random.uniform(0.1, 0.5)
        h_rand = [1.0 if random.random() < density else 0.0 for _ in range(N)]
        s_rand = _score(h_rand)
        
        # Local search on this starting point
        current = list(h_rand)
        current_score = s_rand
        for step in range(500):
            if time.time() >= deadline:
                break
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            h_new, s_new = try_swap(current, i, j)
            if s_new > current_score:
                current = h_new
                current_score = s_new
            elif random.random() < 0.05:
                current = h_new
                current_score = s_new
        if current_score > best_score:
            best_score = current_score
            best_h = list(current)
    
    # Also try continuous optimization on a subset
    # Fix support size k, optimize the heights
    # For a given support S, maximize ||h||_2^2 / (||h||_1 * peak)
    # This is a non-trivial optimization. Let's try a simple approach:
    # Start from the best binary vector and try small perturbations.
    
    h_opt = np.array(best_h, dtype=np.float64)
    lr = 0.01
    for epoch in range(200):
        if time.time() >= deadline:
            break
        # Compute gradient numerically
        grad = np.zeros(N)
        for i in range(N):
            if h_opt[i] < 1e-10:
                continue  # Keep nonnegative, skip zeros
            h_plus = h_opt.copy()
            h_plus[i] += 0.01
            s_plus = _score(h_plus)
            grad[i] = (s_plus - best_score) / 0.01
        # Update
        h_new = h_opt + lr * grad
        h_new = np.clip(h_new, 0, 1.0)
        s_new = _score(h_new)
        if s_new > best_score:
            h_opt = h_new
            best_score = s_new
            best_h = list(h_new)
    
    if time.time() < deadline:
        # Final check: try the pure Sidon set approach more aggressively
        # Try all subsets of size up to 12 (combinatorial search is feasible for small sizes)
        from itertools import combinations
        best_sidon_score = 0
        best_sidon_set = None
        for k in range(2, min(13, N + 1)):
            for subset in combinations(range(N), k):
                if time.time() >= deadline:
                    return best_h
                # Check if it's a Sidon set
                sums = {}
                valid = True
                for idx_a in range(k):
                    for idx_b in range(idx_a, k):
                        a, b = subset[idx_a], subset[idx_b]
                        s = a + b
                        key = (a, b)
                        if s in sums:
                            if sums[s] != key:
                                valid = False
                                break
                        else:
                            sums[s] = key
                    if not valid:
                        break
                if valid:
                    h_test = [0.0] * N
                    for idx in subset:
                        h_test[idx] = 1.0
                    s_test = _score(h_test)
                    if s_test > best_sidon_score:
                        best_sidon_score = s_test
                        best_sidon_set = subset
        
        if best_sidon_set is not None and best_sidon_score > best_score:
            best_h = [0.0] * N
            for idx in best_sidon_set:
                best_h[idx] = 1.0
            best_score = best_sidon_score
    
    return best_h

# EVOLVE-BLOCK-END
