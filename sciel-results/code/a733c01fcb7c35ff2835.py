import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_fft(U):
    """
    Fast score calculation using FFT.
    U: list of non-negative integers.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: A * A
    F_A = np.fft.fft(A)
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    
    # Count non-zero elements in S[0 : 2*M+1]
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Use autocorrelation
    F_R = F_A * np.conj(F_A)
    R = np.fft.ifft(F_R).real
    
    # The differences a-b range from -M to M.
    # In the circular convolution/autocorrelation of size L, the difference d corresponds to index d % L.
    # Since L > 2M, indices 0..M correspond to differences 0..M, and L-M..L-1 correspond to -M..-1.
    # We need the count of non-zero entries in R for these indices.
    diff_indices = list(range(0, M+1)) + list(range(L-M, L))
    diff_count = np.sum(R[diff_indices] > 0.1)
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _score_small(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _get_score(U):
    if max(U) < 1000:
        return _score_small(U)
    return _score_fft(U)

def _greedy_sidon_construct(target_size):
    """
    Construct a Sidon set (B2 sequence) greedily.
    This minimizes |U+U| by ensuring all pairwise sums are distinct.
    """
    U = [0]
    # Keep track of all sums to check for collisions
    sums = set()
    for u in U:
        sums.add(u + u) # Only 0+0 initially
    
    # We want to add elements. 
    # The Mian-Chowla sequence is a classic Sidon set.
    # Greedy: pick smallest x > max(U) such that x+u not in sums for all u in U.
    # Note: We must also add new sums x+u to the set.
    
    while len(U) < target_size:
        if _remaining() < 5.0:
            break
        candidate = max(U) + 1
        found = False
        while not found:
            # Check if candidate creates any sum collision
            valid = True
            new_sums = set()
            for u in U:
                s = candidate + u
                if s in sums:
                    valid = False
                    break
                new_sums.add(s)
            
            if valid:
                # Accept candidate
                U.append(candidate)
                sums.update(new_sums)
                found = True
            else:
                candidate += 1
                # Safety check: if candidate gets too large, we might be stuck or slow
                # But Sidon sets grow roughly quadratically, so this should be fine for n<100
                if candidate > 100000: 
                    break
        if not found:
            break
            
    return U

def _local_search_refine(U, time_budget):
    """
    Refine the set to reduce max(U) and improve the ratio.
    We allow elements to move, but we penalize sum collisions heavily.
    """
    U = sorted(list(U))
    n = len(U)
    current_score = _get_score(U)
    best_U = list(U)
    best_score = current_score
    
    rng = random.Random(42)
    start_time = time.time()
    
    # We perform local search by trying to "compress" the set.
    # Since Sidon sets are sparse, we can try to shift elements down if possible.
    
    while time.time() - start_time < time_budget and _remaining() > 2.0:
        # Pick a random element to try to move
        i = rng.randint(0, n - 1)
        if U[i] == 0:
            continue # Keep 0 fixed
            
        orig_val = U[i]
        # Try to move it down slightly to reduce max(U)
        # Or move it to a "better" spot that maintains low collision
        
        # Strategy: Try all values from max(0, orig_val - 10) to orig_val
        # This attempts to compress the set.
        # If moving down causes a sum collision, the score will drop, and we reject.
        
        improved = False
        for new_val in range(max(0, orig_val - 15), orig_val):
            if new_val == 0 and i != 0:
                continue
            # Check if new_val is already in U (excluding i)
            if new_val in U and U.index(new_val) != i:
                continue
                
            U[i] = new_val
            U.sort()
            
            new_score = _get_score(U)
            
            if new_score > current_score + 1e-10:
                current_score = new_score
                best_U = list(U)
                best_score = current_score
                improved = True
                break
            else:
                # Revert
                U[i] = orig_val
                U.sort()
        
        if not improved:
            # If no local improvement, try a random perturbation to escape local minima
            # This is a "kick"
            if rng.random() < 0.1:
                # Pick a random element and move it randomly within a small range
                j = rng.randint(0, n - 1)
                if U[j] == 0:
                    continue
                orig = U[j]
                delta = rng.randint(-5, 5)
                new_val = orig + delta
                if new_val >= 0 and new_val not in U:
                    U[j] = new_val
                    U.sort()
                    new_score = _get_score(U)
                    # Accept with some probability (Simulated Annealing style)
                    if new_score > current_score:
                        current_score = new_score
                        best_U = list(U)
                        best_score = current_score
                    else:
                        # Revert for now, as we want to stay close to good structures
                        U[j] = orig
                        U.sort()

    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1]
    best_score = 0.0
    
    # 1. Generate initial candidates using Greedy Sidon Construction
    # Sidon sets have distinct sums, which minimizes |U+U|.
    # We try different sizes to see which yields the best score.
    
    # The score depends on n and max(U).
    # For Sidon sets, max(U) ~ n^2.
    # Score ~ 1 + log(n^2 / n^2) / log(n^2) ~ 1.
    # Wait, if |U+U| ~ n^2 and |U-U| ~ n^2, the ratio is ~1.
    # log(1) = 0. Score = 1.
    # We need ratio > 1.
    # |U-U| is always at least 2n-1.
    # |U+U| for Sidon is ~ n^2/2.
    # Ratio ~ (2n) / (n^2/2) = 4/n.
    # This is < 1 for n > 4.
    # So Sidon sets might score < 1?
    # Let's check:
    # U = {0, 1, 3, 7} (Sidon? 0+1=1, 0+3=3, 0+7=7, 1+3=4, 1+7=8, 3+7=10. Sums: 0,1,2,3,4,6,7,8,10. Distinct? 0,1,2,3,4,6,7,8,10. Yes.)
    # |U+U| = 9.
    # |U-U|: diffs of {0,1,3,7}: 0, -1, 1, -3, 3, -7, 7, -6, 6, -2, 2, -4, 4.
    # Set: {-7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7}. Size 13.
    # Ratio = 13/9 = 1.44.
    # log2(1.44) = 0.53.
    # Max U = 7. 2*7+1 = 15. log2(15) = 3.9.
    # Score = 1 + 0.53/3.9 = 1 + 0.13 = 1.13.
    # This is > 1.
    
    # So Sidon sets are good. Let's generate them.
    
    candidates = []
    
    # Try sizes 5 to 30
    for n in [5, 8, 10, 12, 15, 20]:
        if _remaining() < 10.0:
            break
        U = _greedy_sidon_construct(n)
        if len(U) >= 2:
            candidates.append(U)
            
    # 2. Also try a "Random Sparse" set as a fallback/comparison
    # Sometimes random sets with low density work well.
    for n in [10, 20]:
        if _remaining() < 10.0:
            break
        U = [0]
        max_val = n * n * 3
        while len(U) < n:
            x = random.randint(1, max_val)
            if x not in U:
                U.append(x)
        U.sort()
        candidates.append(U)
        
    # 3. Score all candidates
    scored = []
    for U in candidates:
        if _remaining() < 5.0:
            break
        s = _get_score(U)
        scored.append((s, U))
        
    if not scored:
        return [0, 1]
        
    scored.sort(key=lambda x: -x[0])
    best_s, best_U = scored[0]
    
    # 4. Refine the top candidates using Local Search
    # Allocate remaining time to refinement.
    
    time_per_refine = min(30.0, _remaining() / 3.0)
    
    for s, U in scored[:3]:
        if _remaining() < time_per_refine + 2.0:
            break
            
        res_U, res_s = _local_search_refine(U, time_per_refine)
        
        if res_s > best_s:
            best_s = res_s
            best_U = res_U
            
    # 5. Final Polish: Try to reduce max(U) by shifting the entire set (except 0) down?
    # No, 0 is fixed. We can try to shift other elements.
    
    if _remaining() > 5.0:
        U = list(best_U)
        U.sort()
        current_s = best_s
        
        # Try to shift the largest element down
        if len(U) > 1:
            max_idx = len(U) - 1
            orig_max = U[max_idx]
            for new_max in range(orig_max - 1, 0, -1):
                if new_max in U:
                    continue
                U[max_idx] = new_max
                U.sort()
                new_s = _get_score(U)
                if new_s > current_s:
                    current_s = new_s
                    best_U = list(U)
                    best_s = current_s
                else:
                    U[max_idx] = orig_max
                    U.sort()
                    break # If moving down doesn't help, stop (assuming convexity-ish)

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
