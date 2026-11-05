import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_len = max_sum - min_sum + 1  # 129

    # For each target sum s in [128, 256], precompute all valid pairs (a, b) with a <= b, a+b=s, 0<=a,b<=128
    # These are the "pair representations" we need to cover
    sum_to_pairs = {}
    for s in range(min_sum, max_sum + 1):
        pairs = []
        for a in range(max(0, s - n), min(n, s) + 1):
            b = s - a
            if a <= b and 0 <= b <= n:
                pairs.append((a, b))
        sum_to_pairs[s] = pairs

    # Precompute: for each element x in [0,128], which sums can it participate in
    # and which pairs involve it
    element_to_sums = {x: [] for x in range(n + 1)}
    for s in range(min_sum, max_sum + 1):
        for (a, b) in sum_to_pairs[s]:
            element_to_sums[a].append(s)
            if a != b:
                element_to_sums[b].append(s)

    def compute_representation_counts(A_set):
        """Given a set A, compute unordered pair representation counts for [128,256]."""
        x_bin = np.zeros(n + 1, dtype=np.int8)
        for x in A_set:
            x_bin[x] = 1
        conv = np.convolve(x_bin, x_bin, mode='full')
        counts = conv[min_sum:max_sum + 1].astype(np.float64)
        # Diagonal correction: for even s, pair (s/2, s/2) is counted once in conv but represents 1 unordered pair
        # conv counts ordered pairs, so for a!=b: conv[s] = 2 * (# unordered pairs), for a==b: conv[s] includes 1
        # Unordered count = (conv[s] + (1 if s even and s/2 in A else 0)) / 2
        diag = np.zeros(target_len, dtype=np.float64)
        for i, s in enumerate(range(min_sum, max_sum + 1)):
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= n and x_bin[mid]:
                    diag[i] = 1.0
        unordered = (counts + diag) / 2.0
        unordered = np.round(unordered).astype(np.int32)
        return unordered

    def is_valid_and_get_peak(A_set):
        """Check if A covers [128,256] and return peak multiplicity, or inf if not covered."""
        if len(A_set) == 0:
            return float('inf')
        counts = compute_representation_counts(A_set)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    # === CSP BACKTRACKING APPROACH ===
    # Strategy: For a given K (max multiplicity), try to find A such that every sum in [128,256]
    # has at least one representation, and no sum has more than K representations.
    # 
    # Key insight: We can think of this as selecting elements A such that the pair-sum
    # coverage is complete and bounded.
    #
    # Alternative structural approach: Use a "covering design" mindset.
    # Start from small candidate sets and use backtracking to ensure coverage.
    
    # Phase 1: Try structured sets with backtracking refinement
    # A set of size ~64-80 should be enough to cover [128,256] with low multiplicity.
    
    best_peak = float('inf')
    best_A = None

    # Phase 1a: Try arithmetic progression based structures
    # If A = {0, d, 2d, ..., md}, then pair sums cover a range with regular structure
    for d in range(1, 8):
        for start in range(0, min(d, 5)):
            # A = {start + k*d : k=0..m} where start+m*d <= 128
            m_max = (128 - start) // d
            for m in range(20, m_max + 1):
                if m < 20:
                    continue
                A_set = set(start + k * d for k in range(m + 1))
                if len(A_set) == 0:
                    continue
                peak = is_valid_and_get_peak(A_set)
                if peak < best_peak:
                    best_peak = peak
                    best_A = A_set.copy()
                    if best_peak <= 3:
                        break
            if best_peak <= 3:
                break
        if best_peak <= 3:
            break

    # Phase 1b: Try two-interval structures with gaps
    for a in range(10, 60):
        for b in range(a + 5, 129):
            if time.time() > deadline - 30:
                break
            A_set = set(range(0, a + 1)) | set(range(b, 129))
            if len(A_set) < 20:
                continue
            peak = is_valid_and_get_peak(A_set)
            if peak < best_peak:
                best_peak = peak
                best_A = A_set.copy()
        if best_peak <= 3:
            break
    if best_peak <= 3:
        # Quick return if we already have good answer
        pass

    # Phase 1c: Randomized structured initialization
    random.seed(42)
    np.random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline - 35:
            break
        # Generate a random set of size ~50-80
        size = random.randint(45, 85)
        A_set = set(random.sample(range(0, 129), size))
        peak = is_valid_and_get_peak(A_set)
        if peak < best_peak:
            best_peak = peak
            best_A = A_set.copy()
        if best_peak <= 3:
            break

    if best_A is None:
        best_A = set(range(0, 70))
        best_peak = is_valid_and_get_peak(best_A)
        if best_peak == float('inf'):
            best_A = set(range(0, 129))
            best_peak = is_valid_and_get_peak(best_A)

    # Phase 2: Backtracking-based local improvement
    # For the current best_A, try to reduce peak by carefully adding/removing elements
    # using forward-checking style reasoning
    
    def try_improve(A_set, max_iters=200):
        """Try to reduce peak by single add/remove operations with lookahead."""
        cur_peak = is_valid_and_get_peak(A_set)
        if cur_peak == float('inf'):
            return A_set, cur_peak
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try removing elements (to reduce peak)
            for elem in list(A_set):
                if time.time() > deadline - 10:
                    break
                A_test = A_set - {elem}
                if len(A_test) < 15:
                    continue
                p_test = is_valid_and_get_peak(A_test)
                if p_test != float('inf') and p_test < cur_peak:
                    A_set = A_test
                    cur_peak = p_test
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try adding elements (may help redistribute)
            zeros = [x for x in range(0, 129) if x not in A_set]
            random.shuffle(zeros)
            for elem in zeros[:30]:  # limit candidates
                if time.time() > deadline - 10:
                    break
                A_test = A_set | {elem}
                p_test = is_valid_and_get_peak(A_test)
                if p_test != float('inf') and p_test < cur_peak:
                    A_set = A_test
                    cur_peak = p_test
                    improved = True
                    break
            
            if not improved:
                break
        
        return A_set, cur_peak

    if best_peak > 3:
        best_A, best_peak = try_improve(best_A, max_iters=100)

    # Phase 3: Multi-start with structured perturbations
    # Use the current best as seed and create diverse perturbations
    if best_peak > 3:
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            # Perturb best_A: remove k random, add k random
            k = random.randint(3, 10)
            A_list = list(best_A)
            random.shuffle(A_list)
            A_perturbed = set(A_list[k:])  # remove first k
            # Add k new random elements
            candidates = [x for x in range(0, 129) if x not in A_perturbed]
            if candidates:
                additions = random.sample(candidates, min(k, len(candidates)))
                A_perturbed = A_perturbed | set(additions)
            
            if len(A_perturbed) < 15:
                continue
                
            peak_p = is_valid_and_get_peak(A_perturbed)
            if peak_p < best_peak:
                best_peak = peak_p
                best_A = A_perturbed.copy()
            
            # Try to improve the perturbed version
            A_improved, peak_improved = try_improve(A_perturbed, max_iters=50)
            if peak_improved < best_peak:
                best_peak = peak_improved
                best_A = A_improved.copy()
            
            if best_peak <= 3:
                break

    # Phase 4: Final polish - exhaustive single-step improvement
    if best_peak > 3:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try all single removals
            for elem in list(best_A):
                A_test = best_A - {elem}
                if len(A_test) < 15:
                    continue
                p_test = is_valid_and_get_peak(A_test)
                if p_test != float('inf') and p_test < best_peak:
                    best_A = A_test
                    best_peak = p_test
                    improved = True
                    break
            if not improved:
                # Try all single additions
                for elem in range(0, 129):
                    if elem in best_A:
                        continue
                    A_test = best_A | {elem}
                    p_test = is_valid_and_get_peak(A_test)
                    if p_test != float('inf') and p_test < best_peak:
                        best_A = A_test
                        best_peak = p_test
                        improved = True
                        break
            if not improved:
                break

    final_A = sorted(best_A)
    if len(final_A) == 0:
        final_A = list(range(0, 129))
    
    return final_A
