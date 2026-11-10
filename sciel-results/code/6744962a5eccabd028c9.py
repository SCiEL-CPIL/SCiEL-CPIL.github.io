import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def update_best(c, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = c.copy()
            return best_score <= 12

    # GF(2) analysis: for support S, count pairs for each sum degree
    def analyze_support(support):
        """Return dict: degree -> (num_pairs, num_diag_pairs) where i+j=degree"""
        pair_count = {}
        diag_count = {}
        n = len(support)
        for i in range(n):
            for j in range(i, n):
                k = support[i] + support[j]
                pair_count[k] = pair_count.get(k, 0) + 1
                if i == j:
                    diag_count[k] = diag_count.get(k, 0) + 1
        return pair_count, diag_count

    # For a given support, find optimal signs using structured search
    def solve_signs_optimal(support):
        """Find signs for support that minimize nonzero terms in square.
        
        For support S = {s_0, ..., s_11}, square coefficient at degree k is:
          sum_{i+j=k, i<j} 2*c_i*c_j + sum_{2i=k} c_i^2
        = 2 * sum_{i+j=k, i<j} c_i*c_j + (# of i with 2i=k)
        
        For c_i = ±1: c_i*c_j = (-1)^(x_i+x_j) where x_i in {0,1}
        
        We want sum_{i+j=k, i<j} (-1)^(x_i+x_j) to be as close to 
        -(#diag at k)/2 as possible.
        
        If #diag at k is 0: need sum of even # of ±1 to be 0 -> equal +1 and -1
        If #diag at k is 1: need sum of (odd # of pairs) ±1 to be -1/2 -> impossible with ±1
          Actually: 2*sum + 1 = 0 -> sum = -1/2, impossible. So this term is always nonzero.
        If #diag at k is 2: need 2*sum + 2 = 0 -> sum = -1 -> need more -1 than +1
        
        So the minimum possible nonzero depends on parity of #pairs and #diag.
        """
        n = len(support)
        pair_count, diag_count = analyze_support(support)
        
        # Determine which degrees CAN be zeroed
        zeroable = set()
        for k, pc in pair_count.items():
            dc = diag_count.get(k, 0)
            # Square coeff = 2*sum_{pairs} ±1 + dc
            # Can be zero if: 2*sum = -dc, i.e., sum = -dc/2
            # sum is sum of pc values each ±1, so sum has same parity as pc
            # and |sum| <= pc
            if dc % 2 == 0:  # dc must be even for -dc/2 to be integer
                target = -dc // 2
                if abs(target) <= pc and (pc - target) % 2 == 0:
                    # Can achieve sum = target if we have (pc+target)/2 of +1 and (pc-target)/2 of -1
                    zeroable.add(k)
        
        # Now find signs that zero out as many zeroable degrees as possible
        # Use meet-in-the-middle or smart enumeration
        # For n=12, split into two halves of 6
        half = n // 2
        left_support = support[:half]
        right_support = support[half:]
        
        # Precompute for each half: for each possible sign pattern, 
        # what's the contribution to each degree?
        # This is complex; instead, use a greedy + local search approach
        
        best_signs = None
        best_nz = float('inf')
        
        # Try structured patterns first
        patterns = [
            [1]*n,
            [1 if i%2==0 else -1 for i in range(n)],
            [1]*half + [-1]*(n-half),
            [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1] if n==12 else [1]*n,
            [1,1,1,1,1,-1,1,1,1,1,1,-1] if n==12 else [1]*n,
            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1] if n==12 else [1]*n,
        ]
        
        for pat in patterns:
            if len(pat) != n:
                continue
            coeff = np.zeros(24, dtype=np.int64)
            for i, s in enumerate(support):
                coeff[s] = pat[i]
            sq = poly_square(coeff)
            nz = count_nonzero(sq)
            if nz < best_nz:
                best_nz = nz
                best_signs = pat[:]
        
        # Local search from best structured pattern
        if best_signs is not None:
            signs = best_signs[:]
            for _ in range(500):
                if time.time() > deadline - 30:
                    break
                idx = random.randint(0, n-1)
                signs[idx] = -signs[idx]
                coeff = np.zeros(24, dtype=np.int64)
                for i, s in enumerate(support):
                    coeff[s] = signs[i]
                sq = poly_square(coeff)
                nz = count_nonzero(sq)
                if nz < best_nz:
                    best_nz = nz
                    best_signs = signs[:]
                else:
                    signs[idx] = -signs[idx]  # revert
                    if best_nz <= 12:
                        break
        
        if best_signs is not None:
            coeff = np.zeros(24, dtype=np.int64)
            for i, s in enumerate(support):
                coeff[s] = best_signs[i]
            return coeff, best_nz
        
        return None, float('inf')

    # Find supports with good GF(2) properties
    def find_good_supports(max_degree=11, num_terms=12, max_support_size=24):
        all_supports = []
        
        # Systematic search over subsets of range(0, max_degree+1)
        # Focus on supports where many sum-degrees have even pair count
        # and the zeroable set is large
        
        # Try arithmetic progressions and their unions
        for start in range(0, 8):
            for step in range(1, 8):
                support = set()
                for i in range(num_terms):
                    support.add(start + i * step)
                if len(support) == num_terms and max(support) <= max_degree:
                    support = sorted(support)
                    pair_count, diag_count = analyze_support(support)
                    zeroable = 0
                    for k, pc in pair_count.items():
                        dc = diag_count.get(k, 0)
                        if dc % 2 == 0:
                            target = -dc // 2
                            if abs(target) <= pc and (pc - target) % 2 == 0:
                                zeroable += 1
                    if zeroable >= 8:  # At least 8 zeroable degrees
                        all_supports.append((support, zeroable))
        
        # Union of two APs
        for start1 in range(0, 6):
            for step1 in range(1, 4):
                for start2 in range(6, 12):
                    for step2 in range(1, 4):
                        s1 = set()
                        for i in range(6):
                            s1.add(start1 + i * step1)
                        s2 = set()
                        for i in range(6):
                            s2.add(start2 + i * step2)
                        support = sorted(s1 | s2)
                        if len(support) == num_terms and max(support) <= max_degree:
                            pair_count, diag_count = analyze_support(support)
                            zeroable = 0
                            for k, pc in pair_count.items():
                                dc = diag_count.get(k, 0)
                                if dc % 2 == 0:
                                    target = -dc // 2
                                    if abs(target) <= pc and (pc - target) % 2 == 0:
                                        zeroable += 1
                            if zeroable >= 8:
                                all_supports.append((support, zeroable))
        
        # Random supports with pruning
        random.seed(42)
        for _ in range(500):
            support = sorted(random.sample(range(max_degree + 1), num_terms))
            pair_count, diag_count = analyze_support(support)
            zeroable = 0
            for k, pc in pair_count.items():
                dc = diag_count.get(k, 0)
                if dc % 2 == 0:
                    target = -dc // 2
                    if abs(target) <= pc and (pc - target) % 2 == 0:
                        zeroable += 1
            if zeroable >= 8:
                all_supports.append((support, zeroable))
        
        # Sort by zeroable count descending
        all_supports.sort(key=lambda x: -x[1])
        
        # Deduplicate
        seen = set()
        unique = []
        for support, zc in all_supports:
            key = tuple(support)
            if key not in seen:
                seen.add(key)
                unique.append(support)
        
        return unique[:30]  # Limit to avoid timeout

    # Generate diverse initial candidates
    candidates = []

    # Known good patterns
    candidates.append([1]*12)
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    candidates.append([1]*6 + [-1]*6)
    candidates.append([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    candidates.append([1,1,1,1,1,-1,1,1,1,1,1,-1])
    candidates.append([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])

    # BMM-based support search with GF(2) analysis
    good_supports = find_good_supports()
    
    for support in good_supports:
        if time.time() > deadline - 30:
            break
        coeff, nonzero = solve_signs_optimal(support)
        if coeff is not None:
            # Extract the 12-term polynomial
            trimmed = np.zeros(12, dtype=np.int64)
            for i, s in enumerate(support):
                if s < 12:
                    trimmed[s] = coeff[s]
            if count_nonzero(trimmed) == 12 and trimmed[-1] != 0:
                candidates.append(trimmed.tolist())

    # More factor-based
    random.seed(42)
    for _ in range(100):
        i = random.randint(1, 11)
        a = random.choice([-2, -1, 1, 2])
        B = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        A = np.zeros(12, dtype=np.int64)
        A[0] = 1
        A[i] = a
        B_arr = np.array(B, dtype=np.int64)
        P = np.convolve(A, B_arr)[:12]
        if count_nonzero(P) == 12 and P[-1] != 0 and np.all((P >= -64) & (P <= 64)):
            candidates.append(P.tolist())

    # Structured patterns
    for _ in range(100):
        c = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)

    # Symmetric and antisymmetric
    for _ in range(50):
        base = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        c = base + base[::-1]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)
    for _ in range(50):
        base = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        c = base + [-x for x in base[::-1]]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)

    # Evaluate all candidates
    for init_c in candidates:
        if time.time() > deadline - 20:
            break
        curr_c = np.array(init_c, dtype=np.int64)
        if not validate(curr_c):
            continue
        curr_score = evaluate(curr_c)
        if update_best(curr_c, curr_score):
            break

    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
        best_score = evaluate(best_poly)

    # Constraint-guided local search with targeted coefficient changes
    def targeted_local_search(curr_c, curr_score, max_iters=2000):
        nonlocal best_score, best_poly
        no_improvement = 0
        temp = 3.0

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            sq = poly_square(curr_c)
            nz_sq = np.where(sq != 0)[0]

            if len(nz_sq) == 0:
                break

            # Targeted: find which coefficient change would zero the most terms
            if random.random() < 0.3:
                # Identify the most impactful coefficient to change
                # For each coefficient idx, compute how many square terms it affects
                best_idx = None
                best_impact = -1
                for idx in range(12):
                    if curr_c[idx] == 0:
                        continue
                    # This coefficient affects square degrees: idx+0, idx+1, ..., idx+11
                    affected = [idx + j for j in range(12) if 0 <= idx + j < len(sq)]
                    affected_nz = sum(1 for k in affected if sq[k] != 0)
                    if affected_nz > best_impact:
                        best_impact = affected_nz
                        best_idx = idx
                if best_idx is not None:
                    idx = best_idx
                else:
                    idx = random.randint(0, 11)
            else:
                idx = random.randint(0, 11)

            new_val = random.randint(-4, 4)
            if new_val == 0:
                new_val = random.choice([-1, 1])

            if curr_c[idx] == new_val:
                continue

            old_val = curr_c[idx]
            curr_c[idx] = new_val

            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                if update_best(curr_c, curr_score):
                    return True
            else:
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temp, 0.5)):
                    curr_score = new_score
                    no_improvement = 0
                    if curr_score < best_score:
                        update_best(curr_c, curr_score)
                else:
                    curr_c[idx] = old_val
                    no_improvement += 1

            temp *= 0.999
            if no_improvement > 500:
                break

        return best_score <= 12

    # Run targeted local search
    starting_points = [best_poly.copy()]

    for _ in range(10):
        c = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        if c[-1] == 0:
            c[-1] = 1
        c_arr = np.array(c, dtype=np.int64)
        if validate(c_arr):
            starting_points.append(c_arr)

    for sp in starting_points:
        if time.time() > deadline - 10:
            break
        if targeted_local_search(sp.copy(), evaluate(sp), max_iters=1500):
            break

    # Multi-point mutation search
    if time.time() < deadline - 10:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)
        no_improvement = 0

        while time.time() < deadline - 5 and no_improvement < 3000:
            num_mods = random.choice([1, 1, 2, 2, 3])
            indices = random.sample(range(12), num_mods)
            old_vals = [curr_c[i] for i in indices]

            for i in indices:
                val = random.randint(-5, 5)
                if val == 0:
                    val = random.choice([-1, 1])
                curr_c[i] = val

            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                if update_best(curr_c, curr_score):
                    break
            else:
                for i, val in zip(indices, old_vals):
                    curr_c[i] = val
                no_improvement += 1

    # Final refinement with finer steps
    if time.time() < deadline - 3 and best_poly is not None:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)

        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, 11)
            delta_val = random.choice([-2, -1, 1, 2])
            new_val = curr_c[idx] + delta_val
            if new_val == 0:
                new_val = 1
            if not (-64 <= new_val <= 64):
                continue

            old_val = curr_c[idx]
            curr_c[idx] = new_val
            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                if update_best(curr_c, curr_score):
                    break
            else:
                curr_c[idx] = old_val

    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
