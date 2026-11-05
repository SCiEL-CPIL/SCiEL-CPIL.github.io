import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0

    best_poly = None
    best_score = float('inf')

    def count_square_terms(d):
        """Count nonzero terms in square of sparse dict {exp: coeff}."""
        if not d:
            return 0
        square = defaultdict(int)
        items = list(d.items())
        n = len(items)
        for i in range(n):
            ei, ci = items[i]
            for j in range(i, n):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    square[key] += ci * ci
                else:
                    square[key] += 2 * ci * cj
        return sum(1 for v in square.values() if v != 0)

    def dict_to_list(d):
        if not d:
            return [1]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        while len(lst) > 1 and lst[-1] == 0:
            lst.pop()
        return lst

    def validate(d):
        """Check all constraints. Returns score or None."""
        if not d:
            return None
        max_exp = max(d.keys())
        if max_exp > 40:
            return None
        n = max_exp + 1
        if n > 41:
            return None
        lst = [0] * n
        for e, c in d.items():
            lst[e] = c
        while len(lst) > 1 and lst[-1] == 0:
            lst.pop()
        if len(lst) > 41:
            return None
        if lst[-1] == 0:
            return None
        nz = sum(1 for c in lst if c != 0)
        if nz != 12:
            return None
        for c in lst:
            if c < -64 or c > 64:
                return None
        return count_square_terms(d)

    # ====================================================================
    # STRATEGY: Constructive decomposition via (A+B)^2 = A^2 + 2AB + B^2
    #
    # Key idea: Choose A (6 terms) and B (6 terms) such that:
    #   1. Supports of A^2, 2AB, B^2 are pairwise disjoint (no merging)
    #   2. P = A + B has exactly 12 nonzero terms (disjoint supports, no zero coeffs)
    #   3. The total square has as few terms as possible
    #
    # If supports are disjoint: |P^2| = |A^2| + |2AB| + |B^2|
    # A^2 has at most 6*7/2 = 21 terms (fewer if exponents are in AP)
    # AB has at most 36 terms
    # B^2 has at most 21 terms
    #
    # To minimize: use exponents in arithmetic progressions so that
    # sums collide. If A has exponents {a + i*da : i=0..5}, then A^2 has
    # exponents {2a + (i+j)*da : 0<=i<=j<=5}, i.e., 2a + k*da for k=0..10
    # -> 11 terms (if all coefficients are nonzero and no cancellation).
    #
    # For AB: exponents {a + i*da + b + j*db : 0<=i,j<=5}.
    # If da == db and a - b is a multiple of da, then AB exponents are
    # in an AP -> 11 terms. Otherwise more.
    #
    # The disjointness constraint: max(A^2) < min(2AB) and max(2AB) < min(B^2)
    # i.e., 2a+10*da < a+b and a+b+10*da < 2b (if same step)
    #
    # Let's parameterize: A exponents = {s, s+d, ..., s+5d}, B exponents = {t, t+d, ..., t+5d}
    # A^2 support: {2s, 2s+d, ..., 2s+10d} (11 terms)
    # B^2 support: {2t, 2t+d, ..., 2t+10d} (11 terms)
    # AB support: {s+t, s+t+d, ..., s+t+10d} (11 terms)
    #
    # Disjointness: 2s+10d < s+t  AND  s+t+10d < 2t
    #   => s+t > 2s+10d  => t > s+10d
    #   => s+t+10d < 2t  => s+10d < t
    # Both give t > s+10d. So if t >= s+10d+1 (and t-s is a multiple of d for AP alignment),
    # all three supports are disjoint!
    #
    # Total square terms: 11 + 11 + 11 = 33 (if no internal cancellation).
    #
    # To get FEWER than 33, we need internal cancellation within A^2, AB, or B^2.
    # A^2 coefficient at degree 2s+k*d: sum_{i+j=k, 0<=i,j<=5} c_i * c_j (with factor 2 for i!=j)
    # For k=0: c_0^2 (always > 0)
    # For k=10: c_5^2 (always > 0)
    # For k=1..9: can be zero with right coefficients!
    #
    # So the problem reduces to: find 6 coefficients c_0..c_5 (integers, nonzero)
    # such that S_k = sum_{i+j=k} c_i*c_j (with 2x for i!=j) is zero for as many
    # k in {1,...,9} as possible. Same for B.
    #
    # Additionally, AB coefficients: T_k = sum_{i+j=k} c_i^A * c_j^B, and we want
    # 2*T_k = 0 for as many k as possible (i.e., T_k = 0).
    #
    # This is a system of quadratic equations. Let's search for good coefficient
    # vectors of length 6 that minimize the number of nonzero "convolution" entries.
    # ====================================================================

    def conv_self(c):
        """Compute self-convolution of coefficient list c (length 6).
        Returns list of 11 values (degrees 0..10)."""
        n = len(c)
        result = [0] * (2*n - 1)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    result[i+j] += c[i] * c[j]
                else:
                    result[i+j] += 2 * c[i] * c[j]
        return result

    def conv_cross(ca, cb):
        """Compute cross-convolution of ca and cb (each length 6).
        Returns list of 11 values."""
        n = len(ca)
        result = [0] * (2*n - 1)
        for i in range(n):
            for j in range(n):
                result[i+j] += ca[i] * cb[j]
        return result

    def count_nonzero(lst):
        return sum(1 for x in lst if x != 0)

    # Phase 1: Search for 6-coefficient vectors that minimize self-convolution sparsity
    # c = (c0, c1, c2, c3, c4, c5), each in {-3,-2,-1,1,2,3}
    # S0 = c0^2, S10 = c5^2 (always nonzero)
    # Want S1..S9 to be zero.
    
    best_self = []  # list of (count, coeffs)
    best_self_count = 11  # max possible nonzero in self-conv
    
    # Exhaustive search over small coefficient sets
    # 6^6 = 46656 combinations - very fast
    coeff_options = [-3, -2, -1, 1, 2, 3]
    
    for c0 in coeff_options:
        for c1 in coeff_options:
            for c2 in coeff_options:
                for c3 in coeff_options:
                    if time.time() > deadline:
                        break
                    for c4 in coeff_options:
                        for c5 in coeff_options:
                            c = [c0, c1, c2, c3, c4, c5]
                            if any(x == 0 for x in c):
                                continue
                            sc = conv_self(c)
                            cnt = count_nonzero(sc)
                            if cnt < best_self_count:
                                best_self_count = cnt
                                best_self = [(cnt, c[:])]
                            elif cnt == best_self_count:
                                best_self.append((cnt, c[:]))
    
    print(f"Best self-conv nonzero count: {best_self_count}, found {len(best_self)} patterns")
    
    # Phase 2: For pairs of good coefficient vectors (ca, cb), check cross-conv sparsity
    # and overall construction validity.
    
    # Use top patterns for A and B
    # Sort by self-conv count (already sorted since we found them in order)
    # Take top 200 patterns for diversity
    top_patterns = [c for cnt, c in best_self[:500]]
    
    # If we found patterns with very few self-conv terms, great.
    # Now search for (ca, cb) pairs where cross-conv also has few nonzeros.
    
    best_total = 999
    best_AB = None
    
    # Try pairs from top patterns
    # Limit to avoid too many pairs
    n_pat = min(len(top_patterns), 200)
    
    for i in range(n_pat):
        if time.time() > deadline:
            break
        ca = top_patterns[i]
        for j in range(n_pat):
            if time.time() > deadline:
                break
            cb = top_patterns[j]
            
            sc_a = conv_self(ca)
            sc_b = conv_self(cb)
            cr = conv_cross(ca, cb)
            
            total = count_nonzero(sc_a) + count_nonzero(cr) + count_nonzero(sc_b)
            
            if total < best_total:
                best_total = total
                best_AB = (ca[:], cb[:])
    
    print(f"Best total square terms (disjoint): {best_total}")
    
    # Phase 3: Build the actual polynomial and verify
    # A exponents: {s, s+d, ..., s+5d}
    # B exponents: {t, t+d, ..., t+5d} where t > s + 10d
    # P = A + B: 12 terms at exponents {s..s+5d} U {t..t+5d}
    
    if best_AB is not None:
        ca, cb = best_AB
        # Try various (s, d, t) to fit within degree 40
        for d in [1, 2, 3]:
            for s in range(0, 5):
                # A occupies s to s+5d
                max_a = s + 5 * d
                # B must start at t > s + 10d (for disjointness)
                # Also need t + 5d <= 40
                min_t = s + 10 * d + 1
                max_t = 40 - 5 * d
                if min_t > max_t:
                    continue
                for t in range(min_t, max_t + 1):
                    # Build P
                    d_poly = {}
                    for i in range(6):
                        d_poly[s + i * d] = ca[i]
                    for i in range(6):
                        d_poly[t + i * d] = cb[i]
                    
                    if len(d_poly) != 12:
                        continue
                    
                    score = validate(d_poly)
                    if score is not None and score < best_score:
                        best_score = score
                        best_poly = dict_to_list(d_poly)
    
    # Phase 4: Also try the simple consecutive case (d=1, s=0, t=11)
    # which is just 12 consecutive terms with coefficients from our patterns
    if best_AB is not None:
        ca, cb = best_AB
        d_poly = {}
        for i in range(6):
            d_poly[i] = ca[i]
        for i in range(6):
            d_poly[6 + i] = cb[i]
        if len(d_poly) == 12:
            score = validate(d_poly)
            if score is not None and score < best_score:
                best_score = score
                best_poly = dict_to_list(d_poly)
    
    # Phase 5: Local search to improve - try modifying individual coefficients
    # of the best polynomial found so far
    
    if best_poly is not None:
        # Convert to dict for local search
        d_cur = {}
        for i, c in enumerate(best_poly):
            if c != 0:
                d_cur[i] = c
        
        cur_score = validate(d_cur)
        if cur_score is None:
            cur_score = 999
        
        # Simulated annealing on coefficients
        random.seed(123)
        T = 5.0
        for step in range(30000):
            if time.time() > deadline:
                break
            
            # Pick a random nonzero term
            exps = list(d_cur.keys())
            idx = random.randint(0, len(exps) - 1)
            e = exps[idx]
            old_val = d_cur[e]
            new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_val == old_val:
                continue
            
            d_cur[e] = new_val
            new_score = validate(d_cur)
            
            if new_score is not None:
                if new_score < cur_score:
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_poly = dict_to_list(d_cur)
                else:
                    delta = new_score - cur_score
                    if random.random() < pow(2.71828, -delta / max(T, 0.1)):
                        cur_score = new_score
                    else:
                        d_cur[e] = old_val
            else:
                d_cur[e] = old_val
            
            T *= 0.998
    
    # Phase 6: Try the known good pattern from knowledge: consecutive 0..11
    # with specific coefficients that give 13-term square
    # The knowledge says 1.533 -> 1.769 means 13 nonzero terms in square.
    # Let's try to find it via targeted search.
    
    # For consecutive exponents 0..11, S_k = sum_{i+j=k} c_i*c_j
    # We want to minimize nonzero S_k for k=0..22.
    # S_0 = c_0^2, S_22 = c_11^2 (always nonzero)
    # 21 remaining S_k values.
    
    # Use local search with focus on making S_k = 0
    random.seed(456)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        c = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        def square_count_12(c):
            sq = [0] * 23
            for i in range(12):
                for j in range(i, 12):
                    if i == j:
                        sq[i+j] += c[i] * c[j]
                    else:
                        sq[i+j] += 2 * c[i] * c[j]
            return sum(1 for x in sq if x != 0)
        
        cur_sc = square_count_12(c)
        
        T = 3.0
        for step in range(3000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old = c[idx]
            new = random.choice([-3, -2, -1, 1, 2, 3])
            if new == old:
                continue
            c[idx] = new
            new_sc = square_count_12(c)
            
            if new_sc < cur_sc:
                cur_sc = new_sc
                if cur_sc < best_score:
                    best_score = cur_sc
                    d_tmp = {i: c[i] for i in range(12)}
                    best_poly = dict_to_list(d_tmp)
            else:
                delta = new_sc - cur_sc
                if random.random() < pow(2.71828, -delta / max(T, 0.1)):
                    cur_sc = new_sc
                else:
                    c[idx] = old
            
            T *= 0.997
    
    # Phase 7: Brute force sign patterns for consecutive 0..11
    # 2^12 = 4096 patterns with coefficients +/-1
    for signs in itertools.product([-1, 1], repeat=12):
        if time.time() > deadline:
            break
        c = list(signs)
        sq = [0] * 23
        for i in range(12):
            for j in range(i, 12):
                if i == j:
                    sq[i+j] += c[i] * c[j]
                else:
                    sq[i+j] += 2 * c[i] * c[j]
        cnt = sum(1 for x in sq if x != 0)
        if cnt < best_score:
            best_score = cnt
            d_tmp = {i: c[i] for i in range(12)}
            best_poly = dict_to_list(d_tmp)
    
    # Phase 8: Try coefficients in {-2,-1,1,2} for consecutive 0..11
    # 4^12 = 16M - too many. Use random sampling.
    random.seed(789)
    for _ in range(200000):
        if time.time() > deadline:
            break
        c = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        sq = [0] * 23
        for i in range(12):
            for j in range(i, 12):
                if i == j:
                    sq[i+j] += c[i] * c[j]
                else:
                    sq[i+j] += 2 * c[i] * c[j]
        cnt = sum(1 for x in sq if x != 0)
        if cnt < best_score:
            best_score = cnt
            d_tmp = {i: c[i] for i in range(12)}
            best_poly = dict_to_list(d_tmp)
    
    # Phase 9: Try non-consecutive exponent patterns with the constructive approach
    # Use different step sizes for A and B
    
    if best_AB is not None:
        ca, cb = best_AB
        for d_a in [1, 2]:
            for d_b in [1, 2]:
                for s in range(0, 4):
                    max_a = s + 5 * d_a
                    if max_a > 30:
                        continue
                    for t in range(max_a + 1, 36):
                        max_b = t + 5 * d_b
                        if max_b > 40:
                            break
                        d_poly = {}
                        for i in range(6):
                            d_poly[s + i * d_a] = ca[i]
                        for i in range(6):
                            e = t + i * d_b
                            if e in d_poly:
                                d_poly[e] += cb[i]
                                if d_poly[e] == 0:
                                    del d_poly[e]
                            else:
                                d_poly[e] = cb[i]
                        
                        if len(d_poly) != 12:
                            continue
                        
                        score = validate(d_poly)
                        if score is not None and score < best_score:
                            best_score = score
                            best_poly = dict_to_list(d_poly)
    
    # Final fallback
    if best_poly is None:
        best_poly = [1] * 12
    
    # Final validation
    d_final = {}
    for i, c in enumerate(best_poly):
        if c != 0:
            d_final[i] = c
    final_score = validate(d_final)
    if final_score is None:
        best_poly = [1] * 12
    
    return best_poly
