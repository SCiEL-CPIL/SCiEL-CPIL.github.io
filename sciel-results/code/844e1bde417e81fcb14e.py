import time
import itertools
from collections import defaultdict

def _square_nonzero_count(exponents, coefficients):
    """Compute P(x)^2 from sparse representation and count nonzero coefficients."""
    n = len(exponents)
    square = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = exponents[i] + exponents[j]
            c = coefficients[i] * coefficients[j]
            if i != j:
                c *= 2
            square[e] += c
    return sum(1 for c in square.values() if c != 0)

def _build_coeff_list(exponents, coefficients, max_degree=40):
    """Build a coefficient list in ascending degree from sparse representation."""
    if not exponents:
        return None
    max_exp = max(exponents)
    if max_exp > max_degree:
        return None
    result = [0] * (max_exp + 1)
    for e, c in zip(exponents, coefficients):
        result[e] = c
    nonzero = sum(1 for x in result if x != 0)
    if nonzero != 12:
        return None
    if result[-1] == 0:
        return None
    if any(abs(x) > 64 for x in result):
        return None
    if len(result) > 41:
        return None
    return result

def _compute_square_sparse(exponents, coefficients):
    """Compute P(x)^2 and return sparse list of (exp, coeff) for nonzero terms."""
    n = len(exponents)
    square = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = exponents[i] + exponents[j]
            c = coefficients[i] * coefficients[j]
            if i != j:
                c *= 2
            square[e] += c
    return [(e, c) for e, c in sorted(square.items()) if c != 0]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Fallback: 12 consecutive terms, square has 21 nonzero terms
    fallback = [1] * 12
    best_score = 21
    best_candidate = fallback

    # Strategy: P(x) = A(x) * B(x) * C(x) where A, B, C are 4-term polynomials.
    # P has at most 4*4*4 = 64 terms, but we need exactly 12.
    # Key insight: if the exponent supports of A, B, C are "disjoint" in a structured way,
    # the product P can have exactly 12 terms.
    # Then P^2 = (A*B*C)^2 = A^2 * B^2 * C^2.
    # Each of A^2, B^2, C^2 has at most 10 terms (4-term square has at most 10 distinct exponents).
    # The product of three 10-term polynomials can have up to 1000 terms, but with
    # structured exponents, many collisions occur.
    #
    # Better approach: Pre-compute the collision structure for a fixed exponent set,
    # then exhaustively search over all 2^12 sign assignments.

    # Generate candidate exponent sets for A, B, C (each 4-term)
    # such that the product P = A*B*C has exactly 12 nonzero terms.
    #
    # Strategy: Use arithmetic progressions with carefully chosen steps.
    # If A has exponents {a0, a0+d, a0+2d, a0+3d},
    # B has exponents {b0, b0+e, b0+2e, b0+3e},
    # C has exponents {c0, c0+f, c0+2f, c0+3f},
    # then P's exponents are all sums ai + bj + ck.
    # For P to have exactly 12 terms, the sumset must have exactly 12 elements.

    def gen_structured_triples(max_exp_total=40):
        """Generate structured (A_exp, B_exp, C_exp) triples where product has ~12 terms."""
        triples = []
        seen = set()

        # Pattern: arithmetic progressions with coprime steps
        for a_start in range(0, 3):
            for a_step in range(1, 5):
                a_exps = [a_start + i * a_step for i in range(4)]
                if a_exps[-1] > max_exp_total:
                    continue
                for b_start in range(0, 3):
                    for b_step in range(1, 5):
                        b_exps = [b_start + i * b_step for i in range(4)]
                        if b_exps[-1] > max_exp_total:
                            continue
                        for c_start in range(0, 3):
                            for c_step in range(1, 5):
                                c_exps = [c_start + i * c_step for i in range(4)]
                                if c_exps[-1] > max_exp_total:
                                    continue

                                # Compute sumset
                                sumset = set()
                                for ea in a_exps:
                                    for eb in b_exps:
                                        for ec in c_exps:
                                            sumset.add(ea + eb + ec)

                                if len(sumset) == 12:
                                    max_e = max(sumset)
                                    if max_e <= 40:
                                        key = (tuple(sorted(a_exps)), tuple(sorted(b_exps)), tuple(sorted(c_exps)))
                                        if key not in seen:
                                            seen.add(key)
                                            triples.append((sorted(a_exps), sorted(b_exps), sorted(c_exps)))

        # Pattern: geometric-like (powers of 2)
        base_sets = [
            [0, 1, 2, 4],
            [0, 1, 2, 3],
            [0, 2, 4, 8],
            [0, 1, 3, 7],
            [0, 1, 2, 5],
        ]
        for i, a_base in enumerate(base_sets):
            for j, b_base in enumerate(base_sets):
                for k, c_base in enumerate(base_sets):
                    # Shift to avoid overlap issues
                    for a_shift in range(0, 2):
                        for b_shift in range(0, 2):
                            for c_shift in range(0, 2):
                                a_exps = [x + a_shift for x in a_base]
                                b_exps = [x + b_shift for x in b_base]
                                c_exps = [x + c_shift for x in c_base]

                                sumset = set()
                                for ea in a_exps:
                                    for eb in b_exps:
                                        for ec in c_exps:
                                            sumset.add(ea + eb + ec)

                                if len(sumset) == 12:
                                    max_e = max(sumset)
                                    if max_e <= 40:
                                        key = (tuple(sorted(a_exps)), tuple(sorted(b_exps)), tuple(sorted(c_exps)))
                                        if key not in seen:
                                            seen.add(key)
                                            triples.append((sorted(a_exps), sorted(b_exps), sorted(c_exps)))

        return triples

    # Generate the candidate triples
    candidate_triples = gen_structured_triples(max_exp_total=40)

    # Also add some hand-crafted good candidates
    # (1 + x + x^2 + x^3) * (1 + x^4 + x^8 + x^12) * (1 + x^16)
    # This gives 4*4*1 = 16 terms... not 12.
    # Let's try: A = {0,1,2,3}, B = {0,4,8,12}, C = {0, 16}
    # No, C must be 4-term.

    # Try: A = {0,1,2,3}, B = {0,4,8,12}, C = {0, 16, 32, 48} -- too big
    # A = {0,1,2,3}, B = {0,4,8,12}, C = {0, 1, 2, 3} shifted
    # Sumset of {0,1,2,3} + {0,4,8,12} = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16} -- 17 terms

    # Better: use the generated ones. Let's also add a few more patterns.
    # A = {0, 1, 3, 7}, B = {0, 2, 4, 8}, C = {0, 1, 2, 3}
    # This is covered by the base_sets above.

    # If we didn't find any, use a simple fallback structure
    if not candidate_triples:
        # Simple: A={0,1,2,3}, B={0,4,8,12}, C={0, 16, 32, 64} -- too big
        # Just use the fallback
        return fallback

    # For each candidate triple, exhaustively search over all 2^12 sign assignments
    # (4 signs for A, 4 for B, 4 for C)
    # But we can reduce: the overall sign of P doesn't matter (square is same),
    # so we can fix one sign. Also, the relative signs matter.

    # Pre-compute the collision map for each triple
    # For a given (A_exp, B_exp, C_exp), compute all (sum_exp, list_of_(ia,ib,ic))
    # where ia, ib, ic are indices into A_exp, B_exp, C_exp.

    def precompute_collisions(a_exps, b_exps, c_exps):
        """Map each exponent in the sumset to list of (ia, ib, ic) indices."""
        collision_map = defaultdict(list)
        for ia, ea in enumerate(a_exps):
            for ib, eb in enumerate(b_exps):
                for ic, ec in enumerate(c_exps):
                    collision_map[ea + eb + ec].append((ia, ib, ic))
        return collision_map

    # For each candidate triple, do exhaustive sign search
    for a_exps, b_exps, c_exps in candidate_triples:
        if time.time() > deadline:
            break

        collision_map = precompute_collisions(a_exps, b_exps, c_exps)

        # For each exponent in the sumset, we have a list of (ia, ib, ic) tuples.
        # The coefficient at that exponent in P is:
        #   sum over (ia, ib, ic) in collision_map[e] of A_c[ia] * B_c[ib] * C_c[ic]
        #
        # For P to have exactly 12 nonzero terms, each of the 12 exponents must
        # have a nonzero coefficient.
        #
        # For P^2, the coefficient at exponent 2e is:
        #   sum over (e1, e2) with e1+e2=2e of P_c[e1] * P_c[e2]
        #   = P_c[e]^2 + 2 * sum_{e1<e2, e1+e2=2e} P_c[e1]*P_c[e2]
        #
        # We want to minimize the number of nonzero coefficients in P^2.

        # Exhaustive search over 2^12 sign assignments
        # Fix A_c[0] = 1 (overall sign invariance)
        # So we search over 2^11 = 2048 assignments.

        # Pre-compute the structure: for each of the 12 exponents, the list of (ia, ib, ic)
        # For a given sign assignment, P_c[e] = sum of signs[ia]*signs[ib]*signs[ic] for (ia,ib,ic) in collision_map[e]

        # Let's represent signs as a tuple of 12 integers (each +/-1)
        # A_c = signs[0:4], B_c = signs[4:8], C_c = signs[8:12]

        # For speed, pre-compute for each exponent e, the list of (ia, ib, ic)
        # and then for each sign assignment, compute P_c[e] quickly.

        # Actually, let's just iterate over all 2^12 sign assignments and compute P, then P^2.
        # 4096 assignments * (12 terms for P + up to 78 terms for P^2) = ~310k ops per triple.
        # With ~50 triples, that's ~15M ops. Should be fast enough.

        # But we can prune: if any P_c[e] == 0, skip (need exactly 12 nonzero).

        # Let's iterate
        num_assignments = 1 << 12  # 4096

        # For speed, pre-compute the contribution of each (ia, ib, ic) to each exponent
        # Actually, let's just compute directly.

        best_local_score = best_score
        best_local_candidate = None

        # Iterate over all sign assignments
        # To speed up, we can note that flipping all signs of A, B, or C doesn't change P^2
        # (since it's a square). So we can fix A_c[0]=1, B_c[0]=1, C_c[0]=1.
        # That reduces to 2^9 = 512 assignments.

        # Actually, flipping A_c doesn't change P^2 because P^2 = (A*B*C)^2 = A^2*B^2*C^2,
        # and A^2 is the same regardless of signs of A (since (-a)^2 = a^2).
        # Wait, no! P = A*B*C, and P^2 = (A*B*C)^2. If we flip a sign in A, P changes sign
        # for that term, but P^2 is the square of the polynomial, not the product of squares.
        # P^2[x^e] = sum_{i+j=e} P[i]*P[j]. So signs DO matter.
        #
        # However, if we multiply the entire polynomial P by -1, P^2 is unchanged.
        # So we can fix the overall sign: e.g., A_c[0] = 1.
        # That gives 2^11 = 2048 assignments.

        # Let's do 2^11 assignments (fix A_c[0] = 1)
        num_assignments = 1 << 11  # 2048

        # For each assignment, compute P_c and then P^2
        # To speed up, pre-compute the list of (ia, ib, ic) for each exponent

        # Let's use a different approach: for each assignment, compute P as a dict,
        # then compute P^2.

        for mask in range(num_assignments):
            if time.time() > deadline:
                break

            # Decode signs
            # A_c[0] = 1 (fixed)
            # A_c[1] = 1 if mask & 1 else -1
            # A_c[2] = 1 if mask & 2 else -1
            # A_c[3] = 1 if mask & 4 else -1
            # B_c[0] = 1 if mask & 8 else -1
            # B_c[1] = 1 if mask & 16 else -1
            # B_c[2] = 1 if mask & 32 else -1
            # B_c[3] = 1 if mask & 64 else -1
            # C_c[0] = 1 if mask & 128 else -1
            # C_c[1] = 1 if mask & 256 else -1
            # C_c[2] = 1 if mask & 512 else -1
            # C_c[3] = 1 if mask & 1024 else -1

            A_c = [1]
            for i in range(1, 4):
                A_c.append(1 if (mask & (1 << (i-1))) else -1)
            B_c = []
            for i in range(4):
                B_c.append(1 if (mask & (1 << (3+i))) else -1)
            C_c = []
            for i in range(4):
                C_c.append(1 if (mask & (1 << (7+i))) else -1)

            # Compute P = A * B * C
            # First compute AB = A * B
            AB = defaultdict(int)
            for ia in range(4):
                for ib in range(4):
                    AB[a_exps[ia] + b_exps[ib]] += A_c[ia] * B_c[ib]

            # Then P = AB * C
            P = defaultdict(int)
            for e_ab, c_ab in AB.items():
                for ic in range(4):
                    P[e_ab + c_exps[ic]] += c_ab * C_c[ic]

            # Check if P has exactly 12 nonzero terms
            P_items = [(e, c) for e, c in sorted(P.items()) if c != 0]
            if len(P_items) != 12:
                continue
            if max(e for e, c in P_items) > 40:
                continue

            # Compute P^2
            P_exp = [e for e, c in P_items]
            P_c = [c for e, c in P_items]
            score = _square_nonzero_count(P_exp, P_c)

            if score < best_local_score:
                best_local_score = score
                # Build candidate
                candidate = _build_coeff_list(P_exp, P_c)
                if candidate:
                    best_local_candidate = candidate

        if best_local_score < best_score:
            best_score = best_local_score
            best_candidate = best_local_candidate

    # Return the best candidate found
    if best_score < 21:
        return best_candidate
    else:
        return fallback
