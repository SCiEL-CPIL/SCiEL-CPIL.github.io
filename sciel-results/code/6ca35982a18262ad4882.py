"""Structural approach: use P = A*B where A and B are sparse factors with
disjoint or carefully chosen exponent supports. P^2 = A^2 * B^2.
The key insight: if A^2 and B^2 have minimal overlap in their support,
the product A^2*B^2 will have minimal nonzero terms.

We search over factor structures where:
- A has k nonzero terms (k from 3 to 12)
- B has (12-k) nonzero terms
- The exponent supports are chosen to minimize collisions in A^2 and B^2
- Then we verify the full product P has exactly 12 terms and count P^2

Also try direct random search with smart initialization and local refinement."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    N = 41  # max length
    NZ_TARGET = 12

    def square_nonzero_count(poly):
        """Compute the number of nonzero coefficients in poly^2 (exact integer)."""
        nz = [(i, poly[i]) for i in range(len(poly)) if poly[i] != 0]
        n = len(nz)
        if n == 0:
            return 0
        max_deg = 2 * (len(poly) - 1)
        sq = [0] * (max_deg + 1)
        for ii in range(n):
            i, ai = nz[ii]
            for jj in range(ii, n):
                j, aj = nz[jj]
                prod = ai * aj
                if ii == jj:
                    sq[i + j] += prod
                else:
                    sq[i + j] += 2 * prod
        return sum(1 for c in sq if c != 0)

    def poly_mul(p1, p2):
        """Multiply two polynomials (coefficient lists, ascending degree)."""
        r1 = [(i, p1[i]) for i in range(len(p1)) if p1[i] != 0]
        r2 = [(i, p2[i]) for i in range(len(p2)) if p2[i] != 0]
        if not r1 or not r2:
            return [0]
        result_len = len(p1) + len(p2) - 1
        result = [0] * result_len
        for i, ci in r1:
            for j, cj in r2:
                result[i + j] += ci * cj
        # Trim trailing zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    def make_poly_from_terms(terms):
        """terms: list of (exponent, coeff). Returns coefficient list."""
        if not terms:
            return [0]
        max_exp = max(e for e, c in terms)
        poly = [0] * (max_exp + 1)
        for e, c in terms:
            poly[e] = c
        return poly

    best_poly = None
    best_score = float('inf')

    def try_candidate(poly):
        nonlocal best_poly, best_score
        # Trim trailing zeros
        last = -1
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                last = i
                break
        if last == -1:
            return
        trimmed = poly[:last + 1]
        nz_count = sum(1 for c in trimmed if c != 0)
        if nz_count != NZ_TARGET:
            return
        if len(trimmed) > N:
            return
        if any(abs(c) > 64 for c in trimmed):
            return
        score = square_nonzero_count(trimmed)
        if score < best_score:
            best_score = score
            best_poly = trimmed[:]

    # Phase 1: Factor-based construction
    # P = A * B, where A has k terms and B has (12-k) terms
    # A^2 has at most k^2 nonzero terms (with collisions reducing this)
    # B^2 has at most (12-k)^2 nonzero terms
    # P^2 = A^2 * B^2, so support size <= |supp(A^2)| * |supp(B^2)|
    # But we want P to have exactly 12 terms, so A*B must have 12 terms
    
    # Key idea: choose A and B with geometric progression exponents
    # so that A*B naturally has few terms, and A^2, B^2 have few terms
    
    # Try small factor sizes
    for k in range(2, 11):  # A has k terms, B has 12-k terms
        if 12 - k < 1:
            continue
        m = 12 - k
        
        # Try geometric progressions for A and B
        for ratio_a in range(1, 10):
            for ratio_b in range(1, 10):
                # A: exponents 0, ra, 2*ra, ..., (k-1)*ra
                # B: exponents ob, ob+rb, ob+2*rb, ..., ob+(m-1)*rb
                # Choose offset ob to separate supports
                
                max_exp_a = (k - 1) * ratio_a
                max_exp_b = (m - 1) * ratio_b
                
                # Total max exponent of P = max_exp_a + ob + max_exp_b
                # Need <= 40
                for ob in range(0, 10):
                    total_max = max_exp_a + ob + max_exp_b
                    if total_max >= N:
                        continue
                    
                    # Build A and B with various coefficient patterns
                    for sign_a in [1, -1]:
                        for sign_b in [1, -1]:
                            # Try all ±1 coefficient patterns for A
                            # But that's 2^k which is too many for large k
                            # Instead, use structured patterns
                            
                            # Pattern 1: all same sign
                            terms_a = [(i * ratio_a, sign_a) for i in range(k)]
                            terms_b = [(ob + i * ratio_b, sign_b) for i in range(m)]
                            
                            poly_a = make_poly_from_terms(terms_a)
                            poly_b = make_poly_from_terms(terms_b)
                            P = poly_mul(poly_a, poly_b)
                            
                            if len(P) <= N:
                                nz = sum(1 for c in P if c != 0)
                                if nz == NZ_TARGET and all(abs(c) <= 64 for c in P):
                                    try_candidate(P)
                    
                    # Pattern 2: alternating signs in A
                    terms_a_alt = [(i * ratio_a, sign_a * (1 if i % 2 == 0 else -1)) for i in range(k)]
                    terms_b_alt = [(ob + i * ratio_b, sign_b * (1 if i % 2 == 0 else -1)) for i in range(m)]
                    
                    poly_a = make_poly_from_terms(terms_a_alt)
                    poly_b = make_poly_from_terms(terms_b_alt)
                    P = poly_mul(poly_a, poly_b)
                    
                    if len(P) <= N:
                        nz = sum(1 for c in P if c != 0)
                        if nz == NZ_TARGET and all(abs(c) <= 64 for c in P):
                            try_candidate(P)
                    
                    # Pattern 3: A has some zero coefficients to reduce terms
                    # Actually A must have exactly k nonzero terms by construction
                    
                    if time.time() > deadline - 10:
                        break
                
                if time.time() > deadline - 10:
                    break
            
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Random search with structured initialization
    # Start with random 12-term polynomials and refine
    
    def random_poly():
        """Generate a random 12-term polynomial with coefficients in [-64, 64]."""
        # Choose 12 distinct exponents from 0 to 40
        exps = sorted(random.sample(range(N), NZ_TARGET))
        coeffs = [random.choice([-1, 1]) * random.randint(1, 64) for _ in range(NZ_TARGET)]
        poly = [0] * (exps[-1] + 1)
        for e, c in zip(exps, coeffs):
            poly[e] = c
        return poly

    def local_search(poly, max_iters=5000):
        """Local search to minimize square_nonzero_count while maintaining 12 nonzero terms."""
        current = poly[:]
        current_score = square_nonzero_count(current)
        best_local = current[:]
        best_local_score = current_score
        
        nz_positions = [i for i in range(len(current)) if current[i] != 0]
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Perturb coefficient
                idx = random.choice(nz_positions)
                old_val = current[idx]
                delta = random.randint(-5, 5)
                if delta == 0:
                    delta = 1
                new_val = old_val + delta
                if new_val == 0:
                    new_val = 1 if old_val > 0 else -1
                if abs(new_val) > 64:
                    new_val = 64 if new_val > 0 else -64
                if new_val == old_val:
                    continue
                current[idx] = new_val
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = current[:]
                else:
                    current[idx] = old_val
                    
            elif move_type < 0.7:
                # Move term to new position
                idx = random.choice(nz_positions)
                coeff = current[idx]
                empty = [i for i in range(len(current)) if current[i] == 0]
                if not empty:
                    continue
                new_pos = random.choice(empty)
                current[idx] = 0
                current[new_pos] = coeff
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = current[:]
                else:
                    current[new_pos] = 0
                    current[idx] = coeff
                    # Update nz_positions
                    nz_positions.remove(idx)
                    nz_positions.append(new_pos)
                    
            else:
                # Swap coefficients between two positions
                if len(nz_positions) < 2:
                    continue
                i1, i2 = random.sample(nz_positions, 2)
                v1, v2 = current[i1], current[i2]
                current[i1], current[i2] = v2, v1
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = current[:]
                else:
                    current[i1], current[i2] = v1, v2
        
        return best_local, best_local_score

    # Run random search with local refinement
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Try different initialization strategies
        if r < 10:
            # Geometric progression based
            ratio = random.randint(2, 6)
            exps = []
            for i in range(NZ_TARGET):
                e = i * ratio
                if e < N:
                    exps.append(e)
            if len(exps) < NZ_TARGET:
                # Fill remaining with random
                while len(exps) < NZ_TARGET:
                    e = random.randint(0, N - 1)
                    if e not in exps:
                        exps.append(e)
            exps = sorted(exps[:NZ_TARGET])
            coeffs = [random.choice([-1, 1]) * random.randint(1, 8) for _ in range(NZ_TARGET)]
            poly = [0] * (exps[-1] + 1)
            for e, c in zip(exps, coeffs):
                poly[e] = c
        elif r < 20:
            # Block structure: terms in two or three blocks
            n_blocks = random.randint(2, 3)
            block_sizes = [NZ_TARGET // n_blocks] * n_blocks
            block_sizes[0] += NZ_TARGET - sum(block_sizes)
            exps = []
            base = 0
            for bs in block_sizes:
                gap = random.randint(3, 8)
                for i in range(bs):
                    e = base + i * random.randint(1, 3)
                    if e < N:
                        exps.append(e)
                base += gap
            while len(exps) < NZ_TARGET:
                e = random.randint(0, N - 1)
                if e not in exps:
                    exps.append(e)
            exps = sorted(exps[:NZ_TARGET])
            coeffs = [random.choice([-1, 1]) * random.randint(1, 16) for _ in range(NZ_TARGET)]
            poly = [0] * (exps[-1] + 1)
            for e, c in zip(exps, coeffs):
                poly[e] = c
        else:
            # Pure random
            poly = random_poly()
        
        # Local search
        refined, score = local_search(poly, max_iters=3000)
        try_candidate(refined)
    
    # Phase 3: Try specific known-good structures
    # P with terms at positions that are multiples of a large number
    # e.g., positions 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38
    # The square will have terms at even multiples, reducing collisions
    
    structured_candidates = [
        # Geometric with ratio 4
        [(0,1),(4,1),(8,1),(12,1),(16,1),(20,1),(24,1),(28,1),(32,1),(36,1),(37,1),(38,1)],
        # With some negative coefficients
        [(0,1),(4,-1),(8,1),(12,-1),(16,1),(20,-1),(24,1),(28,-1),(32,1),(36,1),(37,-1),(38,1)],
        # Ratio 3
        [(0,1),(3,1),(6,1),(9,1),(12,1),(15,1),(18,1),(21,1),(24,1),(27,1),(30,1),(33,1)],
        # Mixed
        [(0,1),(3,1),(6,-1),(9,1),(12,-1),(15,1),(18,-1),(21,1),(24,-1),(27,1),(30,-1),(33,1)],
        # Sparse with gaps
        [(0,1),(5,1),(10,1),(15,1),(20,1),(25,1),(30,1),(35,1),(36,1),(37,1),(38,1),(39,1)],
        # With larger coefficients for cancellation
        [(0,1),(4,2),(8,1),(12,2),(16,1),(20,2),(24,1),(28,2),(32,1),(36,1),(37,2),(38,1)],
    ]
    
    for terms in structured_candidates:
        if time.time() > deadline - 3:
            break
        poly = make_poly_from_terms(terms)
        if len(poly) <= N and all(abs(c) <= 64 for c in poly):
            nz = sum(1 for c in poly if c != 0)
            if nz == NZ_TARGET:
                try_candidate(poly)
                # Also try local refinement
                refined, score = local_search(poly, max_iters=2000)
                try_candidate(refined)
    
    # Phase 4: Exhaustive search over sign patterns for specific exponent sets
    # For a fixed set of 12 exponents, try all 2^12 = 4096 sign patterns
    # with coefficient magnitudes 1 or 2
    
    fixed_exponent_sets = [
        list(range(0, 41, 4))[:12],  # 0,4,8,...,44 -> truncate to 40
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 38, 40],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 5, 10, 15, 20, 25, 30, 35, 36, 37, 38, 39],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 6, 12, 18, 24, 30, 32, 34, 36, 38, 39, 40],
    ]
    
    for exps in fixed_exponent_sets:
        if len(exps) != NZ_TARGET or max(exps) >= N:
            continue
        if time.time() > deadline - 3:
            break
        
        # Try all sign patterns with magnitude 1
        from itertools import product
        for signs in product([1, -1], repeat=NZ_TARGET):
            if time.time() > deadline - 3:
                break
            terms = [(exps[i], signs[i]) for i in range(NZ_TARGET)]
            poly = make_poly_from_terms(terms)
            try_candidate(poly)
        
        # Try some magnitude 2 patterns
        for mag_pattern in [
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2,1,2,1,2,1],
            [1,1,2,2,1,1,2,2,1,1,2,2],
        ]:
            for signs in product([1, -1], repeat=NZ_TARGET):
                if time.time() > deadline - 3:
                    break
                terms = [(exps[i], signs[i] * mag_pattern[i]) for i in range(NZ_TARGET)]
                poly = make_poly_from_terms(terms)
                if all(abs(c) <= 64 for c in poly):
                    try_candidate(poly)
    
    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    
    # Final validation
    last = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last = i
            break
    if last == -1:
        return [1] + [0] * 10 + [1]

    result = best_poly[:last + 1]
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != NZ_TARGET:
        # Force fix
        nz_positions = [i for i in range(len(result)) if result[i] != 0]
        if len(nz_positions) >= NZ_TARGET:
            fixed = [0] * len(result)
            for i in nz_positions[:NZ_TARGET]:
                fixed[i] = result[i]
            result = fixed
        else:
            result = [1] * NZ_TARGET

    return result
