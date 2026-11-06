import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Fallback: the seed program's answer
    fallback = [1] * 12
    fallback_score = 21  # (1+x+...+x^11)^2 has 21 nonzero terms

    best_score = fallback_score
    best_candidate = fallback

    # Strategy: For structured exponent sets with high additive collision,
    # find coefficients that maximize cancellation in P^2.
    # 
    # Key insight: If exponents form an arithmetic progression, then e_i + e_j
    # depends only on i+j, giving natural groups. Within each group, we need
    # sum of c_i*c_j (for pairs in that group) = 0.
    #
    # For a 12-term AP: e_i = a + i*d for i=0..11
    # e_i + e_j = 2a + (i+j)*d, so sum s = i+j ranges from 1 to 21
    # Group s has pairs (i,j) with i<j and i+j=s
    # 
    # We search over APs and nearby structures, then optimize coefficients.

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        score = _square_nonzero_count(exps, coeffs)
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(exps, coeffs)
            if candidate is not None:
                best_candidate = candidate
            return True
        return False

    # Generate candidate exponent sets
    candidate_exps = []

    # 1. Pure arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(tuple(exps))

    # 2. Two interleaved APs of 6
    for s1 in range(0, 6):
        for step1 in range(1, 5):
            for s2 in range(4, 16):
                for step2 in range(1, 5):
                    exps1 = [s1 + i * step1 for i in range(6)]
                    exps2 = [s2 + i * step2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(tuple(exps))

    # 3. Three interleaved APs of 4
    for s1 in range(0, 5):
        for step in range(1, 4):
            for s2 in range(3, 14):
                for s3 in range(11, 25):
                    exps1 = [s1 + i * step for i in range(4)]
                    exps2 = [s2 + i * step for i in range(4)]
                    exps3 = [s3 + i * step for i in range(4)]
                    exps = sorted(exps1 + exps2 + exps3)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(tuple(exps))

    # 4. Four interleaved APs of 3
    for s1 in range(0, 4):
        for step in range(1, 3):
            for s2 in range(2, 11):
                for s3 in range(9, 20):
                    for s4 in range(17, 30):
                        exps1 = [s1 + i * step for i in range(3)]
                        exps2 = [s2 + i * step for i in range(3)]
                        exps3 = [s3 + i * step for i in range(3)]
                        exps4 = [s4 + i * step for i in range(3)]
                        exps = sorted(exps1 + exps2 + exps3 + exps4)
                        if len(set(exps)) == 12 and max(exps) <= 40:
                            candidate_exps.append(tuple(exps))

    # 5. Symmetric: pairs summing to constant T
    # e.g., {a, T-a, b, T-b, c, T-c, d, T-d, e, T-e, f, T-f}
    for T in range(14, 41):
        for a in range(0, T // 2):
            for b in range(a + 1, T // 2):
                for c in range(b + 1, T // 2):
                    for d in range(c + 1, T // 2):
                        for e in range(d + 1, T // 2):
                            for f in range(e + 1, T // 2):
                                exps = [a, T-a, b, T-b, c, T-c, d, T-d, e, T-e, f, T-f]
                                if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                                    candidate_exps.append(tuple(sorted(exps)))

    # 6. Quadratic patterns
    for a in range(1, 3):
        for b in range(-5, 6):
            for c in range(0, 15):
                exps = []
                valid = True
                for i in range(12):
                    e = a * i * i + b * i + c
                    if e < 0 or e > 40:
                        valid = False
                        break
                    exps.append(e)
                if valid and len(set(exps)) == 12:
                    candidate_exps.append(tuple(sorted(exps)))

    # 7. Near-AP: AP with one perturbation
    for start in range(0, 10):
        for step in range(1, 8):
            base = [start + i * step for i in range(12)]
            if max(base) > 40:
                continue
            for pos in range(12):
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    exps = base[:]
                    exps[pos] += delta
                    if min(exps) >= 0 and max(exps) <= 40 and len(set(exps)) == 12:
                        candidate_exps.append(tuple(sorted(exps)))

    # Deduplicate
    seen = set()
    unique_exps = []
    for exps in candidate_exps:
        if exps not in seen:
            seen.add(exps)
            unique_exps.append(exps)

    # Score exponent sets by additive energy (more collisions = more cancellation potential)
    def additive_energy_score(exps):
        sums = defaultdict(int)
        n = len(exps)
        for i in range(n):
            for j in range(i + 1, n):
                sums[exps[i] + exps[j]] += 1
        # Count pairs of pairs with same sum
        energy = 0
        for cnt in sums.values():
            if cnt > 1:
                energy += cnt * (cnt - 1) // 2
        return energy

    # Sort by additive energy descending
    scored = [(additive_energy_score(e), e) for e in unique_exps]
    scored.sort(key=lambda x: -x[0])

    # Optimize coefficients for a given exponent set
    def optimize_coeffs(exps, max_iter=50):
        """Local search for best coefficients in small range."""
        n = 12
        # Start with all +1
        coeffs = [1] * n
        current = _square_nonzero_count(exps, coeffs)

        # Try sign patterns first (fast)
        # Then try small integer coefficients

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False

            # Single coefficient changes
            for i in range(n):
                old = coeffs[i]
                for new in [-4, -3, -2, -1, 1, 2, 3, 4]:
                    if new == old:
                        continue
                    coeffs[i] = new
                    new_score = _square_nonzero_count(exps, coeffs)
                    if new_score < current:
                        current = new_score
                        improved = True
                        break
                    else:
                        coeffs[i] = old
                        break

            # Double flip (sign changes)
            for i in range(n):
                for j in range(i + 1, n):
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
                    new_score = _square_nonzero_count(exps, coeffs)
                    if new_score < current:
                        current = new_score
                        improved = True
                    else:
                        coeffs[i] = -coeffs[i]
                        coeffs[j] = -coeffs[j]

            if not improved:
                break

        return current, coeffs

    # Process top exponent sets
    num_to_try = 200
    for idx, (ae, exps) in enumerate(scored[:num_to_try]):
        if time.time() > deadline:
            break

        # Quick sign-only check first
        signs = [1] * 12
        sign_score = _square_nonzero_count(exps, signs)

        if sign_score < best_score:
            candidate = _build_coeff_list(exps, signs)
            if candidate is not None:
                best_score = sign_score
                best_candidate = candidate

        # Full optimization
        score, coeffs = optimize_coeffs(exps, max_iter=30)
        if score < best_score:
            candidate = _build_coeff_list(exps, coeffs)
            if candidate is not None:
                best_score = score
                best_candidate = candidate

    # Also try a few specific known-good structures
    # P(x) = x^0 - x^1 + x^2 - x^3 + ... alternating in AP
    for start in range(0, 5):
        for step in range(1, 5):
            if time.time() > deadline:
                break
            exps = [start + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            # Alternating signs
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            score = _square_nonzero_count(exps, coeffs)
            if score < best_score:
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_score = score
                    best_candidate = candidate

    # Try P with coefficients [1, -1, 1, -1, ...] in various structured sets
    # and also try [1, 1, -1, -1, 1, 1, -1, -1, ...] patterns

    # Final: if we have time, do a broader random search
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Random structured exponent set
            base_step = random.randint(1, 4)
            base_start = random.randint(0, 5)
            exps = []
            for i in range(12):
                e = base_start + i * base_step + random.randint(-1, 1)
                exps.append(e)
            exps = sorted(set(exps))
            if len(exps) != 12 or max(exps) > 40 or min(exps) < 0:
                continue
            # Random small coefficients
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            score = _square_nonzero_count(exps, coeffs)
            if score < best_score:
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_score = score
                    best_candidate = candidate

    if best_score < fallback_score:
        return best_candidate
    return fallback

# EVOLVE-BLOCK-END
