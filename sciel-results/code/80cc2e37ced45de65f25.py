import time
import itertools
import random
from collections import defaultdict

def _square_nonzero_count(exponents, coefficients):
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
    deadline = start_time + 85.0

    fallback = [1] * 12
    fallback_score = 21
    best_score = fallback_score
    best_candidate = fallback

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        score = _square_nonzero_count(exps, coeffs)
        if score < best_score:
            candidate = _build_coeff_list(exps, coeffs)
            if candidate is not None:
                best_score = score
                best_candidate = candidate
                return True
        return False

    # Strategy 1: P = A + B where A is 3 terms, B is 9 terms
    # A^2 has at most 6 terms, B^2 has at most 45 terms, 2AB has at most 27 terms
    # Total: 78, but many collisions reduce this

    # Try to minimize by making A^2 and B^2 have many cancellations internally
    # and 2AB have many collisions

    # Start with structured exponent sets for A and B
    # A: 3 terms in AP, B: 9 terms in AP, disjoint or overlapping

    # Generate many structured candidates

    candidate_exps = []
    candidate_coeffs = []

    # Pure AP with 12 terms
    for start in range(0, 8):
        for step in range(1, 6):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(tuple(exps))
                # Try various coefficient patterns
                for pattern in [
                    [1]*12,
                    [1,-1]*6,
                    [1,1,-1,-1]*3,
                    [1,-1,1,-1]*3,
                    [1]*6 + [-1]*6,
                ]:
                    candidate_coeffs.append(pattern[:12])

    # Symmetric: pairs summing to T
    for T in range(12, 41):
        half = T // 2
        for i in range(0, min(6, half)):
            for j in range(i+1, min(6, half)):
                for k in range(j+1, min(6, half)):
                    for l in range(k+1, min(6, half)):
                        for m in range(l+1, min(6, half)):
                            for n in range(m+1, min(6, half)):
                                exps = [i, T-i, j, T-j, k, T-k, l, T-l, m, T-m, n, T-n]
                                if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                                    candidate_exps.append(tuple(sorted(exps)))
                                    # Alternating signs on the sorted version
                                    sorted_exps = sorted(exps)
                                    coeffs = [1 if idx % 2 == 0 else -1 for idx in range(12)]
                                    candidate_coeffs.append(coeffs)

    # 2 interleaved APs of 6
    for s1 in range(0, 5):
        for st1 in range(1, 4):
            for s2 in range(5, 15):
                for st2 in range(1, 4):
                    exps1 = [s1 + i*st1 for i in range(6)]
                    exps2 = [s2 + i*st2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(tuple(exps))
                        # First 6 positive, last 6 negative
                        candidate_coeffs.append([1]*6 + [-1]*6)
                        # Alternating
                        candidate_coeffs.append([1 if i%2==0 else -1 for i in range(12)])

    # 3 interleaved APs of 4
    for s1 in range(0, 4):
        for st in range(1, 3):
            for s2 in range(4, 12):
                for s3 in range(8, 20):
                    exps1 = [s1 + i*st for i in range(4)]
                    exps2 = [s2 + i*st for i in range(4)]
                    exps3 = [s3 + i*st for i in range(4)]
                    exps = sorted(exps1 + exps2 + exps3)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(tuple(exps))
                        candidate_coeffs.append([1]*4 + [-1]*4 + [1]*4)
                        candidate_coeffs.append([1,-1,1,-1]*3)

    # 4 interleaved APs of 3
    for s1 in range(0, 3):
        for st in range(1, 3):
            for s2 in range(3, 10):
                for s3 in range(6, 16):
                    for s4 in range(9, 22):
                        exps1 = [s1 + i*st for i in range(3)]
                        exps2 = [s2 + i*st for i in range(3)]
                        exps3 = [s3 + i*st for i in range(3)]
                        exps4 = [s4 + i*st for i in range(3)]
                        exps = sorted(exps1 + exps2 + exps3 + exps4)
                        if len(set(exps)) == 12 and max(exps) <= 40:
                            candidate_exps.append(tuple(exps))
                            candidate_coeffs.append([1,1,1,-1,-1,-1,1,1,1,-1,-1,-1])

    # Deduplicate
    seen = set()
    unique_pairs = []
    for exps, coeffs in zip(candidate_exps, candidate_coeffs):
        if exps not in seen:
            seen.add(exps)
            unique_pairs.append((exps, coeffs))

    # Score by additive energy
    def additive_energy_score(exps):
        sums = defaultdict(int)
        n = len(exps)
        for i in range(n):
            for j in range(i + 1, n):
                sums[exps[i] + exps[j]] += 1
        energy = 0
        for cnt in sums.values():
            if cnt > 1:
                energy += cnt * (cnt - 1) // 2
        return energy

    # Add all unique exps with all-ones coeff too
    all_exps = list(seen)
    for e in all_exps:
        unique_pairs.append((e, [1]*12))

    # Score and sort
    scored = []
    for exps, coeffs in unique_pairs:
        ae = additive_energy_score(exps)
        scored.append((ae, exps, coeffs))

    scored.sort(key=lambda x: -x[0])

    # Optimize coefficients for top candidates
    def optimize_coeffs(exps, deadline):
        n = 12
        # Start with [1]*12
        coeffs = [1] * n
        current = _square_nonzero_count(exps, coeffs)

        # Also try [1,-1]*6
        alt_coeffs = [1 if i%2==0 else -1 for i in range(12)]
        alt_score = _square_nonzero_count(exps, alt_coeffs)
        if alt_score < current:
            coeffs = alt_coeffs[:]
            current = alt_score

        # Try [1]*6 + [-1]*6
        half_coeffs = [1]*6 + [-1]*6
        half_score = _square_nonzero_count(exps, half_coeffs)
        if half_score < current:
            coeffs = half_coeffs[:]
            current = half_score

        # Local search
        for iteration in range(20):
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

            # Pairwise sign flips
            for i in range(n):
                for j in range(i+1, n):
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

    # Process top 300 by energy
    num_to_try = min(300, len(scored))
    for idx in range(num_to_try):
        if time.time() > deadline:
            break
        ae, exps, init_coeffs = scored[idx]

        # Quick check with initial coeffs
        score = _square_nonzero_count(exps, init_coeffs)
        if score < best_score:
            try_candidate(exps, init_coeffs)

        # Full optimization
        score, coeffs = optimize_coeffs(exps, deadline - 2)
        if score < best_score:
            try_candidate(exps, coeffs)

    # Strategy 2: Random search with structured biases
    if time.time() < deadline - 15:
        random.seed(42)
        for _ in range(3000):
            if time.time() > deadline:
                break

            # Choose a structured base
            choice = random.random()
            if choice < 0.3:
                # AP
                start = random.randint(0, 6)
                step = random.randint(1, 4)
                exps = [start + i*step for i in range(12)]
            elif choice < 0.5:
                # Symmetric
                T = random.randint(14, 40)
                half = T // 2
                if half < 6:
                    continue
                picks = random.sample(range(0, half), 6)
                exps = [p for p in picks] + [T-p for p in picks]
            elif choice < 0.7:
                # 2 interleaved APs
                s1 = random.randint(0, 4)
                st1 = random.randint(1, 3)
                s2 = random.randint(5, 14)
                st2 = random.randint(1, 3)
                exps = sorted([s1+i*st1 for i in range(6)] + [s2+i*st2 for i in range(6)])
            elif choice < 0.85:
                # 3 interleaved APs
                s1 = random.randint(0, 3)
                st = random.randint(1, 2)
                s2 = random.randint(4, 11)
                s3 = random.randint(8, 19)
                exps = sorted([s1+i*st for i in range(4)] + [s2+i*st for i in range(4)] + [s3+i*st for i in range(4)])
            else:
                # Random
                exps = sorted(random.sample(range(0, 41), 12))

            if len(set(exps)) != 12 or max(exps) > 40 or min(exps) < 0:
                continue

            # Random coeffs with small values
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]

            score = _square_nonzero_count(exps, coeffs)
            if score < best_score:
                try_candidate(exps, coeffs)

    # Strategy 3: Direct construction - try to find P where P^2 has exactly 12 terms
    # This requires extreme cancellation. Try small coefficient patterns.
    if time.time() < deadline - 5:
        # P = 1 - x + x^2 - x^3 + ... (alternating)
        for start in range(0, 5):
            for step in range(1, 5):
                if time.time() > deadline:
                    break
                exps = [start + i*step for i in range(12)]
                if max(exps) > 40:
                    continue
                # Alternating
                coeffs = [1 if i%2==0 else -1 for i in range(12)]
                try_candidate(exps, coeffs)

                # Blocks of 2
                coeffs = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                try_candidate(exps, coeffs)

                # Blocks of 3
                coeffs = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
                try_candidate(exps, coeffs)

    if best_score < fallback_score:
        return best_candidate
    return fallback

# EVOLVE-BLOCK-END
