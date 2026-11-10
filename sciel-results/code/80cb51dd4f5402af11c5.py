import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def evaluate(coeffs):
        if not is_valid(coeffs):
            return float('inf')
        # Compute square
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(n):
                if coeffs[j] == 0:
                    continue
                sq[i + j] += coeffs[i] * coeffs[j]
        return sum(1 for c in sq if c != 0)

    def square_nonzero_count(support, signs, mags):
        """Fast computation of nonzero count in square using index sums."""
        n = len(support)
        # Map degree -> coefficient
        # Build sparse representation
        # For each pair (i, j), the term goes to degree support[i] + support[j]
        # Coefficient at that degree += signs[i]*mags[i] * signs[j]*mags[j]
        # We need to count how many distinct degrees have nonzero sum.
        from collections import defaultdict
        deg_sum = defaultdict(int)
        for i in range(n):
            for j in range(n):
                d = support[i] + support[j]
                deg_sum[d] += signs[i] * mags[i] * signs[j] * mags[j]
        return sum(1 for v in deg_sum.values() if v != 0)

    def greedy_sign_assignment(support, mags):
        """
        For a given support and magnitudes, find sign assignment that minimizes
        nonzero count in square. Use greedy: start with all +1, then try flipping
        each sign to see if it helps.
        """
        n = len(support)
        best_signs = [1] * n
        best_count = square_nonzero_count(support, best_signs, mags)

        # Greedy improvement
        improved = True
        while improved:
            improved = False
            for i in range(n):
                new_signs = best_signs[:]
                new_signs[i] = -new_signs[i]
                cnt = square_nonzero_count(support, new_signs, mags)
                if cnt < best_count:
                    best_signs = new_signs
                    best_count = cnt
                    improved = True

        return best_signs, best_count

    def state_to_coeffs(support, signs, mags):
        max_deg = max(support)
        coeffs = [0] * (max_deg + 1)
        for i in range(len(support)):
            coeffs[support[i]] = signs[i] * mags[i]
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    best_score = float('inf')
    best_poly = [1] * 12  # fallback

    def try_result(support, signs, mags):
        nonlocal best_score, best_poly
        coeffs = state_to_coeffs(support, signs, mags)
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Strategy:
    # Phase 1: Exhaustive search over supports in [0, 24] (25 choose 12 = 5.2M)
    # For each support, use greedy sign assignment with mags=[1]*12
    # Phase 2: For the best supports found, try small magnitude variations
    # Phase 3: Local search on positions for the best result

    # Phase 1: Exhaustive over [0, 24]
    phase1_deadline = start_time + 55
    max_degree = 24
    n_positions = max_degree + 1  # 25

    # To make this feasible, we'll sample if too slow
    # 25 choose 12 = 5,200,300
    # For each, we do O(12^2) = 144 operations for square count + greedy signs
    # Let's estimate: 5.2M * ~200 ops = ~1B ops... too slow in Python
    # Let's restrict further: try [0, 20] first (21 choose 12 = 293,930)
    # Then expand if time allows

    # Actually, let's be smarter:
    # The square of a 12-term polynomial has at most C(12,2)+12 = 78 terms
    # but many can cancel. The minimum possible is 12 (if P^2 = c*P for some c,
    # but that requires P to be a monomial times a constant... not possible with 12 terms)
    # Actually the theoretical minimum for n terms is n (achieved by e.g. (1+x)^n type? No.)
    # For a polynomial with n nonzero terms, the square has at least n nonzero terms
    # (since the diagonal terms a_i^2 are all nonzero and at distinct positions if support is such).
    # Wait, no: if support[i] + support[j] = support[i'] + support[j'] for different pairs,
    # they can cancel.
    # The minimum number of nonzero terms in P^2 where P has n terms is at least ceil(n/1) = n?
    # Actually no. Consider P = 1 + x + x^2. P^2 = 1 + 2x + 3x^2 + 2x^3 + x^4 (5 terms, n=3).
    # So it's at least n. Can it be exactly n?
    # P = 1 + x: P^2 = 1 + 2x + x^2 (3 terms, n=2). So no, it's at least 2n-1 for consecutive.
    # For sparse supports, we can do better.
    # If support is {0, 1, 3, 4, ...} with gaps, we can get fewer.

    # Let's do a structured search:
    # 1. Try all supports in [0, 20] with greedy signs, mags=[1]*12
    # 2. Keep top 50 supports
    # 3. For those, try magnitude variations
    # 4. Local search

    # Phase 1: [0, 20], 21 choose 12 = 293,930
    top_supports = []  # (count, support)

    for support in combinations(range(21), 12):
        if time.time() > phase1_deadline:
            break
        support_list = list(support)
        mags = [1] * 12
        signs, cnt = greedy_sign_assignment(support_list, mags)
        top_supports.append((cnt, support_list, signs))

    # Sort and keep top
    top_supports.sort(key=lambda x: x[0])
    top_supports = top_supports[:100]

    # Evaluate top supports
    for cnt, support, signs in top_supports:
        mags = [1] * 12
        try_result(support, signs, mags)

    # Phase 2: For top 20 supports, try magnitude variations
    phase2_deadline = start_time + 70
    for cnt, support, signs in top_supports[:20]:
        if time.time() > phase2_deadline:
            break
        # Try changing one magnitude at a time
        for i in range(12):
            if time.time() > phase2_deadline:
                break
            for new_m in range(2, 11):
                mags = [1] * 12
                mags[i] = new_m
                # Re-greedy signs
                new_signs, new_cnt = greedy_sign_assignment(support, mags)
                try_result(support, new_signs, mags)

    # Phase 3: Local search on the best result
    phase3_deadline = start_time + 82
    if best_score < float('inf'):
        # Reconstruct best state
        # Find the state that gave best_score
        # Actually, let's just do local search from the best poly
        best_coeffs = best_poly
        # Convert to state
        support = [i for i, c in enumerate(best_coeffs) if c != 0]
        signs = [1 if best_coeffs[i] > 0 else -1 for i in support]
        mags = [abs(best_coeffs[i]) for i in support]

        current_support = support
        current_signs = signs
        current_mags = mags
        current_score = best_score

        improved = True
        while improved and time.time() < phase3_deadline:
            improved = False
            # Try moving each position
            for i in range(12):
                if time.time() > phase3_deadline:
                    break
                for new_pos in range(0, 41):
                    if new_pos == current_support[i]:
                        continue
                    if new_pos in current_support[:i] + current_support[i+1:]:
                        continue
                    new_support = current_support[:]
                    new_support[i] = new_pos
                    new_support.sort()
                    new_signs, new_cnt = greedy_sign_assignment(new_support, current_mags)
                    if new_cnt < current_score:
                        current_support = new_support
                        current_signs = new_signs
                        current_score = new_cnt
                        improved = True
                        try_result(new_support, new_signs, current_mags)
                        break
            # Try flipping signs
            for i in range(12):
                if time.time() > phase3_deadline:
                    break
                new_signs = current_signs[:]
                new_signs[i] = -new_signs[i]
                new_cnt = square_nonzero_count(current_support, new_signs, current_mags)
                if new_cnt < current_score:
                    current_signs = new_signs
                    current_score = new_cnt
                    improved = True
                    try_result(current_support, current_signs, current_mags)
            # Try changing magnitudes
            for i in range(12):
                if time.time() > phase3_deadline:
                    break
                for new_m in range(1, 11):
                    if new_m == current_mags[i]:
                        continue
                    new_mags = current_mags[:]
                    new_mags[i] = new_m
                    new_signs, new_cnt = greedy_sign_assignment(current_support, new_mags)
                    if new_cnt < current_score:
                        current_mags = new_mags
                        current_signs = new_signs
                        current_score = new_cnt
                        improved = True
                        try_result(current_support, current_signs, current_mags)
                        break

    # Ensure valid result
    if not is_valid(best_poly) or best_score == float('inf'):
        # Fallback: 12 ones
        best_poly = [1] * 12
        best_score = evaluate(best_poly)

    return best_poly
