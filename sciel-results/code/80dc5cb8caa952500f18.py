import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def is_valid(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def count_sq_nonzero(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    # Initial best: [1]*12 gives 23 nonzero terms in square
    best_poly = [1] * 12
    best_score = count_sq_nonzero(best_poly)

    def try_poly(coeffs):
        nonlocal best_poly, best_score
        if not is_valid(coeffs):
            return
        s = count_sq_nonzero(coeffs)
        if s < best_score:
            best_score = s
            best_poly = list(coeffs)

    # ===== Spectral / FFT-based support scoring =====
    # For a given support (12 exponents), the square has a term at each
    # distinct position p_i + p_j. The number of distinct positions is the
    # structural minimum. We want supports with MANY collisions (fewer distinct
    # positions), which gives more opportunities for cancellation.
    #
    # FFT trick: if we place 1's at the support positions and compute the
    # autocorrelation, the number of non-zero entries in the autocorrelation
    # equals the number of distinct pair-sums. We can use FFT to compute this
    # quickly for many candidate supports.

    def support_collision_score(support):
        """Return number of distinct pair-sums for a 12-element support."""
        if len(support) != 12 or len(set(support)) != 12:
            return float('inf')
        if min(support) < 0 or max(support) > 40:
            return float('inf')
        max_deg = max(support)
        # Build boolean array and compute autocorrelation via FFT
        arr = np.zeros(2 * max_deg + 1, dtype=np.float64)
        for e in support:
            arr[e] = 1.0
        # Autocorrelation: FFT, square magnitude, inverse FFT
        n_fft = 1 << (2 * max_deg + 1).bit_length()
        fft = np.fft.rfft(arr, n=n_fft)
        acorr = np.fft.irfft(fft * np.conj(fft), n=n_fft)
        # The relevant part is acorr[0 : 2*max_deg+1]
        acorr_relevant = acorr[:2 * max_deg + 1]
        # Count non-zero entries (with small tolerance)
        return int(np.count_nonzero(np.abs(acorr_relevant) > 0.5))

    # ===== Graph-based sign search with local refinement =====
    # For a fixed support, build the collision graph: for each degree d in P^2,
    # which pairs (i,j) contribute. Then search for sign assignments that
    # maximize cancellations (zero out as many degrees as possible).

    def build_collision_map(support):
        """Map degree -> list of (i, j, weight) where weight is 1 if i==j else 2."""
        degree_pairs = defaultdict(list)
        for i in range(12):
            for j in range(i, 12):
                d = support[i] + support[j]
                w = 1 if i == j else 2
                degree_pairs[d].append((i, j, w))
        return degree_pairs

    def eval_signs_for_support(support, signs, mags):
        """Given support, signs (+1/-1), and magnitudes, count nonzero square terms."""
        degree_pairs = build_collision_map(support)
        nonzero = 0
        for d, pairs in degree_pairs.items():
            val = 0
            for (i, j, w) in pairs:
                val += w * mags[i] * mags[j] * signs[i] * signs[j]
            if val != 0:
                nonzero += 1
        return nonzero

    def sign_search_support(support, max_trials=4096):
        """Search for best sign pattern (all mags=1) for a given support."""
        degree_pairs = build_collision_map(support)
        best_local = float('inf')
        best_signs = None

        for mask in range(max_trials):
            signs = [1 if ((mask >> i) & 1) == 0 else -1 for i in range(12)]
            nonzero = 0
            for d, pairs in degree_pairs.items():
                val = 0
                for (i, j, w) in pairs:
                    val += w * signs[i] * signs[j]
                if val != 0:
                    nonzero += 1
                    if nonzero >= best_local:
                        break
            if nonzero < best_local:
                best_local = nonzero
                best_signs = signs[:]
                if nonzero <= 2:
                    break
        return best_local, best_signs

    def local_sign_refinement(support, signs, mags, iterations=200):
        """Iteratively flip signs to reduce nonzero count."""
        degree_pairs = build_collision_map(support)
        current = eval_signs_for_support(support, signs, mags)
        for _ in range(iterations):
            improved = False
            for i in range(12):
                signs[i] = -signs[i]
                new_score = eval_signs_for_support(support, signs, mags)
                if new_score < current:
                    current = new_score
                    improved = True
                else:
                    signs[i] = -signs[i]
            if not improved:
                break
        return current, signs

    # ===== Generate candidate supports =====
    # Start with structured supports, then use FFT scoring to prioritize
    # supports with high collision density.

    def generate_supports(n_structured=30, n_random=200):
        supports = []

        # Structured: APs and near-APs
        for d in [1, 2, 3]:
            for a in range(0, 41 - 11 * d + 1):
                support = [a + i * d for i in range(12)]
                if support[11] <= 40:
                    supports.append(support)

        # Structured: two-block supports
        for a in range(0, 10):
            for b in range(a + 6, 25):
                support = list(range(a, a + 6)) + list(range(b, b + 6))
                if len(set(support)) == 12 and max(support) <= 40:
                    supports.append(support)

        # Structured: interleaved
        for a in range(0, 10):
            support = [a + 2 * i for i in range(6)] + [a + 1 + 2 * i for i in range(6)]
            support = sorted(support)
            if len(set(support)) == 12 and max(support) <= 40:
                supports.append(support)

        # Random supports with high collision potential
        random.seed(42)
        for _ in range(n_random):
            # Start with AP and perturb
            d = random.choice([1, 1, 2, 3])
            a = random.randint(0, max(0, 40 - 11 * d))
            support = [a + i * d for i in range(12)]
            for _ in range(random.randint(0, 3)):
                idx = random.randint(0, 11)
                support[idx] += random.choice([-2, -1, 1, 2])
            support = sorted(set(support))
            if len(support) == 12 and 0 <= support[0] and support[11] <= 40:
                supports.append(support)

        # Deduplicate
        seen = set()
        unique = []
        for s in supports:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    # ===== Main search loop =====
    all_supports = generate_supports()

    # Score supports by collision count (lower is better = more collisions)
    scored_supports = []
    for support in all_supports:
        coll = support_collision_score(support)
        scored_supports.append((coll, support))

    # Sort by collision count (most collisions first)
    scored_supports.sort(key=lambda x: x[0])

    # Evaluate top supports with sign search
    tried = set()
    for coll, support in scored_supports:
        if time.time() > deadline - 15:
            break
        key = tuple(support)
        if key in tried:
            continue
        tried.add(key)

        # Quick sign search
        best_local, best_signs = sign_search_support(support, max_trials=4096)
        if best_signs:
            max_deg = max(support)
            coeffs = [0] * (max_deg + 1)
            for i in range(12):
                coeffs[support[i]] = best_signs[i]
            try_poly(coeffs)

            # Local refinement with varied magnitudes
            if time.time() < deadline - 10:
                for mag_set in [[1]*12, [1,1,1,1,1,1,2,2,2,2,2,2], [1,1,1,2,2,2,2,2,2,2,2,2]]:
                    signs_ref = list(best_signs)
                    score_ref, signs_ref = local_sign_refinement(support, signs_ref, mag_set, iterations=100)
                    if score_ref < best_local:
                        max_deg = max(support)
                        coeffs = [0] * (max_deg + 1)
                        for i in range(12):
                            coeffs[support[i]] = signs_ref[i] * mag_set[i]
                        try_poly(coeffs)

    # ===== Magnitude optimization on best found support =====
    if time.time() < deadline - 5:
        best_support = [i for i, c in enumerate(best_poly) if c != 0]
        if len(best_support) == 12:
            best_signs_cur = [1 if best_poly[i] > 0 else -1 for i in best_support]
            best_mags_cur = [abs(best_poly[i]) for i in best_support]

            random.seed(12345)
            for trial in range(5000):
                if time.time() > deadline - 3:
                    break
                mags = list(best_mags_cur)
                n_changes = random.randint(1, 4)
                for _ in range(n_changes):
                    idx = random.randint(0, 11)
                    mags[idx] = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64])

                signs = list(best_signs_cur)
                n_flips = random.randint(0, 2)
                for _ in range(n_flips):
                    idx = random.randint(0, 11)
                    signs[idx] = -signs[idx]

                max_deg = max(best_support)
                coeffs = [0] * (max_deg + 1)
                for i in range(12):
                    coeffs[best_support[i]] = signs[i] * mags[i]
                try_poly(coeffs)

    # ===== Broader random search =====
    if time.time() < deadline - 2:
        random.seed(99999)
        for _ in range(8000):
            if time.time() > deadline - 1:
                break
            d = random.choice([1, 1, 1, 2, 3])
            a = random.randint(0, max(0, 40 - 11 * d))
            support = [a + i * d for i in range(12)]
            for _ in range(random.randint(0, 2)):
                idx = random.randint(0, 11)
                support[idx] += random.randint(-2, 2)
            support = sorted(set(support))
            if len(support) != 12 or support[0] < 0 or support[11] > 40:
                continue

            signs = [random.choice([-1, 1]) for _ in range(12)]
            mags = [random.choice([1, 1, 1, 2, 3]) for _ in range(12)]

            max_deg = max(support)
            coeffs = [0] * (max_deg + 1)
            for i in range(12):
                coeffs[support[i]] = signs[i] * mags[i]
            try_poly(coeffs)

    # Ensure valid result
    if not is_valid(best_poly):
        best_poly = [1] * 12

    return best_poly
