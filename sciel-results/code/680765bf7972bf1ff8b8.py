"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def compute_score_fft(U):
        """Compute score using FFT convolution for exact cardinalities."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        n = len(U)
        # Sumset: convolution of indicator vector with itself
        # Size needed: 2*max_u + 1
        M = 2 * max_u + 1
        # Next power of 2 for FFT efficiency
        fft_size = 1
        while fft_size < M:
            fft_size <<= 1
        vec = np.zeros(fft_size, dtype=np.float64)
        for u in U:
            vec[u] = 1.0
        # Sumset via FFT
        f = np.fft.rfft(vec)
        conv_sum = np.fft.irfft(f * f, n=fft_size)
        n_sum = int(np.sum(conv_sum > 0.5))  # threshold for exact counting
        # Difference set: |U-U| = number of distinct (a-b) for a,b in U
        # This equals the support of autocorrelation
        # autocorr[k] = sum_i vec[i] * vec[i+k] for k in range
        # We need count of k where autocorr[k] > 0
        # autocorr via FFT: irfft(f * conj(f))
        conv_diff = np.fft.irfft(f * np.conj(f), n=fft_size)
        # conv_diff[k] for k in 0..fft_size-1 gives autocorrelation
        # The difference set size = number of positions where conv_diff > 0
        # But we need to account for negative differences too
        # Actually, |U-U| = number of distinct values of a-b
        # autocorr gives count of pairs (i,j) with i-j = k for k >= 0
        # For k < 0, it's symmetric: autocorr[-k] = autocorr[k]
        # So |U-U| = 1 + 2 * (number of k > 0 with autocorr[k] > 0)
        # Wait, let me think again.
        # conv_diff[k] = sum_i vec[i+k] * vec[i] = number of pairs (a,b) with a-b = k
        # For k >= 0: count of (a,b) with a = b + k
        # For k < 0: by symmetry, conv_diff[k] = conv_diff[-k]
        # So |U-U| = number of k in [-max_u, max_u] where conv_diff[k] > 0
        # = 1 + 2 * (number of k in 1..max_u where conv_diff[k] > 0)
        half = conv_diff[1:max_u+1]
        n_pos = int(np.sum(half > 0.5))
        n_diff = 1 + 2 * n_pos

        if n_sum == 0:
            return -1.0
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1.0 + math.log(ratio) / math.log(base)
        return score

    def compute_score_naive(U):
        """Fallback for small sets."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        n_sum = len(sumset)
        n_diff = len(diffset)
        if n_sum == 0:
            return -1.0
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1.0 + math.log(ratio) / math.log(base)
        return score

    def compute_score(U):
        """Dispatch to FFT or naive based on set size."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1.0
        max_u = max(U)
        n = len(U)
        # Use FFT when max_u is large enough to benefit
        if max_u > 50 or n > 20:
            return compute_score_fft(U)
        else:
            return compute_score_naive(U)

    best_score = 0.0
    best_set = [0, 1, 3]

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Pre-compute seed
    best_score = compute_score([0, 1, 3])

    # Phase 1: Structured sparse candidates with large max values
    # Geometric progressions with various ratios
    for ratio in [2, 3, 4, 5, 7, 10]:
        if time_left() < 5:
            break
        geo = [0, 1]
        while geo[-1] * ratio <= 20000:
            geo.append(geo[-1] * ratio)
        for i in range(3, len(geo) + 1):
            if time_left() < 5:
                break
            try_set(geo[:i])

    # Golomb-ruler-like: powers with increasing gaps
    for base_ratio in [2, 3, 4]:
        if time_left() < 5:
            break
        for n_elems in range(4, 20):
            if time_left() < 5:
                break
            # Superincreasing sequence
            seq = [0, 1]
            for i in range(2, n_elems):
                next_val = seq[-1] * base_ratio + 1
                if next_val > 50000:
                    break
                seq.append(next_val)
            if len(seq) >= 3:
                try_set(seq)

    # Random sparse sets with large max
    random.seed(42)
    for trial in range(300):
        if time_left() < 10:
            break
        max_u = random.choice([100, 200, 500, 1000, 2000, 5000, 10000])
        n = random.randint(4, min(30, max_u // 10 + 1))
        # Sample from [1, max_u] with bias toward larger values for sparsity
        inner = random.sample(range(1, max_u + 1), min(n - 1, max_u))
        cand = sorted([0] + inner + [max_u])
        try_set(cand)

    # Phase 2: Simulated annealing on sparse sets
    if time_left() > 30:
        current = list(best_set)
        current_score = best_score
        T = 0.05  # initial temperature
        T_min = 0.001
        cooling = 0.999
        max_iter = 50000

        for it in range(max_iter):
            if time_left() < 15:
                break
            if T < T_min:
                break

            # Propose a move
            move_type = random.random()
            cand = list(current)
            max_cur = max(cand) if cand else 0

            if move_type < 0.3:
                # Insert a new element
                if len(cand) < 50:
                    # Prefer large values for sparsity
                    if random.random() < 0.7:
                        new_el = random.randint(max_cur + 1, max_cur * 2 + 100)
                    else:
                        new_el = random.randint(1, max_cur + 50)
                    if new_el not in cand:
                        cand.append(new_el)
                        cand = sorted(cand)
            elif move_type < 0.6:
                # Delete an element (not 0)
                if len(cand) > 3:
                    idx = random.randint(1, len(cand) - 1)
                    cand = [x for j, x in enumerate(cand) if j != idx]
            elif move_type < 0.85:
                # Shift an element
                if len(cand) >= 2:
                    idx = random.randint(1, len(cand) - 1)
                    shift = random.randint(-max_cur // 10, max_cur // 10)
                    new_val = cand[idx] + shift
                    if new_val >= 0 and new_val not in cand:
                        cand[idx] = new_val
                        cand = sorted(cand)
            else:
                # Replace an element with a much larger one
                if len(cand) >= 2:
                    idx = random.randint(1, len(cand) - 1)
                    new_val = random.randint(max_cur + 1, max_cur * 3 + 100)
                    if new_val not in cand:
                        cand[idx] = new_val
                        cand = sorted(cand)

            if cand == current:
                continue

            s = compute_score(cand)
            if s == -1.0:
                continue

            delta_s = s - current_score
            if delta_s > 0 or random.random() < math.exp(delta_s / T):
                current = cand
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = list(cand)

            T *= cooling

    # Phase 3: Targeted search for very sparse, large sets
    if time_left() > 20:
        # Try sets with exponentially growing gaps
        for start_gap in [2, 3, 5]:
            if time_left() < 10:
                break
            for n_elems in range(3, 25):
                if time_left() < 10:
                    break
                seq = [0]
                gap = start_gap
                for i in range(1, n_elems):
                    seq.append(seq[-1] + gap)
                    gap *= 2
                if max(seq) > 100000:
                    break
                try_set(seq)

        # Try mixed: small cluster + sparse tail
        for cluster_size in [2, 3, 4]:
            if time_left() < 10:
                break
            for tail_n in range(3, 15):
                if time_left() < 10:
                    break
                for base in [10, 50, 100, 500]:
                    if time_left() < 10:
                        break
                    cluster = list(range(cluster_size))
                    tail = [base * (2 ** i) for i in range(tail_n)]
                    if max(tail) > 50000:
                        break
                    cand = sorted(set(cluster + tail))
                    try_set(cand)

    # Phase 4: Final local refinement
    if time_left() > 10:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 200

        while time_left() > 10 and no_improve < max_no_improve:
            improved = False
            max_cur = max(current)

            # Try adding elements at strategic positions
            if len(current) < 40:
                candidates_to_add = []
                # Gaps between consecutive elements
                for i in range(len(current) - 1):
                    gap = current[i + 1] - current[i]
                    if gap > 1:
                        candidates_to_add.append(current[i] + gap // 2)
                # Beyond max
                candidates_to_add.append(max_cur + 1)
                candidates_to_add.append(max_cur * 2)
                candidates_to_add.append(max_cur * 3)

                random.shuffle(candidates_to_add)
                for new_el in candidates_to_add[:20]:
                    if time_left() < 10:
                        break
                    if new_el <= 0 or new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break

            # Try replacing elements with larger values
            for i in range(1, len(current)):
                if time_left() < 10:
                    break
                if not improved:
                    for new_el in [current[i] * 2, current[i] * 3, current[i] + max_cur]:
                        if new_el in current or new_el <= 0:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 in cand:
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
                                improved = True
                                break

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    return list(best_set)
# EVOLVE-BLOCK-END
