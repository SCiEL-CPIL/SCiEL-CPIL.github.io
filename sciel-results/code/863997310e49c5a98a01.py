import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256

    # Build all APs with length >= 2
    aps = []
    ap_membership = [[] for _ in range(n)]
    ap_lengths = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                ap_lengths.append(len(sub_indices))
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)

    def _update_ap(ap_idx, signs):
        """Compute max and min subarray sum for AP ap_idx."""
        ap = aps[ap_idx]
        vals = np.array([signs[i] for i in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sub = int(np.max(cum_ext[1:] - min_prefix[:-1]))
        min_sub = int(np.min(cum_ext[1:] - max_prefix[:-1]))
        return max_sub, min_sub

    def _compute_all(signs):
        """Compute max/min subarray sums for all APs."""
        max_sums = np.zeros(len(aps), dtype=np.int32)
        min_sums = np.zeros(len(aps), dtype=np.int32)
        for i in range(len(aps)):
            max_sums[i], min_sums[i] = _update_ap(i, signs)
        return max_sums, min_sums

    def _get_disc(max_sums, min_sums):
        if len(max_sums) == 0:
            return 1
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _eval_full(signs):
        max_sums, min_sums = _compute_all(signs)
        return _get_disc(max_sums, min_sums)

    # --- Phase 1: Generate diverse candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(('thue_morse', tm))

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(('alternating', alt))

    # All +1
    candidates.append(('all_plus', [1] * n))

    # Direct construction: block pattern
    direct = [0] * n
    for i in range(n):
        block = (i // 4) % 2
        pos = i % 4
        if block == 0:
            direct[i] = 1 if pos < 2 else -1
        else:
            direct[i] = -1 if pos < 2 else 1
    candidates.append(('block_direct', direct))

    # Van der Corput permuted Thue-Morse
    def radical_inverse(i, base=2):
        result = 0.0
        f = 1.0 / base
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    perm = sorted(range(n), key=lambda i: radical_inverse(i))
    tm_perm = [0] * n
    for orig, new in enumerate(perm):
        tm_perm[new] = 1 if bin(orig).count('1') % 2 == 0 else -1
    candidates.append(('tm_permuted', tm_perm))

    # Multiple random
    for s in range(30):
        r = random.Random(s)
        candidates.append((f'rand_{s}', [r.choice((-1, 1)) for _ in range(n)]))

    # Random with bias toward balance
    for s in range(20):
        r = random.Random(100 + s)
        signs = [r.choice((-1, 1)) for _ in range(n)]
        while sum(signs) > 0:
            idx = r.randint(0, n - 1)
            if signs[idx] == 1:
                signs[idx] = -1
                break
        candidates.append((f'bal_rand_{s}', signs))

    # --- Phase 2: Evaluate candidates, pick best ---
    best_signs = candidates[0][1][:]
    best_disc = float('inf')

    for name, cand in candidates:
        if time.time() > deadline - 20:
            break
        d = _eval_full(cand)
        if d < best_disc:
            best_disc = d
            best_signs = cand[:]

    # --- Phase 3: Simulated annealing with incremental updates ---
    signs = best_signs[:]
    max_sums, min_sums = _compute_all(signs)
    current_disc = _get_disc(max_sums, min_sums)

    rng = random.Random(12345)
    no_improve = 0
    max_no_improve = 800
    temp = 3.0
    cool = 0.996

    while time.time() < deadline - 8 and no_improve < max_no_improve:
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        signs[idx] = -old_val

        for ap_idx in ap_membership[idx]:
            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)

        new_disc = _get_disc(max_sums, min_sums)

        if new_disc < current_disc:
            current_disc = new_disc
            no_improve = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            delta = new_disc - current_disc
            if delta > 0 and temp > 0.1:
                if rng.random() < math.exp(-delta / temp):
                    current_disc = new_disc
                    no_improve = 1
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                    no_improve += 1
            else:
                signs[idx] = old_val
                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                no_improve += 1

        temp *= cool

    # --- Phase 4: Strict hill climbing (no temperature) ---
    if time.time() < deadline - 5:
        signs = best_signs[:]
        max_sums, min_sums = _compute_all(signs)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(n):
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)

    # --- Phase 5: Additional restarts if time permits ---
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            signs = best_signs[:]
            rng2 = random.Random(9999 + restart)
            for _ in range(10):
                idx = rng2.randint(0, n - 1)
                signs[idx] = -signs[idx]

            max_sums, min_sums = _compute_all(signs)
            current_disc = _get_disc(max_sums, min_sums)

            # Short SA run
            temp = 2.0
            cool = 0.995
            no_improve = 0
            max_no_improve = 200

            while time.time() < deadline - 5 and no_improve < max_no_improve:
                idx = rng2.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val

                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)

                new_disc = _get_disc(max_sums, min_sums)

                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    delta = new_disc - current_disc
                    if delta > 0 and temp > 0.1:
                        if rng2.random() < math.exp(-delta / temp):
                            current_disc = new_disc
                            no_improve = 1
                        else:
                            signs[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                            no_improve += 1
                    else:
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                        no_improve += 1

                temp *= cool

    # --- Phase 6: Final strict hill climb polish ---
    if time.time() < deadline - 2:
        signs = best_signs[:]
        max_sums, min_sums = _compute_all(signs)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n):
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)

    return best_signs
