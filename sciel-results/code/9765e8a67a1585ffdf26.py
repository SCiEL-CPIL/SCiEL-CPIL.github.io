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
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)

    num_aps = len(aps)

    def _compute_ap_sums(signs):
        """Compute max and min subarray sums for all APs using vectorized approach."""
        max_sums = np.zeros(num_aps, dtype=np.int32)
        min_sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(num_aps):
            ap = aps[i]
            vals = np.array([signs[j] for j in ap], dtype=np.int32)
            cum = np.cumsum(vals)
            cum_ext = np.concatenate(([0], cum))
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            max_sums[i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
            min_sums[i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))
        return max_sums, min_sums

    def _get_disc(max_sums, min_sums):
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _find_worst_ap(max_sums, min_sums):
        """Find the AP index with the worst (max abs) discrepancy."""
        worst_val = -1
        worst_idx = 0
        for i in range(num_aps):
            v = max(abs(int(max_sums[i])), abs(int(min_sums[i])))
            if v > worst_val:
                worst_val = v
                worst_idx = i
        return worst_idx, worst_val

    def _try_flip_worst(signs, max_sums, min_sums, current_disc, rng):
        """Try to improve by flipping a sign in the worst AP."""
        worst_idx, worst_val = _find_worst_ap(max_sums, min_sums)
        if worst_val <= 1:
            return False, signs, max_sums, min_sums, current_disc

        ap = aps[worst_idx]
        # Try flipping each position in the worst AP
        best_new_disc = current_disc
        best_flip_idx = -1
        best_new_max = max_sums.copy()
        best_new_min = min_sums.copy()

        # Sample a subset of positions if AP is long
        if len(ap) > 20:
            candidates = rng.sample(ap, 20)
        else:
            candidates = ap

        for idx in candidates:
            old_val = signs[idx]
            signs[idx] = -old_val

            # Update affected APs
            affected = ap_membership[idx]
            for ap_i in affected:
                ap = aps[ap_i]
                vals = np.array([signs[j] for j in ap], dtype=np.int32)
                cum = np.cumsum(vals)
                cum_ext = np.concatenate(([0], cum))
                min_prefix = np.minimum.accumulate(cum_ext)
                max_prefix = np.maximum.accumulate(cum_ext)
                max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))

            new_disc = _get_disc(max_sums, min_sums)

            if new_disc < best_new_disc:
                best_new_disc = new_disc
                best_flip_idx = idx
                best_new_max = max_sums.copy()
                best_new_min = min_sums.copy()
            else:
                # Revert
                signs[idx] = old_val
                for ap_i in affected:
                    ap = aps[ap_i]
                    vals = np.array([signs[j] for j in ap], dtype=np.int32)
                    cum = np.cumsum(vals)
                    cum_ext = np.concatenate(([0], cum))
                    min_prefix = np.minimum.accumulate(cum_ext)
                    max_prefix = np.maximum.accumulate(cum_ext)
                    max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                    min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))

        if best_flip_idx >= 0:
            # Apply the best flip
            signs[best_flip_idx] = -signs[best_flip_idx]
            for ap_i in ap_membership[best_flip_idx]:
                ap = aps[ap_i]
                vals = np.array([signs[j] for j in ap], dtype=np.int32)
                cum = np.cumsum(vals)
                cum_ext = np.concatenate(([0], cum))
                min_prefix = np.minimum.accumulate(cum_ext)
                max_prefix = np.maximum.accumulate(cum_ext)
                max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))
            return True, signs, max_sums, min_sums, best_new_disc
        return False, signs, max_sums, min_sums, current_disc

    # --- Phase 1: Generate diverse candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)

    # Block pattern
    direct = [0] * n
    for i in range(n):
        block = (i // 4) % 2
        pos = i % 4
        if block == 0:
            direct[i] = 1 if pos < 2 else -1
        else:
            direct[i] = -1 if pos < 2 else 1
    candidates.append(direct)

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
    candidates.append(tm_perm)

    # Multiple random seeds
    for s in range(20):
        r = random.Random(s)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])

    # --- Phase 2: Evaluate candidates, pick best ---
    best_signs = candidates[0][:]
    best_disc = float('inf')

    for cand in candidates:
        if time.time() > deadline - 25:
            break
        max_sums, min_sums = _compute_ap_sums(cand)
        d = _get_disc(max_sums, min_sums)
        if d < best_disc:
            best_disc = d
            best_signs = cand[:]

    # --- Phase 3: Targeted optimization - attack worst AP ---
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_sums(signs)
    current_disc = _get_disc(max_sums, min_sums)

    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 15 and no_improve < max_no_improve:
        improved, signs, max_sums, min_sums, current_disc = _try_flip_worst(
            signs, max_sums, min_sums, current_disc, rng
        )
        if improved:
            no_improve = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            no_improve += 1
            # If stuck, do a random perturbation
            if no_improve % 50 == 0:
                for _ in range(3):
                    idx = rng.randint(0, n - 1)
                    signs[idx] = -signs[idx]
                max_sums, min_sums = _compute_ap_sums(signs)
                current_disc = _get_disc(max_sums, min_sums)
                no_improve = 0

    # --- Phase 4: Simulated annealing ---
    if time.time() < deadline - 10:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_sums(signs)
        current_disc = _get_disc(max_sums, min_sums)

        rng = random.Random(12345)
        no_improve = 0
        max_no_improve = 500
        temp = 3.0
        cool = 0.996

        while time.time() < deadline - 8 and no_improve < max_no_improve:
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val

            for ap_i in ap_membership[idx]:
                ap = aps[ap_i]
                vals = np.array([signs[j] for j in ap], dtype=np.int32)
                cum = np.cumsum(vals)
                cum_ext = np.concatenate(([0], cum))
                min_prefix = np.minimum.accumulate(cum_ext)
                max_prefix = np.maximum.accumulate(cum_ext)
                max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))

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
                        for ap_i in ap_membership[idx]:
                            ap = aps[ap_i]
                            vals = np.array([signs[j] for j in ap], dtype=np.int32)
                            cum = np.cumsum(vals)
                            cum_ext = np.concatenate(([0], cum))
                            min_prefix = np.minimum.accumulate(cum_ext)
                            max_prefix = np.maximum.accumulate(cum_ext)
                            max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                            min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))
                        no_improve += 1
                else:
                    signs[idx] = old_val
                    for ap_i in ap_membership[idx]:
                        ap = aps[ap_i]
                        vals = np.array([signs[j] for j in ap], dtype=np.int32)
                        cum = np.cumsum(vals)
                        cum_ext = np.concatenate(([0], cum))
                        min_prefix = np.minimum.accumulate(cum_ext)
                        max_prefix = np.maximum.accumulate(cum_ext)
                        max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                        min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))
                    no_improve += 1

            temp *= cool

    # --- Phase 5: Strict hill climbing ---
    if time.time() < deadline - 5:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_sums(signs)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(n):
                old_val = signs[idx]
                signs[idx] = -old_val

                for ap_i in ap_membership[idx]:
                    ap = aps[ap_i]
                    vals = np.array([signs[j] for j in ap], dtype=np.int32)
                    cum = np.cumsum(vals)
                    cum_ext = np.concatenate(([0], cum))
                    min_prefix = np.minimum.accumulate(cum_ext)
                    max_prefix = np.maximum.accumulate(cum_ext)
                    max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                    min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))

                new_disc = _get_disc(max_sums, min_sums)

                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_i in ap_membership[idx]:
                        ap = aps[ap_i]
                        vals = np.array([signs[j] for j in ap], dtype=np.int32)
                        cum = np.cumsum(vals)
                        cum_ext = np.concatenate(([0], cum))
                        min_prefix = np.minimum.accumulate(cum_ext)
                        max_prefix = np.maximum.accumulate(cum_ext)
                        max_sums[ap_i] = int(np.max(cum_ext[1:] - min_prefix[:-1]))
                        min_sums[ap_i] = int(np.min(cum_ext[1:] - max_prefix[:-1]))

    # --- Phase 6: Final targeted attack ---
    if time.time() < deadline - 2:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_sums(signs)
        current_disc = _get_disc(max_sums, min_sums)
        rng = random.Random(777)
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            improved, signs, max_sums, min_sums, current_disc = _try_flip_worst(
                signs, max_sums, min_sums, current_disc, rng
            )
            if improved and current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
            if not improved:
                break

    return best_signs
