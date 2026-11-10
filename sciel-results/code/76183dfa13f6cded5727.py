import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s evaluator limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build coverage masks and coverer lists
    masks = [0] * n_blocks
    coverers = [[] for _ in range(n_t)]
    for i, block in enumerate(all_blocks):
        m = 0
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            m |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = m

    full_mask = (1 << n_t) - 1

    def greedy_construct(rng, dl):
        """Randomized greedy set cover with bitmask gain computation."""
        uncovered = full_mask
        selected = []

        while uncovered:
            if time.time() > dl:
                return None

            best_block = -1
            best_gain = 0
            best_gain_mask = 0

            for b_idx in range(n_blocks):
                gain_mask = uncovered & masks[b_idx]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_gain_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    if rng.random() < 0.15:
                        best_block = b_idx
                        best_gain_mask = gain_mask

            if best_block == -1 or best_gain <= 0:
                return None

            selected.append(best_block)
            uncovered &= ~masks[best_block]

        return selected

    def local_search(selected, seed, time_limit, max_pairs_21=5000):
        """
        Local search: redundant removal + 1-1 swap + 2-1 swap.
        Returns (set_of_indices, counts_array).
        """
        rng = random.Random(seed)
        S = set(selected)
        if not S:
            return S, [0] * n_t

        # Build coverage counts
        counts = [0] * n_t
        for b in S:
            m = masks[b]
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        ls_start = time.time()
        improved = True

        while improved and (time.time() - ls_start) < time_limit:
            improved = False

            # --- Pass 1: Remove redundant blocks ---
            to_remove = []
            for b in S:
                m = masks[b]
                redundant = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    S.discard(b)
                    m = masks[b]
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                improved = True
                continue

            # --- Pass 2: 1-1 Swap ---
            list_S = list(S)
            rng.shuffle(list_S)
            swapped = False

            for b_out in list_S:
                if b_out not in S:
                    continue
                if time.time() - ls_start > time_limit - 0.5:
                    break

                # Find t-subsets only covered by b_out
                U = []
                m_out = masks[b_out]
                while m_out:
                    lsb = m_out & (-m_out)
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        U.append(idx)
                    m_out ^= lsb

                if not U:
                    # b_out is redundant (shouldn't happen after pass 1, but safe)
                    S.discard(b_out)
                    m = masks[b_out]
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    improved = True
                    swapped = True
                    break

                # Find candidate blocks covering all of U
                candidates = set(coverers[U[0]])
                for t_idx in U[1:]:
                    candidates.intersection_update(coverers[t_idx])
                candidates -= S

                if candidates:
                    b_in = rng.choice(list(candidates))
                    S.discard(b_out)
                    S.add(b_in)

                    # Update counts
                    m = masks[b_out]
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    m = masks[b_in]
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb

                    improved = True
                    swapped = True
                    break

            if swapped:
                continue

            # --- Pass 3: 2-1 Swap (Remove 2, Add 1) ---
            if time.time() - ls_start < time_limit - 1.0:
                list_S2 = list(S)
                rng.shuffle(list_S2)
                pairs_checked = 0
                found_swap = False

                for i_idx in range(len(list_S2)):
                    if pairs_checked > max_pairs_21:
                        break
                    b1 = list_S2[i_idx]
                    if b1 not in S:
                        continue

                    for j_idx in range(i_idx + 1, len(list_S2)):
                        if pairs_checked > max_pairs_21:
                            break
                        b2 = list_S2[j_idx]
                        if b2 not in S:
                            continue
                        pairs_checked += 1

                        m1 = masks[b1]
                        m2 = masks[b2]

                        # Find t-subsets that become uncovered if we remove both b1 and b2
                        U = []
                        temp = m1 | m2
                        while temp:
                            lsb = temp & (-temp)
                            idx = lsb.bit_length() - 1
                            temp ^= lsb
                            c = counts[idx]
                            if m1 & lsb:
                                c -= 1
                            if m2 & lsb:
                                c -= 1
                            if c == 0:
                                U.append(idx)

                        if not U:
                            # Both blocks are jointly redundant
                            S.discard(b1)
                            S.discard(b2)
                            for bb in (b1, b2):
                                m = masks[bb]
                                while m:
                                    lsb = m & (-m)
                                    idx = lsb.bit_length() - 1
                                    counts[idx] -= 1
                                    m ^= lsb
                            improved = True
                            found_swap = True
                            break

                        # Find candidate covering all of U
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
                        candidates -= S

                        if candidates:
                            b_in = rng.choice(list(candidates))
                            S.discard(b1)
                            S.discard(b2)
                            S.add(b_in)

                            for bb in (b1, b2):
                                m = masks[bb]
                                while m:
                                    lsb = m & (-m)
                                    idx = lsb.bit_length() - 1
                                    counts[idx] -= 1
                                    m ^= lsb
                            m = masks[b_in]
                            while m:
                                lsb = m & (-m)
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                m ^= lsb

                            improved = True
                            found_swap = True
                            break

                    if found_swap:
                        break

                if found_swap:
                    continue

        return S, counts

    # Main loop: multiple restarts with increasing local search time
    best_selected = None
    best_len = float('inf')

    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        seed = 42 + restart_count
        rng_restart = random.Random(seed)

        # Allocate time: greedy gets a fraction, local search gets the rest
        greedy_deadline = time.time() + min(15.0, max(3.0, (deadline - time.time()) * 0.4))
        selected = greedy_construct(rng_restart, greedy_deadline)

        if selected is None:
            if time.time() > deadline - 5.0:
                break
            continue

        # Local search with adaptive time limit
        remaining = deadline - time.time() - 3.0
        ls_time = min(remaining * 0.7, 20.0)
        if ls_time < 1.0:
            ls_time = 1.0

        S, counts = local_search(selected, seed, ls_time, max_pairs_21=3000)

        if len(S) < best_len:
            best_len = len(S)
            best_selected = S

        if time.time() > deadline - 5.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_selected is None:
        uncovered = full_mask
        selected = []
        while uncovered:
            if time.time() > deadline:
                break
            lsb = uncovered & (-uncovered)
            t_idx = lsb.bit_length() - 1
            best_block = -1
            best_gain = -1
            for b_idx in coverers[t_idx]:
                gain_mask = uncovered & masks[b_idx]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block != -1:
                selected.append(best_block)
                uncovered &= ~masks[best_block]
        best_selected = set(selected)

    return [list(all_blocks[i]) for i in best_selected]
