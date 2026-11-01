"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Multi-start greedy + aggressive local search + limited B&B.
- Bitset coverage masks for fast gain computation.
- Multiple greedy starts with random tie-breaking for diversity.
- Local search with coverage-count-based redundancy removal and 1-for-1 swaps.
- B&B with tight lower bound as final polish.
"""
import time
import itertools
import math
import random

_START = None
_DEADLINE = 118.0  # Safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def construct(v, k, t):
    global _START
    _START = time.time()

    # 1. Generate all blocks (k-subsets) and all t-subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_t_subs = list(itertools.combinations(points, t))

    n_blocks = len(all_blocks)
    n_t = len(all_t_subs)

    # Map t-subsets to indices
    t_index = {ts: i for i, ts in enumerate(all_t_subs)}
    full_mask = (1 << n_t) - 1

    # 2. Precompute coverage masks for each block
    masks = [0] * n_blocks
    coverers = [[] for _ in range(n_t)]

    for i, blk in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(blk, t):
            j = t_index[ts]
            mask |= (1 << j)
            coverers[j].append(i)
        masks[i] = mask

    freq = [len(c) for c in coverers]

    # 3. Greedy construction with random tie-breaking
    def greedy_solve(randomize=False, seed=None):
        if randomize and seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        uncovered = full_mask
        selected = []
        point_count = [0] * v

        while uncovered:
            if _elapsed() > _DEADLINE - 5:
                return None

            best_block = -1
            best_score = -1.0
            best_gain = 0

            for i in range(n_blocks):
                m = masks[i]
                inter = m & uncovered
                if inter:
                    gain = inter.bit_count()
                    if randomize:
                        # Add small random perturbation for diversity
                        score = gain + rng.random() * 0.1
                    else:
                        # Deficiency-ratio: prefer less-saturated points
                        pen = 0.0
                        for p in all_blocks[i]:
                            pen += 1.0 / (1.0 + point_count[p])
                        score = gain / (1.0 + pen * 0.03)

                    if score > best_score:
                        best_score = score
                        best_block = i
                        best_gain = gain

            if best_block == -1:
                break

            selected.append(best_block)
            uncovered &= ~masks[best_block]
            for p in all_blocks[best_block]:
                point_count[p] += 1

        return selected

    # 4. Local search with coverage counts (fast)
    def local_search(sel):
        if sel is None:
            return None
        sel = list(sel)
        n_sel = len(sel)

        # Build coverage counts
        cov_count = [0] * n_t
        for i in sel:
            m = masks[i]
            u = m
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                cov_count[j] += 1
                u ^= lsb

        def is_redundant(idx):
            m = masks[sel[idx]]
            u = m
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                if cov_count[j] <= 1:
                    return False
                u ^= lsb
            return True

        def remove_block(idx):
            nonlocal n_sel
            m = masks[sel[idx]]
            u = m
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                cov_count[j] -= 1
                u ^= lsb
            sel.pop(idx)
            n_sel -= 1

        def add_block(idx, block_i):
            nonlocal n_sel
            m = masks[block_i]
            u = m
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                cov_count[j] += 1
                u ^= lsb
            sel[idx] = block_i

        improved = True
        while improved:
            if _elapsed() > _DEADLINE - 5:
                return sel
            improved = False

            # Phase 1: Redundancy removal
            for idx in range(n_sel):
                if _elapsed() > _DEADLINE - 5:
                    return sel
                if is_redundant(idx):
                    remove_block(idx)
                    improved = True
                    break

            # Phase 2: 1-for-1 swaps
            if not improved:
                for idx in range(n_sel):
                    if _elapsed() > _DEADLINE - 5:
                        return sel

                    # Coverage of all other blocks
                    other_cov = 0
                    for j in range(n_sel):
                        if j != idx:
                            other_cov |= masks[sel[j]]

                    need = full_mask & ~other_cov
                    if need == 0:
                        remove_block(idx)
                        improved = True
                        break

                    # Current block's unique contribution
                    cur_extra = (masks[sel[idx]] & ~other_cov).bit_count()

                    # Find best swap: maximize unique coverage
                    best_swap = -1
                    best_gain = cur_extra

                    for i in range(n_blocks):
                        if i == sel[idx]:
                            continue
                        if masks[i] & need:
                            gain = (masks[i] & ~other_cov).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = i

                    if best_swap != -1:
                        # Remove old, add new
                        m_old = masks[sel[idx]]
                        u = m_old
                        while u:
                            lsb = u & -u
                            j = lsb.bit_length() - 1
                            cov_count[j] -= 1
                            u ^= lsb

                        m_new = masks[best_swap]
                        u = m_new
                        while u:
                            lsb = u & -u
                            j = lsb.bit_length() - 1
                            cov_count[j] += 1
                            u ^= lsb

                        sel[idx] = best_swap
                        improved = True
                        break

        return sel

    # 5. Multi-start greedy + local search
    best_solution = None
    best_len = float('inf')

    # First: deterministic greedy
    sol0 = greedy_solve(randomize=False)
    if sol0 is not None:
        sol0 = local_search(sol0)
        if sol0 is not None:
            best_solution = sol0
            best_len = len(sol0)

    # Then: several randomized starts
    n_starts = 8
    for s in range(n_starts):
        if _elapsed() > _DEADLINE - 10:
            break
        sol = greedy_solve(randomize=True, seed=42 + s)
        if sol is not None:
            sol = local_search(sol)
            if sol is not None and len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol[:]

    if best_solution is None:
        return []

    # 6. Lower bound for B&B
    def calc_lb(uncovered):
        if uncovered == 0:
            return 0
        # Fractional LB
        s = 0.0
        u = uncovered
        while u:
            lsb = u & -u
            j = lsb.bit_length() - 1
            s += 1.0 / freq[j]
            u ^= lsb
        return math.ceil(s)

    # 7. Limited B&B to try to improve
    if _elapsed() < _DEADLINE - 20:
        stack = [(full_mask, 0, [])]
        nodes_explored = 0

        while stack and _elapsed() < _DEADLINE - 5:
            uncovered, depth, sol = stack.pop()
            nodes_explored += 1

            lb = calc_lb(uncovered)
            if depth + lb >= best_len:
                continue

            if uncovered == 0:
                if depth < best_len:
                    best_len = depth
                    best_solution = sol[:]
                continue

            # Select variable: uncovered t-subset with minimum freq
            u = uncovered
            min_freq = 10**9
            target_j = -1
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                f = freq[j]
                if f < min_freq:
                    min_freq = f
                    target_j = j
                u ^= lsb

            if target_j == -1:
                continue

            candidates = coverers[target_j]

            scored_candidates = []
            for i in candidates:
                m = masks[i]
                gain = (m & uncovered).bit_count()
                scored_candidates.append((gain, i))

            scored_candidates.sort(reverse=True)

            # Limit branching factor
            for gain, i in scored_candidates[:20]:
                new_uncovered = uncovered & ~masks[i]
                new_depth = depth + 1

                if new_depth >= best_len:
                    continue

                if new_uncovered == 0:
                    if new_depth < best_len:
                        best_len = new_depth
                        best_solution = sol + [i]
                    continue

                nlb = calc_lb(new_uncovered)
                if new_depth + nlb >= best_len:
                    continue

                stack.append((new_uncovered, new_depth, sol + [i]))

    # 8. Convert result
    if best_solution:
        result = [list(all_blocks[i]) for i in best_solution]
        return result
    else:
        return []
