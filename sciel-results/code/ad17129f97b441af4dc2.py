"""Rewritten covering design C(v,k,t) with bitmask greedy + local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s/150s kill

    # Precompute all t-subset bitmasks and an index map
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {m: i for i, m in enumerate(all_tsubsets)}

    # Precompute, for each k-block, the list of t-subset indices it covers
    block_list = list(itertools.combinations(range(v), k))
    n_blocks = len(block_list)

    # For each block, list of t-subset indices
    block_tsets = []
    for b in block_list:
        ts = []
        for c in itertools.combinations(b, t):
            tm = 0
            for x in c:
                tm |= (1 << x)
            ts.append(tset_index[tm])
        block_tsets.append(ts)

    # Build inverse: for each tset, list of blocks covering it.
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts in enumerate(block_tsets):
        for ti in ts:
            tset_to_blocks[ti].append(bi)

    # Precompute block_tsets as arrays of indices for faster iteration
    block_tsets_arr = [np.array(ts, dtype=np.int32) for ts in block_tsets]

    def reduce_solution(sel):
        """Iteratively remove redundant blocks."""
        if not sel:
            return []
        # Build coverage count over selected blocks
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_arr[bi]:
                cov[ti] += 1

        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for bi in sel:
                # Check if removing bi is safe
                ts = block_tsets_arr[bi]
                safe = True
                for ti in ts:
                    if cov[ti] <= 1:
                        safe = False
                        break
                if safe:
                    # Remove bi
                    for ti in ts:
                        cov[ti] -= 1
                    changed = True
                else:
                    new_selected.append(bi)
            sel = new_selected
        return sel

    def local_search_kick(sel, cov, seed):
        """Kick local search: remove a random block and greedily re-cover."""
        rng = random.Random(seed)
        if not sel:
            return sel, cov, len(sel)

        best_size = len(sel)
        best_sel = list(sel)
        best_cov = cov.copy()

        # Try a few kicks
        max_kicks = 50
        kicks = 0
        while kicks < max_kicks and time.time() < deadline - 5.0:
            kicks += 1
            # Pick random block to kick
            idx = rng.randint(0, len(sel) - 1)
            bi_kick = sel[idx]

            # Save state
            saved_sel = list(sel)
            saved_cov = cov.copy()

            # Remove it
            sel.pop(idx)
            for ti in block_tsets_arr[bi_kick]:
                cov[ti] -= 1

            # Find uncovered tsets
            uncovered_tsets = [ti for ti in block_tsets_arr[bi_kick] if cov[ti] == 0]

            if not uncovered_tsets:
                # It was redundant, keep it removed
                sel = reduce_solution(sel)
                # Recompute cov
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in sel:
                    for ti in block_tsets_arr[bi]:
                        cov[ti] += 1
                if len(sel) < best_size:
                    best_size = len(sel)
                    best_sel = list(sel)
                    best_cov = cov.copy()
                continue

            # Greedily cover uncovered_tsets
            new_blocks = []
            uncovered_set = set(uncovered_tsets)

            while uncovered_set and time.time() < deadline - 5.0:
                # Find block covering max uncovered_set elements
                best_bj = -1
                max_cover = 0
                # Candidates: blocks covering any element in uncovered_set
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])

                for bj in candidates:
                    # Count how many elements of uncovered_set are in block_tsets[bj]
                    cnt = 0
                    for ti in block_tsets_arr[bj]:
                        if ti in uncovered_set:
                            cnt += 1
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj

                if best_bj == -1:
                    break

                new_blocks.append(best_bj)
                for ti in block_tsets_arr[best_bj]:
                    if ti in uncovered_set:
                        uncovered_set.remove(ti)

            # Add new blocks
            for bj in new_blocks:
                if bj not in sel:
                    sel.append(bj)
                    for ti in block_tsets_arr[bj]:
                        cov[ti] += 1

            # Reduce
            sel = reduce_solution(sel)
            # Recompute cov
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in sel:
                for ti in block_tsets_arr[bi]:
                    cov[ti] += 1

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = list(sel)
                best_cov = cov.copy()
            else:
                # Revert
                sel = saved_sel
                cov = saved_cov

        return best_sel, best_cov, best_size

    # Main greedy set cover
    uncovered = np.ones(n_tsub, dtype=bool)
    remaining = n_tsub
    selected = []

    # gain[bi] = number of currently-uncovered tsets in block bi
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi, ts in enumerate(block_tsets):
        gain[bi] = len(ts)

    # We'll use a simple loop: pick argmax gain, select it, update.
    for _ in range(n_tsub):  # at most n_tsub steps
        if time.time() > deadline - 30.0:
            break
        if remaining == 0:
            break
        # pick block with max gain
        bi = int(np.argmax(gain))
        if gain[bi] <= 0:
            break
        selected.append(bi)
        # update
        ts = block_tsets_arr[bi]
        for ti in ts:
            if uncovered[ti]:
                uncovered[ti] = False
                remaining -= 1
                # update gains of all blocks covering ti
                for bj in tset_to_blocks[ti]:
                    gain[bj] -= 1
        gain[bi] = 0  # mark as used (we won't reselect)

    if remaining > 0:
        # Fallback: if we ran out of time, do a naive completion
        unc_list = np.where(uncovered)[0]
        for ti in unc_list:
            if time.time() > deadline - 25.0:
                break
            # pick any block covering ti not already selected
            for bj in tset_to_blocks[ti]:
                if bj not in selected:
                    selected.append(bj)
                    for ti2 in block_tsets_arr[bj]:
                        uncovered[ti2] = False
                    break

    # Reduction pass
    selected = reduce_solution(selected)

    # Local search with kicks
    if time.time() < deadline - 10.0:
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            for ti in block_tsets_arr[bi]:
                cov[ti] += 1

        # Try multiple restarts of kick local search
        best_sel = selected
        best_cov = cov
        best_size = len(selected)

        n_restarts = 3
        for rs in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            seed = rs * 12345 + 678
            sel, cov, size = local_search_kick(best_sel, best_cov, seed)
            if size < best_size:
                best_size = size
                best_sel = sel
                best_cov = cov

        selected = best_sel

    # Convert selected block indices to actual blocks (sorted lists)
    blocks = []
    for bi in selected:
        blocks.append(list(block_list[bi]))

    return blocks
