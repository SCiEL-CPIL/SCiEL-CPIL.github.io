import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 143.0
    start_time = time.time()

    # Precompute all t-subsets and index
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all blocks and the t-subsets each covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    block_covered = []
    for block in all_blocks:
        covered = [t_idx[ts] for ts in itertools.combinations(block, t)]
        covered.sort()
        block_covered.append(covered)

    # Reverse index: t-subset -> list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, cov in enumerate(block_covered):
        for j in cov:
            t_to_blocks[j].append(b_idx)

    # ---------- Greedy seed (fast, vectorized-ish) ----------
    def greedy_seed(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        chosen = []
        while uncovered and time.time() < deadline - 55.0:
            # Pick a random uncovered t-subset
            start = rng.choice(tuple(uncovered))
            cands = t_to_blocks[start]
            best_b = -1
            best_gain = 0
            for b in cands:
                gain = 0
                for j in block_covered[b]:
                    if j in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
                    if gain == len(uncovered):
                        break
            if best_b < 0:
                break
            chosen.append(best_b)
            for j in block_covered[best_b]:
                if j in uncovered:
                    uncovered.discard(j)
        return chosen

    # ---------- Coverage helpers ----------
    def coverage_counts(block_list):
        counts = np.zeros(n_t, dtype=np.int32)
        for b in block_list:
            for j in block_covered[b]:
                counts[j] += 1
        return counts

    def is_valid(counts):
        return np.all(counts >= 1)

    def redundancy(counts):
        # sum of (count - 1) over all t-subsets = total coverage - n_t
        return int(counts.sum() - n_t)

    # ---------- SA over fixed-size list ----------
    def sa_search(block_list, time_budget):
        """
        State: list of block indices (fixed length n).
        Always valid: every t-subset covered >= 1.
        Move types:
          1. Swap: remove block i (must keep validity), add block j (not already in list).
          2. Replace: remove block i, add block j (same as swap, but we allow any j).
        Objective: minimize redundancy.
        """
        n = len(block_list)
        if n == 0:
            return block_list, 0

        counts = coverage_counts(block_list)
        if not is_valid(counts):
            return block_list, redundancy(counts)

        cur_redund = redundancy(counts)
        best_list = list(block_list)
        best_redund = cur_redund

        rng = random.Random()
        T = 50.0
        cooling = 0.9995
        min_T = 0.1

        # Precompute: for each position, which blocks could replace it while keeping validity
        # We'll compute on the fly but cache candidate pools per t-subset.

        while time.time() < time_budget and T > min_T:
            # Pick a random position to modify
            pos = rng.randrange(n)
            old_block = block_list[pos]

            # Determine which t-subsets would become uncovered if we remove old_block
            # i.e., t-subsets where counts == 1 and old_block covers them
            critical = []
            for j in block_covered[old_block]:
                if counts[j] == 1:
                    critical.append(j)

            if critical:
                # The new block MUST cover all critical t-subsets
                # Find blocks that cover all of them
                # Start with blocks covering the first critical, intersect with others
                candidates = set(t_to_blocks[critical[0]])
                for c in critical[1:]:
                    candidates &= set(t_to_blocks[c])
                    if not candidates:
                        break
                # Exclude blocks already in the list (except old_block itself, which we're replacing)
                in_list = set(block_list)
                candidates -= in_list
                candidates.add(old_block)  # allow "no-op" (keep same block)
                candidates = list(candidates)
                if not candidates:
                    T *= cooling
                    continue
                new_block = rng.choice(candidates)
            else:
                # No critical t-subsets: any block not already in the list works
                in_list = set(block_list)
                # Sample a random block not in the list
                # To avoid scanning all n_blocks, sample a few
                new_block = old_block
                for _ in range(20):
                    cand = rng.randrange(n_blocks)
                    if cand not in in_list:
                        new_block = cand
                        break
                # If we just picked the same block, that's a no-op; that's fine
                # But let's also allow picking from a diverse set
                if new_block == old_block:
                    # Try to pick a different one
                    for _ in range(10):
                        cand = rng.randrange(n_blocks)
                        if cand not in in_list:
                            new_block = cand
                            break

            # Apply the move
            block_list[pos] = new_block
            # Update counts
            for j in block_covered[old_block]:
                counts[j] -= 1
            for j in block_covered[new_block]:
                counts[j] += 1

            # Check validity
            if not is_valid(counts):
                # Reject: revert
                for j in block_covered[new_block]:
                    counts[j] -= 1
                for j in block_covered[old_block]:
                    counts[j] += 1
                block_list[pos] = old_block
                T *= cooling
                continue

            new_redund = redundancy(counts)
            delta = new_redund - cur_redund

            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cur_redund = new_redund
                if cur_redund < best_redund:
                    best_redund = cur_redund
                    best_list = list(block_list)
            else:
                # Revert
                for j in block_covered[new_block]:
                    counts[j] -= 1
                for j in block_covered[old_block]:
                    counts[j] += 1
                block_list[pos] = old_block

            T *= cooling

        return best_list, best_redund

    # ---------- Reduction: remove redundant blocks ----------
    def reduce_blocks(block_list):
        """Remove blocks that are fully redundant (all their t-subsets have count > 1)."""
        current = list(block_list)
        for _ in range(100):
            if time.time() > deadline - 15.0:
                break
            counts = coverage_counts(current)
            removable = []
            for i, b in enumerate(current):
                ok = True
                for j in block_covered[b]:
                    if counts[j] <= 1:
                        ok = False
                        break
                if ok:
                    removable.append(i)
            if not removable:
                break
            i = removable.pop()
            current.pop(i)
        return current

    # ---------- Main search ----------
    # Generate several greedy seeds
    seeds = []
    for s in range(8):
        if time.time() > deadline - 50.0:
            break
        sol = greedy_seed(s)
        if sol:
            counts = coverage_counts(sol)
            if is_valid(counts):
                seeds.append(sol)

    if not seeds:
        # Fallback: one more greedy
        seeds = [greedy_seed(0)]

    # Start with the best seed
    best_sol = min(seeds, key=len)
    best_len = len(best_sol)

    # Run SA on the best seed with the remaining time budget
    # We'll run SA multiple times with different starting points
    remaining = deadline - 20.0 - time.time()
    n_sa_runs = max(1, int(remaining / 15.0))  # ~15s per SA run

    for run in range(n_sa_runs):
        if time.time() > deadline - 20.0:
            break

        # Pick a starting point: best so far or a random seed
        if run == 0:
            start = best_sol
        else:
            start = seeds[run % len(seeds)]

        time_budget = min(deadline - 20.0, time.time() + 15.0)
        sa_result, sa_redund = sa_search(list(start), time_budget)

        # Reduce the SA result
        reduced = reduce_blocks(sa_result)
        counts = coverage_counts(reduced)
        if is_valid(counts) and len(reduced) < best_len:
            best_len = len(reduced)
            best_sol = reduced

    # Final reduction
    best_sol = reduce_blocks(best_sol)
    counts = coverage_counts(best_sol)
    if not is_valid(counts):
        # Fallback to best seed
        best_sol = min(seeds, key=len)

    # Convert to result
    result = [all_blocks[i] for i in best_sol]
    result = list(set(result))

    # Final validation and fix
    final_covered = np.zeros(n_t, dtype=bool)
    for b in result:
        for ts in itertools.combinations(b, t):
            final_covered[t_idx[ts]] = True

    if not final_covered.all():
        uncovered = np.where(~final_covered)[0]
        for ts_idx in uncovered:
            for b_idx in t_to_blocks[ts_idx]:
                b = all_blocks[b_idx]
                if b not in result:
                    result.append(b)
                    for ts in itertools.combinations(b, t):
                        final_covered[t_idx[ts]] = True
                    break

    return [list(b) for b in result]
