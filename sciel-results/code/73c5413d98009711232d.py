"""
Covering design solver: structural change to a randomized local-search /
simulated-annealing style block-pool optimizer over a precomputed universe of
k-subsets, combined with a fast greedy seed and aggressive 1-for-1 swaps and
deletions.

Key structural differences from the previous pure greedy approach:
1. Instead of committing to a greedy ordering, we build an initial covering with
   a randomized greedy (fast) and then run a long local-search phase that
   (a) deletes any redundant block,
   (b) tries to replace a block with another that covers all of its unique
       t-subsets while increasing overall "overlap" (i.e. reducing the number
       of t-subsets that are covered exactly once),
   (c) occasionally performs random block additions to escape local minima,
       followed by deletion passes.
2. The local search uses a coverage-count array and a "critical" set (t-subsets
   covered exactly once) to guide swaps.
3. Multiple diverse greedy seeds are used, and the best local-search result
   wins.

This is a different optimization family: local search / stochastic hill-climbing
with a different representation (coverage counts + critical set) rather than
pure greedy with heap-based gain updates.
"""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets and their indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # For each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # Precompute: for each block, a frozenset of its points for fast membership
    block_sets = [frozenset(blk) for blk in all_blocks]

    def greedy_seed(seed, time_budget):
        """Fast randomized greedy to produce an initial valid covering."""
        local_rng = random.Random(seed)
        local_start = time.time()

        uncovered = [True] * n_ts
        uncovered_count = n_ts
        block_gain = [len(ts_list) for ts_list in block_ts]
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            for s_idx in block_ts[b_idx]:
                if not uncovered[s_idx]:
                    continue
                uncovered[s_idx] = False
                uncovered_count -= 1
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        # If not complete, brute-force fill
        if uncovered_count > 0:
            unc_set = set(i for i in range(n_ts) if uncovered[i])
            while unc_set and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                selected.append(best_b)
                in_block[best_b] = True
                for s in block_ts[best_b]:
                    unc_set.discard(s)

        return [list(all_blocks[b]) for b in selected]

    def is_valid(blocks):
        """Check if blocks form a valid covering."""
        cov = [0] * n_ts
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                cov[ts_index[ts]] += 1
        return all(c > 0 for c in cov)

    def local_search(blocks, seed, time_limit):
        """
        Local search on a valid covering:
        - Repeatedly delete redundant blocks.
        - Try 1-for-1 swaps: replace a block with another that covers all
          unique t-subsets of the original, preferring replacements that
          increase redundancy (decrease critical count).
        - Occasionally add a random block and then delete redundants to
          escape local minima.
        """
        local_rng = random.Random(seed)
        local_start = time.time()

        if not blocks:
            return blocks

        # Build coverage counts
        coverage = [0] * n_ts
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Count of critical t-subsets (covered exactly once)
        critical_count = sum(1 for c in coverage if c == 1)

        def delete_redundants():
            """Delete all redundant blocks. Returns new critical_count."""
            nonlocal critical_count
            changed = True
            while changed and time.time() - local_start < time_limit:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    if time.time() - local_start > time_limit:
                        return
                    blk = blocks[i]
                    redundant = True
                    for ts in itertools.combinations(blk, t):
                        if coverage[ts_index[ts]] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for ts in itertools.combinations(blk, t):
                            old_c = coverage[ts_index[ts]]
                            coverage[ts_index[ts]] = old_c - 1
                            if old_c == 1:
                                critical_count -= 1
                            if old_c - 1 == 1:
                                critical_count += 1
                        blocks.pop(i)
                        changed = True
                        break

        def try_swap():
            """Try to find and apply a beneficial 1-for-1 swap."""
            nonlocal critical_count
            if not blocks:
                return False

            idx = local_rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_set = frozenset(old_block)

            # Find unique t-subsets of old_block
            unique = []
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 1:
                    unique.append(ts_index[ts])

            if not unique:
                # Redundant - just remove
                for ts in itertools.combinations(old_block, t):
                    old_c = coverage[ts_index[ts]]
                    coverage[ts_index[ts]] = old_c - 1
                    if old_c == 1:
                        critical_count -= 1
                    if old_c - 1 == 1:
                        critical_count += 1
                blocks.pop(idx)
                return True

            # Union of points in unique t-subsets
            union_pts = set()
            for s_idx in unique:
                for p in all_ts[s_idx]:
                    union_pts.add(p)

            if len(union_pts) > k:
                return False

            # Find best replacement: covers all unique t-subsets, maximizes
            # number of t-subsets already covered (minimizes new critical)
            best_repl = -1
            best_new_critical = float('inf')
            best_overlap = -1

            for b_idx in range(n_blocks):
                if b_idx == idx:
                    continue
                blk_set = block_sets[b_idx]
                # Must contain all points in union_pts
                if not union_pts.issubset(blk_set):
                    continue

                # Count how many t-subsets of this block are already covered
                # (coverage > 0 before adding) and how many would become critical
                new_critical_delta = 0
                for ts in itertools.combinations(all_blocks[b_idx], t):
                    s_idx = ts_index[ts]
                    # After removing old_block, coverage of s_idx decreases by 1
                    # if old_block covers it, then we add new_block
                    old_covers = 1 if s_idx in [ts_index[ts2] for ts2 in itertools.combinations(old_block, t)] else 0
                    # This is expensive; approximate: just count overlap with current coverage
                    if coverage[s_idx] > 0:
                        pass  # already covered, adding increases coverage
                    else:
                        new_critical_delta += 1

                # Simpler heuristic: count how many t-subsets of new block
                # are currently covered (coverage >= 1)
                overlap = 0
                for ts in itertools.combinations(all_blocks[b_idx], t):
                    if coverage[ts_index[ts]] > 0:
                        overlap += 1

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_repl = b_idx

            if best_repl == -1:
                return False

            # Apply swap
            new_block = list(all_blocks[best_repl])

            # Remove old_block's coverage
            for ts in itertools.combinations(old_block, t):
                s_idx = ts_index[ts]
                coverage[s_idx] -= 1
                if coverage[s_idx] == 0:
                    return False  # would become uncovered, abort
                if coverage[s_idx] == 1:
                    critical_count += 1
                elif coverage[s_idx] == 2:
                    critical_count -= 1

            # Add new_block's coverage
            for ts in itertools.combinations(new_block, t):
                s_idx = ts_index[ts]
                old_c = coverage[s_idx]
                coverage[s_idx] = old_c + 1
                if old_c == 0:
                    critical_count += 1
                elif old_c == 1:
                    critical_count -= 1

            blocks[idx] = new_block
            return True

        # Main local search loop
        delete_redundants()
        attempts = 0
        max_attempts = 2000
        while (time.time() - local_start < time_limit and attempts < max_attempts):
            attempts += 1
            if local_rng.random() < 0.7:
                # Try a swap
                try_swap()
            else:
                # Random addition + deletion to escape local minima
                if len(blocks) < n_blocks:
                    # Add a random block not already present
                    existing = set(frozenset(b) for b in blocks)
                    candidates = [b for b in range(n_blocks) if frozenset(all_blocks[b]) not in existing]
                    if candidates:
                        b_idx = local_rng.choice(candidates)
                        blocks.append(list(all_blocks[b_idx]))
                        for ts in itertools.combinations(all_blocks[b_idx], t):
                            s_idx = ts_index[ts]
                            old_c = coverage[s_idx]
                            coverage[s_idx] = old_c + 1
                            if old_c == 0:
                                critical_count += 1
                            elif old_c == 1:
                                critical_count -= 1
                        # Now delete redundants
                        delete_redundants()

        return blocks

    # Multi-restart: different greedy seeds, then local search
    best_result = None
    best_count = float('inf')

    # Determine restart count
    n_restarts = 6
    greedy_time = 15.0  # seconds per greedy seed
    ls_time = 15.0  # seconds per local search

    for restart in range(n_restarts):
        if _remaining() < (greedy_time + ls_time + 5.0):
            break

        seed = 42 + restart * 104729
        result = greedy_seed(seed, greedy_time)

        if not result:
            continue

        if not is_valid(result):
            # Emergency fill
            cov = [0] * n_ts
            for blk in result:
                for ts in itertools.combinations(blk, t):
                    cov[ts_index[ts]] += 1
            unc = set(i for i in range(n_ts) if cov[i] == 0)
            while unc and _remaining() > 5.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    gain = sum(1 for s in block_ts[b_idx] if s in unc)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                for s in block_ts[best_b]:
                    unc.discard(s)

        if not is_valid(result):
            continue

        # Local search
        ls_budget = min(ls_time, _remaining() - 5.0)
        if ls_budget > 2.0:
            result = local_search(result, seed + 1, ls_budget)

        if len(result) < best_count:
            best_count = len(result)
            best_result = result

    if best_result is None:
        # Fallback: single greedy
        best_result = greedy_seed(12345, _DEADLINE - 5.0)

    # Final verification and emergency fix
    if not is_valid(best_result):
        cov = [0] * n_ts
        for blk in best_result:
            for ts in itertools.combinations(blk, t):
                cov[ts_index[ts]] += 1
        unc = set(i for i in range(n_ts) if cov[i] == 0)
        while unc and _remaining() > 1.0:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = sum(1 for s in block_ts[b_idx] if s in unc)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            best_result.append(list(all_blocks[best_b]))
            for s in block_ts[best_b]:
                unc.discard(s)

    return best_result
# EVOLVE-BLOCK-END
