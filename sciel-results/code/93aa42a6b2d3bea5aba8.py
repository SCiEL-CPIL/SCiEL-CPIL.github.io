import itertools
import random
import time
import heapq
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit

    def time_ok():
        return time.time() < deadline

    # Fallback for non-target parameters
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time_ok():
            best_block = None
            best_count = -1
            for block_tuple in itertools.combinations(range(v), k):
                block_set = frozenset(block_tuple)
                count = sum(1 for tsub in uncovered if frozenset(tsub) <= block_set)
                if count > best_count:
                    best_count = count
                    best_block = list(block_tuple)
                if time_ok() is False:
                    break
            if best_block is None or best_count == 0:
                break
            blocks.append(best_block)
            block_set = frozenset(best_block)
            uncovered = {tsub for tsub in uncovered if not frozenset(tsub) <= block_set}
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    n_t = len(all_tsubs)

    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    # Precompute block -> t-subset indices and t-subset -> block indices
    block_tsubs = [None] * n_blocks
    tsub_blocks = [[] for _ in range(n_t)]

    for i, block in enumerate(all_blocks):
        idxs = [tsub_to_id[sub] for sub in itertools.combinations(block, t)]
        block_tsubs[i] = idxs
        for idx in idxs:
            tsub_blocks[idx].append(i)

    # Convert to numpy arrays for faster access
    # Each block covers C(7,4)=35 t-subsets
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)
    # tsub_blocks as list of numpy arrays for faster iteration
    tsub_blocks_np = [np.array(tb, dtype=np.int32) for tb in tsub_blocks]

    best_blocks = None
    best_count = float('inf')

    n_restarts = 8

    for restart in range(n_restarts):
        if not time_ok():
            break

        rng = random.Random(42 + restart * 7919)

        # Heap-based lazy deletion greedy with randomized tie-breaking
        gains = np.array([len(block_tsubs[i]) for i in range(n_blocks)], dtype=np.int32)
        covered = np.zeros(n_t, dtype=np.bool_)
        covered_count = 0
        selected_blocks = []

        # Heap: (-gain, random_tiebreak, block_id)
        # random_tiebreak ensures diversity across restarts
        heap = [(-int(gains[i]), rng.random(), i) for i in range(n_blocks)]
        heapq.heapify(heap)

        # Track which blocks are in the heap (for lazy deletion validity)
        # We'll use the gain value to check validity

        while covered_count < n_t and time_ok():
            # Pop the best block
            neg_gain, tiebreak, b_idx = heapq.heappop(heap)
            current_gain = -neg_gain

            # Check if this gain is still valid (lazy deletion)
            if current_gain != gains[b_idx]:
                continue

            if current_gain == 0:
                # No more gains to be had
                break

            # Select this block
            selected_blocks.append(b_idx)

            # Update coverage
            tids = block_tsubs[b_idx]
            for idx in tids:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
                    # Update gains of all blocks covering this t-subset
                    for other_b_idx in tsub_blocks[idx]:
                        if gains[other_b_idx] > 0:
                            gains[other_b_idx] -= 1
                            # Push updated gain to heap
                            heapq.heappush(heap, (-int(gains[other_b_idx]), rng.random(), other_b_idx))

        if covered_count == n_t:
            # Deletion phase: remove redundant blocks
            if time_ok():
                # Count how many selected blocks cover each t-subset
                tsub_block_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in selected_blocks:
                    tids = block_tsubs[b_idx]
                    tsub_block_count[tids] += 1

                changed = True
                while changed and time_ok():
                    changed = False
                    for i in range(len(selected_blocks)):
                        if not time_ok():
                            break
                        b_idx = selected_blocks[i]
                        tids = block_tsubs[b_idx]
                        # Check if all t-subsets in this block are covered by >= 2 blocks
                        if np.all(tsub_block_count[tids] > 1):
                            # Remove this block
                            tsub_block_count[tids] -= 1
                            selected_blocks.pop(i)
                            changed = True
                            break

            # Local search: 1-for-1 swaps
            if time_ok():
                selected_set = set(selected_blocks)
                tsub_block_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in selected_blocks:
                    tids = block_tsubs[b_idx]
                    tsub_block_count[tids] += 1

                improved = True
                while improved and time_ok():
                    improved = False
                    for i in range(len(selected_blocks)):
                        if not time_ok():
                            break
                        b_idx = selected_blocks[i]
                        tids = block_tsubs[b_idx]
                        # Check if removing b_idx would uncover any t-subset
                        if np.any(tsub_block_count[tids] <= 1):
                            continue
                        # Try to replace with a better block
                        # Find a block not in selected_set that covers at least one
                        # t-subset currently covered only by b_idx, and has higher gain
                        # Actually, for 1-for-1, we just need to ensure coverage
                        # Let's try a simpler approach: just do deletion, which we already did
                        # For swaps, we need to find a block that covers the "critical" t-subsets
                        # This is complex; let's rely on MILP for the heavy lifting
                        pass

            current_count = len(selected_blocks)
            if current_count < best_count:
                best_count = current_count
                best_blocks = [all_blocks[i] for i in selected_blocks]

            if best_count <= 145:
                break

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time_ok():
            best_block = None
            best_count = -1
            for block_tuple in itertools.combinations(range(v), k):
                block_set = frozenset(block_tuple)
                count = sum(1 for tsub in uncovered if frozenset(tsub) <= block_set)
                if count > best_count:
                    best_count = count
                    best_block = list(block_tuple)
            if best_block is None or best_count == 0:
                break
            blocks.append(best_block)
            block_set = frozenset(best_block)
            uncovered = {tsub for tsub in uncovered if not frozenset(tsub) <= block_set}
        return blocks

    # MILP polish
    if _HAS_MILP and time_ok():
        best_blocks = _milp_polish(v, k, t, best_blocks, tsub_to_id, all_tsubs, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks)

    return best_blocks


def _milp_polish(v, k, t, blocks, tsub_to_idx, all_t_subsets, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks):
    n_t = len(all_t_subsets)
    n_blocks_in = len(blocks)
    if n_blocks_in == 0:
        return blocks

    # Build the constraint matrix A (n_t x n_blocks_in)
    A = np.zeros((n_t, n_blocks_in), dtype=np.int8)
    for j in range(n_blocks_in):
        b = blocks[j]
        for sub in itertools.combinations(b, t):
            idx = tsub_to_idx[sub]
            A[idx, j] = 1

    c = np.ones(n_blocks_in)
    constraint = LinearConstraint(A, 1, None)
    bounds = Bounds(0, 1)

    # Compute remaining time
    elapsed = time.time() - (deadline - 110)  # deadline was start+110
    remaining = max(10, 110 - elapsed)

    try:
        res = milp(c, integrality=np.ones(n_blocks_in), constraints=constraint,
                   bounds=bounds, options={'time_limit': int(remaining), 'mip_rel_gap': 0.0,
                                           'mip_absgap': 0.0, 'verbose': 0})
    except Exception:
        return blocks

    if res.x is not None:
        sel = [j for j in range(n_blocks_in) if res.x[j] > 0.5]
        new_blocks = [blocks[i] for i in sel]

        # Verify coverage
        if len(new_blocks) < len(blocks):
            cov = np.zeros(n_t, dtype=np.bool_)
            for b in new_blocks:
                for sub in itertools.combinations(b, t):
                    idx = tsub_to_idx[sub]
                    cov[idx] = True
            if np.all(cov):
                return new_blocks

    return blocks
