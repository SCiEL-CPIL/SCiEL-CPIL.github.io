import itertools
import random
import time
import sys
import heapq
from collections import defaultdict

_cache = {}

def _precompute(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]

    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)

    all_k_blocks = list(itertools.combinations(points, k))
    n_k = len(all_k_blocks)

    block_to_subs = []
    sub_to_blocks = [[] for _ in range(n_t)]

    for b_idx, block in enumerate(all_k_blocks):
        subs_in_block = []
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_map[t_sub]
            subs_in_block.append(s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        block_to_subs.append(subs_in_block)

    # Fast lookup: block tuple -> index
    block_to_idx = {block: b_idx for b_idx, block in enumerate(all_k_blocks)}

    _cache[(v, k, t)] = (all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k, block_to_idx)
    return _cache[(v, k, t)]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s margin for safety

    all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k, block_to_idx = _precompute(v, k, t)

    best_blocks = None
    best_count = float('inf')

    def run_greedy(seed):
        """Run greedy with heap-based gain tracking and randomized tie-breaking."""
        random.seed(seed)

        uncovered = bytearray([1] * n_t)
        remaining_uncovered = n_t

        initial_gain = len(block_to_subs[0]) if n_k > 0 else 0
        gain = [initial_gain] * n_k
        version = [0] * n_k
        heap = [(-initial_gain, b, 0) for b in range(n_k)]
        heapq.heapify(heap)

        selected = []

        while remaining_uncovered > 0:
            while heap:
                neg_g, b, ver = heap[0]
                if ver == version[b] and gain[b] == -neg_g:
                    break
                heapq.heappop(heap)

            if not heap:
                break

            neg_g, b_sel, ver = heapq.heappop(heap)
            max_gain = -neg_g

            if max_gain == 0:
                break

            max_blocks = [b_sel]
            temp_heap = []
            while heap:
                ng, b, v2 = heap[0]
                if v2 == version[b] and gain[b] == max_gain:
                    heapq.heappop(heap)
                    max_blocks.append(b)
                    temp_heap.append((ng, b, v2))
                else:
                    break
            for item in temp_heap:
                heapq.heappush(heap, item)

            if len(max_blocks) > 1:
                b_sel = random.choice(max_blocks)

            selected.append(all_k_blocks[b_sel])

            for s in block_to_subs[b_sel]:
                if uncovered[s]:
                    uncovered[s] = 0
                    remaining_uncovered -= 1
                    for b_prime in sub_to_blocks[s]:
                        if gain[b_prime] > 0:
                            gain[b_prime] -= 1
                            version[b_prime] += 1
                            heapq.heappush(heap, (-gain[b_prime], b_prime, version[b_prime]))

        return selected

    def local_search(blocks, seed, max_iters=300):
        """Reduce block count by removing redundant blocks and swapping."""
        random.seed(seed)
        blocks = list(blocks)
        n = len(blocks)

        if n == 0:
            return blocks

        # Build coverage count using fast lookup
        cover_count = [0] * n_t
        for block in blocks:
            b_idx = block_to_idx[block]
            for s in block_to_subs[b_idx]:
                cover_count[s] += 1

        def is_redundant(block):
            b_idx = block_to_idx[block]
            for s in block_to_subs[b_idx]:
                if cover_count[s] <= 1:
                    return False
            return True

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Pass 1: Remove redundant blocks
            new_blocks = []
            for block in blocks:
                if is_redundant(block):
                    b_idx = block_to_idx[block]
                    for s in block_to_subs[b_idx]:
                        cover_count[s] -= 1
                    improved = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
            if not blocks:
                break

            # Pass 2: Swap - replace a block with one that covers its unique t-subsets
            random.shuffle(blocks)
            for i in range(len(blocks)):
                old_block = blocks[i]
                old_b_idx = block_to_idx[old_block]

                only_covered_by_old = []
                for s in block_to_subs[old_b_idx]:
                    if cover_count[s] == 1:
                        only_covered_by_old.append(s)

                if not only_covered_by_old:
                    # Redundant, remove
                    for s in block_to_subs[old_b_idx]:
                        cover_count[s] -= 1
                    blocks.pop(i)
                    improved = True
                    break

                # Find candidates that cover all only_covered_by_old
                candidate_blocks = None
                for s in only_covered_by_old:
                    if candidate_blocks is None:
                        candidate_blocks = set(sub_to_blocks[s])
                    else:
                        candidate_blocks.intersection_update(sub_to_blocks[s])

                if candidate_blocks is None:
                    continue
                candidate_blocks.discard(old_b_idx)

                if not candidate_blocks:
                    continue

                # Try a few random candidates
                candidates = list(candidate_blocks)
                random.shuffle(candidates)
                swapped = False
                for cand_idx in candidates[:10]:
                    cand_block = all_k_blocks[cand_idx]

                    # Check if candidate is already in blocks
                    cand_already_in = cand_block in blocks

                    # Remove old
                    for s in block_to_subs[old_b_idx]:
                        cover_count[s] -= 1

                    # Check validity
                    valid = True
                    for s in only_covered_by_old:
                        if cover_count[s] < 1:
                            valid = False
                            break

                    if valid:
                        # Add candidate if not already present
                        if not cand_already_in:
                            for s in block_to_subs[cand_idx]:
                                cover_count[s] += 1
                            blocks[i] = cand_block
                        else:
                            # Just remove old (candidate already covers)
                            blocks.pop(i)
                        improved = True
                        swapped = True
                        break
                    else:
                        # Revert
                        for s in block_to_subs[old_b_idx]:
                            cover_count[s] += 1

                if swapped:
                    break

        return blocks

    # Phase 1: Run multiple greedy attempts
    attempts = 0
    while time.time() < deadline - 25.0:
        attempts += 1
        blocks = run_greedy(attempts)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        if best_count <= 208:
            break

    # Phase 2: Local search on best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        for ls_seed in range(15):
            if time.time() > deadline - 5.0:
                break
            ls_blocks = local_search(best_blocks, ls_seed, max_iters=300)
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = ls_blocks

    if best_blocks is None:
        return []

    return best_blocks
