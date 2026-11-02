"""
Covering design solver using dynamic greedy with heap-based priority updates.
Strategy: maintain a max-heap of candidate blocks keyed by number of uncovered t-subsets
they would cover. After each block selection, efficiently update only affected candidates.
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

    rng = random.Random(12345)

    # All t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Track coverage: which t-subsets are uncovered
    uncovered = [True] * n_ts
    uncovered_count = n_ts

    # For each point, store which t-subsets it appears in
    # point_to_ts[point] = list of indices of t-subsets containing that point
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_ts):
        for p in ts:
            point_to_ts[p].append(i)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Current uncovered status
    uncovered = [True] * n_ts
    uncovered_count = n_ts

    # For each block, track how many of its t-subsets are still uncovered
    # block_gain[i] = number of uncovered t-subsets in block i
    block_gain = []
    for ts_list in block_ts:
        block_gain.append(len(ts_list))  # initially all uncovered

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # Heap with lazy deletion
    # heap entries: (-gain, block_index, version)
    # We track a version for each block to detect stale entries
    block_version = [0] * n_blocks
    heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
    heapq.heapify(heap)

    selected_blocks = []
    in_block = [False] * n_blocks  # whether block is already selected

    while uncovered_count > 0 and _remaining() > 1.0:
        # Pop from heap until we find a valid entry
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
        selected_blocks.append(list(all_blocks[b_idx]))
        in_block[b_idx] = True

        # Update: for each t-subset in this block, mark as covered and update gains
        for s_idx in block_ts[b_idx]:
            if not uncovered[s_idx]:
                continue
            uncovered[s_idx] = False
            uncovered_count -= 1

            # For each block that contains this t-subset, decrement its gain
            for other_b in ts_to_blocks[s_idx]:
                if in_block[other_b]:
                    continue
                block_gain[other_b] -= 1
                block_version[other_b] += 1
                # Push updated entry
                if block_gain[other_b] > 0:
                    heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

    # Convert to list of lists
    result = [list(blk) for blk in selected_blocks]

    # Local search: try to remove redundant blocks
    if _remaining() > 5.0 and result:
        # Build coverage counts
        coverage = [0] * n_ts
        for blk in result:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Try to remove blocks that are redundant
        changed = True
        while changed and _remaining() > 2.0:
            changed = False
            for i in range(len(result) - 1, -1, -1):
                if _remaining() < 1.0:
                    break
                blk = result[i]
                redundant = True
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        coverage[ts_index[ts]] -= 1
                    result.pop(i)
                    changed = True
                    break

        # 2-to-1 Swap Search (Structural Change)
        # Try to replace two blocks with one block that covers all unique t-subsets
        if _remaining() > 10.0:
            # We need to map result blocks back to their indices in all_blocks for fast lookup
            # Or just use the block tuples.
            # Let's create a set of block tuples in result for O(1) lookup
            result_set = set(result)
            
            # Map block tuple to its index in result? Not needed, we just need to remove them.
            # We will iterate over indices in result.
            
            improved = True
            while improved and _remaining() > 5.0:
                improved = False
                n_res = len(result)
                for i in range(n_res):
                    if _remaining() < 1.0:
                        break
                    for j in range(i + 1, n_res):
                        if _remaining() < 0.5:
                            break
                        
                        blk_i = result[i]
                        blk_j = result[j]
                        
                        # Find t-subsets covered ONLY by blk_i or blk_j
                        # i.e., coverage[ts] == 1 and (ts in blk_i or ts in blk_j)
                        # Actually, if coverage[ts] == 1, it is covered by exactly one block.
                        # If that block is blk_i or blk_j, then removing both will uncover it.
                        # So U = { ts | coverage[ts] == 1 and (ts in blk_i or ts in blk_j) }
                        
                        # To check "ts in blk_i", we can precompute the set of t-subsets for each block in result?
                        # That's expensive.
                        # Alternative: 
                        # 1. Compute the set of t-subsets for blk_i and blk_j.
                        # 2. Iterate through them.
                        
                        ts_i_set = set(ts_index[ts] for ts in itertools.combinations(blk_i, t))
                        ts_j_set = set(ts_index[ts] for ts in itertools.combinations(blk_j, t))
                        
                        # Unique subsets to be covered by the new block
                        U = []
                        for s_idx in ts_i_set:
                            if coverage[s_idx] == 1:
                                U.append(s_idx)
                        for s_idx in ts_j_set:
                            if coverage[s_idx] == 1:
                                U.append(s_idx)
                                
                        if not U:
                            # If no unique subsets, we can just delete one of them? 
                            # The deletion loop above should have caught this.
                            # If U is empty, it means all subsets covered by i or j are covered by others.
                            # So we can delete i and j? No, we can delete i, then j might become redundant?
                            # If U is empty, it means removing i and j leaves no gaps.
                            # So we can remove both? That would be a 2-to-0 swap.
                            # Let's handle 2-to-0 first.
                            if _remaining() > 1.0:
                                # Remove i and j
                                # But we must be careful with indices.
                                # Since we are in a nested loop, let's just mark for deletion.
                                # Actually, if U is empty, we can just remove i. Then re-run deletion?
                                # Or remove both.
                                # Let's try to remove both.
                                # Decrement coverage for all ts in i and j.
                                # But wait, if coverage[ts] == 1 for ts in i, and we remove i, it becomes 0.
                                # If U is empty, it means there are NO ts in i or j with coverage 1.
                                # So all ts in i and j have coverage >= 2.
                                # So removing both is safe.
                                for s_idx in ts_i_set:
                                    coverage[s_idx] -= 1
                                for s_idx in ts_j_set:
                                    coverage[s_idx] -= 1
                                result.pop(j)
                                result.pop(i)
                                improved = True
                                break
                                
                        # If U is not empty, we need a block that covers all of U.
                        # Candidate blocks must contain all s in U.
                        # Intersection of ts_to_blocks[s] for s in U.
                        
                        # Optimization: If |U| is large, intersection is likely empty.
                        # If |U| is small, it's fast.
                        
                        # Pick the s in U with the smallest ts_to_blocks list to start intersection
                        if U:
                            # Sort U by size of ts_to_blocks[s]
                            U_sorted = sorted(U, key=lambda x: len(ts_to_blocks[x]))
                            # Start with the smallest list
                            candidates = set(ts_to_blocks[U_sorted[0]])
                            for s_idx in U_sorted[1:]:
                                if not candidates:
                                    break
                                candidates.intersection_update(ts_to_blocks[s_idx])
                                
                            if candidates:
                                # Find a candidate not in result
                                valid_candidate = None
                                for c_idx in candidates:
                                    if not in_block[c_idx]: # in_block is stale, use result_set
                                        if all_blocks[c_idx] not in result_set:
                                            valid_candidate = c_idx
                                            break
                                    
                                if valid_candidate is not None:
                                    # Perform swap: remove i, j; add valid_candidate
                                    # Decrement coverage for i and j
                                    for s_idx in ts_i_set:
                                        coverage[s_idx] -= 1
                                    for s_idx in ts_j_set:
                                        coverage[s_idx] -= 1
                                    
                                    # Increment coverage for valid_candidate
                                    for s_idx in block_ts[valid_candidate]:
                                        coverage[s_idx] += 1
                                        
                                    # Update result
                                    result.pop(j)
                                    result.pop(i)
                                    new_blk = list(all_blocks[valid_candidate])
                                    result.append(new_blk)
                                    result_set.discard(tuple(blk_i))
                                    result_set.discard(tuple(blk_j))
                                    result_set.add(tuple(new_blk))
                                    
                                    # Update in_block for consistency (though we use result_set for swap logic)
                                    in_block[i] = False
                                    in_block[j] = False
                                    in_block[valid_candidate] = True
                                    
                                    improved = True
                                    break
                            
                            if improved:
                                break

    # Final verification
    final_cov = [0] * n_ts
    for blk in result:
        for ts in itertools.combinations(blk, t):
            final_cov[ts_index[ts]] += 1
    if not all(c > 0 for c in final_cov):
        # Fallback: add greedy blocks for uncovered subsets
        unc_set = set(i for i in range(n_ts) if final_cov[i] == 0)
        while unc_set and _remaining() > 0.5:
            # Find block covering most uncovered
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = 0
                for s_idx in block_ts[b_idx]:
                    if s_idx in unc_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            blk = list(all_blocks[best_b])
            result.append(blk)
            for s_idx in block_ts[best_b]:
                unc_set.discard(s_idx)

    return result
# EVOLVE-BLOCK-END
