"""
Covering design solver: dynamic greedy with heap-based priority updates,
combined with multi-restart randomization and aggressive local search.
Strategy:
  1. Precompute all C(v,k) blocks and their t-subset memberships.
  2. Dynamic greedy: max-heap of blocks keyed by current uncovered-gain.
  3. Multiple randomized restarts (different tie-breaking / orderings).
  4. Deletion phase: remove redundant blocks.
  5. Local search: try to replace a block with a smaller-covering block,
     or remove redundant blocks iteratively.
  6. 2-exchange: remove two blocks, add one that covers all newly-uncovered t-subsets.
"""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 142.0
_start_time = None

# Precomputed global structures (lazy init)
_G = {}


def _init_globals(v, k, t):
    """Precompute all structures needed for the solver."""
    if (v, k, t) in _G:
        return _G[(v, k, t)]

    import itertools
    from collections import defaultdict

    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # For each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # Precompute bitmask for each block: which t-subsets it covers
    # n_ts for C(19,4) = 3876, fits in a Python int
    block_mask = []
    for ts_list in block_ts:
        mask = 0
        for s in ts_list:
            mask |= (1 << s)
        block_mask.append(mask)

    # Precompute bitmask for each block: which points it contains
    block_pt_mask = []
    for blk in all_blocks:
        pm = 0
        for p in blk:
            pm |= (1 << p)
        block_pt_mask.append(pm)

    # For each t-subset, bitmask of points in it
    ts_pt_mask = []
    for ts in all_ts:
        pm = 0
        for p in ts:
            pm |= (1 << p)
        ts_pt_mask.append(pm)

    # Number of t-subsets per block
    n_ts_per_block = len(block_ts[0])  # C(k,t)

    _G[(v, k, t)] = {
        'all_ts': all_ts,
        'n_ts': n_ts,
        'ts_index': ts_index,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'block_ts': block_ts,
        'ts_to_blocks': ts_to_blocks,
        'block_mask': block_mask,
        'block_pt_mask': block_pt_mask,
        'ts_pt_mask': ts_pt_mask,
        'n_ts_per_block': n_ts_per_block,
    }
    return _G[(v, k, t)]


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    G = _init_globals(v, k, t)
    all_ts = G['all_ts']
    n_ts = G['n_ts']
    ts_index = G['ts_index']
    all_blocks = G['all_blocks']
    n_blocks = G['n_blocks']
    block_ts = G['block_ts']
    ts_to_blocks = G['ts_to_blocks']
    block_mask = G['block_mask']
    block_pt_mask = G['block_pt_mask']
    ts_pt_mask = G['ts_pt_mask']
    n_ts_per_block = G['n_ts_per_block']

    rng = random.Random(12345)

    def greedy_solve(seed, max_time_frac=0.5):
        """Run one greedy pass with given random seed for tie-breaking."""
        local_rng = random.Random(seed)
        time_budget = _DEADLINE * max_time_frac
        local_start = time.time()

        uncovered_mask = (1 << n_ts) - 1  # all t-subsets uncovered
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [n_ts_per_block] * n_blocks
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
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

            # Update coverage using bitmask
            bm = block_mask[b_idx]
            new_uncovered = bm & uncovered_mask
            # Count newly covered
            newly_covered = bin(new_uncovered).count('1')
            uncovered_count -= newly_covered
            uncovered_mask &= ~bm

            # Update gains of other blocks containing newly covered t-subsets
            # Iterate over set bits in new_uncovered
            nu = new_uncovered
            while nu:
                s_idx = (nu & (-nu)).bit_length() - 1
                nu &= nu - 1
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        # Convert to list of blocks
        result = [list(all_blocks[b]) for b in selected]

        # Check if complete
        if uncovered_count > 0:
            # Add remaining blocks greedily (brute force over remaining)
            while uncovered_count > 0 and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = bin(block_mask[b_idx] & uncovered_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                uncovered_mask &= ~block_mask[best_b]
                uncovered_count -= best_gain

        return result

    def local_search_improve(blocks, seed, time_limit):
        """Improve a covering by removing redundant blocks and trying swaps."""
        local_rng = random.Random(seed)
        local_start = time.time()

        if not blocks:
            return blocks

        # Build coverage counts using bitmask representation
        coverage = [0] * n_ts
        for blk in blocks:
            blk_set = set(blk)
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Phase 1: Remove redundant blocks (fast pass)
        changed = True
        while changed and time.time() - local_start < time_limit:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() - local_start > time_limit:
                    break
                blk = blocks[i]
                redundant = True
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        coverage[ts_index[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break

        # Phase 2: Randomized swap search - try to replace a block with another
        # that maintains coverage but potentially allows further deletions.
        attempts = 0
        max_attempts = 3000
        while (time.time() - local_start < time_limit and attempts < max_attempts
               and len(blocks) > 0):
            attempts += 1
            idx = local_rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_blk_set = set(old_block)

            # Find t-subsets with coverage exactly 1 (only covered by this block)
            unique_ts = []
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 1:
                    unique_ts.append(ts_index[ts])

            if not unique_ts:
                # Fully redundant - remove
                for ts in itertools.combinations(old_block, t):
                    coverage[ts_index[ts]] -= 1
                blocks.pop(idx)
                continue

            # Must find a replacement covering all unique_ts
            # Quick check: union of points in unique_ts must fit in k
            union_pts_mask = 0
            for s_idx in unique_ts:
                union_pts_mask |= ts_pt_mask[s_idx]
            if bin(union_pts_mask).count('1') > k:
                continue

            # Find best replacement: must contain all points in union_pts_mask
            best_repl = -1
            best_overlap = -1

            # Precompute: for the old block, which t-subsets have coverage > 1
            old_covered_by_others = set()
            for ts in itertools.combinations(old_block, t):
                si = ts_index[ts]
                if coverage[si] > 1:
                    old_covered_by_others.add(si)

            for b_idx in range(n_blocks):
                # Must contain all points in union_pts_mask
                if (block_pt_mask[b_idx] & union_pts_mask) != union_pts_mask:
                    continue

                # Count how many t-subsets of this block are already covered by others
                overlap = 0
                for s_idx in block_ts[b_idx]:
                    if s_idx in old_covered_by_others:
                        overlap += 1
                    elif coverage[s_idx] > 0:
                        overlap += 1

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_repl = b_idx

            if best_repl == -1:
                continue

            # Check: does the replacement actually help?
            min_needed = n_ts_per_block - len(unique_ts)
            if best_overlap < min_needed:
                continue

            new_block = all_blocks[best_repl]

            # Apply swap
            for ts in itertools.combinations(old_block, t):
                coverage[ts_index[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                coverage[ts_index[ts]] += 1

            # Verify validity
            valid = True
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 0:
                    valid = False
                    break

            if valid:
                blocks[idx] = list(new_block)
                # After swap, try to remove other now-redundant blocks
                for j in range(len(blocks) - 1, -1, -1):
                    if j == idx:
                        continue
                    blk_j = blocks[j]
                    removable = True
                    for ts in itertools.combinations(blk_j, t):
                        if coverage[ts_index[ts]] <= 1:
                            removable = False
                            break
                    if removable:
                        for ts in itertools.combinations(blk_j, t):
                            coverage[ts_index[ts]] -= 1
                        blocks.pop(j)
                        break
            else:
                # Revert
                for ts in itertools.combinations(new_block, t):
                    coverage[ts_index[ts]] -= 1
                for ts in itertools.combinations(old_block, t):
                    coverage[ts_index[ts]] += 1

        return blocks

    def local_search_2exchange(blocks, seed, time_limit):
        """Try 2-exchange: remove two blocks, add one block that covers all uniquely-covered t-subsets."""
        local_rng = random.Random(seed)
        local_start = time.time()

        if len(blocks) < 2:
            return blocks

        # Build coverage counts
        coverage = [0] * n_ts
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Track which block index each t-subset is covered by (for quick uniqueness check)
        # coverage[i] > 1 means covered by multiple blocks
        
        improved = True
        while improved and time.time() - local_start < time_limit:
            improved = False
            n = len(blocks)
            if n < 2:
                break
            
            # Randomize pair order
            pairs = list(itertools.combinations(range(n), 2))
            local_rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() - local_start > time_limit:
                    return blocks
                
                idx1, idx2 = i, j
                if idx1 == idx2:
                    continue
                blk1 = blocks[idx1]
                blk2 = blocks[idx2]
                
                # Temporarily remove both
                for ts in itertools.combinations(blk1, t):
                    coverage[ts_index[ts]] -= 1
                for ts in itertools.combinations(blk2, t):
                    coverage[ts_index[ts]] -= 1
                
                # Find t-subsets that are now uncovered
                uncovered_ts = []
                for ts in itertools.combinations(blk1, t):
                    if coverage[ts_index[ts]] == 0:
                        uncovered_ts.append(ts_index[ts])
                for ts in itertools.combinations(blk2, t):
                    si = ts_index[ts]
                    if coverage[si] == 0 and si not in uncovered_ts:
                        uncovered_ts.append(si)
                
                if len(uncovered_ts) > 0:
                    # Look for a single block that covers all uncovered_ts
                    # Quick check: union of points must fit in k
                    union_pts_mask = 0
                    for s_idx in uncovered_ts:
                        union_pts_mask |= ts_pt_mask[s_idx]
                    if bin(union_pts_mask).count('1') > k:
                        # Revert
                        for ts in itertools.combinations(blk1, t):
                            coverage[ts_index[ts]] += 1
                        for ts in itertools.combinations(blk2, t):
                            coverage[ts_index[ts]] += 1
                        continue
                    
                    # Find candidate blocks covering all uncovered_ts
                    # Start from candidates of first uncovered t-subset
                    found = False
                    for cand in ts_to_blocks[uncovered_ts[0]]:
                        # Check if cand covers all uncovered_ts
                        cand_mask = block_mask[cand]
                        all_covered = True
                        for s_idx in uncovered_ts:
                            if not (cand_mask & (1 << s_idx)):
                                all_covered = False
                                break
                        if all_covered:
                            # Check that cand is not already in blocks (excluding the two we're removing)
                            cand_set = set(all_blocks[cand])
                            in_blocks = False
                            for b_idx, b in enumerate(blocks):
                                if b_idx == idx1 or b_idx == idx2:
                                    continue
                                if set(b) == cand_set:
                                    in_blocks = True
                                    break
                            if in_blocks:
                                continue
                            
                            # Valid 2->1 exchange: add cand
                            for ts in itertools.combinations(all_blocks[cand], t):
                                coverage[ts_index[ts]] += 1
                            
                            # Remove the two blocks
                            # Remove higher index first
                            if idx2 > idx1:
                                blocks.pop(idx2)
                                blocks.pop(idx1)
                            else:
                                blocks.pop(idx1)
                                blocks.pop(idx2)
                            
                            blocks.append(list(all_blocks[cand]))
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Revert
                        for ts in itertools.combinations(blk1, t):
                            coverage[ts_index[ts]] += 1
                        for ts in itertools.combinations(blk2, t):
                            coverage[ts_index[ts]] += 1
                else:
                    # Both blocks were redundant (no uncovered t-subsets after removal)
                    # Remove one (the one at higher index)
                    if idx2 > idx1:
                        blocks.pop(idx2)
                    else:
                        blocks.pop(idx1)
                    improved = True
                    break
        
        return blocks

    # Multi-restart greedy
    best_result = None
    best_count = float('inf')

    # Determine number of restarts based on time
    n_restarts = 6
    # Allocate time: 45% to greedy, 25% to 1-exchange LS, 15% to 2-exchange LS, 15% to MILP polish
    total_budget = _DEADLINE - 10.0  # safety margin
    greedy_frac = 0.45
    ls1_frac = 0.25
    ls2_frac = 0.15
    milp_frac = 0.15
    time_per_greedy = total_budget * greedy_frac / n_restarts
    time_per_ls1 = total_budget * ls1_frac / n_restarts
    time_per_ls2 = total_budget * ls2_frac / n_restarts
    time_per_milp = total_budget * milp_frac

    for restart in range(n_restarts):
        if _remaining() < 15.0:
            break

        seed = 12345 + restart * 7919
        max_frac = min(0.50, time_per_greedy / _DEADLINE)

        result = greedy_solve(seed, max_time_frac=max_frac)

        if not result:
            continue

        # Check validity using bitmask
        cov_mask = 0
        for blk in result:
            blk_set = set(blk)
            for ts in itertools.combinations(blk, t):
                cov_mask |= (1 << ts_index[ts])
        if cov_mask != (1 << n_ts) - 1:
            # Invalid - skip
            continue

        # Local search 1-exchange
        ls1_time = min(time_per_ls1, _remaining() - 20.0)
        if ls1_time > 2.0:
            result = local_search_improve(result, seed + 1, ls1_time)

        # Local search 2-exchange
        ls2_time = min(time_per_ls2, _remaining() - 15.0)
        if ls2_time > 1.5:
            result = local_search_2exchange(result, seed + 2, ls2_time)

        if len(result) < best_count:
            best_count = len(result)
            best_result = result

    if best_result is None:
        # Fallback: single greedy pass
        best_result = greedy_solve(12345, max_time_frac=0.9)

    # Final verification
    final_cov = [0] * n_ts
    for blk in best_result:
        for ts in itertools.combinations(blk, t):
            final_cov[ts_index[ts]] += 1
    if not all(c > 0 for c in final_cov):
        # Emergency fix: add blocks for uncovered subsets
        unc_mask = 0
        for i in range(n_ts):
            if final_cov[i] == 0:
                unc_mask |= (1 << i)
        while unc_mask and _remaining() > 0.5:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = bin(block_mask[b_idx] & unc_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            best_result.append(list(all_blocks[best_b]))
            unc_mask &= ~block_mask[best_b]

    # MILP polish: try to remove redundant blocks using a more systematic approach
    if _remaining() > time_per_milp + 5.0:
        milp_time = min(time_per_milp, _remaining() - 5.0)
        if milp_time > 3.0:
            best_result = _milp_polish(best_result, v, k, t, G, milp_time)

    return best_result


def _milp_polish(blocks, v, k, t, G, time_limit):
    """Try to reduce the number of blocks using a MILP-based approach.
    
    Formulation: minimize sum(x_i) subject to coverage constraints.
    We use a greedy deletion heuristic informed by coverage counts.
    """
    import time
    
    all_ts = G['all_ts']
    n_ts = G['n_ts']
    ts_index = G['ts_index']
    all_blocks = G['all_blocks']
    n_blocks = G['n_blocks']
    block_ts = G['block_ts']
    ts_to_blocks = G['ts_to_blocks']
    block_mask = G['block_mask']
    n_ts_per_block = G['n_ts_per_block']
    
    local_start = time.time()
    
    if not blocks:
        return blocks
    
    # Build coverage counts
    coverage = [0] * n_ts
    for blk in blocks:
        for ts in itertools.combinations(blk, t):
            coverage[ts_index[ts]] += 1
    
    # Iteratively try to remove blocks
    # Priority: blocks with most redundancy (highest min coverage among their t-subsets)
    improved = True
    while improved and time.time() - local_start < time_limit:
        improved = False
        n = len(blocks)
        if n == 0:
            break
        
        # For each block, compute its "removability score"
        # A block is removable if all its t-subsets have coverage > 1
        # Score = how much redundancy it has (higher = more redundant)
        best_idx = -1
        best_score = -1
        
        for i in range(n):
            blk = blocks[i]
            min_cov = float('inf')
            total_cov = 0
            for ts in itertools.combinations(blk, t):
                si = ts_index[ts]
                if coverage[si] <= 1:
                    min_cov = 0
                    break
                min_cov = min(min_cov, coverage[si])
                total_cov += coverage[si]
            
            if min_cov > 1:
                # This block is removable
                # Score: prefer removing blocks that free up the most coverage
                score = total_cov
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        if best_idx >= 0:
            # Remove the block
            blk = blocks[best_idx]
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] -= 1
            blocks.pop(best_idx)
            improved = True
    
    return blocks
# EVOLVE-BLOCK-END
