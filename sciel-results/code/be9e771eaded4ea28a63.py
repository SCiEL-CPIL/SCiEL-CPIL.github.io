"""
Covering design C(19,7,4) solver.

Strategy:
  1. Fast greedy construction (seed-like) to get an initial covering.
  2. Local search with block removal and swap operations to reduce count.
  3. Simulated-annealing-like acceptance for swaps to escape local minima.
"""
import itertools
import random
import time
import math

def construct(v, k, t):
    deadline = time.time() + 125.0  # safety margin: 125s of 150s wall-clock
    rng = random.Random(42)

    # ---------- helpers ----------
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map t-subset to index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For a block (frozen tuple of k points), precompute which t-subsets it covers
    def block_tsubs(block):
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]

    # ---------- Phase 1: Greedy construction ----------
    # uncovered[i] = number of blocks still covering t-subset i (start at 0)
    uncovered = [0] * n_tsubs
    blocks = []
    block_tsub_lists = []  # for each block, list of t-subset indices it covers

    while any(u == 0 for u in uncovered):
        if time.time() > deadline - 60.0:
            break  # leave time for local search

        # Pick the t-subset with the most "demand" (all uncovered, so pick first uncovered)
        # Actually, pick a random uncovered one for variety, or the one with most co-coverage potential
        uncovered_indices = [i for i in range(n_tsubs) if uncovered[i] == 0]
        if not uncovered_indices:
            break
        seed_tsub = all_tsubs[rng.choice(uncovered_indices)]
        block = list(seed_tsub)

        while len(block) < k:
            # Score each candidate point by how many additional uncovered t-subsets it would cover
            best_x = -1
            best_score = -1
            block_set = set(block)
            for x in all_points:
                if x in block_set:
                    continue
                new_block = block + [x]
                # Count how many t-subsets of new_block are currently uncovered
                cnt = 0
                for ts in itertools.combinations(new_block, t):
                    idx = tsub_to_idx[ts]
                    if uncovered[idx] == 0:
                        cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_x = x
            block.append(best_x)

        block.sort()
        bt = frozenset(block)
        blocks.append(bt)
        bts = block_tsubs(bt)
        block_tsub_lists.append(bts)
        for idx in bts:
            uncovered[idx] += 1

    # ---------- Phase 2: Local search ----------
    best_blocks = list(blocks)
    best_block_tsub_lists = [list(x) for x in block_tsub_lists]
    best_count = len(blocks)

    # Maintain coverage count array
    def build_counts(bts_lists):
        counts = [0] * n_tsubs
        for bts in bts_lists:
            for idx in bts:
                counts[idx] += 1
        return counts

    current_counts = build_counts(block_tsub_lists)
    cur_blocks = [set(b) for b in blocks]
    cur_bts_lists = [list(x) for x in block_tsub_lists]

    def is_valid(counts):
        return all(c >= 1 for c in counts)

    # Remove redundant blocks
    removed_any = True
    while removed_any:
        removed_any = False
        for i in range(len(cur_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if block i is redundant: all its t-subsets are covered by other blocks
            redundant = True
            for idx in cur_bts_lists[i]:
                if current_counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in cur_bts_lists[i]:
                    current_counts[idx] -= 1
                cur_blocks.pop(i)
                cur_bts_lists.pop(i)
                best_count -= 1
                best_blocks = [frozenset(b) for b in cur_blocks]
                best_block_tsub_lists = [list(x) for x in cur_bts_lists]
                removed_any = True
                break

    # Swap-based local search
    temperature = 1.0
    cooling = 0.9995
    min_temp = 0.01

    def try_swap(i, counts, cur_blocks, cur_bts_lists):
        """Try to find a swap that improves or (with SA) accepts. Return (new_block_set, new_bts, delta) or None."""
        if time.time() > deadline:
            return None
        # T-subsets that would become uncovered if we remove block i
        critical = [idx for idx in cur_bts_lists[i] if counts[idx] == 1]
        if not critical:
            return None  # should have been removed
        
        # Find candidate blocks that cover all critical t-subsets
        # A candidate block B must contain all points of each critical t-subset
        # That means B must be a superset of the union of critical t-subsets
        crit_union = set()
        for idx in critical:
            crit_union.update(all_tsubs[idx])
        
        if len(crit_union) > k:
            return None
        
        # Build candidate: start with crit_union, add random points to reach size k
        base = list(crit_union)
        remaining = [p for p in all_points if p not in crit_union]
        rng.shuffle(remaining)
        for _ in range(k - len(base)):
            base.append(remaining.pop())
        base.sort()
        cand = frozenset(base)
        
        if cand in cur_blocks:
            return None
        
        cand_bts = block_tsubs(cand)
        
        # Compute delta: change in coverage counts
        # Remove block i: decrement its t-subsets
        # Add cand: increment its t-subsets
        # Net effect on validity: for each t-subset, count changes by
        #   -1 if in block i only, +1 if in cand only, 0 if in both
        delta_valid = 0  # number of t-subsets that go from 0 to 1 or 1 to 0
        # We only care about whether the result is valid
        # Simulate:
        ok = True
        for idx in cur_bts_lists[i]:
            if counts[idx] == 1:
                # will be decremented to 0
                # check if cand covers it
                if idx not in cand_bts:
                    ok = False
                    break
        if not ok:
            return None
        
        # This swap keeps validity (critical ones are covered by cand)
        # Count improvement: -1 block
        return (cand, cand_bts, -1)

    # Precompute block membership for faster swap search
    # For each t-subset index, which blocks cover it
    tsub_block_map = [[] for _ in range(n_tsubs)]
    for bi, bts in enumerate(cur_bts_lists):
        for idx in bts:
            tsub_block_map[idx].append(bi)

    while time.time() < deadline and temperature > min_temp:
        # Attempt a random swap
        if not cur_blocks:
            break
        i = rng.randrange(len(cur_blocks))
        
        # Find critical t-subsets for block i
        critical = [idx for idx in cur_bts_lists[i] if current_counts[idx] == 1]
        if not critical:
            # Block is redundant, remove it
            for idx in cur_bts_lists[i]:
                current_counts[idx] -= 1
            cur_blocks.pop(i)
            cur_bts_lists.pop(i)
            best_count -= 1
            best_blocks = [frozenset(b) for b in cur_blocks]
            best_block_tsub_lists = [list(x) for x in cur_bts_lists]
            continue
        
        # Build candidate from critical union
        crit_union = set()
        for idx in critical:
            crit_union.update(all_tsubs[idx])
        
        if len(crit_union) > k:
            temperature *= cooling
            continue
        
        base = list(crit_union)
        remaining = [p for p in all_points if p not in crit_union]
        rng.shuffle(remaining)
        for _ in range(k - len(base)):
            base.append(remaining.pop())
        base.sort()
        cand = frozenset(base)
        
        if cand in cur_blocks:
            temperature *= cooling
            continue
        
        cand_bts = block_tsubs(cand)
        
        # Verify validity after swap
        # Decrement block i's t-subsets, increment cand's
        valid = True
        for idx in cur_bts_lists[i]:
            if current_counts[idx] == 1 and idx not in cand_bts:
                valid = False
                break
        if not valid:
            temperature *= cooling
            continue
        
        # Apply swap
        for idx in cur_bts_lists[i]:
            current_counts[idx] -= 1
        for idx in cand_bts:
            current_counts[idx] += 1
        
        cur_blocks[i] = cand
        cur_bts_lists[i] = cand_bts
        
        # Update best
        best_blocks = [frozenset(b) for b in cur_blocks]
        best_block_tsub_lists = [list(x) for x in cur_bts_lists]
        best_count = len(cur_blocks)
        
        # Try to remove any newly redundant blocks
        for j in range(len(cur_blocks) - 1, -1, -1):
            redundant = all(current_counts[idx] > 1 for idx in cur_bts_lists[j])
            if redundant:
                for idx in cur_bts_lists[j]:
                    current_counts[idx] -= 1
                cur_blocks.pop(j)
                cur_bts_lists.pop(j)
                best_count -= 1
                best_blocks = [frozenset(b) for b in cur_blocks]
                best_block_tsub_lists = [list(x) for x in cur_bts_lists]
        
        temperature *= cooling
    
    # Final cleanup: remove redundant blocks
    for i in range(len(cur_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break
        redundant = all(current_counts[idx] > 1 for idx in cur_bts_lists[i])
        if redundant:
            for idx in cur_bts_lists[i]:
                current_counts[idx] -= 1
            cur_blocks.pop(i)
            cur_bts_lists.pop(i)
            best_count -= 1
            best_blocks = [frozenset(b) for b in cur_blocks]
            best_block_tsub_lists = [list(x) for x in cur_bts_lists]

    # Final validation
    final_counts = [0] * n_tsubs
    for b in best_blocks:
        for ts in itertools.combinations(b, t):
            final_counts[tsub_to_idx[ts]] += 1
    if not all(c >= 1 for c in final_counts):
        # Fallback to greedy result
        return [sorted(b) for b in blocks]
    
    return [sorted(b) for b in best_blocks]
