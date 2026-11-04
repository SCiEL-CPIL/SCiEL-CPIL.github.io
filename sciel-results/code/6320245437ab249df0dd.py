import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute (t-1)-subset to t-subsets mapping
    t1_to_tsets = {}
    for tid, ts in enumerate(all_tsets):
        for sub in itertools.combinations(ts, t - 1):
            t1_to_tsets.setdefault(sub, []).append(tid)

    def get_id(ts_tuple):
        return tset_to_id[ts_tuple]

    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_tsets))
        blocks = []

        while uncovered:
            # Pick a random uncovered t-set to start
            start_id = rng.choice(list(uncovered))
            start_ts = all_tsets[start_id]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                # Compute gain for each candidate point
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                subs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p

                if max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                uncovered.discard(tid)

        return blocks

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    num_seeds = 20
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local Search
    count = np.zeros(n_tsets, dtype=np.int32)
    current_blocks = list(best_blocks)
    block_set_list = [set(b) for b in current_blocks]
    
    # Helper to update counts and block_set_list
    def update_counts(blocks, delta=1):
        for b in blocks:
            for ts in itertools.combinations(b, t):
                tid = get_id(ts)
                count[tid] += delta

    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[get_id(ts)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        new_block_sets = []
        for i, b in enumerate(blocks):
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
                new_block_sets.append(block_set_list[i])
        return new_blocks, new_block_sets, removed

    def try_swap_2_for_1(blocks, block_sets):
        n = len(blocks)
        if n < 2:
            return blocks, block_sets, False

        for i in range(n):
            if time.time() > deadline - 10:
                return blocks, block_sets, False
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    return blocks, block_sets, False

                b1 = blocks[i]
                b2 = blocks[j]

                ids_b1 = set(get_id(ts) for ts in itertools.combinations(b1, t))
                ids_b2 = set(get_id(ts) for ts in itertools.combinations(b2, t))
                
                u_need_ids = []
                for tid in (ids_b1 | ids_b2):
                    c = count[tid]
                    cov_b1 = 1 if tid in ids_b1 else 0
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        u_need_ids.append(tid)

                if not u_need_ids:
                    update_counts([b1, b2], -1)
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_block_sets = [s for idx, s in enumerate(block_sets) if idx != i and idx != j]
                    return new_blocks, new_block_sets, True

                p_need = set()
                for tid in u_need_ids:
                    p_need.update(all_tsets[tid])

                if len(p_need) <= k:
                    new_block = list(p_need)
                    rng = random.Random(i * 1000 + j)
                    while len(new_block) < k:
                        p = rng.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Check if new_block is already in the set (excluding i, j)
                    new_block_set = set(new_block)
                    if any(new_block_set == s for s in block_sets if s != block_sets[i] and s != block_sets[j]):
                        continue

                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_block_sets = [s for idx, s in enumerate(block_sets) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    new_block_sets.append(new_block_set)
                    return new_blocks, new_block_sets, True

        return blocks, block_sets, False

    def try_swap_1_for_1(blocks, block_sets):
        n = len(blocks)
        if n < 1:
            return blocks, block_sets, False

        for i in range(n):
            if time.time() > deadline - 10:
                return blocks, block_sets, False

            b = blocks[i]
            ids_b = set(get_id(ts) for ts in itertools.combinations(b, t))
            
            u_need_ids = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need_ids.append(tid)

            if not u_need_ids:
                update_counts([b], -1)
                return [x for idx, x in enumerate(blocks) if idx != i], [s for idx, s in enumerate(block_sets) if idx != i], True

            p_need = set()
            for tid in u_need_ids:
                p_need.update(all_tsets[tid])

            if len(p_need) <= k:
                new_block = list(p_need)
                rng = random.Random(i * 7919)
                while len(new_block) < k:
                    p = rng.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()
                
                new_block_set = set(new_block)
                if new_block_set == block_sets[i]:
                    continue
                
                # Check if new_block is already in the set (excluding i)
                if any(new_block_set == s for idx, s in enumerate(block_sets) if idx != i):
                    continue

                update_counts([b], -1)
                update_counts([new_block], 1)
                
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                new_block_sets = [s for idx, s in enumerate(block_sets) if idx != i]
                new_blocks.append(new_block)
                new_block_sets.append(new_block_set)
                return new_blocks, new_block_sets, True

        return blocks, block_sets, False

    # New: 2-for-2 Perturbation
    def try_swap_2_for_2(blocks, block_sets):
        n = len(blocks)
        if n < 2:
            return blocks, block_sets, False
        
        rng = random.Random(int(time.time() * 1000) % 10000)
        # Try a few random pairs
        for _ in range(10):
            if time.time() > deadline - 10:
                return blocks, block_sets, False
            
            i, j = rng.sample(range(n), 2)
            b1 = blocks[i]
            b2 = blocks[j]
            
            # Temporarily remove
            update_counts([b1, b2], -1)
            
            # Find uncovered triplets
            uncovered = set()
            for tid in range(n_tsets):
                if count[tid] == 0:
                    uncovered.add(tid)
            
            if not uncovered:
                # Both were redundant? Should have been caught by remove_redundant
                update_counts([b1, b2], 1)
                continue

            # Try to cover uncovered with 2 blocks
            # Greedy 1: Pick block covering most uncovered
            best_block1 = None
            max_gain1 = -1
            # Candidate blocks: We can't iterate all C(24,7). 
            # We must construct them.
            # Strategy: Pick a random uncovered triplet to start block 1
            # Then greedily fill it to maximize coverage of 'uncovered'
            
            start_tid = rng.choice(list(uncovered))
            start_ts = all_tsets[start_tid]
            block1 = list(start_ts)
            block1_set = set(block1)
            
            while len(block1) < k:
                best_p = -1
                max_g = -1
                for p in range(v):
                    if p in block1_set: continue
                    gain = 0
                    for sub in itertools.combinations(block1, t-1):
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid in uncovered:
                            gain += 1
                    if gain > max_g:
                        max_g = gain
                        best_p = p
                if best_p == -1:
                    break
                block1.append(best_p)
                block1_set.add(best_p)
            
            block1.sort()
            
            # Update uncovered based on block1
            new_uncovered = set(uncovered)
            for ts in itertools.combinations(block1, t):
                tid = get_id(ts)
                new_uncovered.discard(tid)
            
            if not new_uncovered:
                # Block 1 covered everything. We need 1 more block? No, 2-for-2 means we replace 2 with 2.
                # If 1 block is enough, we can do 2-for-1. But we are in 2-for-2.
                # We can just create a dummy second block? No, must be distinct and valid.
                # If 1 block covers all, we can try to find a second block that doesn't break anything?
                # Actually, if 1 block covers all, we can just use 2-for-1 logic. 
                # But let's assume we need 2 blocks.
                # If new_uncovered is empty, we can pick any valid block for B2?
                # No, B2 must be distinct from existing blocks (except the ones removed).
                # And it must not cause issues.
                # Let's just try to find a second block that covers *something* or is valid.
                # Actually, if 1 block covers all, we can just return a 2-for-1 move.
                # But this function is 2-for-2. Let's handle 2-for-1 separately.
                # For now, if new_uncovered is empty, we failed 2-for-2 (since we want 2 blocks).
                # We can try to make block1 "smaller" or different? No.
                # Let's just revert and try another pair.
                update_counts([b1, b2], 1)
                continue

            # Greedy 2: Pick block covering most of new_uncovered
            start_tid2 = rng.choice(list(new_uncovered))
            start_ts2 = all_tsets[start_tid2]
            block2 = list(start_ts2)
            block2_set = set(block2)
            
            while len(block2) < k:
                best_p = -1
                max_g = -1
                for p in range(v):
                    if p in block2_set: continue
                    gain = 0
                    for sub in itertools.combinations(block2, t-1):
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid in new_uncovered:
                            gain += 1
                    if gain > max_g:
                        max_g = gain
                        best_p = p
                if best_p == -1:
                    break
                block2.append(best_p)
                block2_set.add(best_p)
            
            block2.sort()
            
            # Check if block2 covers remaining
            final_uncovered = set(new_uncovered)
            for ts in itertools.combinations(block2, t):
                tid = get_id(ts)
                final_uncovered.discard(tid)
            
            if not final_uncovered:
                # Check distinctness
                s1 = set(block1)
                s2 = set(block2)
                if s1 == s2:
                    update_counts([b1, b2], 1)
                    continue
                if s1 == block_sets[i] or s1 == block_sets[j]:
                    update_counts([b1, b2], 1)
                    continue
                if s2 == block_sets[i] or s2 == block_sets[j]:
                    update_counts([b1, b2], 1)
                    continue
                if any(s1 == s for idx, s in enumerate(block_sets) if idx != i and idx != j):
                    update_counts([b1, b2], 1)
                    continue
                if any(s2 == s for idx, s in enumerate(block_sets) if idx != i and idx != j):
                    update_counts([b1, b2], 1)
                    continue
                
                # Accept
                update_counts([block1, block2], 1)
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_block_sets = [s for idx, s in enumerate(block_sets) if idx != i and idx != j]
                new_blocks.append(block1)
                new_blocks.append(block2)
                new_block_sets.append(s1)
                new_block_sets.append(s2)
                return new_blocks, new_block_sets, True
            
            # Revert
            update_counts([b1, b2], 1)

        return blocks, block_sets, False

    # Main Optimization Loop
    progress_made = True
    rounds_no_improve = 0
    while progress_made and time.time() < deadline - 2:
        progress_made = False

        current_blocks, block_set_list, removed = try_remove_redundant(current_blocks, block_set_list)
        if removed:
            progress_made = True
            rounds_no_improve = 0
            continue

        current_blocks, block_set_list, swapped = try_swap_2_for_1(current_blocks, block_set_list)
        if swapped:
            progress_made = True
            rounds_no_improve = 0
            continue

        current_blocks, block_set_list, swapped = try_swap_1_for_1(current_blocks, block_set_list)
        if swapped:
            progress_made = True
            rounds_no_improve = 0
            continue

        # Try 2-for-2
        current_blocks, block_set_list, swapped = try_swap_2_for_2(current_blocks, block_set_list)
        if swapped:
            progress_made = True
            rounds_no_improve = 0
            continue

        # If no progress, perturb
        rounds_no_improve += 1
        if rounds_no_improve > 20:
            # Remove 2 random blocks and repair
            if len(current_blocks) > 10:
                rng = random.Random()
                idxs = rng.sample(range(len(current_blocks)), 2)
                # Remove them
                blocks_to_remove = [current_blocks[i] for i in idxs]
                update_counts(blocks_to_remove, -1)
                
                # Reconstruct block lists
                new_blocks = [b for i, b in enumerate(current_blocks) if i not in idxs]
                new_block_sets = [s for i, s in enumerate(block_set_list) if i not in idxs]
                
                # Repair: Add blocks to cover uncovered
                uncovered = set(tid for tid in range(n_tsets) if count[tid] == 0)
                while uncovered:
                    # Greedy add
                    start_id = rng.choice(list(uncovered))
                    start_ts = all_tsets[start_id]
                    block = list(start_ts)
                    block_set = set(block)
                    while len(block) < k:
                        best_p = -1
                        max_g = -1
                        for p in range(v):
                            if p in block_set: continue
                            gain = 0
                            for sub in itertools.combinations(block, t-1):
                                ts = tuple(sorted(sub + (p,)))
                                tid = get_id(ts)
                                if tid in uncovered:
                                    gain += 1
                            if gain > max_g:
                                max_g = gain
                                best_p = p
                        if best_p == -1:
                            break
                        block.append(best_p)
                        block_set.add(best_p)
                    block.sort()
                    new_blocks.append(block)
                    new_block_sets.append(set(block))
                    update_counts([block], 1)
                    for ts in itertools.combinations(block, t):
                        tid = get_id(ts)
                        uncovered.discard(tid)
                
                current_blocks = new_blocks
                block_set_list = new_block_sets
                rounds_no_improve = 0
                progress_made = True

    return current_blocks

if __name__ == "__main__":
    pass
