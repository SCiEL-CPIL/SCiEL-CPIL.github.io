"""Combined program: Fast Greedy + Precomputed Lookup + Directed Reconstruction Local Search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138  # Safety margin for 150s limit

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

    def greedy_construct(seed=0, strategy='max_gain', existing_uncovered=None):
        """
        Greedy construction.
        If existing_uncovered is provided, it only tries to cover those.
        """
        rng = random.Random(seed)
        if existing_uncovered is not None:
            uncovered = set(existing_uncovered)
        else:
            uncovered = set(range(n_tsets))
        
        blocks = []

        while uncovered:
            # Pick a random uncovered t-set to start
            start_id = rng.choice(list(uncovered))
            start_ts = all_tsets[start_id]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                best_point = -1
                best_score = -1
                candidates = [p for p in range(v) if p not in block_set]

                # Compute gains
                subs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if tid in uncovered:
                            gain += 1

                    if strategy == 'max_gain':
                        score = float(gain)
                    elif strategy == 'ratio':
                        score = gain / (1 + 0.1 * (k - len(block) - 1))
                    elif strategy == 'ratio2':
                        progress = len(block) / k
                        score = gain / (1 + progress)
                    elif strategy == 'ratio_penalty':
                        # New strategy: penalize redundancy more heavily
                        # The denominator grows as the block gets fuller, 
                        # encouraging blocks to be "tight" and not just grab any point
                        remaining_slots = k - len(block) - 1
                        # If few slots left, be more selective (higher penalty)
                        # If many slots left, be more aggressive (lower penalty)
                        penalty = 1.0 + (0.5 / max(1, remaining_slots))
                        score = gain / penalty
                    else:  # random
                        score = rng.random() * gain if gain > 0 else rng.random() * 0.1

                    score += rng.random() * 0.01

                    if score > best_score:
                        best_score = score
                        best_point = p

                if best_score <= 0.01:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                tid = tset_to_id[ts]
                uncovered.discard(tid)

        return blocks

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    # Increased variety of strategies, including the new ratio_penalty
    strategies = ['max_gain', 'max_gain', 'ratio', 'ratio2', 'ratio_penalty', 'ratio_penalty', 'random']
    num_seeds = 40  # Increased seeds to leverage the new strategy
    for seed in range(num_seeds):
        if time.time() > deadline - 25:
            break
        strategy = strategies[seed % len(strategies)]
        blocks = greedy_construct(seed, strategy)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0, 'max_gain')

    # Phase 2: Local Search
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tid = tset_to_id[ts]
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[tset_to_id[ts]] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute block t-set IDs
        block_ids = []
        for b in blocks:
            ids = set()
            for ts in itertools.combinations(b, t):
                ids.add(tset_to_id[ts])
            block_ids.append(ids)

        # Heuristic: prioritize pairs with high overlap
        pair_scores = []
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    break
                overlap = len(block_ids[i] & block_ids[j])
                total = len(block_ids[i] | block_ids[j])
                if total > 0:
                    score = overlap / total
                    unique_i = len(block_ids[i] - block_ids[j])
                    unique_j = len(block_ids[j] - block_ids[i])
                    low_unique = (unique_i < 5 or unique_j < 5)
                    if low_unique:
                        score += 0.5
                    pair_scores.append((score, i, j))

        pair_scores.sort(reverse=True)
        max_pairs_to_try = min(len(pair_scores), 300)

        for score, i, j in pair_scores[:max_pairs_to_try]:
            if time.time() > deadline - 10:
                break

            b1 = blocks[i]
            b2 = blocks[j]
            ids_b1 = block_ids[i]
            ids_b2 = block_ids[j]

            check_ids = ids_b1 | ids_b2

            u_need = []
            for tid in check_ids:
                c = count[tid]
                cov_b1 = 1 if tid in ids_b1 else 0
                cov_b2 = 1 if tid in ids_b2 else 0
                if c == cov_b1 + cov_b2:
                    u_need.append(tid)

            if not u_need:
                update_counts([b1, b2], -1)
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 1000 + j)
            candidates = [p for p in range(v) if p not in new_block_set]

            remaining_slots = k - len(new_block)
            for _ in range(remaining_slots):
                if not candidates:
                    break
                best_p = -1
                best_gain = -1
                subs_in_block = list(itertools.combinations(new_block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if tid not in ids_b1 and tid not in ids_b2:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p

                if best_p == -1 or best_gain == 0:
                    best_p = rng.choice(candidates)

                new_block.append(best_p)
                new_block_set.add(best_p)
                candidates.remove(best_p)

            new_block.sort()

            new_ids = set()
            for ts in itertools.combinations(new_block, t):
                new_ids.add(tset_to_id[ts])

            if all(tid in new_ids for tid in u_need):
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    def try_reconstruct_block(blocks, i):
        """
        Max-SAT inspired move: Reconstruct block i to better cover its unique triplets
        and potentially increase global efficiency.
        """
        n = len(blocks)
        if n < 1:
            return blocks, False

        b = blocks[i]
        ids_b = set()
        for ts in itertools.combinations(b, t):
            ids_b.add(tset_to_id[ts])

        # Find unique triplets
        u_need = [tid for tid in ids_b if count[tid] == 1]

        if not u_need:
            # Redundant, handled by removal
            return blocks, False

        p_need = set()
        for tid in u_need:
            p_need.update(all_tsets[tid])

        if len(p_need) > k:
            return blocks, False

        # Try to build a new block that covers p_need and maximizes additional coverage
        # This is a constrained greedy construction
        new_block = list(p_need)
        new_block_set = set(new_block)
        rng = random.Random(i * 7919 + int(time.time() % 1000))
        candidates = [p for p in range(v) if p not in new_block_set]

        remaining_slots = k - len(new_block)
        for _ in range(remaining_slots):
            if not candidates:
                break
            best_p = -1
            best_gain = -1
            subs_in_block = list(itertools.combinations(new_block, t - 1))

            for p in candidates:
                gain = 0
                for sub in subs_in_block:
                    ts = tuple(sorted(sub + (p,)))
                    tid = tset_to_id[ts]
                    # We want to cover triplets that are NOT already well-covered by others
                    # or just maximize total coverage
                    if count[tid] == 0: # Should not happen if valid, but good for safety
                        gain += 10
                    elif count[tid] == 1:
                        gain += 1
                    else:
                        gain += 0.1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p

            if best_p == -1 or best_gain == 0:
                best_p = rng.choice(candidates)

            new_block.append(best_p)
            new_block_set.add(best_p)
            candidates.remove(best_p)

        new_block.sort()

        if new_block == b:
            return blocks, False

        new_ids = set()
        for ts in itertools.combinations(new_block, t):
            new_ids.add(tset_to_id[ts])

        if all(tid in new_ids for tid in u_need):
            # Check if this improves the "score" (e.g., reduces unique coverage of other blocks? 
            # Actually, we just swap. The count update handles the rest.)
            update_counts([b], -1)
            update_counts([new_block], 1)
            new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
            new_blocks.append(new_block)
            return new_blocks, True

        return blocks, False

    # Iteratively apply local search moves
    improved = True
    max_rounds = 100
    stall_count = 0
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        if not improved:
            stall_count += 1
            if stall_count >= 3:
                # Try perturbation/reconstruction to escape local optimum
                if time.time() < deadline - 20:
                    # Pick a random block to reconstruct
                    if current_blocks:
                        idx = random.Random(int(time.time()*1000) % 10000).randrange(len(current_blocks))
                        new_blocks, pert = try_reconstruct_block(current_blocks, idx)
                        if pert:
                            current_blocks = new_blocks
                            improved = True
                            stall_count = 0
                            # Try removals after perturbation
                            for _pass in range(3):
                                if time.time() > deadline - 15:
                                    break
                                new_blocks, rem = try_remove_redundant(current_blocks)
                                if rem:
                                    current_blocks = new_blocks
                                    improved = True
                                else:
                                    break
                if not improved:
                    break
                continue
            else:
                break
        else:
            stall_count = 0
        improved = False

        # Try removing redundant blocks
        for _pass in range(3):
            if time.time() > deadline - 15:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            else:
                break

        # Try 2-for-1 swaps
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        # Try reconstruction (1-for-1 but smarter)
        if current_blocks and time.time() < deadline - 15:
            # Try a few random reconstructions
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                idx = random.Random(int(time.time()*10000) % 100000).randrange(len(current_blocks))
                new_blocks, swp = try_reconstruct_block(current_blocks, idx)
                if swp:
                    current_blocks = new_blocks
                    improved = True
                    break

    # Final verification
    covered = set()
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(tset_to_id[ts])

    if len(covered) < n_tsets:
        return best_blocks

    # Final polish
    if time.time() < deadline - 5:
        for _pass in range(5):
            if time.time() > deadline - 3:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
            else:
                break

    return current_blocks
# EVOLVE-BLOCK-END
