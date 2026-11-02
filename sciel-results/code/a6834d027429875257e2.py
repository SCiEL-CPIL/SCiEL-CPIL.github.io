"""
Direct structural construction for covering design C(24,7,3).
Combines greedy bitmask construction with iterative greedy set cover using
bitmask-based gain computation, plus simulated annealing local search.
"""
import itertools
import random
import time
import math


def _precompute_triples(v, t):
    """Precompute all t-subsets and map them to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triple_to_idx[combo] = idx
        triples.append(combo)
    return triples, triple_to_idx


def _block_triple_indices(block, triple_to_idx):
    """Get the set of triple indices covered by a block."""
    idxs = set()
    for combo in itertools.combinations(block, 3):
        if combo in triple_to_idx:
            idxs.add(triple_to_idx[combo])
    return idxs


def _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx):
    """
    Bitmask-based greedy construction (point-by-point extension).
    Returns a list of blocks (tuples).
    """
    n_triples = len(triples)
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()

    while uncovered_count > 0:
        seed_idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                seed_idx = i
                break

        if seed_idx < 0:
            break

        seed_triple = triples[seed_idx]
        block = list(seed_triple)

        while len(block) < k:
            best_x = None
            best_gain = -1

            for x in range(v):
                if x in block:
                    continue

                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(new_triple, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break

            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            for combo in itertools.combinations(b_tuple, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1

    return blocks


def _iterative_greedy_set_cover(v, k, t, rng, triples, triple_to_idx,
                                block_masks, all_blocks, time_limit=30.0):
    """
    Iterative greedy set cover using precomputed block masks.
    Selects blocks by maximizing newly covered triples (bitmask gain).
    """
    start_time = time.time()
    n_triples = len(triples)
    full_mask = (1 << n_triples) - 1

    best_blocks = None
    best_n = float('inf')

    n_blocks_total = len(all_blocks)
    max_restarts = 8

    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break

        uncovered_mask = full_mask
        selected = []
        used = set()

        while uncovered_mask != 0:
            if time.time() > start_time + time_limit:
                break

            best_gain = -1
            best_idx = -1
            best_ties = []

            # Sample a subset of blocks for speed if too many
            if n_blocks_total > 50000:
                candidates = rng.sample(range(n_blocks_total), min(50000, n_blocks_total))
            else:
                candidates = range(n_blocks_total)

            for i in candidates:
                if i in used:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_ties = [i]
                elif gain == best_gain and gain > 0:
                    best_ties.append(i)

            if best_gain <= 0:
                break

            # Random tie-breaking for diversity
            if len(best_ties) > 1:
                best_idx = rng.choice(best_ties)

            used.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            selected.append(best_idx)

        if uncovered_mask == 0 and len(selected) < best_n:
            best_n = len(selected)
            best_blocks = [all_blocks[i] for i in selected]

    return best_blocks, best_n


def _local_search_sa(blocks, v, k, t, triples, triple_to_idx, time_limit=60.0):
    """
    Simulated annealing local search to minimize block count.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(42)

    best_blocks = list(blocks)
    best_n = len(best_blocks)
    current_blocks = list(blocks)
    current_n = len(current_blocks)

    def compute_counts(blk_list):
        counts = [0] * len(triples)
        for b in blk_list:
            for combo in itertools.combinations(b, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def is_valid_counts(counts):
        return all(c >= 1 for c in counts)

    current_counts = compute_counts(current_blocks)

    T_start = 1.0
    T_end = 0.01
    cooling = 0.9995

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        T = T_start * (cooling ** iteration)
        if T < T_end:
            T = T_end

        move_type = rng.random()
        if move_type < 0.4:
            # Try to remove a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            b = current_blocks[i]
            removable = True
            for combo in itertools.combinations(b, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and current_counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                new_blocks = current_blocks[:i] + current_blocks[i+1:]
                new_counts = compute_counts(new_blocks)
                if is_valid_counts(new_counts):
                    current_blocks = new_blocks
                    current_counts = new_counts
                    current_n = len(current_blocks)
                    if current_n < best_n:
                        best_n = current_n
                        best_blocks = list(current_blocks)
                    continue
        elif move_type < 0.7:
            # Try to swap a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            old_b = current_blocks[i]
            old_set = set(old_b)

            new_set = set()
            for _ in range(k):
                candidates = [x for x in range(v) if x not in new_set]
                old_cands = [x for x in old_set if x not in new_set]
                if old_cands and rng.random() < 0.6:
                    x = rng.choice(old_cands)
                else:
                    x = rng.choice(candidates)
                new_set.add(x)

            new_b = tuple(sorted(new_set))
            if new_b == old_b:
                continue

            if new_b in current_blocks:
                continue

            new_blocks = current_blocks[:]
            new_blocks[i] = new_b
            new_counts = compute_counts(new_blocks)

            if is_valid_counts(new_counts):
                current_blocks = new_blocks
                current_counts = new_counts
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
        else:
            # Try to add a block (to escape local minima)
            if current_n > 100:
                continue
            new_b = tuple(sorted(rng.sample(range(v), k)))
            if new_b in current_blocks:
                continue

            new_blocks = current_blocks + [new_b]
            new_counts = compute_counts(new_blocks)
            current_blocks = new_blocks
            current_counts = new_counts
            current_n = len(current_blocks)

        if iteration % 5000 == 0 and current_n > best_n + 5:
            current_blocks = list(best_blocks)
            current_counts = compute_counts(current_blocks)
            current_n = best_n

    return best_blocks, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)
    n_triples = len(triples)

    # Precompute all blocks and their bitmask coverage
    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for combo in itertools.combinations(block, t):
            idx = triple_to_idx.get(combo, -1)
            if idx >= 0:
                mask |= (1 << idx)
        block_masks.append(mask)

    # Step 1: Multiple greedy restarts (point-by-point)
    best_greedy = None
    best_greedy_n = float('inf')

    greedy_time = 15.0
    greedy_deadline = start_time + greedy_time

    for restart in range(10):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover_bitmask(v, k, t, r, triples, triple_to_idx)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx)

    # Step 2: Iterative greedy set cover with precomputed masks
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        igc_time = min(remaining * 0.4, 30.0)
        igc_blocks, igc_n = _iterative_greedy_set_cover(
            v, k, t, rng, triples, triple_to_idx,
            block_masks, all_blocks, time_limit=igc_time
        )
        if igc_blocks is not None and igc_n < best_greedy_n:
            best_greedy = igc_blocks
            best_greedy_n = igc_n

    # Step 3: Simulated annealing local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        sa_time = remaining - 5.0
        best_blocks, best_n = _local_search_sa(
            best_greedy, v, k, t, triples, triple_to_idx,
            time_limit=sa_time
        )
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    return [list(b) for b in best_blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    # Verify
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
