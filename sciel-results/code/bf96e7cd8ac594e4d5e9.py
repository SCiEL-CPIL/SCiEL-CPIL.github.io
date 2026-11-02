import itertools
import random
import time
import numpy as np

_PRECOMPUTED = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]

    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each triple, list of blocks that contain it
    blocks_by_triple = [[] for _ in range(num_triples)]
    for b_idx, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            blocks_by_triple[t_idx].append(b_idx)

    # Precompute the list of triple indices covered by each block (as a frozenset of ints)
    block_triples = []
    for block in all_blocks:
        tris = frozenset(triple_to_idx[sub] for sub in itertools.combinations(block, t))
        block_triples.append(tris)

    # Precompute, for each block, the set of blocks that share at least one triple with it
    # (for local search: which other blocks could be affected when we remove/replace this one)
    # We'll compute lazily per-block as a set of block indices to save memory.

    _PRECOMPUTED[key] = {
        'all_blocks': all_blocks,
        'blocks_by_triple': blocks_by_triple,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx,
        'num_blocks': num_blocks,
        'block_triples': block_triples,
        'all_triples': all_triples,
    }
    return _PRECOMPUTED[key]

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        return _simple_greedy(v, k, t)

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    block_triples = data['block_triples']

    best_blocks = None
    best_count = float('inf')

    run_id = 0
    while time.time() < deadline:
        # Alternate between different greedy strategies
        mode = run_id % 3
        if mode == 0:
            blocks = _greedy_bitset(data, random_seed=run_id, mode='rarest')
        elif mode == 1:
            blocks = _greedy_bitset(data, random_seed=run_id, mode='random')
        else:
            blocks = _greedy_bitset(data, random_seed=run_id, mode='weighted')

        run_id += 1

        # Apply reduction
        blocks = _reduce_blocks(blocks, v, t)

        # Local search polish to try to drop a few more blocks
        blocks = _local_search_polish(blocks, data, deadline)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        # If we already beat known best, keep searching a bit more for further gains
        if best_count <= 74:
            # tighten deadline to still finish
            pass

    if best_blocks is None:
        best_blocks = _simple_greedy(v, k, t)

    # Final reduction
    best_blocks = _reduce_blocks(best_blocks, v, t)

    return [list(b) for b in best_blocks]

def _greedy_bitset(data, random_seed, mode):
    rng = random.Random(random_seed)
    all_blocks = data['all_blocks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    block_triples = data['block_triples']
    triple_to_idx = data['triple_to_idx']

    # uncovered as a set of triple indices
    uncovered = set(range(num_triples))

    # triple_freq: number of blocks covering each triple (for rarity)
    triple_freq = np.array([len(blocks_by_triple[i]) for i in range(num_triples)], dtype=np.int32)

    blocks = []

    while uncovered:
        # Select target triple based on mode
        if mode == 'rarest':
            # pick uncovered triple with fewest covering blocks
            # sample a handful of uncovered triples for efficiency
            sample = set(uncovered)
            if len(sample) > 60:
                sample = set(rng.sample(list(uncovered), 60))
            best_t = min(sample, key=lambda x: triple_freq[x])
            t_idx = best_t
        elif mode == 'random':
            t_idx = rng.choice(tuple(uncovered))
        else:  # weighted
            # weight by inverse frequency, sample
            sample = list(uncovered)
            if len(sample) > 200:
                sample = rng.sample(sample, 200)
            weights = [1.0 / max(1, triple_freq[x]) for x in sample]
            t_idx = rng.choices(sample, weights=weights, k=1)[0]

        candidates = blocks_by_triple[t_idx]

        # Score each candidate block by how many new (uncovered) triples it covers
        best_block_idx = None
        best_gain = -1
        for b_idx in candidates:
            gain = 0
            for tri in block_triples[b_idx]:
                if tri in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and rng.random() < 0.25:
                best_block_idx = b_idx

        selected_block = all_blocks[best_block_idx]
        blocks.append(selected_block)

        # Update uncovered
        for tri in block_triples[best_block_idx]:
            uncovered.discard(tri)

    return blocks

def _reduce_blocks(blocks, v, t):
    if not blocks:
        return []

    # Map each triple to list of block indices covering it
    triple_to_blocks = {}
    for i, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            if sub not in triple_to_blocks:
                triple_to_blocks[sub] = []
            triple_to_blocks[sub].append(i)

    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            redundant = True
            for sub in itertools.combinations(blocks[i], t):
                covered = False
                for j in triple_to_blocks[sub]:
                    if j != i and active[j]:
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True

    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _local_search_polish(blocks, data, deadline):
    # Try to reduce block count by removing redundant blocks and by replacing
    # one block with another that covers its unique triples while not losing coverage.
    # We use a coverage-count approach: count how many times each triple is covered.
    all_blocks = data['all_blocks']
    block_triples = data['block_triples']
    triple_to_idx = data['triple_to_idx']
    num_triples = data['num_triples']

    # Build coverage counts over current blocks
    # We'll represent current blocks as a list of block indices into all_blocks
    # But current blocks may not be in all_blocks? They are 7-subsets of 24, so they are in all_blocks.
    # Map each current block (tuple) to its index in all_blocks.
    block_set = set(all_blocks)
    current = []
    for b in blocks:
        tb = tuple(b)
        if tb in block_set:
            current.append(all_blocks.index(tb))
        else:
            # Should not happen; fallback
            current.append(None)

    # coverage count per triple
    cov = np.zeros(num_triples, dtype=np.int32)
    for b_idx in current:
        if b_idx is None:
            continue
        for tri in block_triples[b_idx]:
            cov[tri] += 1

    # Precompute for each block the set of triples it covers as a set of ints for quick diff
    # We already have block_triples as frozensets.

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current)
        for i in range(n):
            if current[i] is None:
                continue
            b_idx = current[i]
            # Check if removing this block leaves all its triples still covered
            removable = True
            for tri in block_triples[b_idx]:
                if cov[tri] <= 1:
                    removable = False
                    break
            if removable:
                # Remove it
                for tri in block_triples[b_idx]:
                    cov[tri] -= 1
                current.pop(i)
                improved = True
                break  # restart loop
        # If no removal, try a 1-for-1 replacement: find a block not in current that
        # covers all triples that would become uncovered if we remove current[i].
        # To limit cost, only attempt for a few i per pass.
        if not improved:
            # Try replacement for up to a few positions
            attempts = 0
            for i in range(len(current)):
                if current[i] is None:
                    continue
                b_idx = current[i]
                # Triples that would become uncovered if we remove b_idx
                critical = [tri for tri in block_triples[b_idx] if cov[tri] == 1]
                if not critical:
                    # then it's removable, but we already tried; skip
                    continue
                # Find a candidate block (not in current) that covers all critical triples
                # and doesn't introduce any new uncovered triple (i.e., every triple it covers
                # is either already covered or in critical set).
                # To keep cost low, only check candidates that share a triple with b_idx.
                # We'll just scan a random sample of all_blocks.
                found = False
                # Build set of current block indices for quick membership
                current_set = set(current)
                # Candidates: blocks that cover at least one critical triple
                cand_set = set()
                for tri in critical:
                    for cb in data['blocks_by_triple'][tri]:
                        if cb not in current_set:
                            cand_set.add(cb)
                # Sample up to 200 candidates
                cand_list = list(cand_set)
                if len(cand_list) > 200:
                    cand_list = random.Random(i * 1000 + attempts).sample(cand_list, 200)
                for cb in cand_list:
                    # Check cb covers all critical
                    ok = True
                    for tri in critical:
                        if tri not in block_triples[cb]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check cb doesn't cover any triple with cov == 0 (would be fine, but we
                    # must ensure after swap no triple becomes uncovered). Actually after swap,
                    # triples covered only by b_idx (critical) are now covered by cb.
                    # Triples covered by cb but not by b_idx: they were covered by others
                    # (cov>=1) or not covered at all (cov==0) -> if cov==0 that's fine, it
                    # becomes covered. The risk is triples covered by b_idx but not cb that
                    # had cov==1 (critical) -> those are exactly critical, and cb covers them.
                    # So the swap is safe iff cb covers all critical triples.
                    # Perform swap
                    for tri in block_triples[b_idx]:
                        cov[tri] -= 1
                    for tri in block_triples[cb]:
                        cov[tri] += 1
                    current[i] = cb
                    improved = True
                    found = True
                    break
                if found:
                    break
                attempts += 1
                if attempts >= 5:
                    break

    # Convert back to list of lists
    result = []
    for b_idx in current:
        if b_idx is None:
            continue
        result.append(list(all_blocks[b_idx]))
    return result

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks
