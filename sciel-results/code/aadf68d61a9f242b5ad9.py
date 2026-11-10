# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_build(v, k, t, rng):
    """Greedy covering with randomized tie-breaking. Returns list of blocks."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        start = rng.choice(sorted(uncovered))
        block = list(start)
        while len(block) < k:
            # For each candidate point, count how many uncovered t-subsets it would cover
            # A t-subset {a1..at} is covered by adding x iff it contains x and t-1 elements from block
            # i.e., the t-subset minus x is a (t-1)-subset of block
            block_set = set(block)
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that contain x and whose other t-1 elements are in block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if (tuple(sorted(rest + (x,))) in uncovered):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if gain == 0:
                        break
            if best_x == -1 or best_gain == 0:
                # Pick a random point not in block
                candidates = [x for x in range(v) if x not in block_set]
                best_x = rng.choice(candidates)
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks


def _build_cover_index(blocks, v, t):
    """Build a dict mapping each t-subset to the list of block indices covering it."""
    cover = {}
    for bi, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            cover.setdefault(sub, []).append(bi)
    return cover


def _local_search(v, k, t, blocks, deadline, rng):
    """Try to remove blocks by finding replacements that cover all their unique t-subsets."""
    cover = _build_cover_index(blocks, v, t)
    active = list(range(len(blocks)))
    # We'll track which blocks are removed
    removed = set()

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try each active block in random order
        for bi in rng.sample(active, len(active)):
            if bi in removed:
                continue
            if time.time() > deadline - 5.0:
                break
            # Find t-subsets uniquely covered by block bi
            unique_subs = []
            for sub in itertools.combinations(blocks[bi], t):
                covs = cover.get(sub, [])
                # Count active coverers
                active_covs = [c for c in covs if c not in removed]
                if len(active_covs) == 1 and active_covs[0] == bi:
                    unique_subs.append(sub)
            if not unique_subs:
                # Can safely remove
                removed.add(bi)
                improved = True
                continue
            # Try to find a single replacement block that covers all unique_subs
            # The replacement must contain all points that appear in unique_subs
            # Actually, the replacement block must cover each unique sub, meaning for each sub,
            # the block must contain all t elements of sub.
            # So the block must contain the union of all unique_subs' elements.
            # But that union might have more than k elements.
            # Let's find the minimal set of points that must be in the block:
            # For each unique sub, the block must contain all t elements.
            # So the block must contain union of all unique_subs.
            required = set()
            for sub in unique_subs:
                required.update(sub)
            if len(required) > k:
                continue
            # Now find a block of size k containing all required points that covers all unique_subs
            # (it automatically covers them since it contains all required points which include each sub)
            # We just need to fill the remaining k - len(required) points optimally.
            # For simplicity, try a few random completions
            remaining_points = [x for x in range(v) if x not in required]
            need_more = k - len(required)
            if need_more < 0:
                continue
            # Try to find a good completion by greedy: pick points that cover most other uncovered subs
            # But for now, just try random completions a few times
            found = False
            for _attempt in range(5):
                if need_more == 0:
                    new_block = sorted(required)
                else:
                    extra = rng.sample(remaining_points, need_more)
                    new_block = sorted(required | set(extra))
                # Check: does this new block cover all unique_subs?
                # Yes, by construction it contains all required points.
                # Add the new block and remove bi
                new_block_idx = len(blocks)
                blocks.append(new_block)
                # Update cover index
                for sub in itertools.combinations(new_block, t):
                    cover.setdefault(sub, []).append(new_block_idx)
                removed.add(bi)
                active.append(new_block_idx)
                improved = True
                found = True
                break
            if found:
                # Don't break, continue trying other blocks
                pass
    # Rebuild the final list
    result = [blocks[i] for i in range(len(blocks)) if i not in removed]
    return result


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Phase 1: Greedy baseline
    blocks = _greedy_build(v, k, t, rng)

    # Phase 2: Local search - try to remove redundant blocks
    if time.time() < deadline - 10.0:
        blocks = _local_search(v, k, t, blocks, deadline, rng)

    # Phase 3: Try additional greedy runs with different seeds and keep the best
    best = blocks
    best_len = len(blocks)
    seed = 43
    while time.time() < deadline - 10.0:
        rng2 = random.Random(seed)
        candidate = _greedy_build(v, k, t, rng2)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            # Try local search on the better one
            if time.time() < deadline - 5.0:
                candidate = _local_search(v, k, t, candidate, deadline, rng2)
                if len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)
        seed += 1

    return best
# EVOLVE-BLOCK-END
