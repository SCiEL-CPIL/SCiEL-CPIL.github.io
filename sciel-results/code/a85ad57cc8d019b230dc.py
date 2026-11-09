import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # safety margin: 150s kill, leave 15s buffer

    rng = _rng(_SEED)
    all_triples = list(itertools.combinations(range(v), 3))
    n_all = len(all_triples)

    # Map each triple to an index for fast counting
    triple_to_idx = {}
    for i, tr in enumerate(all_triples):
        triple_to_idx[tr] = i

    # Precompute for each point, the list of triples containing it (for faster gain computation)
    # point -> list of triple indices containing that point
    point_triples = defaultdict(list)
    for tr_idx, tr in enumerate(all_triples):
        for p in tr:
            point_triples[p].append(tr_idx)

    # For a set of points S, gain = number of triples with all points in S and count==0
    # We can compute this by intersecting triple lists, but that's expensive.
    # Instead, for a candidate block B, we compute gain directly.

    def compute_gain(block, count):
        """Compute number of newly covered triples by this block."""
        gain = 0
        for tr in itertools.combinations(block, 3):
            if count[triple_to_idx[tr]] == 0:
                gain += 1
        return gain

    def build_block_greedy(start_triple, count, remaining_points=None):
        """Build a block greedily starting from start_triple, maximizing gain."""
        if remaining_points is None:
            remaining_points = set(range(v))
        block = list(start_triple)
        for p in start_triple:
            remaining_points.discard(p)

        for _ in range(k - len(block)):
            best_cand = None
            best_s = -1
            # Sample candidates for speed
            cands = list(rng.choice(sorted(remaining_points), size=min(40, len(remaining_points)), replace=False))
            for cand in cands:
                b = tuple(sorted(block + [cand]))
                s = compute_gain(b, count)
                if s > best_s:
                    best_s = s
                    best_cand = cand
            if best_cand is None:
                best_cand = remaining_points.pop()
            block.append(best_cand)
            remaining_points.discard(best_cand)

        return tuple(sorted(block))

    def build_greedy_set_cover():
        """Greedy set cover: repeatedly add block covering most uncovered triples."""
        count = np.zeros(n_all, dtype=np.int32)
        blocks = []

        for _ in range(300):
            if time.time() > deadline - 3:
                break

            # Find uncovered triples
            uncovered_mask = count == 0
            n_uncovered = int(np.sum(uncovered_mask))
            if n_uncovered == 0:
                break

            # Sample several starting uncovered triples
            uncovered_indices = np.where(uncovered_mask)[0]
            n_sample = min(25, len(uncovered_indices))
            sample_idx = rng.choice(uncovered_indices, size=n_sample, replace=False)

            best_block = None
            best_gain = -1

            for idx in sample_idx:
                tr = all_triples[idx]
                block = build_block_greedy(tr, count)
                gain = compute_gain(block, count)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block

            if best_block is None or best_gain <= 0:
                break

            blocks.append(best_block)
            for tr in itertools.combinations(best_block, 3):
                count[triple_to_idx[tr]] += 1

        return blocks, count

    def local_improve(blocks, count, max_rounds=100):
        """Local search: try replacing blocks to reduce total count while maintaining coverage."""
        blocks = [list(b) for b in blocks]
        count = count.copy()

        for round_num in range(max_rounds):
            if time.time() > deadline - 3:
                break

            any_improved = False
            for bi in range(len(blocks)):
                if time.time() > deadline - 3:
                    break

                # Temporarily remove block bi
                removed = blocks[bi]
                for tr in itertools.combinations(removed, 3):
                    count[triple_to_idx[tr]] -= 1

                # Find uncovered triples after removal
                uncovered_mask = count == 0
                n_uncovered = int(np.sum(uncovered_mask))

                if n_uncovered == 0:
                    # Block was redundant - remove it
                    blocks.pop(bi)
                    any_improved = True
                    break

                # Try to find a better replacement
                uncovered_indices = np.where(uncovered_mask)[0]
                n_sample = min(20, len(uncovered_indices))
                sample_idx = rng.choice(uncovered_indices, size=n_sample, replace=False)

                best_repl = None
                best_gain = -1

                for idx in sample_idx:
                    tr = all_triples[idx]
                    block = build_block_greedy(tr, count)
                    gain = compute_gain(block, count)
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = block

                if best_repl is not None and best_gain > 0:
                    blocks[bi] = best_repl
                    for tr in itertools.combinations(best_repl, 3):
                        count[triple_to_idx[tr]] += 1
                    any_improved = True
                else:
                    # Restore original block
                    for tr in itertools.combinations(removed, 3):
                        count[triple_to_idx[tr]] += 1

            if not any_improved:
                break

        return blocks, count

    def ensure_coverage(blocks, count):
        """Add blocks until all triples are covered."""
        blocks = [list(b) for b in blocks]
        count = count.copy()

        while True:
            if time.time() > deadline - 2:
                break

            uncovered_mask = count == 0
            n_uncovered = int(np.sum(uncovered_mask))
            if n_uncovered == 0:
                break

            uncovered_indices = np.where(uncovered_mask)[0]
            tr = all_triples[uncovered_indices[0]]
            block = build_block_greedy(tr, count)
            blocks.append(block)
            for tr2 in itertools.combinations(block, 3):
                count[triple_to_idx[tr2]] += 1

        return blocks, count

    # Phase 1: Greedy construction
    blocks, count = build_greedy_set_cover()

    # Phase 2: Local improvement
    blocks, count = local_improve(blocks, count)

    # Phase 3: Ensure full coverage
    blocks, count = ensure_coverage(blocks, count)

    # Phase 4: Final local improvement pass with more iterations
    if time.time() < deadline - 5:
        blocks, count = local_improve(blocks, count, max_rounds=50)

    return blocks
