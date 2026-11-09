import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}

    # Count of how many times each triple is covered
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []

    def add_block(block):
        blocks.append(tuple(block))
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] += 1

    def remove_block(block):
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] -= 1

    def is_covered():
        return np.all(count >= 1)

    def build_block_greedy(start_triple, count_arr, rng_obj):
        """Build a block greedily starting from a given triple."""
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            best_cand = None
            best_s = -1
            # Try all remaining candidates (v - len(block) candidates)
            for cand in sorted(remaining):
                b = tuple(sorted(block + [cand]))
                # Count new coverage: triples with all points in b that are currently uncovered
                s = 0
                for tr2 in itertools.combinations(b, t):
                    if count_arr[triple_to_idx[tr2]] == 0:
                        s += 1
                if s > best_s:
                    best_s = s
                    best_cand = cand
            if best_cand is None:
                best_cand = remaining.pop()
            block.append(best_cand)
            remaining.discard(best_cand)
        return tuple(sorted(block))

    # Phase 1: Greedy construction
    for _ in range(300):
        if time.time() > deadline - 10:
            break
        uncovered = [tr for tr in all_triples if count[triple_to_idx[tr]] == 0]
        if not uncovered:
            break

        # Sample uncovered triples to find best starting point
        sample_size = min(30, len(uncovered))
        sample_idx = list(rng.choice(len(uncovered), size=sample_size, replace=False))

        best_block = None
        best_gain = -1

        for idx in sample_idx:
            tr = uncovered[idx]
            block = build_block_greedy(tr, count, rng)
            # Compute gain
            gain = 0
            for tr2 in itertools.combinations(block, t):
                if count[triple_to_idx[tr2]] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            break

        add_block(best_block)

    # Phase 2: Local improvement - try replacing blocks
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            # Temporarily remove block bi
            removed = blocks[bi]
            remove_block(removed)

            # Find uncovered triples
            uncovered = [tr for tr in all_triples if count[triple_to_idx[tr]] == 0]

            if not uncovered:
                # Can remove this block
                blocks.pop(bi)
                improved = True
                break

            # Try to find a better replacement
            best_repl = None
            best_gain = -1
            sample_size = min(20, len(uncovered))
            sample_idx = list(rng.choice(len(uncovered), size=sample_size, replace=False))

            for idx in sample_idx:
                tr = uncovered[idx]
                block = build_block_greedy(tr, count, rng)
                gain = 0
                for tr2 in itertools.combinations(block, t):
                    if count[triple_to_idx[tr2]] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_repl = block

            if best_repl is not None and best_gain >= 0:
                blocks[bi] = best_repl
                add_block(best_repl)
                improved = True
                break
            else:
                # Restore original
                add_block(removed)

        if not improved:
            break

    # Phase 3: Ensure full coverage
    while time.time() < deadline - 2:
        uncovered = [tr for tr in all_triples if count[triple_to_idx[tr]] == 0]
        if not uncovered:
            break
        tr = uncovered[0]
        block = build_block_greedy(tr, count, rng)
        add_block(block)

    # Final deduplication
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)

    return [list(b) for b in unique_blocks]
