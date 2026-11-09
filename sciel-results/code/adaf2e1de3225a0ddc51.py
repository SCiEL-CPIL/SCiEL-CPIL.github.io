"""Covering design C(19,7,4) via greedy set cover with restarts."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    num_blocks = len(all_ksubs)

    # Precompute coverage matrix: block i covers tsub j
    # Use numpy for speed
    coverage = np.zeros((num_blocks, num_tsubs), dtype=np.uint8)
    for bi, block in enumerate(all_ksubs):
        for ts in itertools.combinations(block, t):
            coverage[bi, tsub_index[ts]] = 1

    # Greedy set cover with randomization
    def greedy_cover(seed):
        rng = random.Random(seed)
        remaining = np.ones(num_tsubs, dtype=np.uint8)
        remaining_count = num_tsubs
        chosen = []
        used_blocks = set()

        while remaining_count > 0:
            if time.time() > deadline:
                return None

            # Compute gain for each unused block
            unused_mask = np.ones(num_blocks, dtype=bool)
            for b in chosen:
                unused_mask[b] = False

            # Gain = number of newly covered tsubsets
            gains = coverage[:, remaining.astype(np.int32)].sum(axis=1) if remaining_count < num_tsubs else coverage.sum(axis=1)
            # Actually, gain for block i = sum over j where remaining[j]==1 of coverage[i,j]
            gains = (coverage * remaining[None, :]).sum(axis=1)
            gains = gains.astype(np.int32)
            gains[~unused_mask] = -1

            # Add small random noise to break ties and enable diversity
            noise = np.random.default_rng(seed + len(chosen)).integers(0, 1000, size=num_blocks)
            gains_noisy = gains * 1000 + noise
            gains_noisy[~unused_mask] = -1

            best_idx = int(np.argmax(gains_noisy))
            if gains[best_idx] <= 0:
                break

            chosen.append(best_idx)
            # Update remaining
            new_covered = coverage[best_idx] & remaining
            remaining[new_covered.astype(bool)] = 0
            remaining_count -= int(new_covered.sum())

        return [all_ksubs[i] for i in chosen]

    # Try multiple seeds, keep best
    best_blocks = None
    best_len = float('inf')

    # First try deterministic greedy
    det_result = greedy_cover(0)
    if det_result is not None:
        best_blocks = det_result
        best_len = len(det_result)

    # Try random restarts
    seed_counter = 1
    while time.time() < deadline - 5.0:
        result = greedy_cover(seed_counter)
        if result is not None and len(result) < best_len:
            best_blocks = result
            best_len = len(result)
        seed_counter += 1

    if best_blocks is None:
        # Fallback: should not happen, but return something
        return [list(range(k))]

    # Convert to list of lists
    return [list(b) for b in best_blocks]
