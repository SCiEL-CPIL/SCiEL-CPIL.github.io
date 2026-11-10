import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 112.0


def _generic_greedy(v, k, t, start_time):
    """Generic randomized greedy for small instances."""
    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    n_t = len(all_t)
    idx_map = {s: i for i, s in enumerate(all_t)}
    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)

    # coverage[b][s] = 1 if block b covers t-subset s
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    for b, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            cov[b, idx_map[sub]] = 1

    uncovered = np.ones(n_t, dtype=bool)
    chosen = []
    deadline = start_time + DEADLINE_SECONDS

    while uncovered.any() and time.time() < deadline:
        gains = cov[:, uncovered].sum(axis=1)
        if gains.max() <= 0:
            break
        b = int(np.argmax(gains))
        chosen.append(b)
        uncovered[cov[b] == 1] = False

    return [list(all_blocks[i]) for i in chosen]


def construct(v, k, t):
    start_time = time.time()

    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precompute for v=19, k=8, t=5
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)  # 11628
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)  # 75582

    # Build coverage matrix: cov[b, s] = 1 if block b covers t-subset s
    # Each block of size 8 covers C(8,5)=56 t-subsets
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    for b, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            cov[b, idx_map[sub]] = 1

    # For each t-subset, list of blocks that cover it
    # (stored as lists for quick access)
    blocks_for_subset = []
    for s in range(n_t):
        blocks_for_subset.append(np.where(cov[:, s] == 1)[0])

    best_blocks = None
    best_count = float('inf')
    deadline = start_time + DEADLINE_SECONDS

    rng = random.Random(42)

    while time.time() < deadline - 2.0:
        # Randomized greedy with random tie-breaking
        uncovered = np.ones(n_t, dtype=bool)
        chosen = []

        while uncovered.any():
            # Pick a random uncovered t-subset
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            s = int(rng.choice(unc_idx))

            # Candidate blocks covering this subset
            cands = blocks_for_subset[s]
            if len(cands) == 0:
                # Should not happen; skip
                break

            # Compute gain for each candidate: number of uncovered subsets it covers
            # Use numpy for speed
            gains = cov[cands, :].sum(axis=1)  # total coverage per block
            # But we want coverage of UNCOVERED subsets
            # Faster: for each candidate, count uncovered it covers
            # cov[cands, :][uncovered] -> shape (len(cands), n_t) masked
            # Use: cov[cands][:, uncovered].sum(axis=1)
            # But that's a big slice. Alternative:
            # gains = np.array([cov[b, uncovered].sum() for b in cands])
            # That's slow in Python. Use numpy:
            gains = np.einsum('ij,j->i', cov[cands], uncovered)

            # Add small random perturbation to break ties
            gains_perturbed = gains + rng.random() * 0.5

            best_local = int(np.argmax(gains_perturbed))
            b = int(cands[best_local])

            chosen.append(b)
            uncovered[cov[b] == 1] = False

        # Local improvement: try to remove redundant blocks
        chosen_set = set(chosen)
        # Try removing each block and see if coverage is still complete
        # This is expensive; do a few passes
        changed = True
        passes = 0
        while changed and passes < 3 and time.time() < deadline - 1.0:
            changed = False
            passes += 1
            # Recompute coverage counts for current set
            # count[s] = number of blocks in chosen covering s
            if len(chosen) == 0:
                break
            counts = np.zeros(n_t, dtype=np.int16)
            for b in chosen:
                counts += cov[b]

            # Try removing blocks in random order
            order = list(range(len(chosen)))
            rng.shuffle(order)
            new_chosen = []
            for i in order:
                b = chosen[i]
                # If any subset is covered only by this block, keep it
                # Check: for subsets covered by b, is count > 1?
                # Subsets covered by b: where cov[b]==1
                covered_by_b = cov[b] == 1
                # For those, count must be > 1 to remove
                if np.all(counts[covered_by_b] > 1):
                    # Safe to remove
                    counts -= cov[b]
                    changed = True
                else:
                    new_chosen.append(b)

            if changed:
                chosen = new_chosen

        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen[:]
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = np.ones(n_t, dtype=bool)
        chosen = []
        while uncovered.any() and time.time() < deadline:
            gains = np.einsum('ij,j->i', cov, uncovered)
            b = int(np.argmax(gains))
            if gains[b] <= 0:
                break
            chosen.append(b)
            uncovered[cov[b] == 1] = False
        best_blocks = chosen

    return [list(all_blocks[i]) for i in best_blocks]
