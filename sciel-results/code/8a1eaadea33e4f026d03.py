"""
Optimized covering design C(19,7,4) using vectorized greedy set cover with
NumPy-accelerated incidence matrix, randomized restarts, and iterative
redundancy removal.

Strategy:
  1. Build the full incidence matrix M (n_k_blocks x n_t_subsets) where
     M[b, s] = 1 iff t-subset s is contained in k-block b.
  2. Run a greedy set-cover heuristic: at each step pick the block with the
     highest number of still-uncovered t-subsets (M @ uncovered_mask).
  3. After each greedy run, perform iterative redundancy removal: a block is
     redundant if every t-subset it covers is covered by at least one other
     selected block.
  4. Repeat with different random seeds / perturbations within the time budget,
     keeping the best (fewest-blocks) valid covering found.
"""
import itertools
import time
import random
import sys

import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 40 s safety margin before 150 s kill

    points = list(range(v))

    # --- Precomputation -------------------------------------------------
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_idx = {sub: i for i, sub in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(points, k))
    n_k = len(k_blocks)

    # For each k-block, list of t-subset indices it covers (as a set for fast intersection)
    block_cover_sets = []
    block_cover_arrays = []
    for block in k_blocks:
        idxs = [t_idx[sub] for sub in itertools.combinations(block, t)]
        block_cover_sets.append(set(idxs))
        block_cover_arrays.append(np.array(idxs, dtype=np.int32))

    # Dense incidence matrix M: shape (n_k, n_t), dtype uint8
    # This allows fast matrix-vector products for greedy gain calculation
    M = np.zeros((n_k, n_t), dtype=np.uint8)
    for b, inds in enumerate(block_cover_arrays):
        M[b, inds] = 1

    # --- Helper: greedy construction ------------------------------------
    def greedy(seed=None, noise_scale=0.0):
        """Run one greedy pass. Returns list of block indices."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t, dtype=np.uint8)
        selected = []
        
        # Optimization: Keep track of active blocks? 
        # For now, rely on NumPy speed of M @ uncovered.
        
        while np.any(uncovered):
            gains = M @ uncovered  # (n_k,) array of ints
            
            if gains.max() == 0:
                break  # shouldn't happen for valid params
                
            if noise_scale > 0.0:
                noise = rng.random(n_k) * noise_scale
                gains = gains.astype(np.float64) + noise
                
            best = int(np.argmax(gains))
            selected.append(best)
            
            # Update uncovered
            # M[best] is a row of 0s and 1s
            # We want to set uncovered[i] = 0 where M[best, i] == 1
            # This is equivalent to uncovered &= ~M[best]
            # Or simpler:
            uncovered[M[best] == 1] = 0

        return selected

    # --- Helper: redundancy removal -------------------------------------
    def remove_redundant(selected):
        """Iteratively remove redundant blocks. Returns reduced list."""
        if not selected:
            return selected
        
        # counts[s] = number of selected blocks covering t-subset s
        # Use the precomputed arrays for fast indexing
        counts = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            counts[block_cover_arrays[b]] += 1
            
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_sel = []
            for i, b in enumerate(selected):
                subs = block_cover_arrays[b]
                if len(subs) == 0:
                    new_sel.append(b)
                    continue
                # If min count > 1, it's redundant
                if np.min(counts[subs]) > 1:
                    # redundant: remove it
                    counts[subs] -= 1
                    changed = True
                else:
                    new_sel.append(b)
            if len(new_sel) < len(selected):
                selected = new_sel
            else:
                break
        return selected

    # --- Helper: validate -----------------------------------------------
    def is_valid(selected):
        if not selected:
            return n_t == 0
        # Check if every t-subset is covered
        # We can reconstruct coverage from selected blocks
        covered = np.zeros(n_t, dtype=np.uint8)
        for b in selected:
            covered[block_cover_arrays[b]] = 1
        return bool(np.all(covered == 1))

    # --- Main search loop ------------------------------------------------
    best_blocks = None
    best_len = float('inf')

    # Phase 1: deterministic greedy + redundancy removal
    sel = greedy(seed=0, noise_scale=0.0)
    sel = remove_redundant(sel)
    if is_valid(sel) and len(sel) < best_len:
        best_len = len(sel)
        best_blocks = sel[:]

    # Phase 2: randomized greedy restarts
    # We vary the noise to break ties differently
    rng_outer = np.random.default_rng(12345)
    
    while time.time() < deadline - 10:
        # Random noise scale to explore different local optima
        noise = rng_outer.random() * 1.0
        seed = int(rng_outer.integers(0, 2**31))
        
        sel = greedy(seed=seed, noise_scale=noise)
        sel = remove_redundant(sel)
        
        if is_valid(sel) and len(sel) < best_len:
            best_len = len(sel)
            best_blocks = sel[:]
            # If we improve, we can try slightly less noise to refine?
            # Or just continue exploring.
            # If we are very good, we might want to tighten search
            if best_len <= 145:
                 # Reduce noise range for subsequent iterations in this branch?
                 # For simplicity, just keep going.
                 pass

    # --- Convert to required output format --------------------------------
    if best_blocks is None:
        # Fallback: single deterministic greedy
        sel = greedy(seed=0, noise_scale=0.0)
        best_blocks = sel

    return [list(k_blocks[b]) for b in best_blocks]


if __name__ == "__main__":
    t0 = time.time()
    blocks = construct(19, 7, 4)
    t1 = time.time()
    print(f"Number of blocks: {len(blocks)}  (time: {t1 - t0:.2f}s)")
    # Verify
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, 4):
            covered.add(ts)
    total = sum(1 for _ in itertools.combinations(range(19), 4))
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "INVALID COVERING"
    # Check distinctness
    assert len(set(tuple(b) for b in blocks)) == len(blocks), "DUPLICATE BLOCKS"
    print("Valid.")
