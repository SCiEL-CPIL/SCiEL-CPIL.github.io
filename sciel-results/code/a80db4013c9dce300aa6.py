import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Cyclic group Z_v: generate orbit representatives of k-subsets under rotation
    # A k-subset S has orbit { (s + d) % v for d in range(v) }
    # We collect distinct orbit representatives (canonical: sorted tuple, min lex)
    
    seen_canonical = set()
    orbit_reps = []  # list of tuples (each is a canonical k-subset)
    orbit_to_blocks = []  # for each rep, list of all blocks in its orbit
    
    # Generate all k-subsets, group by orbit
    # For efficiency, iterate over combinations and compute canonical form
    all_k = list(itertools.combinations(range(v), k))
    
    # Build orbit map: canonical rep -> list of blocks in orbit
    orbit_map = defaultdict(list)
    for block in all_k:
        # Compute canonical representative: min over rotations of sorted tuple
        best = None
        for d in range(v):
            rotated = tuple(sorted((x + d) % v for x in block))
            if best is None or rotated < best:
                best = rotated
        orbit_map[best].append(block)
    
    # Now we have orbit representatives and their member blocks
    # For each orbit rep, we know how many distinct blocks it contains (usually v, unless stabilizer)
    orbit_reps = list(orbit_map.keys())
    num_orbits = len(orbit_reps)
    
    # For each orbit rep, compute which t-subsets its blocks cover
    # Since all blocks in an orbit are rotations, they cover rotated t-subsets
    # We'll precompute for each orbit rep: the set of t-subset indices covered by ANY block in its orbit
    
    # Actually, for the flow formulation, we need: for each orbit rep, which t-subsets are covered
    # by at least one block in the orbit. But for min-cost flow, we want to select individual blocks.
    
    # Alternative: since the problem is symmetric, we can use a greedy approach over orbits
    # with the understanding that selecting an orbit means selecting all v blocks (or a subset).
    
    # Better approach: use the orbit structure to guide a smarter greedy.
    # For each orbit rep, compute the "coverage" of its orbit: union of t-subsets covered by all blocks in orbit.
    # Then do a greedy set cover over orbits, but allow partial orbit selection.
    
    # Precompute: for each orbit rep, the list of t-subset indices covered by each block in the orbit
    orbit_block_ts = {}  # rep -> list of (block, list_of_ts_indices)
    for rep in orbit_reps:
        blocks_in_orbit = orbit_map[rep]
        block_ts_list = []
        for b in blocks_in_orbits:
            pass  # placeholder
        orbit_block_ts[rep] = block_ts_list
    
    # Let me restructure: build a candidate pool of "interesting" blocks
    # Strategy: start with greedy over all blocks, but use orbit structure for pruning
    
    # Phase 1: Fast greedy using numpy
    all_k_blocks = all_k
    num_blocks = len(all_k_blocks)
    
    # Precompute block -> t-subsets
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
    
    # Greedy set cover
    gain = np.full(num_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t, True, dtype=bool)
    is_selected = np.full(num_blocks, False, dtype=bool)
    solution_blocks = []
    
    while True:
        if time.time() > deadline - 20.0:
            break
        max_idx = int(np.argmax(gain))
        if gain[max_idx] <= 0:
            break
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Phase 2: Orbit-aware local search
    # Map each block to its orbit rep
    block_to_orbit = {}
    for rep in orbit_reps:
        for b in orbit_map[rep]:
            block_to_orbit[id(b)] = rep  # use id since tuples are hashed
    
    # Better: build a direct map
    block_idx_to_orbit = [None] * num_blocks
    for rep in orbit_reps:
        for b in orbit_map[rep]:
            # Find index of b in all_k_blocks
            # This is expensive; instead, build a tuple->idx map
            pass
    
    tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    block_idx_to_orbit = [None] * num_blocks
    for rep in orbit_reps:
        for b in orbit_map[rep]:
            idx = tuple_to_idx[tuple(b)]
            block_idx_to_orbit[idx] = rep
    
    # Redundancy removal with orbit awareness
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index[ts]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index[ts]] == 1:
                        redundant = False
                        break
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index[ts]] -= 1
                    changed = True
                    break
        return blocks
    
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)
    
    # Phase 3: Orbit-based swap local search
    # Try replacing a block with another block from the same orbit (rotation)
    # This preserves orbit structure and may improve coverage
    
    def orbit_swap_search(blocks, max_rounds=200):
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index[ts]] += 1
        
        improved = False
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            found = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks, improved
                # Find unique t-subsets of blocks[i]
                unique_ts = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)
                if not unique_ts:
                    continue
                # Try all rotations of blocks[i]
                orig = blocks[i]
                orig_idx = tuple_to_idx.get(tuple(orig), -1)
                if orig_idx < 0:
                    continue
                orbit = block_idx_to_orbit[orig_idx]
                if orbit is None:
                    continue
                # Try each block in the orbit
                for b in orbit_map[orbit]:
                    if b == orig:
                        continue
                    b_idx = tuple_to_idx.get(tuple(b), -1)
                    if b_idx < 0:
                        continue
                    # Check if b covers all unique_ts
                    b_ts_set = set(block_to_ts[b_idx])
                    if all(ts in b_ts_set for ts in unique_ts):
                        # Accept swap
                        for ts in itertools.combinations(orig, t):
                            cover_count[ts_index[ts]] -= 1
                        blocks[i] = list(b)
                        for ts in itertools.combinations(b, t):
                            cover_count[ts_index[ts]] += 1
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return blocks, improved
    
    best_solution, _ = orbit_swap_search(best_solution, max_rounds=100)
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)
    
    # Phase 4: Multi-start greedy with different random orderings
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Randomized greedy
        random.seed(42 + restart)
        gain = np.full(num_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_blocks, False, dtype=bool)
        sol = []
        
        # Add randomness: shuffle blocks with same gain
        while True:
            if time.time() > deadline - 5.0:
                break
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            candidates = np.where(gain == max_gain)[0]
            if len(candidates) > 0:
                # Pick randomly among top candidates
                max_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                break
            sol.append(list(all_k_blocks[max_idx]))
            is_selected[max_idx] = True
            gain[max_idx] = -1
            for ts_idx in block_to_ts[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
        
        if len(sol) < best_len:
            sol = optimize_redundancy(sol)
            sol, _ = orbit_swap_search(sol, max_rounds=50)
            sol = optimize_redundancy(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
    
    # Final cleanup
    best_solution = optimize_redundancy(best_solution)
    
    return best_solution
