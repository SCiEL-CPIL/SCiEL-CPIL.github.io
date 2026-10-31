import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of (t-1)-subsets that contain x
    # Actually, we need: for a partial block B and candidate x, gain = # uncovered t-subsets
    # formed by (t-1)-subset of B plus x.
    
    # Deadline: 105 seconds to be safe (limit is 120, kill is 150)
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple randomized greedy attempts
    while time.time() < deadline:
        # Shuffle the order of subsets to randomize the "first uncovered" choice
        shuffled_subsets = all_t_subsets[:]
        random.shuffle(shuffled_subsets)
        
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        
        for s in shuffled_subsets:
            idx_s = subset_to_idx[s]
            if not uncovered[idx_s]:
                continue
                
            # Build a block containing s
            block = list(s)
            
            # Greedily add points to the block
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain is the number of new t-subsets covered by adding x
                    # These are subsets formed by (t-1) elements from 'block' and x
                    gain = 0
                    # Iterate over all (t-1) subsets of the current block
                    # t=4, so we iterate over 3-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset
                        new_subset = tuple(sorted(list(rest) + [x]))
                        # Check if it's in the mapping and if it's uncovered
                        if new_subset in subset_to_idx:
                            idx = subset_to_idx[new_subset]
                            if uncovered[idx]:
                                gain += 1
                                
                    # Add small random noise to break ties
                    gain += random.random() * 0.1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                block.append(best_point)
            
            # Sort the block for consistency (though not strictly required if we use sets)
            block.sort()
            current_blocks.append(block)
            
            # Update uncovered status
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
            
            # Optional: Early termination if all covered? 
            # Not easy to check without scanning, but we can check if uncovered.sum() == 0
            # This is O(N) per block, which is fine.
            if not uncovered.any():
                break
        
        # Check if this run is better
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
            # If we hit the theoretical lower bound or a very good number, we might want to stop early?
            # Lower bound is 131. If we find 131, we can stop.
            if best_count <= 131:
                break

    if best_blocks is None:
        # Fallback: return a valid covering (should not happen if loop runs at least once)
        # But to be safe, run one deterministic greedy
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        for s in all_t_subsets:
            idx_s = subset_to_idx[s]
            if not uncovered[idx_s]:
                continue
            block = list(s)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(list(rest) + [x]))
                        if new_subset in subset_to_idx:
                            idx = subset_to_idx[new_subset]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            current_blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
        best_blocks = current_blocks

    # Local Search: ILS with exact intersection repair
    if best_blocks is not None and time.time() < deadline - 10.0:
        best_blocks = local_search(best_blocks, v, k, t, subset_to_idx, deadline)
    
    # Spectral Pruning: remove blocks with low eigenvector centrality in the
    # block co-occurrence graph (edges weighted by shared t-subsets).
    if best_blocks is not None and time.time() < deadline - 5.0:
        best_blocks = spectral_prune(best_blocks, v, k, t, subset_to_idx, deadline)
    
    # Final redundant-block sweep (cheap, exact)
    if best_blocks is not None:
        best_blocks = final_sweep(best_blocks, v, k, t, subset_to_idx)
    
    return best_blocks

def spectral_prune(blocks, v, k, t, subset_to_idx, deadline):
    """
    Build a block co-occurrence graph where edge weight = number of shared
    t-subsets between two blocks.  Compute eigenvector centrality and prune
    blocks whose centrality falls below a dynamically adjusted threshold.
    A block is only removed if doing so does not break coverage.
    """
    n = len(blocks)
    if n <= 2:
        return blocks

    # Precompute the list of t-subset indices for each block
    block_subs = []
    for block in blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[sub])
        block_subs.append(subs)

    # Build the co-occurrence (shared t-subset) matrix incrementally.
    # For each t-subset, find all blocks covering it and add pairwise weights.
    # To keep memory/time bounded, cap the number of blocks per subset.
    from collections import defaultdict
    subset_to_blocks = defaultdict(list)
    for bi, subs in enumerate(block_subs):
        for si in subs:
            if len(subset_to_blocks[si]) < 12:  # cap to limit O(m^2) blow-up
                subset_to_blocks[si].append(bi)

    # Weighted adjacency: W[i][j] = # shared t-subsets
    # Use a dict of dicts for sparse storage
    W = defaultdict(lambda: defaultdict(int))
    for si, blks in subset_to_blocks.items():
        m = len(blks)
        for a in range(m):
            for b in range(a + 1, m):
                W[blks[a]][blks[b]] += 1
                W[blks[b]][blks[a]] += 1

    # Power iteration for eigenvector centrality
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        new_x = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i in W:
                s = 0.0
                for j, w in W[i].items():
                    s += w * x[j]
                new_x[i] = s
        norm = np.linalg.norm(new_x)
        if norm < 1e-12:
            break
        new_x /= norm
        if np.max(np.abs(new_x - x)) < 1e-7:
            x = new_x
            break
        x = new_x

    # Compute coverage counts
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    for subs in block_subs:
        for si in subs:
            coverage_count[si] += 1

    # Sort blocks by centrality ascending (least central first)
    order = np.argsort(x)  # ascending
    blocks = [b[:] for b in blocks]
    block_subs = [block_subs[i] for i in order]
    x_sorted = x[order]

    # Dynamically adjusted threshold: start at the 10th percentile, tighten
    # as we remove blocks.
    threshold = np.percentile(x_sorted, 10)

    removed = []
    for i in range(n):
        if time.time() > deadline - 5.0:
            break
        if x_sorted[i] >= threshold:
            # All remaining blocks have centrality >= threshold
            break
        # Try to remove block i
        safe = True
        for si in block_subs[i]:
            if coverage_count[si] <= 1:
                safe = False
                break
        if safe:
            for si in block_subs[i]:
                coverage_count[si] -= 1
            removed.append(i)
            # Slightly tighten threshold to keep removing more
            threshold = max(threshold, x_sorted[i] * 1.1)

    if not removed:
        return blocks

    # Rebuild the block list without removed blocks
    removed_set = set(removed)
    new_blocks = []
    for i in range(n):
        if i not in removed_set:
            new_blocks.append(blocks[i])
    return new_blocks

def final_sweep(blocks, v, k, t, subset_to_idx):
    """
    Exact redundant-block removal: repeatedly scan blocks in a fixed order
    and remove any block whose removal does not break coverage.
    """
    if not blocks:
        return blocks

    # Build coverage counts
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    block_subs = []
    for block in blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            si = subset_to_idx[sub]
            subs.append(si)
            coverage_count[si] += 1
        block_subs.append(subs)

    n = len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            # Check if block i is redundant
            safe = True
            for si in block_subs[i]:
                if coverage_count[si] <= 1:
                    safe = False
                    break
            if safe:
                for si in block_subs[i]:
                    coverage_count[si] -= 1
                blocks.pop(i)
                block_subs.pop(i)
                n -= 1
                changed = True
                break  # restart scan since indices shifted
    return blocks

def local_search(blocks, v, k, t, subset_to_idx, deadline):
    """
    ILS with exact intersection repair: try to remove blocks that are redundant,
    or swap blocks to maintain validity while reducing size.
    """
    blocks = [b[:] for b in blocks]
    current_count = len(blocks)
    
    # Precompute coverage count for each t-subset
    # coverage_count[i] = number of blocks that cover subset i
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            coverage_count[idx] += 1
    
    # Try to remove redundant blocks (those whose removal doesn't break coverage)
    # We can do this in a random order to escape local minima.
    # Use a list of (block, subs) to avoid recomputing combinations.
    block_subs = []
    for block in blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[sub])
        block_subs.append(subs)

    for _ in range(200):  # Max 200 removal attempts
        if time.time() > deadline - 5.0:
            break
        if current_count <= 131:
            break
        if not blocks:
            break

        # Pick a random block to try removing
        idx_to_try = random.randint(0, len(blocks) - 1)
        block_to_remove = blocks[idx_to_try]
        subs_to_remove = block_subs[idx_to_try]

        # Check if removing this block would leave any t-subset uncovered
        safe_to_remove = True
        for idx in subs_to_remove:
            if coverage_count[idx] <= 1:
                safe_to_remove = False
                break

        if safe_to_remove:
            # Remove the block and update coverage counts
            for idx in subs_to_remove:
                coverage_count[idx] -= 1
            blocks.pop(idx_to_try)
            block_subs.pop(idx_to_try)
            current_count = len(blocks)

    return blocks
