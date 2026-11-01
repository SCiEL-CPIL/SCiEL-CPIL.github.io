import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s

    points = list(range(v))
    
    # Precompute all t-subsets and k-blocks
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    n_b = len(k_blocks)
    
    # Bitmasks for fast subset checks
    # block_mask[bi] is a 64-bit int representing the points in k_blocks[bi]
    block_mask = np.zeros(n_b, dtype=np.uint64)
    for bi, block in enumerate(k_blocks):
        m = 0
        for p in block:
            m |= (1 << p)
        block_mask[bi] = m
        
    # subset_mask[si] is a 64-bit int representing the points in t_subsets[si]
    subset_mask = np.zeros(n_t, dtype=np.uint64)
    for si, sub in enumerate(t_subsets):
        m = 0
        for p in sub:
            m |= (1 << p)
        subset_mask[si] = m
        
    # For each block, list of t-subset indices it covers
    block_cover = []
    for bi, block in enumerate(k_blocks):
        subs = list(itertools.combinations(block, t))
        idx = np.array([t_idx[s] for s in subs], dtype=np.int32)
        block_cover.append(idx)
        
    # For each t-subset, list of block indices that cover it
    tsub_blocks = [[] for _ in range(n_t)]
    for bi, inds in enumerate(block_cover):
        for s in inds:
            tsub_blocks[s].append(bi)
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    def time_ok():
        return time.time() < deadline

    def greedy(seed=42):
        """Randomized Greedy Set Cover."""
        random.seed(seed)
        gains = np.full(n_b, len(block_cover[0]), dtype=np.int32)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        unc_count = n_t
        
        while unc_count > 0:
            if not time_ok():
                return selected, False
            
            # Pick a block with max gain, with random tie-breaking
            max_gain = np.max(gains)
            if max_gain == 0:
                return selected, False
                
            candidates = np.where(gains == max_gain)[0]
            b = int(random.choice(candidates))
            
            selected.append(b)
            inds = block_cover[b]
            
            # Update uncovered and gains
            unique = inds[uncovered[inds]]
            if len(unique) > 0:
                unc_count -= len(unique)
                uncovered[unique] = False
                
                # Decrement gains for all blocks covering these subsets
                to_dec = []
                for s in unique:
                    to_dec.append(tsub_blocks_np[s])
                if to_dec:
                    flat = np.concatenate(to_dec)
                    np.add.at(gains, flat, -1)
                    
        return selected, True

    def compute_cover_count(selected):
        cc = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cc[block_cover[b]] += 1
        return cc

    def local_search(selected, seed=42):
        """Local Search with Redundancy Removal and Swaps."""
        random.seed(seed)
        cc = compute_cover_count(selected)
        improved = True
        
        while improved and time_ok():
            improved = False
            
            # 1. Redundancy Removal
            redundant = []
            for b in selected:
                if np.min(cc[block_cover[b]]) > 1:
                    redundant.append(b)
            if redundant:
                for b in redundant:
                    selected.remove(b)
                    cc[block_cover[b]] -= 1
                improved = True
                continue
            
            # 2. Compute unique subsets per block
            uniq = {}
            for b in selected:
                subs = block_cover[b]
                u = [s for s in subs if cc[s] == 1]
                uniq[b] = u
            
            # 3. 1-for-1 Swap
            # Find block with min unique subsets, try to replace it
            min_u = min(len(u) for u in uniq.values()) if uniq else 0
            if min_u > 0:
                cands_out = [b for b, u in uniq.items() if len(u) == min_u]
                random.shuffle(cands_out)
                swap_done = False
                for b_out in cands_out:
                    if not time_ok():
                        break
                    s_uniq = uniq[b_out]
                    if not s_uniq:
                        continue
                    first_s = s_uniq[0]
                    cands_in = tsub_blocks_np[first_s]
                    exclude = set(selected)
                    s_masks = [subset_mask[s] for s in s_uniq]
                    
                    valid = []
                    for c in cands_in:
                        c = int(c)
                        if c in exclude:
                            continue
                        cm = block_mask[c]
                        # Check if block c covers all unique subsets of b_out
                        ok = all((cm & sm) == sm for sm in s_masks)
                        if ok:
                            valid.append(c)
                            
                    if valid:
                        b_in = random.choice(valid)
                        selected.remove(b_out)
                        selected.append(b_in)
                        cc[block_cover[b_out]] -= 1
                        cc[block_cover[b_in]] += 1
                        swap_done = True
                        improved = True
                        break
                if swap_done:
                    continue

            # 4. 2-for-1 Merge
            # Try to replace two blocks with one
            if len(selected) > 2 and time_ok():
                # Limit to blocks with small unique sets to keep it fast
                small = [b for b in selected if len(uniq[b]) <= 10]
                random.shuffle(small)
                merge_done = False
                # Try a limited number of pairs
                for i in range(min(50, len(small) - 1)):
                    if not time_ok():
                        break
                    for j in range(i + 1, min(i + 10, len(small))):
                        b1, b2 = small[i], small[j]
                        u1 = uniq[b1]
                        u2 = uniq[b2]
                        combined = set(u1) | set(u2)
                        if len(combined) == 0:
                            continue
                        
                        # Find a block c that covers all subsets in combined
                        first_s = next(iter(combined))
                        cands = tsub_blocks_np[first_s]
                        exclude = {b1, b2}
                        c_masks = [subset_mask[s] for s in combined]
                        
                        found = False
                        for c in cands:
                            c = int(c)
                            if c in exclude:
                                continue
                            cm = block_mask[c]
                            if all((cm & sm) == sm for sm in c_masks):
                                # Perform merge
                                selected.remove(b1)
                                selected.remove(b2)
                                selected.append(c)
                                cc[block_cover[b1]] -= 1
                                cc[block_cover[b2]] -= 1
                                cc[block_cover[c]] += 1
                                merge_done = True
                                found = True
                                break
                        if found:
                            improved = True
                            break
                    if merge_done:
                        break
                if merge_done:
                    continue

        return selected

    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    si = 0
    while time_ok():
        seed = seeds[si % len(seeds)]
        si += 1
        
        selected, completed = greedy(seed)
        if not completed:
            continue
            
        selected = local_search(selected, seed)
        
        cur_len = len(selected)
        if cur_len < best_len:
            best_len = cur_len
            best_blocks = [list(k_blocks[i]) for i in selected]
            
        # If we have a very good solution, we can stop early if time is tight
        if time.time() > deadline - 5.0:
            break

    # Fallback if no solution found
    if best_blocks is None:
        selected, _ = greedy(42)
        best_blocks = [list(k_blocks[i]) for i in selected]

    return best_blocks
