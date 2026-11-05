import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148  # 2s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute triplet IDs for each block
    # block_triplets[i] = list of triplet IDs covered by block i
    block_triplets = []
    for b in all_blocks:
        tids = [triplet_to_id[triplet] for triplet in itertools.combinations(b, t)]
        block_triplets.append(tids)
    
    # Precompute for each triplet, which blocks contain it
    # This is a list of lists: triplet_blocks[tid] = list of block indices
    triplet_blocks = [[] for _ in range(n_triplets)]
    for bi, tids in enumerate(block_triplets):
        for tid in tids:
            triplet_blocks[tid].append(bi)
    
    # Convert to numpy arrays for speed
    # block_triplets as list of arrays
    block_triplets_arr = [np.array(tids, dtype=np.int32) for tids in block_triplets]
    
    # Helper to get mask of a block (for greedy)
    def get_block_mask(block_idx):
        mask = 0
        for tid in block_triplets[block_idx]:
            mask |= (1 << tid)
        return mask
    
    # Precompute masks for all blocks
    block_masks = [get_block_mask(i) for i in range(n_blocks)]
    
    # Helper to enumerate blocks containing a set of points
    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return []
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))
    
    # Helper to get block index from a block tuple
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Global Greedy Construction
    def greedy_global(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            for i in range(n_blocks):
                mask = block_masks[i]
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if not best_indices:
                break
                
            idx = random.choice(best_indices)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        return [all_blocks[i] for i in selected_indices]
    
    # Local Greedy Construction
    def greedy_local(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        while uncovered:
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]
            
            best_block = None
            best_gain = -1
            best_mask = 0
            
            for block in blocks_containing(start_triplet):
                bi = block_to_idx.get(block)
                if bi is not None:
                    b_mask = block_masks[bi]
                    gain = (b_mask & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                        best_mask = b_mask
            
            if best_block:
                blocks.append(best_block)
                uncovered &= ~best_mask
            else:
                break
                
        return blocks
    
    # Local Search Optimizer (improved)
    def optimize(blocks, max_time=None):
        if max_time is None:
            max_time = min(30.0, time_left() * 0.5)
        
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def update_counts_add(b):
            for tid in block_triplets[block_to_idx[b]]:
                counts[tid] += 1
        
        def update_counts_remove(b):
            for tid in block_triplets[block_to_idx[b]]:
                counts[tid] -= 1
        
        for b in blocks:
            update_counts_add(b)
        
        max_iters = 10000
        start_opt = time.time()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5 or (time.time() - start_opt) > max_time:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    bi = block_to_idx[b]
                    redundant = True
                    for tid in block_triplets[bi]:
                        if counts[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        update_counts_remove(b)
                        blocks.pop(i)
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                # Only try pairs where at least one block is "wasteful"
                # (has low unique coverage)
                unique_counts = []
                for i in range(n):
                    b = blocks[i]
                    bi = block_to_idx[b]
                    unique = 0
                    for tid in block_triplets[bi]:
                        if counts[tid] == 1:
                            unique += 1
                    unique_counts.append(unique)
                
                # Sort by unique count, try pairs with low unique counts first
                indices_sorted = sorted(range(n), key=lambda i: unique_counts[i])
                tried = set()
                
                for pos1 in range(min(20, n)):
                    for pos2 in range(pos1 + 1, min(40, n)):
                        i = indices_sorted[pos1]
                        j = indices_sorted[pos2]
                        if (i, j) in tried:
                            continue
                        tried.add((i, j))
                        
                        if time.time() > deadline - 10:
                            break
                            
                        b1 = blocks[i]
                        b2 = blocks[j]
                        bi1 = block_to_idx[b1]
                        bi2 = block_to_idx[b2]
                        
                        ids_b1 = set(block_triplets[bi1])
                        ids_b2 = set(block_triplets[bi2])
                        
                        needed_triplets = []
                        all_involved = ids_b1 | ids_b2
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                
                        if not needed_triplets:
                            # Both redundant
                            update_counts_remove(b2)
                            update_counts_remove(b1)
                            blocks.pop(j)
                            blocks.pop(i)
                            improved = True
                            break
                        
                        points_needed = set()
                        for tid in needed_triplets:
                            trip = id_to_triplet[tid]
                            points_needed.update(trip)
                        
                        if len(points_needed) > k:
                            continue
                        
                        found_replacement = False
                        for new_block in blocks_containing(points_needed):
                            if new_block == b1 or new_block == b2:
                                continue
                            
                            nb = block_to_idx.get(new_block)
                            if nb is None:
                                continue
                            
                            new_ids = set(block_triplets[nb])
                            covers = all(tid in new_ids for tid in needed_triplets)
                            
                            if covers:
                                update_counts_remove(b2)
                                update_counts_remove(b1)
                                blocks.pop(j)
                                blocks.pop(i)
                                
                                blocks.append(new_block)
                                update_counts_add(new_block)
                                
                                improved = True
                                found_replacement = True
                                break
                                
                        if improved:
                            break
                            
            # 3. Try 1-for-1
            if not improved:
                i = random.randint(0, n - 1)
                b1 = blocks[i]
                bi1 = block_to_idx[b1]
                ids_b1 = set(block_triplets[bi1])
                
                needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                
                if needed_triplets:
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) <= k:
                        for new_block in blocks_containing(points_needed):
                            if new_block == b1:
                                continue
                            nb = block_to_idx.get(new_block)
                            if nb is None:
                                continue
                            new_ids = set(block_triplets[nb])
                            if all(tid in new_ids for tid in needed_triplets):
                                update_counts_remove(b1)
                                blocks.pop(i)
                                blocks.append(new_block)
                                update_counts_add(new_block)
                                improved = True
                                break
                                
            if not improved:
                break
                
        return blocks
    
    # Simulated Annealing Optimizer (improved with numpy)
    def optimize_sa(blocks, max_time=30.0):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def update_counts_add(b):
            bi = block_to_idx[b]
            for tid in block_triplets[bi]:
                counts[tid] += 1
        
        def update_counts_remove(b):
            bi = block_to_idx[b]
            for tid in block_triplets[bi]:
                counts[tid] -= 1
        
        for b in blocks:
            update_counts_add(b)
        
        def compute_energy(b_list):
            n_blocks = len(b_list)
            redundancy = np.sum(np.maximum(counts - 1, 0))
            return n_blocks + 0.1 * redundancy
        
        current_blocks = list(blocks)
        current_energy = compute_energy(current_blocks)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        T = 5.0
        T_min = 0.01
        cooling = 0.998
        
        start = time.time()
        
        while T > T_min and (time.time() - start) < max_time:
            if time.time() > deadline - 5:
                break
            
            n = len(current_blocks)
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Try to add a random block
                new_block = random.choice(all_blocks)
                if new_block in current_blocks:
                    continue
                nb = block_to_idx[new_block]
                trial_counts = counts.copy()
                for tid in block_triplets[nb]:
                    trial_counts[tid] += 1
                trial_blocks = current_blocks + [new_block]
                trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
                
            elif move_type < 0.65:
                # Try to remove a block
                if n == 0:
                    continue
                idx = random.randint(0, n - 1)
                old_block = current_blocks[idx]
                ob = block_to_idx[old_block]
                trial_counts = counts.copy()
                for tid in block_triplets[ob]:
                    trial_counts[tid] -= 1
                
                if np.min(trial_counts) < 0:
                    continue
                if np.min(trial_counts) == 0:
                    uncovered = np.sum(trial_counts == 0)
                    trial_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0)) + 100 * uncovered
                else:
                    trial_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
                    
            else:
                # Try to swap a block
                if n == 0:
                    continue
                idx = random.randint(0, n - 1)
                old_block = current_blocks[idx]
                new_block = random.choice(all_blocks)
                if new_block == old_block or new_block in current_blocks:
                    continue
                
                ob = block_to_idx[old_block]
                nb = block_to_idx[new_block]
                trial_counts = counts.copy()
                for tid in block_triplets[ob]:
                    trial_counts[tid] -= 1
                for tid in block_triplets[nb]:
                    trial_counts[tid] += 1
                
                if np.min(trial_counts) < 0:
                    continue
                if np.min(trial_counts) == 0:
                    uncovered = np.sum(trial_counts == 0)
                    trial_blocks = current_blocks[:]
                    trial_blocks[idx] = new_block
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0)) + 100 * uncovered
                else:
                    trial_blocks = current_blocks[:]
                    trial_blocks[idx] = new_block
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
            
            delta_E = trial_energy - current_energy
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                current_blocks = trial_blocks
                current_energy = trial_energy
                counts = trial_counts.copy()
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = list(current_blocks)
            
            T *= cooling
            
            # Targeted redundant removal
            if random.random() < 0.15:
                for i in range(len(current_blocks)):
                    b = current_blocks[i]
                    bi = block_to_idx[b]
                    redundant = True
                    for tid in block_triplets[bi]:
                        if counts[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        trial_counts = counts.copy()
                        for tid in block_triplets[bi]:
                            trial_counts[tid] -= 1
                        if np.min(trial_counts) > 0:
                            trial_blocks = current_blocks[:i] + current_blocks[i+1:]
                            trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
                            delta_E = trial_energy - current_energy
                            if delta_E < 0:
                                current_blocks = trial_blocks
                                current_energy = trial_energy
                                counts = trial_counts
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_blocks = list(current_blocks)
                        break
        
        return best_blocks
    
    # Final polish: exhaustive local search
    def final_polish(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def update_add(b):
            bi = block_to_idx[b]
            for tid in block_triplets[bi]:
                counts[tid] += 1
        
        def update_remove(b):
            bi = block_to_idx[b]
            for tid in block_triplets[bi]:
                counts[tid] -= 1
        
        for b in blocks:
            update_add(b)
        
        start_polish = time.time()
        max_polish_time = min(20.0, time_left() * 0.4)
        
        while (time.time() - start_polish) < max_polish_time:
            if time.time() > deadline - 3:
                break
            
            improved = False
            n = len(blocks)
            
            # Exhaustive redundant removal
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    bi = block_to_idx[b]
                    redundant = True
                    for tid in block_triplets[bi]:
                        if counts[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        update_remove(b)
                        blocks.pop(i)
                        improved = True
                        break
            
            # Exhaustive 2-for-1
            if not improved:
                n = len(blocks)
                for i in range(n):
                    if improved:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 5:
                            break
                        
                        b1 = blocks[i]
                        b2 = blocks[j]
                        bi1 = block_to_idx[b1]
                        bi2 = block_to_idx[b2]
                        
                        ids_b1 = set(block_triplets[bi1])
                        ids_b2 = set(block_triplets[bi2])
                        
                        needed = []
                        for tid in ids_b1 | ids_b2:
                            if counts[tid] == 1:
                                needed.append(tid)
                        
                        if not needed:
                            update_remove(b2)
                            update_remove(b1)
                            blocks.pop(j)
                            blocks.pop(i)
                            improved = True
                            break
                        
                        pts = set()
                        for tid in needed:
                            pts.update(id_to_triplet[tid])
                        
                        if len(pts) > k:
                            continue
                        
                        for nb_tuple in blocks_containing(pts):
                            if nb_tuple == b1 or nb_tuple == b2:
                                continue
                            nb = block_to_idx.get(nb_tuple)
                            if nb is None:
                                continue
                            new_ids = set(block_triplets[nb])
                            if all(tid in new_ids for tid in needed):
                                update_remove(b2)
                                update_remove(b1)
                                blocks.pop(j)
                                blocks.pop(i)
                                blocks.append(nb_tuple)
                                update_add(nb_tuple)
                                improved = True
                                break
                        
                        if improved:
                            break
            
            if not improved:
                break
        
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    def verify_and_update(blocks):
        nonlocal best_blocks, best_len
        if not blocks:
            return
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            bi = block_to_idx[b]
            for tid in block_triplets[bi]:
                counts[tid] += 1
        if np.min(counts) > 0:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
    
    # Strategy: Generate diverse candidates, optimize, then SA + polish
    candidates = []
    
    # Local greedy - fast, diverse
    for seed in range(8):
        if time.time() > deadline - 40:
            break
        blocks = greedy_local(seed)
        if blocks:
            candidates.append(blocks)
    
    # Global greedy - slower but potentially better
    for seed in range(3):
        if time.time() > deadline - 30:
            break
        blocks = greedy_global(seed)
        if blocks:
            candidates.append(blocks)
    
    # Optimize all candidates
    optimized = []
    for blocks in candidates:
        if time.time() > deadline - 20:
            break
        opt = optimize(blocks, max_time=min(10.0, time_left() * 0.3))
        optimized.append(opt)
    
    # Sort by length, take top for SA
    optimized.sort(key=lambda x: len(x))
    
    # SA on top candidates
    sa_results = []
    for blocks in optimized[:3]:
        if time.time() > deadline - 15:
            break
        sa = optimize_sa(blocks, max_time=min(20.0, time_left() * 0.4))
        sa_results.append(sa)
    
    # Final polish on best results
    all_final = sa_results + optimized[:2]
    
    for blocks in all_final:
        if time.time() > deadline - 8:
            break
        polished = final_polish(blocks)
        verify_and_update(polished)
    
    # Also verify non-polished
    for blocks in optimized:
        verify_and_update(blocks)
    
    if best_blocks is None:
        # Fallback
        blocks = greedy_local(0)
        best_blocks = blocks
    
    return best_blocks

if __name__ == "__main__":
    pass
