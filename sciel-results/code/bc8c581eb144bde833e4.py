import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    rng = random.Random(42)

    def remaining_time():
        return deadline - time.time()

    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast array-based tracking
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    uncovered = np.ones(num_tsubsets, dtype=np.bool_)
    blocks = []
    
    # Precompute t-subset indices for each possible block (lazy cache)
    block_tsub_indices_cache = {}
    
    def get_block_tsub_indices(block):
        """Get array of t-subset indices contained in a block."""
        key = tuple(block)
        if key not in block_tsub_indices_cache:
            indices = []
            for sub in itertools.combinations(block, t):
                idx = tsub_to_idx.get(sub)
                if idx is not None:
                    indices.append(idx)
            block_tsub_indices_cache[key] = np.array(indices, dtype=np.int32)
        return block_tsub_indices_cache[key]
    
    # Track frequency of each point in selected blocks
    point_freq = np.zeros(v, dtype=np.int32)
    
    # Greedy phase with hard-subset prioritization
    while uncovered.any() and remaining_time() > 5.0:
        # Pick a 'hard' uncovered subset: one with points rarely used
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        if len(uncovered_indices) <= 2000:
            # Compute difficulty scores for all uncovered
            best_score = -1
            seed = None
            for idx in uncovered_indices:
                sub = all_tsubsets[idx]
                score = 0.0
                for p in sub:
                    score += 1.0 / (point_freq[p] + 1)
                if score > best_score:
                    best_score = score
                    seed = sub
        else:
            # Sample candidates
            sample_size = min(100, len(uncovered_indices))
            sampled_indices = rng.choice(uncovered_indices, size=sample_size, replace=False)
            best_score = -1
            seed = None
            for idx in sampled_indices:
                sub = all_tsubsets[idx]
                score = 0.0
                for p in sub:
                    score += 1.0 / (point_freq[p] + 1)
                if score > best_score:
                    best_score = score
                    seed = sub
        
        seed_set = set(seed)
        remaining_points = [x for x in range(v) if x not in seed_set]
        num_to_add = k - t
        
        best_gain = -1
        best_block = None
        best_freq_sum = float('inf')
        
        for add_combo in itertools.combinations(remaining_points, num_to_add):
            candidate_block = tuple(sorted(seed + add_combo))
            
            # Compute gain using array operations
            tsub_indices = get_block_tsub_indices(candidate_block)
            gain = uncovered[tsub_indices].sum()
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
                best_freq_sum = sum(point_freq[p] for p in candidate_block)
            elif gain == best_gain:
                freq_sum = sum(point_freq[p] for p in candidate_block)
                if freq_sum < best_freq_sum:
                    best_gain = gain
                    best_block = candidate_block
                    best_freq_sum = freq_sum
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Update uncovered
        tsub_indices = get_block_tsub_indices(best_block)
        uncovered[tsub_indices] = False
        for p in best_block:
            point_freq[p] += 1
    
    # Phase 1: Remove redundant blocks (using coverage counts)
    if remaining_time() > 3.0:
        changed = True
        while changed and remaining_time() > 3.0:
            changed = False
            n = len(blocks)
            if n == 0:
                break
            
            # Build coverage count array
            cover_count = np.zeros(num_tsubsets, dtype=np.int32)
            for block in blocks:
                tsub_indices = get_block_tsub_indices(tuple(block))
                cover_count[tsub_indices] += 1
            
            for i in range(len(blocks)):
                if remaining_time() < 3.0:
                    break
                block_i = tuple(blocks[i])
                tsub_indices_i = get_block_tsub_indices(block_i)
                
                # Check if all t-subsets of block_i are covered by other blocks
                all_covered = True
                for idx in tsub_indices_i:
                    if cover_count[idx] <= 1:
                        all_covered = False
                        break
                
                if all_covered:
                    blocks.pop(i)
                    changed = True
                    break  # Restart since indices changed
    
    # Phase 2: Simulated Annealing
    if remaining_time() > 10.0:
        # State: list of blocks
        # Energy: number of blocks (minimize)
        # Moves: add random block, remove redundant block, swap block
        
        current_blocks = [list(b) for b in blocks]
        best_blocks = [list(b) for b in current_blocks]
        best_energy = len(current_blocks)
        
        # Precompute coverage for current state
        def compute_coverage(block_list):
            cover_count = np.zeros(num_tsubsets, dtype=np.int32)
            for block in block_list:
                tsub_indices = get_block_tsub_indices(tuple(block))
                cover_count[tsub_indices] += 1
            return cover_count
        
        def is_valid(block_list):
            """Check if all t-subsets are covered."""
            cover_count = np.zeros(num_tsubsets, dtype=np.int32)
            for block in block_list:
                tsub_indices = get_block_tsub_indices(tuple(block))
                cover_count[tsub_indices] += 1
            return (cover_count >= 1).all()
        
        def unique_tsubs_for_block(block_idx, block_list, cover_count):
            """Find t-subsets only covered by this block."""
            block = tuple(block_list[block_idx])
            tsub_indices = get_block_tsub_indices(block)
            unique = []
            for idx in tsub_indices:
                if cover_count[idx] == 1:
                    unique.append(idx)
            return unique
        
        cover_count = compute_coverage(current_blocks)
        current_energy = len(current_blocks)
        
        T = 5.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        max_iterations = 5000
        
        iteration = 0
        while T > T_min and remaining_time() > 10.0 and iteration < max_iterations:
            iteration += 1
            move_type = rng.random()
            
            if move_type < 0.3 and len(current_blocks) > 0:
                # Try removing a block
                i = rng.randrange(len(current_blocks))
                block_i = tuple(current_blocks[i])
                tsub_indices_i = get_block_tsub_indices(block_i)
                
                # Check if removal keeps validity
                can_remove = True
                for idx in tsub_indices_i:
                    if cover_count[idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    new_blocks = current_blocks[:i] + current_blocks[i+1:]
                    new_energy = len(new_blocks)
                    delta_e = new_energy - current_energy  # Should be -1
                    
                    # Accept (always improves)
                    current_blocks = new_blocks
                    current_energy = new_energy
                    cover_count = compute_coverage(current_blocks)
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in current_blocks]
                # If can't remove, skip this iteration
                
            elif move_type < 0.5:
                # Try adding a random block
                # Pick a random point and build a block around it
                if v >= k:
                    center = rng.randrange(v)
                    others = [x for x in range(v) if x != center]
                    rng.shuffle(others)
                    new_block = sorted([center] + others[:k-1])
                    new_block_tuple = tuple(new_block)
                    
                    # Check if this block is already present
                    if new_block_tuple in [tuple(b) for b in current_blocks]:
                        T *= cooling_rate
                        continue
                    
                    new_blocks = current_blocks + [list(new_block)]
                    new_energy = len(new_blocks)
                    delta_e = new_energy - current_energy  # Should be +1
                    
                    # Accept with probability exp(-delta_e/T)
                    if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                        current_blocks = new_blocks
                        current_energy = new_energy
                        cover_count = compute_coverage(current_blocks)
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = [list(b) for b in current_blocks]
                
            else:
                # Try swapping a block
                if len(current_blocks) > 0:
                    i = rng.randrange(len(current_blocks))
                    block_i = tuple(current_blocks[i])
                    
                    # Find unique t-subsets for this block
                    unique_tsubs = unique_tsubs_for_block(i, current_blocks, cover_count)
                    
                    if not unique_tsubs:
                        # No unique coverage, can try any replacement
                        must_have = set()
                    else:
                        # Must cover all unique t-subsets
                        must_have = set()
                        for idx in unique_tsubs:
                            must_have.update(all_tsubsets[idx])
                    
                    if len(must_have) > k:
                        T *= cooling_rate
                        continue
                    
                    remaining_for_block = [x for x in range(v) if x not in must_have]
                    num_to_add = k - len(must_have)
                    
                    if num_to_add <= 0:
                        candidate_block = tuple(sorted(must_have))
                    elif num_to_add <= 3:
                        # Try a few random completions
                        found = False
                        for _ in range(20):
                            if num_to_add == 0:
                                add_combo = ()
                            elif num_to_add == 1:
                                add_combo = (rng.choice(remaining_for_block),)
                            elif num_to_add == 2:
                                idxs = rng.sample(range(len(remaining_for_block)), 2)
                                add_combo = tuple(sorted([remaining_for_block[idxs[0]], remaining_for_block[idxs[1]]]))
                            else:
                                idxs = rng.sample(range(len(remaining_for_block)), 3)
                                add_combo = tuple(sorted([remaining_for_block[idxs[0]], remaining_for_block[idxs[1]], remaining_for_block[idxs[2]]]))
                            
                            candidate_block = tuple(sorted(tuple(must_have) + add_combo))
                            
                            # Verify it covers all unique t-subsets
                            valid = True
                            cand_set = set(candidate_block)
                            for idx in unique_tsubs:
                                if not set(all_tsubsets[idx]).issubset(cand_set):
                                    valid = False
                                    break
                            if valid:
                                found = True
                                break
                        
                        if not found:
                            T *= cooling_rate
                            continue
                    else:
                        # Too many to add, skip
                        T *= cooling_rate
                        continue
                    
                    # Check if candidate is same as current
                    if candidate_block == block_i:
                        T *= cooling_rate
                        continue
                    
                    # Check if candidate is already in blocks (other than i)
                    if candidate_block in [tuple(b) for j, b in enumerate(current_blocks) if j != i]:
                        T *= cooling_rate
                        continue
                    
                    # Compute new coverage
                    new_blocks = current_blocks.copy()
                    new_blocks[i] = list(candidate_block)
                    
                    # Check validity
                    new_cover_count = np.zeros(num_tsubsets, dtype=np.int32)
                    for block in new_blocks:
                        tsub_indices = get_block_tsub_indices(tuple(block))
                        new_cover_count[tsub_indices] += 1
                    
                    if (new_cover_count >= 1).all():
                        new_energy = len(new_blocks)
                        delta_e = new_energy - current_energy  # Should be 0
                        
                        # Accept with probability 1 (no energy change) or slight preference
                        if delta_e <= 0 or rng.random() < 0.5:
                            current_blocks = new_blocks
                            current_energy = new_energy
                            cover_count = new_cover_count
                            
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_blocks = [list(b) for b in current_blocks]
            
            T *= cooling_rate
    
        # Use the best found
        if best_energy < len(blocks):
            blocks = best_blocks
    
    # Final cleanup: remove any remaining redundant blocks
    if remaining_time() > 2.0:
        changed = True
        while changed and remaining_time() > 2.0:
            changed = False
            n = len(blocks)
            if n == 0:
                break
            
            cover_count = np.zeros(num_tsubsets, dtype=np.int32)
            for block in blocks:
                tsub_indices = get_block_tsub_indices(tuple(block))
                cover_count[tsub_indices] += 1
            
            for i in range(len(blocks)):
                if remaining_time() < 2.0:
                    break
                block_i = tuple(blocks[i])
                tsub_indices_i = get_block_tsub_indices(block_i)
                
                all_covered = True
                for idx in tsub_indices_i:
                    if cover_count[idx] <= 1:
                        all_covered = False
                        break
                
                if all_covered:
                    blocks.pop(i)
                    changed = True
                    break
    
    return blocks
