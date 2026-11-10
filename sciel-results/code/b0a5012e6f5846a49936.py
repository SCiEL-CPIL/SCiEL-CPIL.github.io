"""Covering design C(19,8,5) using vectorized greedy with bottleneck heuristic and local search."""
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 118.0

def construct(v, k, t):
    start_time = time.time()
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block: the list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_index[t_sub])
        block_t_indices.append(indices)
    
    # Precompute bitsets for each block (using Python ints for fast AND operations)
    block_bitsets = [0] * n_blocks
    for bi in range(n_blocks):
        mask = 0
        for idx in block_t_indices[bi]:
            mask |= (1 << idx)
        block_bitsets[bi] = mask
    
    full_mask = (1 << n_t) - 1
    
    # For each t-subset, precompute which blocks cover it
    blocks_for_subset = [[] for _ in range(n_t)]
    for bi in range(n_blocks):
        for idx in block_t_indices[bi]:
            blocks_for_subset[idx].append(bi)
    
    # Precompute point incidence: for each point, which t-subsets contain it
    point_to_tsubsets = [[] for _ in range(v)]
    for i, t_sub in enumerate(all_t_subsets):
        for p in t_sub:
            point_to_tsubsets[p].append(i)
    
    def compute_counts(block_indices):
        """Compute coverage counts for each t-subset."""
        counts = np.zeros(n_t, dtype=np.int32)
        for bi in block_indices:
            indices = block_t_indices[bi]
            if indices:
                counts[indices] += 1
        return counts
    
    def prune(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return block_indices
        
        counts = compute_counts(block_indices)
        if np.any(counts == 0):
            return list(block_indices)
        
        result = []
        rng = random.Random()
        order = list(range(len(block_indices)))
        rng.shuffle(order)
        
        for pos in order:
            bi = block_indices[pos]
            indices = block_t_indices[bi]
            if indices and np.any(counts[indices] == 1):
                result.append(bi)
            else:
                if indices:
                    counts[indices] -= 1
        
        return result
    
    def run_greedy(seed, mode='gain'):
        """Greedy set cover with different scoring modes."""
        rng = random.Random(seed)
        uncovered = full_mask
        blocks_used = []
        
        while uncovered:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                return None
            
            best_score = -1
            best_blocks = []
            
            for bi in range(n_blocks):
                mask = block_bitsets[bi]
                common = mask & uncovered
                if common == 0:
                    continue
                
                gain = common.bit_count()
                
                if mode == 'gain':
                    score = gain
                elif mode == 'bottleneck':
                    # Find min degree among uncovered subsets this block covers
                    min_deg = float('inf')
                    c = common
                    while c:
                        lsb = c & (-c)
                        idx = lsb.bit_length() - 1
                        deg = len(blocks_for_subset[idx])
                        if deg < min_deg:
                            min_deg = deg
                        c ^= lsb
                    score = gain * 1000 - min_deg
                elif mode == 'soft_bottleneck':
                    # Use a softer measure: sum of inverse degrees
                    inv_sum = 0.0
                    c = common
                    while c:
                        lsb = c & (-c)
                        idx = lsb.bit_length() - 1
                        deg = len(blocks_for_subset[idx])
                        inv_sum += 1.0 / deg
                        c ^= lsb
                    score = gain * 100 + inv_sum
                else:
                    score = gain
                
                if score > best_score:
                    best_score = score
                    best_blocks = [bi]
                elif score == best_score:
                    best_blocks.append(bi)
            
            if not best_blocks or best_score <= 0:
                break
            
            chosen_bi = rng.choice(best_blocks)
            blocks_used.append(chosen_bi)
            uncovered &= ~block_bitsets[chosen_bi]
        
        return blocks_used
    
    def local_improve(block_indices, seed):
        """Local search: remove redundant blocks and try replacements."""
        rng = random.Random(seed)
        block_indices = list(block_indices)
        block_indices = prune(block_indices)
        
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
            
            improved = False
            counts = compute_counts(block_indices)
            
            # Find removable blocks
            removable = []
            for bi in block_indices:
                indices = block_t_indices[bi]
                if indices and np.all(counts[indices] >= 2):
                    removable.append(bi)
            
            if removable:
                bi_to_remove = rng.choice(removable)
                block_indices.remove(bi_to_remove)
                improved = True
                continue
            
            # Try 1-for-2 replacement: remove two blocks, add one
            if len(block_indices) >= 3:
                # Pick two blocks to try removing
                pair_candidates = list(itertools.combinations(range(len(block_indices)), 2))
                rng.shuffle(pair_candidates)
                
                for pos1, pos2 in pair_candidates[:50]:  # Limit attempts
                    bi1 = block_indices[pos1]
                    bi2 = block_indices[pos2]
                    
                    # Check if removing both leaves any t-subset uncovered
                    counts_test = compute_counts(block_indices)
                    for idx in block_t_indices[bi1]:
                        counts_test[idx] -= 1
                    for idx in block_t_indices[bi2]:
                        counts_test[idx] -= 1
                    
                    if np.any(counts_test == 0):
                        continue
                    
                    # Find blocks that cover all the t-subsets that would be uncovered
                    # (i.e., t-subsets with count exactly 1 after removal)
                    uncovered_after = np.where(counts_test == 0)[0]
                    if len(uncovered_after) > 0:
                        # Need a block that covers all these
                        candidate_blocks = set(range(n_blocks))
                        for idx in uncovered_after:
                            candidate_blocks &= set(blocks_for_subset[idx])
                        if not candidate_blocks:
                            continue
                        
                        candidate_blocks -= set(block_indices)
                        if candidate_blocks:
                            new_bi = rng.choice(list(candidate_blocks))
                            # Verify the new block covers everything needed
                            new_mask = block_bitsets[new_bi]
                            all_covered = True
                            for idx in uncovered_after:
                                if not (new_mask >> idx) & 1:
                                    all_covered = False
                                    break
                            if all_covered:
                                block_indices.remove(bi1)
                                block_indices.remove(bi2)
                                block_indices.append(new_bi)
                                improved = True
                                break
            
            break
        
        return block_indices
    
    best_solution = None
    best_size = float('inf')
    
    modes = ['gain', 'bottleneck', 'soft_bottleneck']
    seed = 42
    
    while time.time() - start_time < DEADLINE_SECONDS - 10:
        mode = modes[seed % len(modes)]
        sol = run_greedy(seed, mode)
        if sol:
            sol = local_improve(sol, seed)
            sol = prune(sol)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
        
        seed += 1
    
    if best_solution is None:
        sol = run_greedy(0, 'gain')
        if sol:
            sol = prune(sol)
            best_solution = sol
    
    if best_solution is None:
        return []
    
    return [list(all_blocks[bi]) for bi in best_solution]
