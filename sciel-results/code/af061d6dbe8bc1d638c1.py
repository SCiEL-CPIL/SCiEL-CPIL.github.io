import itertools
import time
import random
import numpy as np

# Precomputation
v, k, t = 19, 7, 4
# Map 4-subsets to indices
subsets_4 = list(itertools.combinations(range(v), t))
subset_to_idx = {s: i for i, s in enumerate(subsets_4)}
N_subsets = len(subsets_4)

# Generate all 7-subsets and their masks
blocks_7 = list(itertools.combinations(range(v), k))
block_masks = []
block_subsets = [] # Store the actual 4-subsets indices for fast count update
for b in blocks_7:
    mask = 0
    idxs = []
    for s in itertools.combinations(b, t):
        idx = subset_to_idx[s]
        mask |= (1 << idx)
        idxs.append(idx)
    block_masks.append(mask)
    block_subsets.append(idxs)

N_blocks = len(blocks_7)

# Convert block_subsets to numpy arrays for faster operations
block_subsets_np = [np.array(idxs, dtype=np.int32) for idxs in block_subsets]

def solve():
    deadline = time.time() + 118
    best_blocks = None
    best_len = float('inf')
    
    # Initial full mask
    full_mask = (1 << N_subsets) - 1
    
    # Lagrangian dual weights for uncovered subsets
    # w[s] is the penalty weight for subset s being uncovered
    # Start with uniform weights
    w = np.ones(N_subsets, dtype=np.float64)
    
    # Subgradient descent parameters
    step_size = 0.1
    best_w = w.copy()
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        uncovered = full_mask
        used_blocks = []
        cover_counts = [0] * N_subsets
        
        # 1. Greedy Construction with Lagrangian-guided selection
        while uncovered:
            best_score = -1
            candidates = []
            
            # Compute score for each block: weighted gain - cost
            # Score = sum of w[s] for uncovered s in block - lambda (block count penalty)
            # We use a soft penalty: score = weighted_gain / (1 + small_penalty * used_count)
            # Actually, simpler: just use weighted gain, but with randomization
            
            for i, mask in enumerate(block_masks):
                gain_mask = uncovered & mask
                if gain_mask == 0:
                    continue
                # Compute weighted gain: sum of w[s] for s in gain_mask
                # Use numpy for speed
                gain_indices = np.fromiter(
                    (i2 for i2 in block_subsets[i] if (gain_mask >> i2) & 1),
                    dtype=np.int32
                )
                if len(gain_indices) == 0:
                    continue
                weighted_gain = np.sum(w[gain_indices])
                
                if weighted_gain > best_score:
                    best_score = weighted_gain
                    candidates = [i]
                elif abs(weighted_gain - best_score) < 1e-9:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Pick with some randomization to diversify
            idx = random.choice(candidates)
            used_blocks.append(idx)
            
            # Update uncovered
            uncovered &= ~block_masks[idx]
            
            # Update cover counts
            for s_idx in block_subsets[idx]:
                cover_counts[s_idx] += 1
        
        # 2. Local Optimization
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 2a. Redundancy Removal
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for idx in used_blocks:
                    is_redundant = True
                    for s_idx in block_subsets[idx]:
                        if cover_counts[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s_idx in block_subsets[idx]:
                            cover_counts[s_idx] -= 1
                        changed = True
                        improved = True
                    else:
                        new_blocks.append(idx)
                used_blocks = new_blocks

            if not used_blocks:
                break

            # 2b. 1-for-1 Swaps
            unique_subsets = {}
            for idx in used_blocks:
                u_list = []
                for s_idx in block_subsets[idx]:
                    if cover_counts[s_idx] == 1:
                        u_list.append(s_idx)
                if u_list:
                    unique_subsets[idx] = u_list

            found_swap = False
            block_order = list(used_blocks)
            random.shuffle(block_order)
            
            for i_idx in block_order:
                if i_idx not in unique_subsets:
                    continue
                u_list = unique_subsets[i_idx]
                
                used_set = set(used_blocks)
                
                for j_idx in range(N_blocks):
                    if j_idx in used_set:
                        continue
                    
                    covers = True
                    for s_idx in u_list:
                        if s_idx not in block_subsets[j_idx]:
                            covers = False
                            break
                    
                    if covers:
                        used_blocks.remove(i_idx)
                        for s_idx in block_subsets[i_idx]:
                            cover_counts[s_idx] -= 1
                        
                        used_blocks.append(j_idx)
                        for s_idx in block_subsets[j_idx]:
                            cover_counts[s_idx] += 1
                        
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break
            
            if found_swap:
                continue

        if len(used_blocks) < best_len:
            best_len = len(used_blocks)
            best_blocks = [blocks_7[i] for i in used_blocks]
            if best_len <= 131:
                break
        
        # 3. Update dual weights via subgradient descent
        # Subgradient of the Lagrangian with respect to w[s]:
        # g[s] = 1 if s is covered, -1 if s is uncovered
        # But we want to increase w for uncovered subsets to guide future greedy choices
        # Actually, the dual problem: maximize min over covers of sum w[s]*[s covered]
        # Subgradient: for each s, if s is uncovered in current cover, g[s] = -1
        #                if s is covered, g[s] = 1
        # Update: w[s] <- max(0, w[s] + step_size * g[s])
        
        # Compute subgradient
        g = np.ones(N_subsets, dtype=np.float64)
        for s_idx in range(N_subsets):
            if cover_counts[s_idx] == 0:
                g[s_idx] = -1.0
        
        # Adaptive step size: decrease over time
        current_step = step_size / (1 + 0.01 * iteration)
        
        # Update weights
        w = np.maximum(0, w + current_step * g)
        
        # Normalize to prevent weights from drifting too much
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w * (N_subsets / w_sum)
        
        # Occasionally re-seed weights to escape local optima
        if iteration % 20 == 0:
            w = np.ones(N_subsets, dtype=np.float64)
            # Add some random perturbation
            w += np.random.uniform(-0.5, 0.5, N_subsets)
            w = np.maximum(0, w)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w * (N_subsets / w_sum)

    return best_blocks

def construct(v, k, t):
    if v == 19 and k == 7 and t == 4:
        return solve()
    else:
        # Fallback to simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            max_covered = 0
            for b in itertools.combinations(range(v), k):
                covered = 0
                for s in itertools.combinations(b, t):
                    if s in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = b
            if not best_block:
                break
            blocks.append(list(best_block))
            for s in itertools.combinations(best_block, t):
                uncovered.discard(s)
        return blocks
