"""Optimized covering design search for C(24,7,3).

Strategy:
1. Fast greedy construction with vectorized gain computation (multiple seeds).
2. Fast reduction (remove redundant blocks).
3. Local search: try to replace blocks with better ones.
"""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute: for each point x and each (t-1)-subset of other points,
    # which t-subset index would be covered if x is added
    # point_t1_map[x][(t-1)-tuple] = t-subset index
    point_t1_map = [{} for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for i in range(t):
            rest = tuple(sorted(subset[:i] + subset[i+1:]))
            p = subset[i]
            point_t1_map[p][rest] = idx
    
    # Precompute all (t-1)-subsets for each possible block size
    # For a block of size s, we need all (t-1)-subsets of that block
    def get_t1_subsets(block):
        """Get all (t-1)-subsets of block as sorted tuples."""
        return list(itertools.combinations(block, t-1))
    
    def greedy_cover(seed):
        """Fast greedy with randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        blocks = []
        
        # Maintain a list of uncovered t-subsets for random selection
        uncovered_list = list(range(n_t_subsets))
        idx_to_pos = {i: i for i in range(n_t_subsets)}
        
        while uncovered_count > 0:
            if time.time() > deadline - 15:
                return None
            
            # Pick a random uncovered t-subset
            pos = rng.randint(0, len(uncovered_list) - 1)
            start_idx = uncovered_list[pos]
            
            # Remove from tracking structures
            last_idx = uncovered_list[-1]
            uncovered_list[pos] = last_idx
            idx_to_pos[last_idx] = pos
            uncovered_list.pop()
            del idx_to_pos[start_idx]
            uncovered[start_idx] = False
            uncovered_count -= 1
            
            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            block_set = set(block)
            
            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 15:
                    return None
                
                best_x = -1
                best_gain = -1
                
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    # Compute gain: how many uncovered t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        rest_sorted = tuple(sorted(rest))
                        idx = point_t1_map[x].get(rest_sorted, -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0 and rng.random() < 0.3:
                        best_x = x
                
                if best_x >= 0:
                    block.append(best_x)
                    block_set.add(best_x)
                else:
                    break
            
            if len(block) < k:
                return None
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered: remove all t-subsets of the new block
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    if idx in idx_to_pos:
                        p = idx_to_pos[idx]
                        last_idx = uncovered_list[-1]
                        uncovered_list[p] = last_idx
                        idx_to_pos[last_idx] = p
                        uncovered_list.pop()
                        del idx_to_pos[idx]
        
        return blocks

    def reduce_blocks(blocks):
        """Remove redundant blocks efficiently."""
        if not blocks:
            return []
        
        blocks = list(blocks)
        if len(blocks) == 0:
            return blocks
        
        # Compute coverage counts
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        for block in blocks:
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                coverage_count[idx] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            
            # Try removing each block (from end to start)
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                
                block = blocks[bi]
                can_remove = True
                affected = []
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] -= 1
                    affected.append(idx)
                    if coverage_count[idx] == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(bi)
                    improved = True
                    break
                else:
                    for a_idx in affected:
                        coverage_count[a_idx] += 1
        
        return blocks

    def local_search(blocks, time_budget=30.0):
        """Try to improve by replacing blocks."""
        if time.time() > deadline - time_budget:
            return blocks
        
        blocks = list(blocks)
        rng = random.Random(42)
        
        # Precompute coverage
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        for block in blocks:
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                coverage_count[idx] += 1
        
        improved = True
        local_deadline = time.time() + time_budget
        
        while improved and time.time() < local_deadline:
            improved = False
            
            for bi in range(len(blocks)):
                if time.time() > local_deadline:
                    break
                
                # Try to find a better block to replace blocks[bi]
                block = blocks[bi]
                
                # Compute what this block covers uniquely
                unique_coverage = []
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    if coverage_count[idx] == 1:
                        unique_coverage.append(idx)
                
                if not unique_coverage:
                    # Can remove this block
                    for subset in itertools.combinations(block, t):
                        idx = t_subset_to_idx[subset]
                        coverage_count[idx] -= 1
                    blocks.pop(bi)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all unique subsets
                # and potentially more
                best_new_block = None
                best_extra_gain = 0
                
                # Generate candidate blocks that contain all points of some 
                # (t-1)-subset from the unique coverage
                candidate_starts = set()
                for idx in unique_coverage:
                    subset = all_t_subsets[idx]
                    for i in range(t-1):
                        rest = tuple(sorted(subset[:i] + subset[i+1:]))
                        candidate_starts.add(rest)
                
                # Limit candidates for speed
                if len(candidate_starts) > 50:
                    candidate_starts = list(candidate_starts)[:50]
                
                for rest in candidate_starts:
                    # For each point not in rest, try adding it
                    rest_set = set(rest)
                    for x in range(v):
                        if x in rest_set:
                            continue
                        
                        # Build a candidate block containing rest + x
                        cand_block = list(rest) + [x]
                        cand_set = set(cand_block)
                        
                        # Expand to size k greedily
                        while len(cand_block) < k:
                            best_y = -1
                            best_g = -1
                            for y in range(v):
                                if y in cand_set:
                                    continue
                                g = 0
                                for r2 in itertools.combinations(cand_block, t-1):
                                    r2_sorted = tuple(sorted(r2))
                                    idx2 = point_t1_map[y].get(r2_sorted, -1)
                                    if idx2 != -1 and coverage_count[idx2] <= 1:
                                        g += 1
                                if g > best_g:
                                    best_g = g
                                    best_y = y
                            if best_y >= 0:
                                cand_block.append(best_y)
                                cand_set.add(best_y)
                            else:
                                break
                        
                        if len(cand_block) < k:
                            continue
                        
                        cand_block.sort()
                        
                        # Check if this covers all unique subsets
                        covers_all = True
                        for idx in unique_coverage:
                            subset = all_t_subsets[idx]
                            if not set(subset).issubset(set(cand_block)):
                                covers_all = False
                                break
                        
                        if not covers_all:
                            continue
                        
                        # Count extra coverage
                        extra = 0
                        for subset in itertools.combinations(cand_block, t):
                            idx = t_subset_to_idx[subset]
                            if coverage_count[idx] == 0:
                                extra += 1
                        
                        if extra > best_extra_gain:
                            best_extra_gain = extra
                            best_new_block = cand_block
                
                if best_new_block is not None and best_extra_gain > 0:
                    # Replace
                    for subset in itertools.combinations(block, t):
                        idx = t_subset_to_idx[subset]
                        coverage_count[idx] -= 1
                    for subset in itertools.combinations(best_new_block, t):
                        idx = t_subset_to_idx[subset]
                        coverage_count[idx] += 1
                    blocks[bi] = best_new_block
                    improved = True
                    break
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds for greedy construction
    seed = 0
    greedy_deadline = deadline - 40  # Leave time for reduction and local search
    while time.time() < greedy_deadline:
        result = greedy_cover(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
        seed += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        uncovered_list = list(range(n_t_subsets))
        idx_to_pos = {i: i for i in range(n_t_subsets)}
        blocks = []
        
        while uncovered_count > 0:
            start_idx = uncovered_list[0]
            
            last_idx = uncovered_list[-1]
            uncovered_list[0] = last_idx
            idx_to_pos[last_idx] = 0
            uncovered_list.pop()
            del idx_to_pos[start_idx]
            uncovered[start_idx] = False
            uncovered_count -= 1
            
            block = list(all_t_subsets[start_idx])
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        rest_sorted = tuple(sorted(rest))
                        idx = point_t1_map[x].get(rest_sorted, -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x >= 0:
                    block.append(best_x)
                    block_set.add(best_x)
                else:
                    break
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    if idx in idx_to_pos:
                        p = idx_to_pos[idx]
                        last_idx = uncovered_list[-1]
                        uncovered_list[p] = last_idx
                        idx_to_pos[last_idx] = p
                        uncovered_list.pop()
                        del idx_to_pos[idx]
        
        best_blocks = blocks
    
    # Apply local search to the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        local_budget = min(30.0, deadline - 5 - time.time())
        if local_budget > 5:
            improved = local_search(best_blocks, local_budget)
            if len(improved) < len(best_blocks):
                best_blocks = improved
    
    return best_blocks

# EVOLVE-BLOCK-END
