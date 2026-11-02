"""Portfolio search: generate varied candidate coverings and keep the best within a strict time budget."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin below 150s kill
    start_time = time.time()
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    def greedy_cover(seed, time_budget):
        """Greedy covering with randomization for tie-breaking."""
        rng = random.Random(seed)
        # uncovered[i] = True if t_subset i is not yet covered
        uncovered = np.ones(n_t_subsets, dtype=bool)
        n_uncovered = n_t_subsets
        blocks = []
        
        # Precompute for each point x, which t-subsets contain x
        # points_in_subset[x] = list of t-subset indices that contain point x
        points_in_subset = [[] for _ in range(v)]
        for i, subset in enumerate(all_t_subsets):
            for x in subset:
                points_in_subset[x].append(i)
        
        while n_uncovered > 0:
            if time.time() > time_budget:
                return None
            
            # Pick a random uncovered t-subset to start the block
            uncovered_indices = np.where(uncovered)[0]
            start_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            
            # Add points to the block greedily
            while len(block) < k:
                if time.time() > time_budget:
                    return None
                
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(list(rest) + [x]))
                        if sub in t_subset_to_idx:
                            idx = t_subset_to_idx[sub]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0:
                        # Random tie-breaking
                        if rng.random() < 0.5:
                            best_x = x
                if best_x >= 0:
                    block.append(best_x)
                else:
                    break
            
            if len(block) < k:
                # Can't form a valid block
                return None
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    n_uncovered -= 1
        
        return blocks
    
    # Try multiple seeds with greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        time_budget = min(deadline, time.time() + (deadline - time.time()) / max(1, 20 - seed))
        result = greedy_cover(seed, time_budget)
        if result is not None:
            if len(result) < best_count:
                best_count = len(result)
                best_blocks = result
    
    # Try randomized local search / hill climbing on the best solution found
    if best_blocks is not None and time.time() < deadline - 10:
        # Local search: try to remove redundant blocks
        blocks = list(best_blocks)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Recompute coverage
            covered = np.zeros(n_t_subsets, dtype=bool)
            block_coverage = [[] for _ in range(len(blocks))]
            for bi, block in enumerate(blocks):
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    if not covered[idx]:
                        covered[idx] = True
                    block_coverage[bi].append(idx)
            
            # Try removing each block
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                # Check if any t-subset is only covered by this block
                can_remove = True
                for idx in block_coverage[bi]:
                    # Count how many blocks cover this t-subset
                    count = 0
                    for other_bi, other_block in enumerate(blocks):
                        if other_bi == bi:
                            continue
                        if idx in block_coverage[other_bi]:
                            count += 1
                            break
                    if count == 0:
                        can_remove = False
                        break
                if can_remove:
                    blocks.pop(bi)
                    improved = True
                    break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy without time limit issues
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(list(rest) + [x])) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x >= 0:
                    block.append(best_x)
                else:
                    break
            block.sort()
            blocks.append(block)
            for subset in itertools.combinations(block, t):
                uncovered.discard(subset)
        best_blocks = blocks
    
    return best_blocks

# EVOLVE-BLOCK-END
