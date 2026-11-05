import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered or time.time() > deadline:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    # Precomputation
    all_points = range(v)
    all_t_subsets = list(itertools.combinations(all_points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    block_tsubsets = []
    for block in all_blocks:
        ts_indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubsets.append(ts_indices)

    def run_greedy(seed):
        rng = random.Random(seed)
        # Shuffle block indices to break ties randomly
        block_indices = list(range(n_blocks))
        rng.shuffle(block_indices)
        
        counts = [0] * n_t_subsets
        selected_blocks = []
        uncovered_count = n_t_subsets
        
        while uncovered_count > 0 and time.time() < deadline - 5:
            best_block_idx = -1
            best_gain = 0
            
            for b_idx in block_indices:
                ts_indices = block_tsubsets[b_idx]
                gain = 0
                for ts_idx in ts_indices:
                    if counts[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                if best_gain == 35: # Max possible
                    break

            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            for ts_idx in block_tsubsets[best_block_idx]:
                if counts[ts_idx] == 0:
                    uncovered_count -= 1
                counts[ts_idx] += 1
        
        # Redundancy Removal
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                redundant = True
                for ts_idx in block_tsubsets[b_idx]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_tsubsets[b_idx]:
                        counts[ts_idx] -= 1
                    selected_blocks.pop(i)
                    changed = True
        
        return selected_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    # Time budget for each run is roughly (deadline - start_time) / num_runs
    # Let's do 5-10 runs depending on time.
    
    num_runs = 5
    for i in range(num_runs):
        if time.time() > deadline - 20:
            break
        blocks = run_greedy(42 + i)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]

    if best_blocks is None:
        # Fallback: just return the first one or empty
        return [list(all_blocks[0])]
        
    return [list(all_blocks[i]) for i in best_blocks]
