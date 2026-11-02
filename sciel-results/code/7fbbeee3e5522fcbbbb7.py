import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute triples for all possible blocks
    # This might take a few seconds but speeds up SA immensely
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    num_blocks = len(all_blocks)
    
    # block_triples[i] is a list of t-subset indices contained in block i
    block_triples = [None] * num_blocks
    for i, b in enumerate(all_blocks):
        block_triples[i] = [t_subset_to_idx[tuple(sorted(trip))] for trip in itertools.combinations(b, t)]

    def get_block_triples(block):
        return block_triples[block_to_idx[tuple(block)]]

    def greedy_cover():
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        while np.any(uncovered):
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0: break
            start_idx = random.choice(unc_idx)
            start_trip = all_t_subsets[start_idx]
            
            block = list(start_trip)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block: continue
                    new_block = block + [x]
                    gain = 0
                    for trip in itertools.combinations(new_block, t):
                        idx = t_subset_to_idx[tuple(sorted(trip))]
                        if uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            for idx in get_block_triples(block):
                uncovered[idx] = False
        
        return blocks

    def run_sa(n_blocks, time_limit):
        # Initialize with random blocks
        blocks = [random.choice(all_blocks) for _ in range(n_blocks)]
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for idx in get_block_triples(b):
                cover_count[idx] += 1
        uncovered_count = int(np.sum(cover_count == 0))
        
        best_blocks = [b[:] for b in blocks]
        best_uncovered = uncovered_count
        if best_uncovered == 0:
            return best_blocks
            
        T = 1.0
        cooling = 0.9995
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < 1e-5:
                break
                
            i = random.randint(0, n_blocks - 1)
            old_block = blocks[i]
            
            # Generate new block
            if random.random() < 0.7:
                # Swap a point
                new_block = list(old_block)
                idx_in = random.randint(0, k - 1)
                old_point = new_block[idx_in]
                while True:
                    new_point = random.randint(0, v - 1)
                    if new_point not in new_block:
                        break
                new_block[idx_in] = new_point
                new_block.sort()
            else:
                new_block = random.choice(all_blocks)
            
            if new_block == old_block:
                continue
                
            prev_uncovered = uncovered_count
            
            old_triples = get_block_triples(old_block)
            new_triples = get_block_triples(new_block)
            
            for idx in old_triples:
                cover_count[idx] -= 1
                if cover_count[idx] == 0:
                    uncovered_count += 1
            
            for idx in new_triples:
                if cover_count[idx] == 0:
                    uncovered_count -= 1
                cover_count[idx] += 1
            
            delta = uncovered_count - prev_uncovered
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                blocks[i] = new_block
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best_blocks = [b[:] for b in blocks]
                    if best_uncovered == 0:
                        return best_blocks
            else:
                # Rollback
                for idx in new_triples:
                    if cover_count[idx] == 0:
                        uncovered_count += 1
                    cover_count[idx] -= 1
                for idx in old_triples:
                    if cover_count[idx] == 0:
                        uncovered_count -= 1
                    cover_count[idx] += 1
                    
            T *= cooling
            
        return None

    start_time = time.time()
    time_limit = 110.0
    
    initial_blocks = greedy_cover()
    n_initial = len(initial_blocks)
    best_valid_blocks = initial_blocks
    
    steps = n_initial - 69
    if steps <= 0:
        return best_valid_blocks
        
    time_per_step = max(0.5, (time_limit - 5.0) / steps)
    
    for n in range(n_initial - 1, 68, -1):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        result = run_sa(n, time_per_step)
        if result is not None:
            best_valid_blocks = result
            # Continue to try lower n
            
    return best_valid_blocks
