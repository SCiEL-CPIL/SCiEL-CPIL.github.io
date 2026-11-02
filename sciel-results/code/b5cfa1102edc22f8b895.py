import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin for safety

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage: block_coverage[i] is a tuple of t-subset indices covered by block i
    block_coverage = []
    for block in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(block, t):
            covered_indices.append(t_subset_to_idx[ts])
        block_coverage.append(tuple(covered_indices))
        
    # Inverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverage):
        for idx in cov:
            t_to_blocks[idx].append(b_idx)

    # Helper: Calculate counts and uncovered set for a list of block indices
    def get_state(blocks):
        counts = [0] * num_t_subsets
        for b_idx in blocks:
            for idx in block_coverage[b_idx]:
                counts[idx] += 1
        uncovered = set()
        for i in range(num_t_subsets):
            if counts[i] == 0:
                uncovered.add(i)
        return counts, uncovered

    # Helper: Greedy Set Cover starting from an initial set of blocks
    # Uses "Least Constraints First" heuristic: pick the uncovered subset with fewest candidates
    def greedy_repair(initial_blocks):
        counts, uncovered = get_state(initial_blocks)
        current_blocks = list(initial_blocks)
        
        while uncovered:
            # Find the uncovered subset with the minimum number of candidate blocks
            min_subset = -1
            min_count = float('inf')
            for u in uncovered:
                cnt = len(t_to_blocks[u])
                if cnt < min_count:
                    min_count = cnt
                    min_subset = u
                if min_count == 1:
                    break
            
            if min_subset == -1:
                break # No candidates? Should not happen if covering exists
                
            candidates = t_to_blocks[min_subset]
            
            # Among candidates, find the one with max gain over ALL uncovered
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                gain = 0
                for idx in block_coverage[b_idx]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                break
                
            current_blocks.append(best_block_idx)
            for idx in block_coverage[best_block_idx]:
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
                    
        return current_blocks, counts, uncovered

    # Helper: Local Search (Reduction)
    def local_search(blocks, counts):
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b_idx = blocks[i]
                is_redundant = True
                for idx in block_coverage[b_idx]:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for idx in block_coverage[b_idx]:
                        counts[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
                    
        return blocks, counts

    # Initial Solution: Greedy from empty
    initial_blocks, counts, uncovered = greedy_repair([])
    
    # Initial Local Search
    current_blocks, counts = local_search(initial_blocks, counts)
    
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    # Iterated Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Update Best
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # 2. Perturbation
        perturbed = list(current_blocks)
        num_perturb = random.randint(1, 2)
        
        for _ in range(num_perturb):
            if not perturbed:
                break
            idx_to_remove = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx_to_remove)
            
            if random.random() < 0.5 and perturbed:
                idx_to_replace = random.randint(0, len(perturbed) - 1)
                new_block = random.randint(0, num_blocks - 1)
                perturbed[idx_to_replace] = new_block
        
        # 3. Repair
        repaired_blocks, counts, uncovered = greedy_repair(perturbed)
        
        # 4. Local Search on repaired
        repaired_blocks, counts = local_search(repaired_blocks, counts)
        
        # 5. Accept if better or equal (to allow drift)
        if len(repaired_blocks) <= len(current_blocks):
            current_blocks = repaired_blocks
            # counts is already updated in local_search

    # Return the best solution found
    return [list(all_blocks[i]) for i in best_blocks]
