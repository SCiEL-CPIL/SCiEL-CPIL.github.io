import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute for each block, the list of t-subset indices it covers
    block_covered_indices = []
    block_covered_set = []
    block_covered_np = []  # numpy arrays for fast vectorized operations
    
    for b in all_blocks:
        idxs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covered_indices.append(idxs)
        block_covered_set.append(set(idxs))
        block_covered_np.append(np.array(idxs, dtype=np.int32))
        
    # For fast lookup of which blocks cover a specific t-subset
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, idxs in enumerate(block_covered_indices):
        for s_idx in idxs:
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute block size (number of t-subsets covered)
    block_sizes = np.array([len(idxs) for idxs in block_covered_indices], dtype=np.int32)

    def calculate_coverage(block_indices):
        """Calculate coverage boolean array from a list of block indices."""
        covered = np.zeros(n_subsets, dtype=bool)
        for idx in block_indices:
            covered[block_covered_indices[idx]] = True
        return covered

    def greedy_construct():
        """Greedy set cover with most-constrained-first heuristic."""
        covered = np.zeros(n_subsets, dtype=bool)
        blocks = []
        
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick the uncovered subset with fewest covering blocks (most constrained)
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered_indices])
            min_count = counts.min()
            candidates = uncovered_indices[counts == min_count]
            target_subset = random.choice(candidates.tolist())
            
            # Get blocks covering this target
            candidate_blocks = subset_to_blocks[target_subset]
            
            best_gain = -1
            best_block = -1
            for b_idx in candidate_blocks:
                gain = np.count_nonzero(~covered[block_covered_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block != -1:
                blocks.append(best_block)
                covered[block_covered_indices[best_block]] = True
            else:
                blocks.append(random.randint(0, n_blocks - 1))
                
        return blocks

    def local_search(blocks, max_time=10):
        """Local search with 1-for-1 swaps and 2-for-1 exchanges."""
        current_blocks = list(blocks)
        start_ls = time.time()
        
        # Maintain coverage count array for efficiency
        count_arr = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            count_arr[block_covered_indices[b_idx]] += 1
        
        improved = True
        while improved and time.time() < start_ls + max_time:
            improved = False
            n_blocks_cur = len(current_blocks)
            
            # 1. Remove redundant blocks
            for i in range(n_blocks_cur - 1, -1, -1):
                b_idx = current_blocks[i]
                redundant = True
                for s_idx in block_covered_indices[b_idx]:
                    if count_arr[s_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    current_blocks.pop(i)
                    count_arr[block_covered_indices[b_idx]] -= 1
                    improved = True
                    break
            
            if not improved and n_blocks_cur > 1:
                # 2. Try 1-for-1 swap: replace block i with a better block j
                # or 3. Try 2-for-1 exchange
                if random.random() < 0.5:
                    # 1-for-1 swap
                    i = random.randint(0, len(current_blocks) - 1)
                    b_i = current_blocks[i]
                    
                    # Find subsets uniquely covered by b_i
                    unique_to_bi = []
                    for s_idx in block_covered_indices[b_i]:
                        if count_arr[s_idx] == 1:
                            unique_to_bi.append(s_idx)
                    
                    if not unique_to_bi:
                        # b_i is redundant, remove it
                        current_blocks.pop(i)
                        count_arr[block_covered_indices[b_i]] -= 1
                        improved = True
                        break
                    
                    # Find a block that covers all unique_to_bi and adds new coverage
                    # Use the most constrained unique subset to find candidates
                    min_cnt = min(len(subset_to_blocks[s]) for s in unique_to_bi)
                    target_s = next(s for s in unique_to_bi if len(subset_to_blocks[s]) == min_cnt)
                    
                    best_block = -1
                    best_gain = -1
                    for b_c in subset_to_blocks[target_s]:
                        if b_c == b_i:
                            continue
                        # Check if b_c covers all unique_to_bi
                        set_c = block_covered_set[b_c]
                        all_covered = all(s in set_c for s in unique_to_bi)
                        if all_covered:
                            gain = np.count_nonzero(~(count_arr > 0)[block_covered_indices[b_c]]) 
                            # Better: count new subsets covered by b_c
                            new_cov = 0
                            for s_idx in block_covered_indices[b_c]:
                                if count_arr[s_idx] == 0:
                                    new_cov += 1
                            if new_cov > best_gain:
                                best_gain = new_cov
                                best_block = b_c
                    
                    if best_block != -1:
                        current_blocks[i] = best_block
                        count_arr[block_covered_indices[b_i]] -= 1
                        count_arr[block_covered_indices[best_block]] += 1
                        improved = True
                        break
                else:
                    # 2-for-1 exchange
                    if len(current_blocks) >= 2:
                        i, j = random.sample(range(len(current_blocks)), 2)
                        b_i = current_blocks[i]
                        b_j = current_blocks[j]
                        
                        set_i = block_covered_set[b_i]
                        set_j = block_covered_set[b_j]
                        
                        # Find critical subsets
                        critical_subs = []
                        for s_idx in block_covered_indices[b_i]:
                            if s_idx in set_j:
                                if count_arr[s_idx] == 2:
                                    critical_subs.append(s_idx)
                            else:
                                if count_arr[s_idx] == 1:
                                    critical_subs.append(s_idx)
                        
                        for s_idx in block_covered_indices[b_j]:
                            if s_idx not in set_i:
                                if count_arr[s_idx] == 1:
                                    critical_subs.append(s_idx)
                        
                        if not critical_subs:
                            if i > j:
                                i, j = j, i
                            current_blocks.pop(j)
                            current_blocks.pop(i)
                            count_arr[block_covered_indices[b_i]] -= 1
                            count_arr[block_covered_indices[b_j]] -= 1
                            improved = True
                            break
                        else:
                            candidate_blocks = set(subset_to_blocks[critical_subs[0]])
                            for s_idx in critical_subs[1:]:
                                candidate_blocks &= set(subset_to_blocks[s_idx])
                                if not candidate_blocks:
                                    break
                            
                            if candidate_blocks:
                                best_c = random.choice(list(candidate_blocks))
                                if best_c not in (b_i, b_j):
                                    if i > j:
                                        i, j = j, i
                                    current_blocks.pop(j)
                                    current_blocks.pop(i)
                                    current_blocks.append(best_c)
                                    count_arr[block_covered_indices[b_i]] -= 1
                                    count_arr[block_covered_indices[b_j]] -= 1
                                    count_arr[block_covered_indices[best_c]] += 1
                                    improved = True
                                    break
                                else:
                                    if i > j:
                                        i, j = j, i
                                    current_blocks.pop(j)
                                    current_blocks.pop(i)
                                    count_arr[block_covered_indices[b_i]] -= 1
                                    count_arr[block_covered_indices[b_j]] -= 1
                                    improved = True
                                    break

        return current_blocks

    # Main Loop: Multiple Restarts / Iterated Local Search
    best_solution = None
    best_size = float('inf')
    
    # 1. Initial Greedy (try multiple times, keep best)
    for trial in range(3):
        if time.time() > deadline - 40:
            break
        init_blocks = greedy_construct()
        improved = local_search(init_blocks, max_time=15)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = improved[:]
    
    if time.time() > deadline - 20:
        return [list(all_blocks[idx]) for idx in best_solution]
        
    # 2. Iterated Local Search
    current_blocks = best_solution[:]
    
    while time.time() < deadline - 8:
        # Perturb: remove random blocks
        n_remove = random.randint(2, 5)
        if len(current_blocks) > n_remove + 5:
            to_remove = random.sample(range(len(current_blocks)), n_remove)
            remove_set = set(to_remove)
            current_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
            
        # Repair with Greedy
        covered = calculate_coverage(current_blocks)
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick most constrained uncovered subset
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered_indices])
            min_count = counts.min()
            candidates = uncovered_indices[counts == min_count]
            target_subset = random.choice(candidates.tolist())
            candidate_blocks = subset_to_blocks[target_subset]
            best_gain = -1
            best_block = -1
            for b_idx in candidate_blocks:
                gain = np.count_nonzero(~covered[block_covered_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block != -1:
                current_blocks.append(best_block)
                covered[block_covered_indices[best_block]] = True
            else:
                break
                
        # Local Search
        improved = local_search(current_blocks, max_time=8)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = improved[:]
            current_blocks = best_solution[:]
        else:
            current_blocks = improved[:]

    # Final validation: ensure all t-subsets are covered
    final_covered = calculate_coverage(best_solution)
    if not np.all(final_covered):
        while not np.all(final_covered):
            if time.time() > deadline - 2:
                break
            uncovered = np.where(~final_covered)[0]
            if len(uncovered) == 0:
                break
            target = random.choice(uncovered.tolist())
            for b_idx in subset_to_blocks[target]:
                if b_idx not in best_solution:
                    best_solution.append(b_idx)
                    final_covered[block_covered_indices[b_idx]] = True
                    break
            else:
                best_solution.append(random.randint(0, n_blocks - 1))
                final_covered = calculate_coverage(best_solution)

    # Remove duplicates
    best_solution = list(dict.fromkeys(best_solution))

    return [list(all_blocks[idx]) for idx in best_solution]
