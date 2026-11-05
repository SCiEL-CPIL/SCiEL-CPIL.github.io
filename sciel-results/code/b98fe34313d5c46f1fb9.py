import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their covered subset indices
all_blocks = []
block_to_subsets = []
subset_to_blocks = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    block_idx = len(all_blocks)
    all_blocks.append(block)
    covered_indices = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        covered_indices.append(idx)
        subset_to_blocks[idx].append(block_idx)
    block_to_subsets.append(covered_indices)

N_BLOCKS = len(all_blocks)

# Weights for greedy: rare subsets are more important
subset_frequencies = [len(lst) for lst in subset_to_blocks]
weights = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)

# Coverage matrix: N_BLOCKS x N_SUBSETS
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, indices in enumerate(block_to_subsets):
    for j in indices:
        coverage_matrix[i, j] = 1

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118

    def run_greedy():
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        
        while True:
            if not uncovered.any():
                break
            
            effective_weights = weights * uncovered
            gains = coverage_matrix.dot(effective_weights)
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
                
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0
            
            if time.time() > deadline - 5:
                break

        return chosen

    def run_packing_greedy():
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        coverage_counts = np.zeros(N_SUBSETS, dtype=np.int32)
        
        while True:
            if not uncovered.any():
                break
            if time.time() > deadline - 15:
                break
            
            new_cov = (coverage_matrix * uncovered).sum(axis=1)
            total_cov = coverage_matrix.sum(axis=1)
            overlap_pen = total_cov - new_cov
            
            score = new_cov * 2.0 - overlap_pen * 0.3
            
            best_idx = np.argmax(score)
            if new_cov[best_idx] == 0:
                break
                
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0
            coverage_counts[coverage_matrix[best_idx] == 1] += 1

        return chosen

    def resolve_deficit(packing_blocks):
        current = list(packing_blocks)
        current_set = set(current)
        
        coverage = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current:
            coverage[block_to_subsets[b_idx]] += 1
        
        deficit_subsets = np.where(coverage == 0)[0]
        if len(deficit_subsets) == 0:
            return current
        
        deficit_mask = np.zeros(N_SUBSETS, dtype=np.int8)
        deficit_mask[deficit_subsets] = 1
        
        while True:
            if time.time() > deadline - 15:
                break
            
            coverage = np.zeros(N_SUBSETS, dtype=np.int32)
            for b_idx in current:
                coverage[block_to_subsets[b_idx]] += 1
            
            deficit_subsets = np.where(coverage == 0)[0]
            if len(deficit_subsets) == 0:
                break
            
            deficit_mask = np.zeros(N_SUBSETS, dtype=np.int8)
            deficit_mask[deficit_subsets] = 1
            
            all_block_indices = np.arange(N_BLOCKS)
            not_in_current = np.array([b not in current_set for b in all_block_indices])
            
            if not not_in_current.any():
                break
            
            deficit_cov = (coverage_matrix * deficit_mask).sum(axis=1)
            
            valid_mask = not_in_current & (deficit_cov > 0)
            if not valid_mask.any():
                break
            
            total_cov_per_block = coverage_matrix.sum(axis=1)
            conflict_approx = total_cov_per_block - deficit_cov
            
            score = deficit_cov * 1.0 - conflict_approx * 0.2
            score[~valid_mask] = -np.inf
            
            best_idx = np.argmax(score)
            if deficit_cov[best_idx] == 0:
                break
            
            current.append(best_idx)
            current_set.add(best_idx)
        
        return current

    def prune_redundant(blocks_list):
        if not blocks_list:
            return blocks_list
        
        current = list(blocks_list)
        if not current:
            return current
        
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current:
            counts[block_to_subsets[b_idx]] += 1
            
        new_blocks = []
        for b_idx in current:
            redundant = True
            for s_idx in block_to_subsets[b_idx]:
                if counts[s_idx] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(b_idx)
            else:
                counts[block_to_subsets[b_idx]] -= 1
        
        return new_blocks

    def local_search(initial_blocks):
        current_blocks = list(initial_blocks)
        current_blocks = list(set(current_blocks))
        current_blocks = prune_redundant(current_blocks)
        
        best_blocks = current_blocks.copy()
        best_len = len(current_blocks)
        
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current_blocks:
            counts[block_to_subsets[b_idx]] += 1
        
        def update_counts_add(b_idx):
            counts[block_to_subsets[b_idx]] += 1
            
        def update_counts_remove(b_idx):
            counts[block_to_subsets[b_idx]] -= 1
            
        current_set = set(current_blocks)
            
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < deadline - 10:
            pruned = prune_redundant(current_blocks)
            if len(pruned) < len(current_blocks):
                counts = np.zeros(N_SUBSETS, dtype=np.int32)
                for b_idx in pruned:
                    counts[block_to_subsets[b_idx]] += 1
                current_blocks = pruned
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve = 0
                continue

            improved = False
            block_indices = list(range(len(current_blocks)))
            random.shuffle(block_indices)
            
            for i in block_indices:
                if time.time() > deadline - 10:
                    break
                    
                b_out = current_blocks[i]
                
                unique_subsets = []
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        unique_subsets.append(s_idx)
                        
                if not unique_subsets:
                    update_counts_remove(b_out)
                    current_blocks.pop(i)
                    current_set.discard(b_out)
                    improved = True
                    no_improve = 0
                    break
                    
                cands = None
                for s_idx in unique_subsets:
                    lst = subset_to_blocks[s_idx]
                    if cands is None:
                        cands = set(lst)
                    else:
                        cands.intersection_update(lst)
                    if not cands:
                        break
                        
                if cands:
                    cands.discard(b_out)
                    if cands:
                        cand_list = list(cands)
                        if len(cand_list) <= 5:
                            b_in = random.choice(cand_list)
                        else:
                            best_cand = cand_list[0]
                            best_score = -1
                            sample = random.sample(cand_list, min(10, len(cand_list)))
                            for b_c in sample:
                                score = 0
                                for s_idx in block_to_subsets[b_c]:
                                    if counts[s_idx] == 1:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_cand = b_c
                            b_in = best_cand

                        update_counts_remove(b_out)
                        current_blocks.pop(i)
                        current_set.discard(b_out)
                        update_counts_add(b_in)
                        current_blocks.append(b_in)
                        current_set.add(b_in)
                        
                        improved = True
                        no_improve = 0
                        break

            if improved:
                continue
                
            no_improve += 1
            if no_improve > max_no_improve:
                for _ in range(random.randint(2, 4)):
                    b_rand = random.randint(0, N_BLOCKS - 1)
                    if b_rand not in current_set:
                        current_blocks.append(b_rand)
                        current_set.add(b_rand)
                        update_counts_add(b_rand)
                no_improve = 0
                pruned = prune_redundant(current_blocks)
                if len(pruned) < len(current_blocks):
                    counts = np.zeros(N_SUBSETS, dtype=np.int32)
                    for b_idx in pruned:
                        counts[block_to_subsets[b_idx]] += 1
                    current_blocks = pruned
                    current_set = set(current_blocks)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()

        return best_blocks

    def polish(best_blocks):
        if not best_blocks:
            return best_blocks
        
        current = list(best_blocks)
        current = list(set(current))
        current = prune_redundant(current)
        
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current:
            counts[block_to_subsets[b_idx]] += 1
        
        improved = True
        while improved:
            improved = False
            
            for i in range(len(current)):
                b_out = current[i]
                can_remove = True
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    counts[block_to_subsets[b_out]] -= 1
                    current.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            for i in range(len(current)):
                b_out = current[i]
                unique_subsets = []
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        unique_subsets.append(s_idx)
                
                if not unique_subsets:
                    counts[block_to_subsets[b_out]] -= 1
                    current.pop(i)
                    improved = True
                    break
                
                cands = None
                for s_idx in unique_subsets:
                    lst = subset_to_blocks[s_idx]
                    if cands is None:
                        cands = set(lst)
                    else:
                        cands.intersection_update(lst)
                    if not cands:
                        break
                
                if cands:
                    cands.discard(b_out)
                    for b_in in cands:
                        if b_in in current:
                            counts[block_to_subsets[b_out]] -= 1
                            current.pop(i)
                            improved = True
                            break
                        else:
                            counts[block_to_subsets[b_out]] -= 1
                            current.pop(i)
                            counts[block_to_subsets[b_in]] += 1
                            current.append(b_in)
                            improved = True
                            break
                    if improved:
                        break
            
            if time.time() > deadline - 3:
                break

        return current

    def run_random_greedy():
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        
        while True:
            if not uncovered.any():
                break
            
            if time.time() > deadline - 5:
                break
            
            effective_weights = uncovered
            gains = coverage_matrix.dot(effective_weights)
            
            valid_indices = np.where(gains > 0)[0]
            if len(valid_indices) == 0:
                break
            
            best_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0

        return chosen

    best_overall = None
    best_len_overall = float('inf')

    while time.time() < deadline - 15:
        r = random.random()
        if r < 0.3:
            initial_blocks = run_greedy()
        elif r < 0.5:
            initial_blocks = run_random_greedy()
        else:
            packing = run_packing_greedy()
            initial_blocks = resolve_deficit(packing)
        
        ls_blocks = local_search(initial_blocks)
        
        if len(ls_blocks) < best_len_overall:
            best_len_overall = len(ls_blocks)
            best_overall = ls_blocks

    if best_overall is not None:
        polished = polish(best_overall)
        if len(polished) < best_len_overall:
            best_overall = polished
            best_len_overall = len(polished)
    else:
        best_overall = run_greedy()

    return [list(all_blocks[i]) for i in best_overall]
