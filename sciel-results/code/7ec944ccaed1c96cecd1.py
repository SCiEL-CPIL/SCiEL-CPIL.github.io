import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    limit = 115.0 
    deadline = start_time + limit

    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    block_covers = []
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(blocks):
        covers = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            sub_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))
        
    if not _deadline_ok(deadline):
        return []

    # 2. Fast Greedy Initialization
    uncovered_mask = np.ones(n_t, dtype=bool)
    gains = np.full(n_blocks, 0, dtype=np.int32)
    
    initial_gain = len(block_covers[0]) if n_blocks > 0 else 0
    gains[:] = initial_gain
    
    selected = []
    selected_set = set()
    
    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break
            
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        best_block = np.argmax(gains)
        
        if best_block in selected_set:
            gains[best_block] = -1
            continue
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        covers_b = block_covers[best_block]
        for s_idx in covers_b:
            if uncovered_mask[s_idx]:
                uncovered_mask[s_idx] = False
                for b_prime in sub_to_blocks[s_idx]:
                    gains[b_prime] -= 1
        
        gains[best_block] = -1

    if not selected:
        return []
        
    # 3. Local Search with ILS (Iterated Local Search)
    
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1
        
    def prune_redundant(selected_list, counts, deadline):
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in range(len(selected_list) - 1, -1, -1):
                b = selected_list[i]
                covers_b = block_covers[b]
                if np.all(counts[covers_b] > 1):
                    selected_list.pop(i)
                    counts[covers_b] -= 1
                    changed = True
        return changed

    selected_list = list(selected)
    prune_redundant(selected_list, counts, deadline)
    
    best_size = len(selected_list)
    best_solution = selected_list.copy()
    best_counts = counts.copy()
    
    rng = random.Random(42)
    all_block_indices = list(range(n_blocks))
    current_set = set(selected_list)
    
    def rebuild_counts(block_list):
        c = np.zeros(n_t, dtype=np.int32)
        for b in block_list:
            c[block_covers[b]] += 1
        return c

    # Helper: greedy repair - add blocks to cover any uncovered subsets
    def greedy_repair(block_list, counts, deadline):
        """Add blocks to cover any uncovered subsets. Returns True if all covered."""
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            return True
        
        # For each uncovered subset, find candidate blocks
        # Use a simple approach: repeatedly add the block that covers the most uncovered subsets
        for _ in range(100):  # safety limit
            if not _deadline_ok(deadline):
                return False
            
            # Recompute uncovered
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                return True
            
            # Find the best block to add
            best_block = -1
            best_gain = 0
            for s_idx in uncovered[:50]:  # sample to keep it fast
                for b_prime in sub_to_blocks[s_idx]:
                    if b_prime not in current_set:
                        # Count how many uncovered subsets this block covers
                        cov = block_covers[b_prime]
                        gain = np.sum(counts[cov] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_block = b_prime
            
            if best_block == -1 or best_gain == 0:
                break
            
            block_list.append(best_block)
            current_set.add(best_block)
            counts[block_covers[best_block]] += 1
        
        # Final check
        return np.all(counts > 0)

    # Main ILS Loop
    perturbation_strength = 3  # number of blocks to randomly remove
    
    while _deadline_ok(deadline):
        # Phase 1: Add-Remove local search
        for _ in range(50):  # limited iterations per ILS round
            if not _deadline_ok(deadline):
                break
            
            # Pick a random block to add, weighted toward blocks covering weakly-covered subsets
            # Find subsets with low coverage
            min_count = np.min(counts)
            if min_count <= 2:
                # Focus on weakly covered subsets
                weak_subs = np.where(counts <= min_count + 1)[0]
                if len(weak_subs) > 0:
                    target_sub = rng.choice(weak_subs.tolist())
                    candidates = [b for b in sub_to_blocks[target_sub] if b not in current_set]
                    if candidates:
                        b_add = rng.choice(candidates)
                    else:
                        b_add = rng.choice(all_block_indices)
                else:
                    b_add = rng.choice(all_block_indices)
            else:
                b_add = rng.choice(all_block_indices)
            
            if b_add in current_set:
                continue
            
            temp_list = selected_list.copy()
            temp_set = current_set.copy()
            temp_list.append(b_add)
            temp_set.add(b_add)
            
            temp_counts = counts.copy()
            temp_counts[block_covers[b_add]] += 1
            
            prune_redundant(temp_list, temp_counts, deadline)
            
            current_size = len(temp_list)
            
            if current_size < best_size:
                best_size = current_size
                best_solution = temp_list.copy()
                best_counts = temp_counts.copy()
                selected_list = temp_list
                current_set = temp_set
                counts = temp_counts
            elif current_size < len(selected_list):
                selected_list = temp_list
                current_set = temp_set
                counts = temp_counts
            
            if len(selected_list) > best_size + 5:
                selected_list = best_solution.copy()
                current_set = set(best_solution)
                counts = best_counts.copy()
        
        # Phase 2: Perturbation (random deletion + greedy repair)
        if not _deadline_ok(deadline):
            break
        
        # Start from best solution
        pert_list = best_solution.copy()
        pert_set = set(pert_list)
        pert_counts = best_counts.copy()
        
        # Remove k random blocks
        num_remove = min(perturbation_strength, len(pert_list) // 3)
        if num_remove > 0:
            remove_indices = rng.sample(range(len(pert_list)), num_remove)
            # Remove in reverse order to preserve indices
            for idx in sorted(remove_indices, reverse=True):
                b = pert_list[idx]
                pert_list.pop(idx)
                pert_set.discard(b)
                pert_counts[block_covers[b]] -= 1
        
        # Greedy repair
        # Reset current_set temporarily for repair
        old_current_set = current_set
        current_set = pert_set
        
        repaired = greedy_repair(pert_list, pert_counts, deadline)
        
        current_set = old_current_set
        
        if repaired:
            prune_redundant(pert_list, pert_counts, deadline)
            
            if len(pert_list) < best_size:
                best_size = len(pert_list)
                best_solution = pert_list.copy()
                best_counts = pert_counts.copy()
                selected_list = pert_list
                current_set = set(best_solution)
                counts = pert_counts
        
        # Gradually increase perturbation strength
        if rng.random() < 0.1:
            perturbation_strength = min(perturbation_strength + 1, 10)
        elif rng.random() < 0.2:
            perturbation_strength = max(2, perturbation_strength - 1)

    # Final Result
    result = [list(blocks[i]) for i in best_solution]
    return result
