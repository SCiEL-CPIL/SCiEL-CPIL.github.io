import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145

    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_triplet_ids = []
    block_masks = []
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    
    for idx, block in enumerate(all_blocks):
        tids = []
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            tids.append(tid)
            mask |= (1 << tid)
            triplet_to_blocks[tid].append(idx)
        block_triplet_ids.append(tids)
        block_masks.append(mask)
        
    def greedy_init():
        uncovered = (1 << n_triplets) - 1
        selected = []
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline - 30:
                break
            best_gain = -1
            candidates = []
            max_gain_val = 0
            for i in range(n_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > max_gain_val:
                    max_gain_val = gain
                    candidates = [i]
                elif gain == max_gain_val and gain > 0:
                    candidates.append(i)
            
            if max_gain_val == 0:
                break
                
            best_idx = random.choice(candidates)
            selected.append(all_blocks[best_idx])
            selected_set.add(all_blocks[best_idx])
            uncovered &= ~block_masks[best_idx]
            
        return selected, selected_set

    selected, selected_set = greedy_init()
    
    count = np.zeros(n_triplets, dtype=np.int32)
    def update_counts(blocks_list, delta):
        for b in blocks_list:
            for triplet in itertools.combinations(b, t):
                tid = triplet_to_id[triplet]
                count[tid] += delta

    update_counts(selected, 1)
    
    def get_unique_triplets(block):
        unique = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            if count[tid] == 1:
                unique.append(tid)
        return unique

    def try_remove_redundant():
        new_selected = []
        removed = False
        for b in selected:
            redundant = True
            for triplet in itertools.combinations(b, t):
                tid = triplet_to_id[triplet]
                if count[tid] == 1:
                    redundant = False
                    break
            if redundant:
                update_counts([b], -1)
                selected_set.remove(b)
                removed = True
            else:
                new_selected.append(b)
        return new_selected, removed

    def try_2_for_1():
        n = len(selected)
        if n < 2:
            return False
        
        uniques = [get_unique_triplets(b) for b in selected]
        
        for i in range(n):
            if time.time() > deadline - 5:
                return False
            if not uniques[i]:
                continue
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return False
                if not uniques[j]:
                    continue
                
                all_unique_tids = set(uniques[i]) | set(uniques[j])
                
                points_needed = set()
                for tid in all_unique_tids:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) <= k:
                    new_block_points = list(points_needed)
                    while len(new_block_points) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block_points:
                            new_block_points.append(p)
                    new_block = tuple(sorted(new_block_points))
                    
                    update_counts([selected[i], selected[j]], -1)
                    selected_set.remove(selected[i])
                    selected_set.remove(selected[j])
                    
                    update_counts([new_block], 1)
                    selected_set.add(new_block)
                    
                    new_selected = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                    new_selected.append(new_block)
                    
                    nonlocal selected
                    selected = new_selected
                    return True
                    
        return False

    def random_walk_step():
        if len(selected) == 0:
            return False
        
        idx = random.randint(0, len(selected) - 1)
        b_remove = selected[idx]
        
        old_selected = selected[:]
        old_set = set(selected_set)
        old_count = count.copy()
        
        update_counts([b_remove], -1)
        selected_set.remove(b_remove)
        selected.pop(idx)
        
        uncovered_tids = []
        for triplet in itertools.combinations(b_remove, t):
            tid = triplet_to_id[triplet]
            if count[tid] == 0:
                uncovered_tids.append(tid)
                
        if uncovered_tids:
            uncovered_mask = 0
            for tid in uncovered_tids:
                uncovered_mask |= (1 << tid)
            
            candidate_indices = set()
            for tid in uncovered_tids:
                candidate_indices.update(triplet_to_blocks[tid])
            
            while uncovered_mask:
                best_gain = 0
                best_idx = -1
                for i in candidate_indices:
                    gain = (block_masks[i] & uncovered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        if gain == len(uncovered_tids):
                            break
                
                if best_gain == 0:
                    break
                
                new_b = all_blocks[best_idx]
                update_counts([new_b], 1)
                selected_set.add(new_b)
                selected.append(new_b)
                uncovered_mask &= ~block_masks[best_idx]
                
        if len(selected) < len(old_selected):
            return True
        else:
            count[:] = old_count
            selected = old_selected
            selected_set = old_set
            return False

    best_count = len(selected)
    
    progress = True
    while progress and time.time() < deadline - 10:
        progress = False
        
        new_sel, removed = try_remove_redundant()
        if removed:
            selected = new_sel
            progress = True
            best_count = min(best_count, len(selected))
            
        if not progress:
            if try_2_for_1():
                progress = True
                best_count = min(best_count, len(selected))
                
        if not progress:
            if random_walk_step():
                progress = True
                best_count = min(best_count, len(selected))
                
    return [list(b) for b in selected]
