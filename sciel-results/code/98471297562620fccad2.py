import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    # Precompute all t-subsets and map them to IDs
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    def greedy_construct(uncovered_set, randomize=False):
        blocks = []
        current_uncovered = set(uncovered_set)
        
        while current_uncovered:
            if randomize:
                target_subset = random.choice(list(current_uncovered))
            else:
                target_subset = min(current_uncovered)
            
            missing_points = [x for x in range(v) if x not in target_subset]
            num_to_pick = k - t
            
            best_gain = -1
            best_combo = None
            
            combos = list(itertools.combinations(missing_points, num_to_pick))
            if randomize:
                random.shuffle(combos)
            
            for combo in combos:
                candidate_block = tuple(sorted(target_subset + combo))
                
                gain = 0
                for sub in itertools.combinations(candidate_block, t):
                    if sub in current_uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_combo = combo
                    if gain == len(current_uncovered):
                        break
            
            final_block = tuple(sorted(target_subset + best_combo))
            blocks.append(final_block)
            
            for sub in itertools.combinations(final_block, t):
                current_uncovered.discard(sub)
                
        return blocks

    def minimize(blocks):
        cover_count = [0] * n_subsets
        block_subsets = []
        for b in blocks:
            subs = []
            for sub in itertools.combinations(b, t):
                subs.append(subset_to_id[sub])
            block_subsets.append(subs)
            for id_ in subs:
                cover_count[id_] += 1
        
        removed = True
        while removed:
            removed = False
            new_blocks = []
            new_block_subsets = []
            for i, b in enumerate(blocks):
                is_redundant = True
                for id_ in block_subsets[i]:
                    if cover_count[id_] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for id_ in block_subsets[i]:
                        cover_count[id_] -= 1
                    removed = True
                else:
                    new_blocks.append(b)
                    new_block_subsets.append(block_subsets[i])
            
            blocks = new_blocks
            block_subsets = new_block_subsets
            
        return blocks

    def try_swap(blocks):
        if not blocks:
            return blocks
        idx = random.randint(0, len(blocks) - 1)
        b = blocks[idx]
        
        cover_count = [0] * n_subsets
        for blk in blocks:
            for sub in itertools.combinations(blk, t):
                cover_count[subset_to_id[sub]] += 1
        
        unique_subs = []
        for sub in itertools.combinations(b, t):
            if cover_count[subset_to_id[sub]] == 1:
                unique_subs.append(sub)
        
        if not unique_subs:
            return blocks
        
        union_points = set()
        for sub in unique_subs:
            union_points.update(sub)
        
        if len(union_points) > k:
            return blocks
            
        missing_points = [x for x in range(v) if x not in union_points]
        num_to_pick = k - len(union_points)
        
        if num_to_pick == 0:
            candidate = tuple(sorted(union_points))
            if candidate != b:
                new_blocks = [blk for i, blk in enumerate(blocks) if i != idx] + [candidate]
                return minimize(new_blocks)
            else:
                return blocks
        else:
            combo = random.sample(missing_points, num_to_pick)
            candidate = tuple(sorted(list(union_points) + combo))
            if candidate != b:
                new_blocks = [blk for i, blk in enumerate(blocks) if i != idx] + [candidate]
                return minimize(new_blocks)
            else:
                return blocks

    # Initial Solution
    initial_uncovered = set(all_subsets)
    initial_blocks = greedy_construct(initial_uncovered, randomize=True)
    best_blocks = minimize(initial_blocks)
    
    # Local Search
    while time.time() < deadline:
        move_type = random.choice(['remove', 'swap'])
        
        if move_type == 'remove':
            num_remove = random.randint(1, 3)
            perturbed_blocks = list(best_blocks)
            for _ in range(num_remove):
                if not perturbed_blocks:
                    break
                idx = random.randint(0, len(perturbed_blocks) - 1)
                perturbed_blocks.pop(idx)
            
            cover_count = [0] * n_subsets
            for b in perturbed_blocks:
                for sub in itertools.combinations(b, t):
                    cover_count[subset_to_id[sub]] += 1
                    
            uncovered = set()
            for i, cnt in enumerate(cover_count):
                if cnt == 0:
                    uncovered.add(all_subsets[i])
                    
            if not uncovered:
                continue
                
            repair_blocks = greedy_construct(uncovered, randomize=True)
            candidate_blocks = perturbed_blocks + repair_blocks
            candidate_blocks = minimize(candidate_blocks)
            
        else: # swap
            candidate_blocks = try_swap(best_blocks)
            if candidate_blocks is None:
                continue

        if len(candidate_blocks) < len(best_blocks):
            best_blocks = candidate_blocks
            
    return [list(b) for b in best_blocks]
