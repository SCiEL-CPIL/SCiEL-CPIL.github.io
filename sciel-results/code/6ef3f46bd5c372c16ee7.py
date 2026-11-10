import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and map to index
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Helper to get index of a t-subset
    def get_idx(sub):
        return tsub_to_idx[sub]

    # Helper to generate all t-subset indices of a block
    def block_tsubs(block):
        return [get_idx(sub) for sub in itertools.combinations(block, t)]

    # 1. Greedy Construction
    uncovered = set(range(num_tsubs))
    blocks = []
    
    while uncovered:
        if time.time() - start_time > deadline:
            break
            
        # Pick a random uncovered t-subset
        seed_idx = random.choice(list(uncovered))
        seed_sub = all_tsubs[seed_idx]
        seed_set = set(seed_sub)
        remaining_points = [x for x in range(v) if x not in seed_set]
        
        best_gain = -1
        best_block = None
        
        # Iterate through all combinations of points to add
        for add_combo in itertools.combinations(remaining_points, k - t):
            candidate_block = tuple(sorted(seed_sub + add_combo))
            # Calculate gain
            gain = 0
            for sub in itertools.combinations(candidate_block, t):
                idx = tsub_to_idx[sub]
                if idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
            elif gain == best_gain and best_block is not None:
                if candidate_block < best_block:
                    best_block = candidate_block
            
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        for sub in itertools.combinations(best_block, t):
            idx = tsub_to_idx[sub]
            uncovered.discard(idx)
            
    if not blocks:
        return []
        
    # 2. Local Search
    # Initialize coverage counts
    cover_count = [0] * num_tsubs
    
    def add_block(block):
        for idx in block_tsubs(block):
            cover_count[idx] += 1
            
    def remove_block(block):
        for idx in block_tsubs(block):
            cover_count[idx] -= 1
            
    for b in blocks:
        add_block(b)
        
    def is_redundant(block):
        for idx in block_tsubs(block):
            if cover_count[idx] <= 1:
                return False
        return True

    # Remove initial redundancies
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                remove_block(b)
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # SA Parameters
    temperature = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    while time.time() - start_time < deadline:
        if temperature < min_temp:
            temperature = 1.0
        
        if not blocks:
            break
            
        i = random.randint(0, len(blocks) - 1)
        b_old = blocks[i]
        
        # Generate neighbor
        if random.random() < 0.5:
            p_out = random.choice(b_old)
            candidates_in = [x for x in range(v) if x not in b_old]
            if not candidates_in:
                continue
            p_in = random.choice(candidates_in)
            b_new = [x for x in b_old if x != p_out]
            b_new.append(p_in)
            b_new.sort()
        else:
            if len(b_old) < 2:
                continue
            p_out1, p_out2 = random.sample(b_old, 2)
            candidates_in = [x for x in range(v) if x not in b_old]
            if len(candidates_in) < 2:
                continue
            p_in1, p_in2 = random.sample(candidates_in, 2)
            b_new = [x for x in b_old if x != p_out1 and x != p_out2]
            b_new.extend([p_in1, p_in2])
            b_new.sort()
        
        if b_new in blocks:
            continue
            
        # Check validity of swap
        unique_indices = []
        for idx in block_tsubs(b_old):
            if cover_count[idx] == 1:
                unique_indices.append(idx)
                
        if not unique_indices:
            # b_old is redundant
            remove_block(b_old)
            blocks.pop(i)
            # Remove others
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for j, b in enumerate(blocks):
                    if is_redundant(b):
                        remove_block(b)
                        changed = True
                    else:
                        new_blocks.append(b)
                blocks = new_blocks
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            continue
            
        valid_swap = True
        new_block_set = set(itertools.combinations(b_new, t))
        for idx in unique_indices:
            sub = all_tsubs[idx]
            if sub not in new_block_set:
                valid_swap = False
                break
                
        if not valid_swap:
            continue
            
        # Perform swap
        remove_block(b_old)
        add_block(b_new)
        blocks[i] = b_new
        
        # Remove redundancies
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for j, b in enumerate(blocks):
                if is_redundant(b):
                    remove_block(b)
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            
        temperature *= cooling_rate
        
    return best_blocks
