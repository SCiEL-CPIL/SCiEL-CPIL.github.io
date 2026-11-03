"""Covering design C(19,8,5) - Optimized Simulated Annealing."""
import numpy as np
import itertools
import time
import random
from math import comb

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    def get_block_t_subs(block):
        return [sub_to_idx[tuple(sub)] for sub in itertools.combinations(block, t)]

    # 1. Fast Greedy Initialization
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=bool)
        blocks = []
        while np.any(uncovered):
            idx = np.where(uncovered)[0][0]
            seed = all_t_subs[idx]
            block = list(seed)
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = 0
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in sub_to_idx:
                            i = sub_to_idx[cand]
                            if uncovered[i]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            for i in get_block_t_subs(block):
                uncovered[i] = False
                
            if time.time() - start_time > time_limit * 0.2:
                break
        return blocks

    initial_blocks = greedy_init()
    
    def is_valid(blocks):
        covered = np.zeros(n_t_subs, dtype=bool)
        for b in blocks:
            for i in get_block_t_subs(b):
                covered[i] = True
        return np.all(covered)

    if not is_valid(initial_blocks):
        # Fallback
        uncovered = np.ones(n_t_subs, dtype=bool)
        blocks = []
        while np.any(uncovered):
            idx = np.where(uncovered)[0][0]
            seed = all_t_subs[idx]
            block = list(seed)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block:
                    block.append(x)
            block.sort()
            blocks.append(block)
            for i in get_block_t_subs(block):
                uncovered[i] = False
        initial_blocks = blocks

    best_blocks = list(initial_blocks)
    best_len = len(best_blocks)
    
    current_blocks = list(initial_blocks)
    cover_counts = np.zeros(n_t_subs, dtype=int)
    for b in current_blocks:
        for i in get_block_t_subs(b):
            cover_counts[i] += 1
            
    current_uncovered = np.sum(cover_counts == 0)
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    def energy():
        if current_uncovered == 0:
            return len(current_blocks)
        return len(current_blocks) + 1000 * current_uncovered

    current_energy = energy()
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                block = current_blocks[idx]
                subs = get_block_t_subs(block)
                
                # Calculate delta_uncovered
                delta_uncovered = 0
                for i in subs:
                    if cover_counts[i] == 1:
                        delta_uncovered += 1
                
                new_uncovered = current_uncovered + delta_uncovered
                new_len = len(current_blocks) - 1
                new_energy = new_len + (1000 * new_uncovered if new_uncovered > 0 else 0)
                delta_E = new_energy - current_energy
                
                if delta_E < 0 or (T > 0 and random.random() < np.exp(-delta_E / T)):
                    # Accept
                    for i in subs:
                        cover_counts[i] -= 1
                    current_blocks.pop(idx)
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if new_uncovered == 0 and new_len < best_len:
                        best_blocks = list(current_blocks)
                        best_len = new_len
                # else reject, do nothing
                
        elif move_type < 0.8:
            # Add
            # Target uncovered if possible
            if current_uncovered > 0:
                uncovered_idxs = np.where(cover_counts == 0)[0]
                target_idx = uncovered_idxs[random.randint(0, len(uncovered_idxs) - 1)]
            else:
                target_idx = random.randint(0, n_t_subs - 1)
                
            seed = all_t_subs[target_idx]
            block = list(seed)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block:
                    block.append(x)
            block.sort()
            subs = get_block_t_subs(block)
            
            # Calculate delta_uncovered
            delta_uncovered = 0
            for i in subs:
                if cover_counts[i] == 0:
                    delta_uncovered -= 1 # Becomes covered
            
            new_uncovered = current_uncovered + delta_uncovered
            new_len = len(current_blocks) + 1
            new_energy = new_len + (1000 * new_uncovered if new_uncovered > 0 else 0)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or (T > 0 and random.random() < np.exp(-delta_E / T)):
                for i in subs:
                    cover_counts[i] += 1
                current_blocks.append(block)
                current_uncovered = new_uncovered
                current_energy = new_energy
                if new_uncovered == 0 and new_len < best_len:
                    best_blocks = list(current_blocks)
                    best_len = new_len
                    
        else:
            # Swap
            if len(current_blocks) > 0:
                b_idx = random.randint(0, len(current_blocks) - 1)
                block = current_blocks[b_idx]
                p_idx = random.randint(0, len(block) - 1)
                old_p = block[p_idx]
                new_p = random.randint(0, v - 1)
                if new_p == old_p:
                    continue
                    
                new_block = list(block)
                new_block[p_idx] = new_p
                new_block.sort()
                
                old_subs = get_block_t_subs(block)
                new_subs = get_block_t_subs(new_block)
                
                # Delta uncovered
                # For old_subs: if count was 1, becomes 0 (uncovered +1)
                # For new_subs: if count was 0, becomes 1 (uncovered -1)
                # Note: Some subs might be in both old and new.
                # We must handle overlaps carefully.
                
                # Simulate changes
                delta_uncovered = 0
                # Temporarily apply old removals
                for i in old_subs:
                    if cover_counts[i] == 1:
                        delta_uncovered += 1
                    # cover_counts[i] -= 1 (conceptually)
                
                # Temporarily apply new additions
                for i in new_subs:
                    # What is the count after removals?
                    # If i was in old_subs, count was decremented.
                    # If i was not in old_subs, count is original.
                    # This is getting complex. 
                    # Simpler: Just calculate full delta by checking specific indices.
                    pass
                
                # Given complexity and low probability of swap helping significantly vs Add/Remove,
                # and time constraints, we skip Swap in this optimized version for safety.
                pass

        T *= cooling_rate

    if not is_valid(best_blocks):
        return initial_blocks
    return best_blocks
