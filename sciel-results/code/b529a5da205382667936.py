import itertools
import sys
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    block_masks = [0] * n_k_blocks
    # Also build inverse index: t_sub_idx -> list of block_idx covering it
    t_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            t_to_blocks[idx].append(i)
        block_masks[i] = mask
        
    # State variables
    selected = set()
    counts = [0] * n_t_subs
    
    def add_block(idx):
        selected.add(idx)
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb
            
    def remove_block(idx):
        selected.remove(idx)
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            m ^= lsb
            
    def is_valid():
        for c in counts:
            if c == 0:
                return False
        return True
        
    def greedy_init():
        # Select blocks to cover all t-subsets
        uncovered = (1 << n_t_subs) - 1
        while uncovered:
            best_idx = -1
            max_gain = 0
            for i in range(n_k_blocks):
                if i in selected:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            add_block(best_idx)
            uncovered &= ~block_masks[best_idx]
            
    def greedy_repair():
        # Add blocks until valid
        while True:
            # Find an uncovered t-subset
            uncovered_idx = -1
            for i in range(n_t_subs):
                if counts[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx == -1:
                break
            
            # Find best block to cover it
            candidates = t_to_blocks[uncovered_idx]
            best_idx = -1
            max_gain = 0
            # We want to maximize coverage of *other* uncovered subsets
            # Or just pick one that covers many uncovered
            for b_idx in candidates:
                if b_idx in selected:
                    continue
                # Count how many uncovered subsets this block covers
                m = block_masks[b_idx]
                # We need to count bits in m where counts[bit] == 0
                # This is slightly expensive, but n_t_subs is small (11628)
                # However, we can just use the mask and check counts?
                # Let's iterate bits of m
                gain = 0
                mm = m
                while mm:
                    lsb = mm & -m
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 0:
                        gain += 1
                    mm ^= lsb
                if gain > max_gain:
                    max_gain = gain
                    best_idx = b_idx
            
            if best_idx != -1:
                add_block(best_idx)
            else:
                break # Should not happen
                
    def try_remove_redundant():
        removed = False
        # Iterate over a copy since we modify selected
        for idx in list(selected):
            # Check if any t-subset is covered ONLY by this block
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
            if not essential:
                remove_block(idx)
                removed = True
        return removed
        
    def try_swap():
        # Try to swap a selected block with an unselected one
        # Condition: The new block must cover all essential subsets of the old block
        for idx in list(selected):
            m = block_masks[idx]
            # Find essential subsets
            essential_subs = []
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential_subs.append(bit_idx)
                m ^= lsb
                
            if not essential_subs:
                continue # Will be removed by try_remove_redundant
                
            # Find candidate blocks that cover ALL essential_subs
            # Intersection of t_to_blocks for each essential sub
            candidates = set(t_to_blocks[essential_subs[0]])
            for sub_idx in essential_subs[1:]:
                candidates.intersection_update(t_to_blocks[sub_idx])
                
            # Filter out selected blocks
            candidates -= selected
            
            if not candidates:
                continue
                
            # Pick a candidate
            # Heuristic: Pick one that covers the most *other* subsets to potentially help redundancy
            # Or just random? Let's try random to explore space.
            # Actually, picking the one with max overlap with current selected set might be good?
            # Let's just pick one.
            new_idx = random.choice(list(candidates))
            
            # Perform swap
            remove_block(idx)
            add_block(new_idx)
            
            # After swap, run redundant removal immediately
            while try_remove_redundant():
                pass
                
            return True
            
        return False

    # Main Logic
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer
    
    greedy_init()
    
    # Initial cleanup
    while try_remove_redundant():
        pass
        
    best_size = len(selected)
    best_solution = set(selected)
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Remove Redundant
        while try_remove_redundant():
            improved = True
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = set(selected)
            
        # Phase 2: Swap
        # Try multiple swaps in one iteration to get out of local minima
        swap_count = 0
        max_swaps_per_iter = 50
        while swap_count < max_swaps_per_iter:
            if try_swap():
                improved = True
                swap_count += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
            else:
                break
                
        if not improved:
            # Perturb
            # Remove some random blocks and repair
            # Number of blocks to remove: proportional to size?
            # Let's remove 5% of blocks or at least 1
            n_remove = max(1, len(selected) // 20)
            
            # Pick random blocks to remove
            # We must ensure we don't remove all blocks
            if n_remove < len(selected):
                to_remove = random.sample(list(selected), n_remove)
                for idx in to_remove:
                    remove_block(idx)
                
                # Repair
                greedy_repair()
                
                # Cleanup
                while try_remove_redundant():
                    pass
                
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                    
            # If we are stuck, maybe try a different perturbation
            # Or just continue loop
            
    # Restore best solution
    # Clear current
    for idx in list(selected):
        remove_block(idx)
    for idx in best_solution:
        add_block(idx)
        
    return [list(k_blocks[i]) for i in best_solution]
