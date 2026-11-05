import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55  # seconds, keep a margin
    
    # Precompute conflict masks for fast checking
    # conflict_masks[x] is a list of bitmasks. 
    # If x is added to S, and (mask & S_mask) == mask, a 4-AP is formed.
    # The mask contains the other 3 elements of the AP.
    conflict_masks = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            # For each element in the AP, the other 3 form the conflict mask
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_masks[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_masks[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_masks[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_masks[term4].append(mask_d)
            
    # Convert to tuples for faster iteration
    conflict_masks = [tuple(lst) for lst in conflict_masks]
    
    # Bit positions for quick membership check
    bit_positions = [0] + [1 << (i - 1) for i in range(1, n + 1)]
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_masks[x]:
            if (mask & S_mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list, order):
        """Try to add all possible elements in the given order."""
        added_any = False
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added_any = True
        return S_mask, S_list, added_any
    
    def generate_block_seed(block_start, block_len=81):
        """
        Generate a 4-AP-free seed for a block [block_start, block_start + block_len - 1]
        using the base-3 (Salem-Spencer) construction.
        
        For a block of length L, we use digits 0 and 1 in base 3.
        The maximum value with k base-3 digits is (3^k - 1)/2.
        For L=81, we need k such that (3^k - 1)/2 >= 80, so 3^k >= 161, k >= 5 (3^5=243).
        But we only use values up to L-1 within the block.
        """
        seed = set()
        # Use base-3 digits 0 and 1
        max_digits = 5  # 3^5 = 243, (243-1)/2 = 121 > 81
        for mask in range(1, 1 << max_digits):
            val = 0
            for i in range(max_digits):
                if (mask >> i) & 1:
                    val += (3 ** i)
            if 1 <= val <= block_len:
                seed.add(block_start + val - 1)  # 1-indexed within block
        return seed
    
    def merge_seeds_conflict_aware(seed_sets):
        """
        Merge multiple 4-AP-free seeds using conflict-aware union.
        For overlapping elements, we resolve conflicts by checking which elements
        can coexist without forming 4-APs.
        """
        merged_mask = 0
        merged_list = []
        
        # Process seeds in order, adding elements that don't conflict
        for seed in seed_sets:
            for x in sorted(seed):
                if not (merged_mask & bit_positions[x]):
                    if can_add(x, merged_mask):
                        merged_mask |= bit_positions[x]
                        merged_list.append(x)
        
        return merged_mask, merged_list
    
    def local_repair(S_mask, S_list, num_iterations=100):
        """
        Local repair: try to fix any issues and add more elements.
        This is a light local search to improve the merged seed.
        """
        for _ in range(num_iterations):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try to add random elements
            added = False
            for _ in range(3):
                x = random.randint(1, n)
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        added = True
                        break
            
            if added:
                continue
            
            # Try a swap
            if len(S_list) > 0:
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                for _ in range(5):
                    x = random.randint(1, n)
                    if S_mask & bit_positions[x]:
                        continue
                    S_mask_without_y = S_mask & ~bit_positions[y]
                    if can_add(x, S_mask_without_y):
                        S_mask = S_mask_without_y | bit_positions[x]
                        S_list[y_idx] = x
                        break
        
        return S_mask, S_list
    
    def density_incremental_construction(block_length=81, overlap=10):
        """
        Density-incremental construction:
        1. Partition [1, n] into overlapping blocks
        2. Generate 4-AP-free seeds for each block
        3. Merge seeds using conflict-aware union
        4. Local repair to improve
        """
        seeds = []
        block_start = 1
        while block_start <= n:
            block_end = min(block_start + block_length - 1, n)
            seed = generate_block_seed(block_start, block_end - block_start + 1)
            seeds.append(seed)
            block_start += block_length - overlap
        
        # Merge all seeds
        S_mask, S_list = merge_seeds_conflict_aware(seeds)
        
        # Local repair
        S_mask, S_list = local_repair(S_mask, S_list, num_iterations=50)
        
        return S_mask, S_list
    
    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # First, try the density-incremental construction
    S_mask, S_list = density_incremental_construction()
    current_size = len(S_list)
    if current_size > best_size:
        best_size = current_size
        best_mask = S_mask
    
    # Also try the original core-based approach for comparison
    core = set()
    for mask in range(1, 1 << 6):  # 6 digits: 3^0 to 3^5
        val = 0
        for i in range(6):
            if (mask >> i) & 1:
                val += (3 ** i)
        if 1 <= val <= n:
            core.add(val)
    
    S_mask_core = 0
    for x in core:
        S_mask_core |= bit_positions[x]
    S_list_core = list(core)
    
    candidates = [i for i in range(1, n + 1) if i not in core]
    random.shuffle(candidates)
    
    for x in candidates:
        if can_add(x, S_mask_core):
            S_mask_core |= bit_positions[x]
            S_list_core.append(x)
    
    if len(S_list_core) > best_size:
        best_size = len(S_list_core)
        best_mask = S_mask_core
        S_mask, S_list = S_mask_core, S_list_core
    
    # Run multiple restarts to find a good solution
    # Each restart: Start from best of density-incremental or core, then SA
    while time.time() - start_time < time_limit:
        # 1. Start from the current best solution
        S_mask = best_mask
        S_list = []
        for i in range(1, n + 1):
            if S_mask & bit_positions[i]:
                S_list.append(i)
        
        current_size = len(S_list)
        
        # 2. Simulated Annealing Local Search
        temperature = 5.0
        min_temperature = 0.01
        cooling_rate = 0.997
        no_improvement_count = 0
        max_no_improvement = 20000
        best_in_sa_size = current_size
        best_in_sa_mask = S_mask
        
        while time.time() - start_time < time_limit and no_improvement_count < max_no_improvement:
            # Try to add an element (always accept if valid)
            added = False
            for _ in range(3):
                x = random.randint(1, n)
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        current_size += 1
                        added = True
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                        if current_size > best_in_sa_size:
                            best_in_sa_size = current_size
                            best_in_sa_mask = S_mask
                        break
            
            if added:
                temperature = max(min_temperature, temperature * 0.999)
                continue
                
            # If no element could be added, try a swap
            y_idx = random.randint(0, len(S_list) - 1)
            y = S_list[y_idx]
            
            swap_made = False
            for _ in range(8):
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    continue
                
                S_mask_without_y = S_mask & ~bit_positions[y]
                if can_add(x, S_mask_without_y):
                    S_mask = S_mask_without_y | bit_positions[x]
                    S_list[y_idx] = x
                    swap_made = True
                    no_improvement_count = 0
                    break
            
            if not swap_made:
                no_improvement_count += 1
                
            temperature = max(min_temperature, temperature * cooling_rate)
            
            # Periodic fill
            if no_improvement_count % 300 == 0:
                S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
                new_size = len(S_list)
                if new_size > current_size:
                    current_size = new_size
                    no_improvement_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                    if current_size > best_in_sa_size:
                        best_in_sa_size = current_size
                        best_in_sa_mask = S_mask

        # Final fill for this restart
        S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask

    # Final polish
    final_list = []
    for i in range(1, n + 1):
        if best_mask & bit_positions[i]:
            final_list.append(i)
    
    if not final_list:
        return [1, 2]
        
    final_mask = best_mask
    for x in range(1, n + 1):
        if not (final_mask & bit_positions[x]):
            if can_add(x, final_mask):
                final_mask |= bit_positions[x]
                final_list.append(x)
    
    if not final_list:
        return [1, 2]
        
    return final_list
