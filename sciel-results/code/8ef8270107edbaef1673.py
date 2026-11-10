import time
import random
import sys

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_size = N + 1
    best_set = []
    
    # Precompute bit operations for speed
    def popcount(x):
        """Fast popcount using bit operations"""
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count
    
    def get_avail_list(mask):
        """Extract list of indices from bitmask."""
        lst = []
        m = mask
        while m:
            lsb = m & -m
            lst.append(lsb.bit_length() - 1)
            m ^= lsb
        return lst
    
    def compute_sums_mask(S_list):
        """Compute sums mask from a list of elements."""
        sums_mask = 0
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            sums_mask |= (1 << (2 * a))
            for j in range(i + 1, n):
                sums_mask |= (1 << (a + S_list[j]))
        return sums_mask
    
    def check_valid_add(S_mask, sums_mask, z):
        """Check if z can be added to S."""
        if (sums_mask >> (2 * z)) & 1:
            return False
        if (sums_mask >> z) & S_mask:
            return False
        return True
    
    def add_element(S_mask, sums_mask, z, S_list):
        """Add element z to S, return new masks and updated list."""
        new_S_mask = S_mask | (1 << z)
        new_sums_mask = sums_mask | (1 << (2 * z))
        
        # Add z+a for all a in S
        temp_S = S_mask
        while temp_S:
            lsb_S = temp_S & -temp_S
            a = lsb_S.bit_length() - 1
            new_sums_mask |= (1 << (z + a))
            temp_S ^= lsb_S
        
        new_S_list = S_list + [z]
        return new_S_mask, new_sums_mask, new_S_list
    
    def remove_element(S_mask, sums_mask, S_list, x):
        """Remove element x from S, return new masks and list."""
        new_S_list = [s for s in S_list if s != x]
        new_S_mask = S_mask ^ (1 << x)
        new_sums_mask = compute_sums_mask(new_S_list)
        return new_S_mask, new_sums_mask, new_S_list
    
    def compute_available_mask(S_mask, sums_mask):
        """Compute which elements are still available to add."""
        avail_mask = 0
        for w in range(1, N + 1):
            if (S_mask >> w) & 1:
                continue
            if check_valid_add(S_mask, sums_mask, w):
                avail_mask |= (1 << w)
        return avail_mask
    
    def greedy_extend(S_mask, sums_mask, available_mask, rng, strategy='max_block'):
        """
        Greedily extend S to a maximal Sidon set.
        strategy: 'max_block', 'min_conflict', or 'random'
        """
        S_list = get_avail_list(S_mask)
        
        while available_mask and time.time() < deadline:
            avail_list = get_avail_list(available_mask)
            if not avail_list:
                break
            
            if strategy == 'random':
                rng.shuffle(avail_list)
            else:
                rng.shuffle(avail_list)
            
            best_z = -1
            best_score = -1.0
            best_blocked = -1
            
            for z in avail_list:
                if not check_valid_add(S_mask, sums_mask, z):
                    continue
                
                # Simulate adding z
                new_S_mask, new_sums_mask, _ = add_element(S_mask, sums_mask, z, S_list)
                
                # Count how many other available elements are blocked
                blocked_count = 0
                for w in avail_list:
                    if w == z:
                        continue
                    if not check_valid_add(new_S_mask, new_sums_mask, w):
                        blocked_count += 1
                
                if strategy == 'max_block':
                    score = float(blocked_count)
                elif strategy == 'min_conflict':
                    # Prefer elements that don't conflict with many others
                    conflict_count = 0
                    for w in avail_list:
                        if w == z:
                            continue
                        if (new_sums_mask >> w) & new_S_mask:
                            conflict_count += 1
                    score = -float(conflict_count)
                else:
                    score = 0.0
                
                score += rng.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_z = z
                    best_blocked = blocked_count
            
            if best_z == -1:
                break
            
            # Commit best_z
            S_mask, sums_mask, S_list = add_element(S_mask, sums_mask, best_z, S_list)
            available_mask &= ~(1 << best_z)
            
            # Update available mask
            available_mask = compute_available_mask(S_mask, sums_mask)
        
        return S_mask, sums_mask, S_list
    
    def backtrack_search(S_mask, sums_mask, available_mask, max_depth=3):
        """
        Deterministic backtracking with MRV heuristic.
        Returns the best maximal Sidon set found.
        """
        S_list = get_avail_list(S_mask)
        best_result = (S_mask, sums_mask, S_list[:])
        
        if max_depth <= 0 or not available_mask:
            return best_result
        
        avail_list = get_avail_list(available_mask)
        
        # MRV: sort by number of conflicts (fewest conflicts first)
        def mrv_score(z):
            if not check_valid_add(S_mask, sums_mask, z):
                return float('inf')
            new_S_mask, new_sums_mask, _ = add_element(S_mask, sums_mask, z, S_list)
            conflicts = 0
            for w in avail_list:
                if w != z and not check_valid_add(new_S_mask, new_sums_mask, w):
                    conflicts += 1
            return conflicts
        
        scored = [(mrv_score(z), z) for z in avail_list if check_valid_add(S_mask, sums_mask, z)]
        scored.sort()
        
        # Try top candidates
        for _, z in scored[:min(5, len(scored))]:
            if time.time() > deadline:
                break
            
            new_S_mask, new_sums_mask, new_S_list = add_element(S_mask, sums_mask, z, S_list)
            new_avail = compute_available_mask(new_S_mask, new_sums_mask)
            
            result = backtrack_search(new_S_mask, new_sums_mask, new_avail, max_depth - 1)
            
            if len(result[2]) < len(best_result[2]):
                best_result = result
            
            # Also try greedy extension from this state
            greedy_result = greedy_extend(new_S_mask, new_sums_mask, new_avail, random.Random(hash((S_mask, z))))
            if len(greedy_result[2]) < len(best_result[2]):
                best_result = greedy_result
        
        return best_result
    
    # Main search loop
    strategies = ['max_block', 'min_conflict', 'random']
    num_iterations = 0
    
    for iter in range(2000):
        if time.time() > deadline:
            break
        
        num_iterations += 1
        rng = random.Random(iter * 7919 + 13)
        
        # Strategy selection
        strategy = strategies[iter % len(strategies)]
        
        # Start with empty set
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1  # Bits 1..N
        
        # Initial greedy extension
        S_mask, sums_mask, S_list = greedy_extend(S_mask, sums_mask, available_mask, rng, strategy)
        
        current_size = len(S_list)
        
        # Local search: try removing elements
        improved = True
        local_iter = 0
        while improved and time.time() < deadline and local_iter < 20:
            improved = False
            local_iter += 1
            
            S_list_copy = S_list[:]
            rng.shuffle(S_list_copy)
            
            for x in S_list_copy:
                if time.time() > deadline:
                    break
                
                new_S_mask, new_sums_mask, new_S_list = remove_element(S_mask, sums_mask, S_list, x)
                new_avail = compute_available_mask(new_S_mask, new_sums_mask)
                
                # Try multiple extension strategies
                best_ext_size = len(new_S_list)
                best_ext = (new_S_mask, new_sums_mask, new_S_list[:])
                
                for strat in strategies:
                    if time.time() > deadline:
                        break
                    
                    ext_rng = random.Random(hash((S_mask, x, strat)))
                    ext_result = greedy_extend(new_S_mask, new_sums_mask, new_avail, ext_rng, strat)
                    
                    if len(ext_result[2]) < best_ext_size:
                        best_ext_size = len(ext_result[2])
                        best_ext = ext_result
                
                # Also try backtracking
                if time.time() < deadline:
                    bt_result = backtrack_search(new_S_mask, new_sums_mask, new_avail, max_depth=2)
                    if len(bt_result[2]) < best_ext_size:
                        best_ext_size = len(bt_result[2])
                        best_ext = bt_result
                
                if best_ext_size < current_size:
                    S_mask, sums_mask, S_list = best_ext
                    current_size = best_ext_size
                    improved = True
                    break
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_list[:]
            
            # Try to improve further with more aggressive local search
            if best_size <= 18:
                # Additional refinement
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    
                    S_list_copy = S_list[:]
                    rng.shuffle(S_list_copy)
                    
                    for x in S_list_copy:
                        if time.time() > deadline:
                            break
                        
                        new_S_mask, new_sums_mask, new_S_list = remove_element(S_mask, sums_mask, S_list, x)
                        new_avail = compute_available_mask(new_S_mask, new_sums_mask)
                        
                        best_ext_size = len(new_S_list)
                        best_ext = (new_S_mask, new_sums_mask, new_S_list[:])
                        
                        for strat in strategies:
                            if time.time() > deadline:
                                break
                            
                            ext_rng = random.Random(hash((S_mask, x, strat, _)))
                            ext_result = greedy_extend(new_S_mask, new_sums_mask, new_avail, ext_rng, strat)
                            
                            if len(ext_result[2]) < best_ext_size:
                                best_ext_size = len(ext_result[2])
                                best_ext = ext_result
                        
                        if best_ext_size < current_size:
                            S_mask, sums_mask, S_list = best_ext
                            current_size = best_ext_size
                            best_size = current_size
                            best_set = S_list[:]
                            break
    
    # Final verification and return
    if not best_set:
        # Fallback: simple greedy
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1
        S_mask, sums_mask, best_set = greedy_extend(S_mask, sums_mask, available_mask, random.Random(42), 'max_block')
    
    return best_set
