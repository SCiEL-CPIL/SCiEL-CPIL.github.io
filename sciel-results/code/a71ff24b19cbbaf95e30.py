import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_size = N + 1
    best_set = []
    
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
                b = S_list[j]
                sums_mask |= (1 << (a + b))
        return sums_mask
    
    def compute_available_mask(S_mask, sums_mask):
        """Compute which elements can be added to S."""
        available = 0
        # For each w not in S, check if w can be added
        # w can be added if: 2w not in sums AND no a in S such that w+a in sums
        # (sums >> w) & S_mask == 0 means no a in S has w+a in sums
        
        # Efficient approach: iterate through all w from 1 to N
        for w in range(1, N + 1):
            if (S_mask >> w) & 1:
                continue
            if (sums_mask >> (2 * w)) & 1:
                continue
            if (sums_mask >> w) & S_mask:
                continue
            available |= (1 << w)
        return available
    
    def greedy_extend(S_mask, sums_mask, available_mask, rng, alpha=1.0, beta=0.3):
        """
        Greedily extend S to a maximal Sidon set.
        alpha: weight for blocking count
        beta: weight for randomness/diversity
        """
        S_list = get_avail_list(S_mask)
        
        while available_mask:
            avail_list = get_avail_list(available_mask)
            if not avail_list:
                break
            
            # Shuffle for randomness
            rng.shuffle(avail_list)
            
            best_z = -1
            best_score = -1.0
            
            for z in avail_list:
                # Check if z is valid to add to current S
                if (sums_mask >> (2 * z)) & 1:
                    continue
                if (sums_mask >> z) & S_mask:
                    continue
                
                # Simulate adding z
                new_S_mask = S_mask | (1 << z)
                new_sums_mask = sums_mask | (1 << (2 * z))
                
                # Add z+a for a in S
                temp_S = S_mask
                add_bits = 0
                while temp_S:
                    lsb_S = temp_S & -temp_S
                    a = lsb_S.bit_length() - 1
                    add_bits |= (1 << (z + a))
                    temp_S ^= lsb_S
                new_sums_mask |= add_bits
                
                # Count how many other available elements are blocked
                blocked_count = 0
                for w in avail_list:
                    if w == z:
                        continue
                    if (new_sums_mask >> (2 * w)) & 1:
                        blocked_count += 1
                        continue
                    if (new_sums_mask >> w) & new_S_mask:
                        blocked_count += 1
                
                # Score: maximize blocking, add noise for diversity
                score = alpha * blocked_count + beta * rng.random()
                
                if score > best_score:
                    best_score = score
                    best_z = z
            
            if best_z == -1:
                break
            
            # Commit best_z
            S_mask = S_mask | (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            sums_mask |= (1 << (2 * best_z))
            temp_S = S_mask ^ (1 << best_z)  # Old S
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                sums_mask |= (1 << (best_z + a))
                temp_S ^= lsb_S
            
            # Update available_mask
            available_mask &= ~(1 << best_z)
            
            # Remove blocked elements
            new_avail_mask = 0
            temp_avail = available_mask
            while temp_avail:
                lsb_w = temp_avail & -temp_avail
                w = lsb_w.bit_length() - 1
                if not ((sums_mask >> (2 * w)) & 1):
                    if not ((sums_mask >> w) & S_mask):
                        new_avail_mask |= lsb_w
                temp_avail ^= lsb_w
            
            available_mask = new_avail_mask
        
        return S_mask, sums_mask, available_mask, S_list
    
    def try_remove_and_extend(S_list, x, rng, alpha, beta):
        """Try removing element x and greedily extending. Returns (new_S_list, improved) or (None, False)."""
        new_S_list = [s for s in S_list if s != x]
        new_S_mask = 0
        for s in new_S_list:
            new_S_mask |= (1 << s)
        
        new_sums_mask = compute_sums_mask(new_S_list)
        new_avail_mask = compute_available_mask(new_S_mask, new_sums_mask)
        
        new_S_mask, new_sums_mask, _, new_S_list_ext = greedy_extend(
            new_S_mask, new_sums_mask, new_avail_mask, rng, alpha, beta
        )
        
        if len(new_S_list_ext) < len(S_list):
            return new_S_list_ext, True
        return None, False
    
    def two_opt_search(S_list, rng, alpha, beta, max_rounds=3):
        """
        2-opt: try swapping pairs of elements.
        Remove two elements, greedily extend, see if we get a smaller set.
        """
        current_S = S_list[:]
        n = len(current_S)
        
        for _ in range(max_rounds):
            improved = False
            # Try all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return current_S
                    
                    x, y = current_S[i], current_S[j]
                    
                    # Remove both x and y
                    new_S_list = [s for s in current_S if s != x and s != y]
                    new_S_mask = 0
                    for s in new_S_list:
                        new_S_mask |= (1 << s)
                    
                    new_sums_mask = compute_sums_mask(new_S_list)
                    new_avail_mask = compute_available_mask(new_S_mask, new_sums_mask)
                    
                    new_S_mask, new_sums_mask, _, new_S_list_ext = greedy_extend(
                        new_S_mask, new_sums_mask, new_avail_mask, rng, alpha, beta
                    )
                    
                    if len(new_S_list_ext) < len(current_S):
                        current_S = new_S_list_ext
                        n = len(current_S)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current_S
    
    # Main Loop
    for iter in range(2000):
        if time.time() > deadline:
            break
        
        rng = random.Random(iter)
        
        # Vary the heuristic parameters for diversity
        alpha = 1.0 + rng.random() * 0.5
        beta = 0.1 + rng.random() * 0.4
        
        # Start with empty set
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1  # Bits 1..N
        
        # Initial Greedy Extension
        S_mask, sums_mask, available_mask, S_list = greedy_extend(S_mask, sums_mask, available_mask, rng, alpha, beta)
        
        current_size = len(S_list)
        
        # Local Search 1: Try to remove elements and re-extend
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list_copy = S_list[:]
            rng.shuffle(S_list_copy)
            
            for x in S_list_copy:
                if time.time() > deadline:
                    break
                
                new_S_list, did_improve = try_remove_and_extend(S_list, x, rng, alpha, beta)
                if did_improve:
                    S_list = new_S_list
                    current_size = len(S_list)
                    improved = True
                    break  # Restart the while loop with new S
        
        # Local Search 2: 2-opt swaps
        if time.time() < deadline:
            S_list = two_opt_search(S_list, rng, alpha, beta, max_rounds=2)
            current_size = len(S_list)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_list[:]
    
    return best_set
