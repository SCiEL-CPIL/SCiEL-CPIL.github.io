import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    
    # Precompute bad triples bitmask B[i][j]
    # B[i][j] is a mask of k such that {i, j, k} is a bad triple.
    B = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Compute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    mask |= (1 << k)
            B[i][j] = mask
            B[j][i] = mask

    # Precompute conflict count for each number: how many other numbers form a bad triple with it
    conflict_count = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if B[i][j] != 0:
                conflict_count[i] += 1
                conflict_count[j] += 1

    def is_valid_addition(x, current_set_mask):
        """
        Check if x can be added to the set represented by current_set_mask.
        """
        if (current_set_mask >> x) & 1:
            return False
            
        temp = current_set_mask
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            if (B[x][y] & current_set_mask) != 0:
                return False
            temp ^= lsb
            
        return True

    def greedy_build(order):
        s = 0
        for x in order:
            if is_valid_addition(x, s):
                s |= (1 << x)
        return s

    def local_search(s, time_limit):
        """
        Try to improve s using 1-for-1, 1-for-2, and 2-for-1 swaps.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            elements_in_s = [i for i in range(1, N + 1) if (s >> i) & 1]
            elements_not_in_s = [i for i in range(1, N + 1) if not ((s >> i) & 1)]
            
            random.shuffle(elements_in_s)
            random.shuffle(elements_not_in_s)
            
            # Try 1-for-2 swaps (grow by 1)
            for u in elements_in_s:
                if time.time() > time_limit:
                    break
                
                s_temp = s & ~(1 << u)
                
                for v in elements_not_in_s:
                    if is_valid_addition(v, s_temp):
                        s_temp_with_v = s_temp | (1 << v)
                        
                        for w in elements_not_in_s:
                            if w == v:
                                continue
                            if is_valid_addition(w, s_temp_with_v):
                                s = s_temp_with_v | (1 << w)
                                improved = True
                                break
                        if improved:
                            break
                            
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-for-1 swaps (shrink by 1 to potentially enable bigger growth later)
            # This helps escape local minima
            for idx_u in range(len(elements_in_s)):
                if time.time() > time_limit:
                    break
                u = elements_in_s[idx_u]
                
                for idx_v in range(idx_u + 1, len(elements_in_s)):
                    v = elements_in_s[idx_v]
                    
                    s_temp = s & ~(1 << u) & ~(1 << v)
                    
                    for w in elements_not_in_s:
                        if is_valid_addition(w, s_temp):
                            s = s_temp | (1 << w)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swaps (maintain size, change structure)
            for u in elements_in_s:
                if time.time() > time_limit:
                    break
                
                s_temp = s & ~(1 << u)
                
                for w in elements_not_in_s:
                    if is_valid_addition(w, s_temp):
                        # Check if this swap actually helps by trying to grow
                        s_candidate = s_temp | (1 << w)
                        
                        # Try to add one more element
                        grew = False
                        for x in elements_not_in_s:
                            if x == w:
                                continue
                            if is_valid_addition(x, s_candidate):
                                s = s_candidate | (1 << x)
                                improved = True
                                grew = True
                                break
                        if grew:
                            break
                        
                        # Even if we didn't grow immediately, accept the swap to change structure
                        # with some probability to escape local minima
                        if random.random() < 0.1:
                            s = s_candidate
                            improved = True
                            break
                            
                if improved:
                    break
                
        return s

    best_size = 0
    best_set = 0
    
    random.seed(42)
    
    # Create multiple orderings
    # 1. Sorted by conflict count (fewest conflicts first) - "safe" numbers first
    order_by_conflict = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
    
    # 2. Reverse conflict order
    order_by_conflict_rev = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
    
    # 3. Natural order
    order_natural = list(range(1, N + 1))
    
    # 4. Reverse natural
    order_natural_rev = list(range(N, 0, -1))
    
    orders = [order_by_conflict, order_by_conflict_rev, order_natural, order_natural_rev]
    
    # 5-14. Random shuffles
    for _ in range(10):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        s = greedy_build(order)
        s = local_search(s, deadline)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s

    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    
    if not result:
        return [1]
        
    return result
