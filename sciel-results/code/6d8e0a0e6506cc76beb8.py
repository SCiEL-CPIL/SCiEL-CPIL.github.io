import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    
    # 1. Precompute conflict masks
    # conflict_mask[i][j] is a bitmask of k such that {i, j, k} is a bad triple.
    # We only need i < j, but storing full matrix is easier and fast enough.
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Build conflict masks
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
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
    
    # Precompute conflict degree: for each element x, how many pairs (y,z) have {x,y,z} bad
    # This helps with removal heuristic
    conflict_degree = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            mask = conflict_mask[i][j]
            # Count how many k are in mask
            cnt = bin(mask).count('1')
            conflict_degree[i] += cnt
            conflict_degree[j] += cnt

    def is_valid(s_mask):
        """
        Check if the set represented by s_mask is valid.
        """
        # Iterate through all pairs in the set
        temp = s_mask
        while temp:
            lsb_i = temp & -temp
            i = lsb_i.bit_length() - 1
            temp ^= lsb_i
            
            # For each i, check against all j > i in the set
            temp_j = s_mask & ~((1 << (i + 1)) - 1) # j > i
            
            while temp_j:
                lsb_j = temp_j & -temp_j
                j = lsb_j.bit_length() - 1
                temp_j ^= lsb_j
                
                # Check if any k in s_mask is in conflict_mask[i][j]
                # We exclude i and j from the check implicitly? 
                # conflict_mask[i][j] contains k such that {i,j,k} is bad.
                # If k is i or j, it's not in the mask because k != i, j in precomputation.
                # So we just check intersection with s_mask.
                if (conflict_mask[i][j] & s_mask) != 0:
                    return False
                    
        return True

    def find_bad_triple(s_mask):
        """
        Find a bad triple in s_mask. Returns (i, j, k) or None.
        """
        temp = s_mask
        while temp:
            lsb_i = temp & -temp
            i = lsb_i.bit_length() - 1
            temp ^= lsb_i
            
            temp_j = s_mask & ~((1 << (i + 1)) - 1)
            while temp_j:
                lsb_j = temp_j & -temp_j
                j = lsb_j.bit_length() - 1
                temp_j ^= lsb_j
                
                # Find k in s_mask that is in conflict_mask[i][j]
                candidates = conflict_mask[i][j] & s_mask
                if candidates:
                    lsb_k = candidates & -candidates
                    k = lsb_k.bit_length() - 1
                    return (i, j, k)
        return None

    def compute_conflict_degree_in(s_mask):
        """
        Compute conflict degree of each element in s_mask.
        Degree of x = number of pairs (y,z) in s_mask such that {x,y,z} is bad.
        """
        degree = [0] * (N + 1)
        temp = s_mask
        while temp:
            lsb_i = temp & -temp
            i = lsb_i.bit_length() - 1
            temp ^= lsb_i
            
            temp_j = s_mask & ~((1 << (i + 1)) - 1)  # j > i
            while temp_j:
                lsb_j = temp_j & -temp_j
                j = lsb_j.bit_length() - 1
                temp_j ^= lsb_j
                
                # Find k in s_mask that is in conflict_mask[i][j]
                candidates = conflict_mask[i][j] & s_mask
                cnt = bin(candidates).count('1')
                degree[i] += cnt
                degree[j] += cnt
        return degree

    def greedy_repair(s_mask):
        """
        Remove elements to make s_mask valid.
        Uses conflict-degree heuristic to choose which element to remove.
        """
        while time.time() < deadline:
            triple = find_bad_triple(s_mask)
            if not triple:
                break
            
            i, j, k = triple
            
            # Compute conflict degrees in current set
            degree = compute_conflict_degree_in(s_mask)
            
            # Remove the element with highest conflict degree
            candidates = [i, j, k]
            to_remove = max(candidates, key=lambda x: degree[x])
            
            s_mask &= ~(1 << to_remove)
            
        return s_mask

    def compute_conflicts_with_set(x, s_mask):
        """
        Count how many elements in s_mask conflict with x.
        An element y in s_mask conflicts with x if there exists z in s_mask
        such that {x, y, z} is a bad triple.
        """
        count = 0
        temp = s_mask
        while temp:
            lsb_y = temp & -temp
            y = lsb_y.bit_length() - 1
            temp ^= lsb_y
            
            if x == y:
                continue
            
            # Check if there's a z in s_mask such that {x, y, z} is bad
            candidates = conflict_mask[x][y] & s_mask
            if candidates:
                count += 1
                
        return count

    def expand(s_mask):
        """
        Try to add elements to s_mask.
        Try elements in order of fewest conflicts with current set.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try adding elements in order of fewest conflicts
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            
            # Sort by conflict count (fewest first)
            def conflict_key(x):
                if (s_mask >> x) & 1:
                    return -1  # Already in set, skip
                return compute_conflicts_with_set(x, s_mask)
            
            elements.sort(key=conflict_key)
            
            for x in elements:
                if (s_mask >> x) & 1:
                    continue
                new_mask = s_mask | (1 << x)
                if is_valid(new_mask):
                    s_mask = new_mask
                    changed = True
                    
        return s_mask

    def local_search(s_mask):
        """
        Try to improve s_mask by swapping elements.
        Focus on swaps that might enable subsequent additions.
        """
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and time.time() < deadline and iteration < max_iterations:
            iteration += 1
            improved = False
            
            elements_in = [i for i in range(1, N + 1) if (s_mask >> i) & 1]
            elements_out = [i for i in range(1, N + 1) if not ((s_mask >> i) & 1)]
            
            random.shuffle(elements_in)
            random.shuffle(elements_out)
            
            # Try 1-for-1 swaps that might enable more additions
            best_swap = None
            best_expanded_size = -1
            
            for u in elements_in:
                for v in elements_out:
                    if time.time() > deadline:
                        break
                    new_mask = (s_mask & ~(1 << u)) | (1 << v)
                    if is_valid(new_mask):
                        # Try to expand from this state
                        expanded = expand(new_mask)
                        expanded_size = bin(expanded).count('1')
                        
                        if expanded_size > best_expanded_size:
                            best_expanded_size = expanded_size
                            best_swap = expanded
            
            if best_swap is not None and best_expanded_size > bin(s_mask).count('1'):
                s_mask = best_swap
                improved = True
                
        return s_mask

    best_size = 0
    best_set = 0
    
    random.seed(42)
    
    # Strategy:
    # 1. Start with full set, repair to valid.
    # 2. Expand.
    # 3. Local search with swaps.
    # 4. Repeat with different random seeds.
    
    for run in range(30):
        if time.time() > deadline:
            break
            
        # Start with full set
        s = (1 << (N + 1)) - 2  # Bits 1..128 set
        
        # Repair
        s = greedy_repair(s)
        
        # Expand
        s = expand(s)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s
            
        # Local search with swap-based improvement
        s = local_search(s)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s

    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    if not result:
        return [1]
    return result
