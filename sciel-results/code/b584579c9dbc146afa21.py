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

    def greedy_repair(s_mask):
        """
        Remove elements to make s_mask valid.
        """
        while time.time() < deadline:
            triple = find_bad_triple(s_mask)
            if not triple:
                break
            
            i, j, k = triple
            # Heuristic: remove the element with the most conflicts?
            # Or just random.
            # Let's try removing the one with highest degree in the conflict graph restricted to S.
            # Degree of x in S: number of pairs (y,z) in S such that {x,y,z} is bad.
            # This is expensive to compute.
            # Simpler: Remove random element from the triple.
            # Or remove the largest number?
            # Let's try removing the one that appears in the most bad triples in S.
            # Approximation: count how many other elements in S conflict with x.
            
            # To be fast, just pick one randomly from the triple.
            # Or pick the one with smallest "value"?
            # Let's pick the one that is "most central".
            # For now, random.
            candidates = [i, j, k]
            random.shuffle(candidates)
            to_remove = candidates[0]
            
            s_mask &= ~(1 << to_remove)
            
        return s_mask

    def expand(s_mask):
        """
        Try to add elements to s_mask.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try adding elements in random order
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            
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
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            elements_in = [i for i in range(1, N + 1) if (s_mask >> i) & 1]
            elements_out = [i for i in range(1, N + 1) if not ((s_mask >> i) & 1)]
            
            random.shuffle(elements_in)
            random.shuffle(elements_out)
            
            # Try 1-for-1 swaps
            for u in elements_in:
                for v in elements_out:
                    if time.time() > deadline:
                        break
                    new_mask = (s_mask & ~(1 << u)) | (1 << v)
                    if is_valid(new_mask):
                        # Check if this swap enables more additions?
                        # For now, just accept if it maintains validity.
                        # But we want to maximize size.
                        # If we just swap, size is same.
                        # We should only swap if it helps in a subsequent expansion.
                        # To keep it simple, let's just do expansion after swap.
                        # But that's expensive.
                        # Let's just do "Add if possible" logic in expand.
                        # So local search should focus on "Unstuck" moves.
                        # If we are stuck (can't add), try a swap.
                        pass
            
            # Better local search:
            # If we can't add any element, try to remove one and add two?
            # Or just restart?
            
        return s_mask

    best_size = 0
    best_set = 0
    
    random.seed(42)
    
    # Strategy:
    # 1. Start with full set, repair to valid.
    # 2. Expand.
    # 3. Repeat with different random seeds / removal heuristics.
    
    for run in range(50):
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
            
        # Local Search: Try to swap elements to escape local optimum
        # Simple hill climbing: try to add. If fail, remove random, try add.
        for step in range(100):
            if time.time() > deadline:
                break
            # Try to add
            elements_out = [i for i in range(1, N + 1) if not ((s >> i) & 1)]
            random.shuffle(elements_out)
            added = False
            for x in elements_out:
                new_s = s | (1 << x)
                if is_valid(new_s):
                    s = new_s
                    added = True
                    break
                    
            if not added:
                # Stuck. Remove a random element and try to add more.
                elements_in = [i for i in range(1, N + 1) if (s >> i) & 1]
                if elements_in:
                    u = random.choice(elements_in)
                    s &= ~(1 << u)
                    
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s

    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    if not result:
        return [1]
    return result
