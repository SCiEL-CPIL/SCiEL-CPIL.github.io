import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute for each pair (i,j), the set of k that form a bad triple
    bad_triple = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                if gcd_table[i][k] == g_ij and gcd_table[j][k] == g_ij:
                    mask |= (1 << k)
            bad_triple[i][j] = mask
            bad_triple[j][i] = mask
    
    def is_valid(s_mask):
        """Check if set represented by s_mask has no bad triple."""
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
                if (bad_triple[i][j] & s_mask) != 0:
                    return False
        return True
    
    def find_bad_triple(s_mask):
        """Find a bad triple in s_mask. Returns (i,j,k) or None."""
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
                candidates = bad_triple[i][j] & s_mask
                if candidates:
                    lsb_k = candidates & -candidates
                    k = lsb_k.bit_length() - 1
                    return (i, j, k)
        return None
    
    def count_conflicts(x, s_mask):
        """Count how many pairs in s_mask conflict with x."""
        cnt = 0
        temp = s_mask
        while temp:
            lsb_y = temp & -temp
            y = lsb_y.bit_length() - 1
            temp ^= lsb_y
            if x == y:
                continue
            if (bad_triple[x][y] & s_mask) != 0:
                cnt += 1
        return cnt
    
    def greedy_construct():
        """Build a valid set greedily, prioritizing elements with low conflict potential."""
        element_badness = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                mask = bad_triple[i][j]
                cnt = bin(mask).count('1')
                element_badness[i] += cnt
                element_badness[j] += cnt
        
        elements = sorted(range(1, N + 1), key=lambda x: element_badness[x])
        s_mask = 0
        for x in elements:
            new_mask = s_mask | (1 << x)
            if is_valid(new_mask):
                s_mask = new_mask
        return s_mask
    
    def local_search(s_mask, max_iter=200):
        """Improve s_mask via 1-for-1 swaps and additions."""
        improved = True
        iteration = 0
        current_size = bin(s_mask).count('1')
        
        while improved and time.time() < deadline and iteration < max_iter:
            iteration += 1
            improved = False
            
            elements_in = [i for i in range(1, N + 1) if (s_mask >> i) & 1]
            elements_out = [i for i in range(1, N + 1) if not ((s_mask >> i) & 1)]
            
            # Try additions first
            random.shuffle(elements_out)
            for v in elements_out:
                new_mask = s_mask | (1 << v)
                if is_valid(new_mask):
                    s_mask = new_mask
                    current_size += 1
                    improved = True
            
            if not improved:
                # Try 1-for-1 swaps
                random.shuffle(elements_in)
                random.shuffle(elements_out)
                best_swap = None
                best_size = current_size
                
                for u in elements_in:
                    for v in elements_out:
                        if time.time() > deadline:
                            break
                        new_mask = (s_mask & ~(1 << u)) | (1 << v)
                        if is_valid(new_mask):
                            # Try to expand from here
                            temp_mask = new_mask
                            temp_size = bin(temp_mask).count('1')
                            # Quick expansion attempt
                            for w in elements_out:
                                if w == v:
                                    continue
                                nm = temp_mask | (1 << w)
                                if is_valid(nm):
                                    temp_mask = nm
                                    temp_size += 1
                            if temp_size > best_size:
                                best_size = temp_size
                                best_swap = temp_mask
                
                if best_swap is not None and best_size > current_size:
                    s_mask = best_swap
                    current_size = best_size
                    improved = True
        
        return s_mask
    
    def simulated_annealing(s_mask, start_temp=10.0, end_temp=0.1, steps=5000):
        """SA-based local search."""
        current = s_mask
        current_size = bin(current).count('1')
        best = current
        best_size = current_size
        temp = start_temp
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            elements_in = [i for i in range(1, N + 1) if (current >> i) & 1]
            elements_out = [i for i in range(1, N + 1) if not ((current >> i) & 1)]
            
            new_mask = current
            
            if move_type < 0.3 and elements_out:
                # Add
                v = random.choice(elements_out)
                new_mask = current | (1 << v)
            elif move_type < 0.5 and elements_in and len(elements_in) > 1:
                # Remove
                u = random.choice(elements_in)
                new_mask = current & ~(1 << u)
            elif elements_in and elements_out:
                # Swap
                u = random.choice(elements_in)
                v = random.choice(elements_out)
                new_mask = (current & ~(1 << u)) | (1 << v)
            
            if new_mask == current:
                continue
            
            if is_valid(new_mask):
                new_size = bin(new_mask).count('1')
                delta = new_size - current_size
                if delta >= 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                    current = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best = current
                        best_size = current_size
            else:
                # Invalid move: penalty
                delta = -1  # Penalty for invalid
                if random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                    current = new_mask
                    current_size = bin(current).count('1')
            
            temp = max(end_temp, temp * 0.999)
        
        return best
    
    def branch_and_bound(initial_mask, time_limit=30.0):
        """
        Branch-and-bound for maximum independent set in 3-uniform hypergraph.
        Uses greedy upper bound for pruning.
        """
        best_mask = initial_mask
        best_size = bin(initial_mask).count('1')
        start_bb = time.time()
        
        # Precompute conflict sets: for each element x, which other elements conflict with it
        # conflict_with[x] = bitmask of y such that there exists z where (x,y,z) is a bad triple
        # Actually, we need: if we have x and y in our set, then we cannot have any z in bad_triple[x][y]
        
        # Greedy upper bound: given a partial solution (current_mask) and remaining candidates (avail_mask),
        # estimate max additional elements we can add
        def greedy_upper_bound(current_mask, avail_mask):
            """Estimate max elements we can still add to current_mask from avail_mask."""
            # Greedily add elements that don't create conflicts
            remaining = avail_mask
            count = 0
            while remaining:
                # Pick the element with fewest conflicts among remaining
                best_x = -1
                min_conflicts = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    x = lsb.bit_length() - 1
                    temp ^= lsb
                    # Count conflicts: how many pairs (x, y) in remaining would be problematic
                    # Actually, just count how many other elements in remaining conflict with x
                    conflict_cnt = 0
                    temp2 = remaining
                    while temp2:
                        lsb2 = temp2 & -temp2
                        y = lsb2.bit_length() - 1
                        temp2 ^= lsb2
                        if y != x and (bad_triple[x][y] & (current_mask | remaining)) != 0:
                            conflict_cnt += 1
                    if conflict_cnt < min_conflicts:
                        min_conflicts = conflict_cnt
                        best_x = x
                if best_x == -1:
                    break
                # Check if adding best_x is valid
                new_mask = current_mask | (1 << best_x)
                if is_valid(new_mask):
                    current_mask = new_mask
                    remaining &= ~(1 << best_x)
                    count += 1
                else:
                    remaining &= ~(1 << best_x)
            return count
        
        # Branch and bound
        def bb_search(current_mask, avail_mask, depth=0):
            nonlocal best_mask, best_size
            if time.time() - start_bb > time_limit:
                return
            
            current_size = bin(current_mask).count('1')
            
            # Upper bound: current_size + greedy_upper_bound
            ub = current_size + greedy_upper_bound(current_mask, avail_mask)
            if ub <= best_size:
                return
            
            if avail_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            # Try adding each available element
            temp = avail_mask
            # Sort by some heuristic: prefer elements with fewer conflicts
            elements = []
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length() - 1
                temp ^= lsb
                # Count conflicts with current set
                conf = 0
                temp2 = current_mask
                while temp2:
                    lsb2 = temp2 & -temp2
                    y = lsb2.bit_length() - 1
                    temp2 ^= lsb2
                    if (bad_triple[x][y] & current_mask) != 0:
                        conf += 1
                elements.append((conf, x))
            elements.sort()
            
            for conf, x in elements:
                if time.time() - start_bb > time_limit:
                    return
                
                # Skip if x conflicts with current set
                if not is_valid(current_mask | (1 << x)):
                    continue
                
                new_current = current_mask | (1 << x)
                new_avail = avail_mask & ~(1 << x)
                # Remove elements that would conflict with x
                # An element y in new_avail conflicts with x if there exists z in new_current such that (x,y,z) is bad
                # But we need to be more careful: y conflicts with x if adding y to new_current creates a bad triple
                # For simplicity, we just keep all and let is_valid check later
                # But for pruning, we can remove elements that definitely conflict
                temp2 = new_avail
                filtered_avail = 0
                while temp2:
                    lsb2 = temp2 & -temp2
                    y = lsb2.bit_length() - 1
                    temp2 ^= lsb2
                    # Check if y can be added to new_current
                    if is_valid(new_current | (1 << y)):
                        filtered_avail |= (1 << y)
                
                bb_search(new_current, filtered_avail, depth + 1)
        
        # Start with the initial solution
        # Available elements are those not in initial_mask
        avail = 0
        for i in range(1, N + 1):
            if not ((initial_mask >> i) & 1):
                avail |= (1 << i)
        
        bb_search(initial_mask, avail)
        
        return best_mask
    
    # Main search loop
    best_set = 0
    best_size = 0
    
    random.seed(42)
    
    # Multiple strategies
    for run in range(20):
        if time.time() > deadline:
            break
        
        # Strategy 1: Greedy from scratch with random ordering
        if run < 5:
            s_mask = greedy_construct()
        else:
            # Start with random subset
            s_mask = 0
            for i in range(1, N + 1):
                if random.random() < 0.5:
                    s_mask |= (1 << i)
            # Repair
            while time.time() < deadline:
                triple = find_bad_triple(s_mask)
                if not triple:
                    break
                i, j, k = triple
                # Remove the one with most conflicts
                deg = [count_conflicts(x, s_mask) for x in (i, j, k)]
                to_remove = (i, j, k)[deg.index(max(deg))]
                s_mask &= ~(1 << to_remove)
        
        # Local search
        s_mask = local_search(s_mask)
        
        size = bin(s_mask).count('1')
        if size > best_size:
            best_size = size
            best_set = s_mask
        
        # Simulated annealing
        if time.time() < deadline - 5.0:
            s_mask = simulated_annealing(s_mask)
            size = bin(s_mask).count('1')
            if size > best_size:
                best_size = size
                best_set = s_mask
    
    # Branch-and-bound polish on the best solution found
    if time.time() < deadline - 10.0:
        bb_result = branch_and_bound(best_set, time_limit=min(25.0, deadline - time.time() - 5.0))
        bb_size = bin(bb_result).count('1')
        if bb_size > best_size:
            best_size = bb_size
            best_set = bb_result
    
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    if not result:
        return [1]
    return result
