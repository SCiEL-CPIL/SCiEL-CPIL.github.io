import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Strategy 1: Enumerate Circulant Graphs
        # Steps are 1 to n//2
        max_step = n // 2
        # We represent the set of steps as a bitmask or a list
        # Number of steps is small (<= 20), so 2^20 ~ 1M combinations is too many for Python if we do it naively for each n.
        # But we can prune triangle-free condition early.
        
        # Actually, let's just try a few specific known good circulants or random subsets of steps.
        # Random subset of steps is faster to generate than full enumeration if we limit attempts.
        
        found = False
        attempts = 0
        max_attempts = 50000
        
        while time.time() < deadline - 5 and attempts < max_attempts and not found:
            attempts += 1
            
            # Generate a random set of steps for a circulant graph
            # To ensure connectivity and density, pick a random number of steps
            # Triangle-free condition: No a,b,c in S such that a+b+c = 0 mod n
            # This is equivalent to: S is a sum-free set in Z_n (with some caveats for 2a+b=0)
            
            # Let's try to construct a "Sum-Free" set of steps.
            # Start with empty set, add random steps if they don't create a triangle.
            adj = [0] * n
            steps = set()
            
            # Randomize the order of potential steps
            potential_steps = list(range(1, max_step + 1))
            random.shuffle(potential_steps)
            
            for s in potential_steps:
                # Check if adding s creates a triangle
                # A triangle is formed if there exist s1, s2 in steps such that:
                # 1. s1 + s2 + s = 0 mod n
                # 2. 2*s1 + s = 0 mod n
                # 3. s1 + 2*s = 0 mod n
                # Basically, for any pair (s1, s2) in steps, check if -s1-s2 is in steps or s.
                
                is_safe = True
                for s1 in steps:
                    # Check s1 + s1 + s = 0
                    if (2 * s1 + s) % n == 0:
                        is_safe = False
                        break
                    # Check s1 + s2 + s = 0 for s2 in steps
                    # We can just check if (-s1 - s) % n is in steps
                    s3 = (-s1 - s) % n
                    # Note: s3 could be 0, but 0 is not in steps.
                    # s3 could be equal to s1 or s.
                    if s3 in steps:
                        # If s3 == s1, then 2*s1 + s = 0, already checked.
                        # If s3 == s, then s1 + 2*s = 0.
                        # If s3 is distinct, then s1 + s + s3 = 0.
                        is_safe = False
                        break
                    # Also check if s3 == s?
                    # If (-s1 - s) % n == s, then s1 + 2s = 0.
                    # This is covered by the loop if we check carefully?
                    # Let's just check all pairs in steps including s itself?
                    # No, we are adding s. So we check interactions with existing steps.
                    # If s3 == s, it means s1 + 2s = 0.
                    if s3 == s:
                        is_safe = False
                        break
                
                if is_safe:
                    steps.add(s)
                    # Add edges to adj
                    for i in range(n):
                        j = (i + s) % n
                        if i < j:
                            adj[i] |= (1 << j)
                        else:
                            adj[j] |= (1 << i)
                        # Wait, circulant graph is undirected.
                        # Edge (i, i+s) is same as (i+s, i).
                        # We just need to set bits correctly.
            
            # If we didn't add enough steps, the graph might be too sparse (large alpha).
            # We want dense graphs to keep alpha small.
            # Let's check the size of steps. If len(steps) < n//4, it's likely too sparse.
            # But let's just check alpha.
            
            if len(steps) < 5:
                continue
                
            # Fast check for alpha >= 10
            # Use the exact checker
            
            def has_independent_set_10():
                all_mask = (1 << n) - 1
                # Precompute non_adj
                non_adj = [0] * n
                for i in range(n):
                    non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
                
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    if mask.bit_count() < 10 - size:
                        return False
                    
                    # Heuristic: Pick vertex with LOWEST degree in mask
                    # Why? Including it removes few vertices, maximizing chance to find a large set.
                    best_v = -1
                    min_deg = n + 1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        deg = (adj[v] & mask).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        temp ^= lsb
                        
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Option 1: Include best_v
                    next_mask = non_adj[best_v] & mask
                    if dfs(next_mask, size + 1):
                        return True
                    
                    # Option 2: Exclude best_v
                    next_mask_excl = mask & ~v_bit
                    if dfs(next_mask_excl, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
            
        # If no circulant found, try random greedy (old method) as fallback
        # But we already spent time.
        
    return best_result
