import time
import random
import sys

DEADLINE = 145.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Target n values to try. We start high and go lower if we can't find a valid solution quickly.
    # Known record is 50. We aim for 50 or 51.
    # If we find a valid 50, we try to push to 51.
    
    def count_triangles(n, colors):
        """
        Fast count of monochromatic triangles using bitmasks.
        colors: list of length n*(n-1)//2, values 0..3.
        Returns: number of monochromatic triangles.
        """
        # Build adjacency bitmasks for each color
        # adj[c][i] is a bitmask of vertices j such that edge (i,j) has color c.
        # For n <= 64, we can use Python integers as bitmasks.
        
        adj = [[0] * n for _ in range(4)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        total_triangles = 0
        for c in range(4):
            for i in range(n):
                # For each vertex i, count triangles (i, j, k) where all edges are color c.
                # This is equivalent to counting edges in the subgraph induced by neighbors of i in color c.
                # Let N_i be the set of neighbors of i in color c.
                # The number of triangles containing i is the number of edges in the induced subgraph on N_i.
                # Summing over all i counts each triangle 3 times.
                ni = adj[c][i]
                # Iterate over neighbors j of i
                # To avoid double counting edges in the induced subgraph, we can just sum degrees / 2?
                # No, simpler: for each j in N_i, count how many k in N_i with k > j?
                # Even simpler: for each j in N_i, count (adj[c][j] & ni) >> j? No.
                # Let's use the property: number of triangles in graph G is sum_{v} (deg(v) choose 2) / 3? No, that's for cliques? No.
                # Number of triangles = 1/3 * sum_{v} (number of edges in N(v)).
                # Let's just iterate.
                # For each j such that (ni >> j) & 1:
                #   common = ni & adj[c][j]
                #   count += common.bit_count()
                # This counts each edge (j,k) in N_i twice (once from j, once from k)?
                # No, if we iterate j, and count bits in (ni & adj[c][j]), we count the edge (j,k) for each j.
                # So sum_{j in N_i} (ni & adj[c][j]).bit_count() counts each edge in N_i twice.
                # So edges_in_Ni = 0.5 * sum...
                # And triangles_containing_i = edges_in_Ni.
                # So total triangles = 1/3 * sum_{i} triangles_containing_i.
                
                # Optimization:
                # sum_deg = 0
                # jmask = ni
                # while jmask:
                #   lb = jmask & -jmask
                #   j = lb.bit_length() - 1
                #   jmask ^= lb
                #   sum_deg += (ni & adj[c][j]).bit_count()
                # edges = sum_deg // 2
                # total_triangles += edges
                
                # Actually, we can just count the sum of bit_counts and divide by 6 at the end?
                # Let's just accumulate the raw sum.
                jmask = ni
                while jmask:
                    lb = jmask & -jmask
                    j = lb.bit_length() - 1
                    jmask ^= lb
                    total_triangles += (ni & adj[c][j]).bit_count()
                    
        # Each triangle is counted 3 times (once for each vertex), and for each vertex, each edge in its neighborhood is counted twice in the sum above?
        # Let's re-verify.
        # For a fixed triangle (u, v, w) in color c.
        # When i=u: j iterates over v, w.
        #   j=v: (ni & adj[c][v]) contains w. bit_count includes 1.
        #   j=w: (ni & adj[c][w]) contains v. bit_count includes 1.
        # So for i=u, contribution is 2.
        # Similarly for i=v, contribution is 2.
        # Similarly for i=w, contribution is 2.
        # Total contribution to total_triangles for this specific triangle is 6.
        # So actual number of triangles = total_triangles // 6.
        
        return total_triangles // 6

    def is_valid(n, colors):
        return count_triangles(n, colors) == 0

    # We will try to find a valid coloring for a target n.
    # Start with n=50. If we can't find one in time, try 49, etc.
    # But we want to maximize n.
    
    best_n = 32
    best_colors = None
    
    # Initial 32-vertex solution (Greenwood-Gleason doubled)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_table, x = {}, 1
    for e in range(15):
        log_table[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log_table[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
                
    # If we have time, we can use the 32-vertex solution as a seed for SA for larger n?
    # No, SA works best on the full vector.
    # We can initialize the vector for n=50 by repeating the pattern or randomizing.
    
    def solve_for_n(target_n, time_budget):
        """
        Use Simulated Annealing to find a valid coloring for K_target_n.
        Returns (n, colors) if found, else (best_n_found, best_colors_found) or None.
        """
        m = target_n * (target_n - 1) // 2
        # Initialize with random colors
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Calculate initial energy
        current_energy = count_triangles(target_n, colors)
        
        if current_energy == 0:
            return target_n, colors
        
        best_energy = current_energy
        best_colors = colors[:]
        
        # SA parameters
        # Temperature should start high enough to accept bad moves.
        # Energy range is roughly [0, target_n^3].
        # Let's start T = 1000 and cool down.
        T = 1000.0
        T_min = 0.1
        cooling_rate = 0.999 # Adjust based on steps
        
        steps = 0
        max_steps = 500000 # Hard cap on steps to avoid infinite loop if time budget is large
        
        while time.time() < start_time + time_budget:
            if steps >= max_steps:
                break
                
            # Pick a random edge to flip
            idx = random.randint(0, m - 1)
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
                
            colors[idx] = new_color
            
            # Calculate new energy
            # This is the bottleneck. count_triangles is O(n^2) or O(n^3) depending on implementation.
            # Our bitset implementation is O(n^2 * n/word_size) roughly?
            # For n=50, it's fast enough?
            # Let's estimate: 50*49/2 = 1225 edges.
            # count_triangles builds adj (1225 ops), then loops 4 * 50 * (avg degree).
            # Avg degree in random graph is ~ n/4 = 12.
            # So 4 * 50 * 12 = 2400 bit operations.
            # Python bit operations on 64-bit ints are fast.
            # Should be ~microseconds.
            # 120s / 10us = 12,000,000 steps. Plenty.
            
            new_energy = count_triangles(target_n, colors)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = colors[:]
                
            # Acceptance criterion
            delta_E = new_energy - current_energy
            if delta_E < 0:
                current_energy = new_energy
            else:
                # Accept with probability exp(-delta_E / T)
                if random.random() < pow(2.718281828, -delta_E / T):
                    current_energy = new_energy
                else:
                    # Reject
                    colors[idx] = old_color
            
            steps += 1
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            # If we found a valid solution, we can try to extend or just return.
            # Since we are targeting a specific n, returning is fine.
            if best_energy == 0:
                return target_n, best_colors
                
        return None

    # Strategy:
    # 1. Try to solve for n=50 with a large time budget.
    # 2. If successful, try to solve for n=51 with remaining time.
    # 3. If n=50 fails, try n=49, etc., to ensure we return at least 32.
    
    # Allocate time
    # We know 32 is valid.
    # Let's try 50 first.
    
    # To improve chances, we can restart SA multiple times with different seeds.
    
    attempts = [50, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    
    # We will iterate through these targets.
    # For each target, we give it a portion of the time.
    # If we find a valid coloring for a higher n, we keep it.
    
    # Let's prioritize 50 and 51.
    # Give 50 seconds to n=50 attempts.
    # Give 20 seconds to n=51 attempts (if 50 succeeded or if we want to gamble).
    # Actually, if we find 50, we can use the rest of the time for 51.
    # If we don't find 50, we should try lower n to secure a score > 32.
    
    # Revised plan:
    # 1. Try n=50 for 40s. If found, update best.
    # 2. If n=50 found, try n=51 for remaining time.
    # 3. If n=50 NOT found, try n=49 for 20s. If found, update best.
    # 4. If n=49 NOT found, try n=48 for 10s.
    # ... and so on.
    # Finally, return best found (at least 32).
    
    # But wait, if we fail at 50, we might find 49. 49 > 32.
    # If we fail at all, we return 32.
    
    # Let's implement a loop that tries targets in descending order,
    # allocating decreasing time budgets.
    
    targets = [50, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    # We need to be careful: if we try 50 and fail, we haven't wasted time if we can still try 49.
    # But if we try 50 for 40s and fail, we have 100s left.
    
    # Let's use a dynamic time allocation.
    # Time per target: 10s for high targets, 5s for lower?
    
    # Actually, SA might need many restarts.
    # Let's just loop through targets, and for each, run SA until time runs out or solution found.
    # But we must ensure we don't spend all time on 50 and miss 49 which might be easier?
    # No, 50 is harder than 49. If 50 is not found in 40s, 49 might be found in 10s.
    
    # Let's assign specific budgets:
    budgets = {
        51: 20,
        50: 40,
        49: 15,
        48: 10,
        47: 5,
        46: 5,
        45: 5,
        44: 5,
        43: 5,
        42: 5,
        41: 5,
        40: 5,
    }
    
    # We will iterate in order of preference (highest n first).
    # If we find a valid n, we record it.
    # We stop trying lower n if we have found a high n and are out of time?
    # No, we want the MAX n.
    # So we should try 51 first? No, 51 is harder.
    # If we try 51 first and fail, we wasted time.
    # Better to try 50 first. If we get 50, then try 51.
    # If we don't get 50, try 49.
    
    # Order: 50, 51, 49, 48, ...
    
    ordered_targets = [50, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    # We will keep track of the best result found so far.
    # Initially 32.
    
    # To make SA more effective, we can use multiple restarts within the budget for a target.
    
    for target in ordered_targets:
        if time.time() > deadline - 5:
            break
            
        if target in budgets:
            budget = budgets[target]
        else:
            budget = 5
            
        # Don't exceed global deadline
        end_time = min(start_time + budget, deadline - 2)
        
        # Run SA for this target
        res = solve_for_n(target, end_time - start_time)
        
        if res:
            n_found, colors_found = res
            if n_found > best_n:
                best_n = n_found
                best_colors = colors_found
                
                # If we found 50 or 51, we can break if we are running low on time?
                # No, if we found 50, we still want to try 51 if we have time.
                # But our loop order is 50, then 51. So that works.
                # If we found 51, we are done (max possible is likely 51 or 52, but 51 is huge).
                # Actually, if we find 51, we can stop.
                if best_n >= 51:
                    break
                    
    # If we still have time and best_n is 32, we might try to do a quick random search for n=33?
    # Unlikely to succeed with random.
    # We return the best we have.
    
    if best_colors is None:
        best_colors = base_colors
        
    return {'n': best_n, 'colors': best_colors}

# Quick sanity check in head:
# count_triangles complexity:
# For n=50, m=1225.
# Building adj: 1225 iterations.
# Loop 4 * 50 * (avg deg ~ 12.5) = 2500 inner iterations.
# Each inner iteration: bit_count on 64-bit int. Very fast.
# Total ~ 4000 Python operations per energy eval.
# 1000 evals/sec easily? Maybe 5000/sec.
# 40 seconds -> 200,000 steps.
# Is 200,000 steps enough to find a valid 50-vertex coloring from random start?
# Random start has ~ n^3/6 * (1/4)^3 * ... triangles?
# Expected triangles in random 4-coloring: C(n,3) * 4 * (1/4)^3 = C(n,3)/16.
# For n=50, C(50,3) = 19600. 19600/16 = 1225 triangles.
# We need to reduce from 1225 to 0.
# SA can do this, but 200k steps might be tight.
# However, we have multiple restarts implicitly if we loop?
# No, my solve_for_n runs ONE long SA chain.
# It might get stuck in local minima.
# We should implement restarts in solve_for_n.

# Let's modify solve_for_n to do multiple short SA runs.

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def count_triangles(n, colors):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        total_triangles = 0
        for c in range(4):
            for i in range(n):
                ni = adj[c][i]
                jmask = ni
                while jmask:
                    lb = jmask & -jmask
                    j = lb.bit_length() - 1
                    jmask ^= lb
                    total_triangles += (ni & adj[c][j]).bit_count()
        return total_triangles // 6

    def solve_for_n(target_n, time_budget):
        m = target_n * (target_n - 1) // 2
        best_energy = float('inf')
        best_colors = None
        
        # Multiple restarts
        while time.time() < start_time + time_budget:
            # Initialize random
            colors = [random.randint(0, 3) for _ in range(m)]
            current_energy = count_triangles(target_n, colors)
            
            if current_energy == 0:
                return target_n, colors
            
            T = 50.0
            T_min = 0.1
            cooling_rate = 0.9995
            
            # Run SA for a short duration
            sa_start = time.time()
            sa_budget = min(5.0, time_budget - (time.time() - start_time)) # 5s per restart?
            
            while time.time() < sa_start + sa_budget:
                idx = random.randint(0, m - 1)
                old_color = colors[idx]
                new_color = random.randint(0, 3)
                if new_color == old_color:
                    continue
                
                colors[idx] = new_color
                new_energy = count_triangles(target_n, colors)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors[:]
                        if best_energy == 0:
                            return target_n, best_colors
                else:
                    delta_E = new_energy - current_energy
                    if random.random() < pow(2.718281828, -delta_E / T):
                        current_energy = new_energy
                    else:
                        colors[idx] = old_color
                
                T *= cooling_rate
                if T < T_min:
                    T = T_min
        
        return None

    # Base 32 solution
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_table, x = {}, 1
    for e in range(15):
        log_table[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log_table[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
                
    best_n = 32
    best_colors = base_colors[:]
    
    ordered_targets = [50, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    budgets = {
        50: 50,
        51: 20,
        49: 20,
        48: 10,
        47: 5,
        46: 5,
        45: 5,
        44: 5,
        43: 5,
        42: 5,
        41: 5,
        40: 5,
    }
    
    for target in ordered_targets:
        if time.time() > deadline - 5:
            break
            
        if target in budgets:
            budget = budgets[target]
        else:
            budget = 5
            
        # Don't exceed global deadline
        end_time = min(start_time + budget, deadline - 2)
        
        res = solve_for_n(target, end_time - start_time)
        
        if res:
            n_found, colors_found = res
            if n_found > best_n:
                best_n = n_found
                best_colors = colors_found
                
                if best_n >= 51:
                    break
                    
    return {'n': best_n, 'colors': best_colors}
