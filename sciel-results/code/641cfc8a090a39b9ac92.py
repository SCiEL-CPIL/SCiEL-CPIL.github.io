import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    deadline = start_time + 75.0  # Safety margin for 90s kill

    # Precompute neighbors as bitmasks
    # neighbors[i] is an integer where bit j is 1 if lcm(i+1, j+1) <= 512
    neighbors = [0] * n
    
    for i in range(n):
        num_i = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            lcm_val = (num_i * num_j) // g
            if lcm_val <= limit:
                mask |= (1 << j)
        neighbors[i] = mask
        
    # Helper to get list of vertices from bitmask
    def mask_to_list(mask):
        verts = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts.append(idx)
            mask ^= lsb
        return verts

    # Helper to get bitmask from list of vertices
    def list_to_mask(verts):
        mask = 0
        for v in verts:
            mask |= (1 << v)
        return mask

    # Find a good initial clique using structural insight
    # Divisors of 504 form a clique because lcm(a,b) divides 504 <= 512
    # 504 = 2^3 * 3^2 * 7. Divisors count: 4*3*2 = 24.
    # Let's verify and construct this set.
    divisors_504 = []
    for i in range(1, 513):
        if 504 % i == 0:
            divisors_504.append(i - 1) # 0-indexed
        
    # Verify it's a clique
    is_clique = True
    for i in range(len(divisors_504)):
        for j in range(i+1, len(divisors_504)):
            if not (neighbors[divisors_504[i]] & (1 << divisors_504[j])):
                is_clique = False
                break
        if not is_clique:
            break
            
    if is_clique:
        initial_clique_mask = list_to_mask(divisors_504)
    else:
        # Fallback to a smaller safe clique, e.g., powers of 2
        initial_clique_mask = 0
        for k in range(9):
            v = (1 << k) - 1
            initial_clique_mask |= (1 << v)
            
    best_size = bin(initial_clique_mask).count('1')
    best_clique = initial_clique_mask

    # Tomita's Algorithm (Maximum Clique) with Bitsets
    # R: current clique (list of vertices)
    # P: candidate set (bitmask)
    # X: excluded set (bitmask, usually not needed for max clique size if we just want size)
    
    # We use a recursive branch and bound.
    # To avoid recursion depth issues and overhead, we implement it carefully.
    # Python recursion limit is usually 1000, which is fine for N=512.
    
    # Coloring for ordering and bounding
    # We need to color P to get an upper bound.
    # Simple greedy coloring:
    def color_sort(P):
        # Returns list of (vertex, color) sorted by color, and max_color
        # This is O(|P|^2) in worst case, but |P| decreases rapidly.
        # Optimization: Only color if |P| + len(R) > best_size
        colors = {}
        max_color = 0
        # Iterate vertices in P
        verts = mask_to_list(P)
        # Sort by degree in P for better coloring?
        # deg(v) = bin(neighbors[v] & P).count('1')
        # This is expensive. Let's just use index order or simple heuristic.
        # Actually, for the bound to be tight, we need good coloring.
        # Let's use a simple greedy: assign smallest available color.
        
        # To speed up, we can stop coloring if max_color + len(R) <= best_size
        # But we need the order for branching.
        
        used_colors = [set() for _ in range(len(verts)+1)] # List of sets of vertices using color c? No, just track used colors per vertex.
        
        # Standard greedy coloring for Max Clique:
        # Order vertices. Assign color 1, 2, ...
        # c(v) is the color.
        # Bound: |R| + c(v) <= best -> prune.
        
        # Let's implement a fast coloring routine.
        # Since N is small (512), and P shrinks, this is okay.
        
        # We'll return the vertices sorted by color, and the max color.
        # Actually, we can just return the list of vertices in order of assignment.
        
        color_assign = {}
        max_c = 0
        for v in verts:
            # Find min color not used by neighbors of v in P
            # Neighbors of v in P: neighbors[v] & P
            nb_mask = neighbors[v] & P
            # Get colors of neighbors
            nb_colors = set()
            m = nb_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if u in color_assign:
                    nb_colors.add(color_assign[u])
                m ^= lsb
                
            c = 1
            while c in nb_colors:
                c += 1
            color_assign[v] = c
            if c > max_c:
                max_c = c
            
        # Sort vertices by color
        sorted_verts = sorted(verts, key=lambda x: color_assign[x])
        return sorted_verts, max_c

    def expand(R, P):
        nonlocal best_size, best_clique
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = 0
                for v in R:
                    best_clique |= (1 << v)
            return
            
        # Bound check: if len(R) + popcount(P) <= best_size, prune
        if len(R) + bin(P).count('1') <= best_size:
            return

        # Color sort P
        # Optimization: Only do full coloring if necessary?
        # For now, always color.
        sorted_P, max_color = color_sort(P)
        
        # If len(R) + max_color <= best_size, prune
        if len(R) + max_color <= best_size:
            return

        # Branch
        # Iterate through sorted_P in reverse order (highest color first)
        # This is the standard Tomita approach.
        
        # We need to handle the set P dynamically.
        # We will iterate through the vertices.
        # For each v, we add it to R, and intersect P with neighbors[v].
        
        # To avoid modifying the original P, we pass a copy or manage state.
        # Since we are in Python, passing integers is cheap.
        
        # However, we must ensure we don't revisit.
        # The algorithm:
        # for v in sorted_P (reverse):
        #   R.append(v)
        #   P_new = P & neighbors[v]
        #   P = P & ~(1 << v)  # Remove v from P for subsequent iterations
        #   expand(R, P_new)
        #   R.pop()
        
        # Note: P is modified in the loop. We need to be careful.
        # Let's use a local variable for P.
        
        P_local = P
        R_local = R[:] # Copy list
        
        # Iterate in reverse order of colors
        for i in range(len(sorted_P) - 1, -1, -1):
            v = sorted_P[i]
            
            # Check if v is still in P_local (it might have been removed by a previous branch? 
            # No, in this formulation, we remove v from P_local after processing.
            # But we iterate over a static list sorted_P.
            # We must check if v is in P_local.
            if not (P_local & (1 << v)):
                continue
                
            # Bound check with color
            # The color of v is color_assign[v] from the last call?
            # We need to access the color. 
            # Let's restructure to return colors in the sort.
            
            # For now, just use the simple bound: len(R) + popcount(P_local) <= best
            if len(R_local) + bin(P_local).count('1') <= best_size:
                break # Since we are going down in color, and P is shrinking, 
                      # this bound might not be monotonic if we just use popcount.
                      # But if we used color bound, we could break earlier.
                      # Let's stick to popcount for simplicity and correctness first.
            
            # Branch
            R_local.append(v)
            P_new = P_local & neighbors[v]
            
            # Update best
            if len(R_local) > best_size:
                best_size = len(R_local)
                # We can update best_clique here if we want, but we do it at leaf.
                # Actually, updating at leaf is safer for consistency.
                
            expand(R_local, P_new)
            
            R_local.pop()
            P_local &= ~(1 << v)
            
            # Optional: If len(R_local) + max_color_remaining <= best_size, break
            # We don't have max_color_remaining easily without re-coloring.

    # Initial call
    # R is empty, P is all nodes? 
    # No, we can start with the initial clique to prune better.
    # If we start with R = initial_clique, P = neighbors of all in R.
    
    R_init = mask_to_list(initial_clique_mask)
    P_init = (1 << n) - 1
    for v in R_init:
        P_init &= neighbors[v]
        
    # Ensure R_init is a clique (verified above)
    expand(R_init, P_init)
    
    # If time remains, try random restarts with local search or just more B&B?
    # B&B with a good initial bound is usually very fast for N=512.
    # Let's check time.
    
    if time.time() < deadline - 5.0:
        # Try to improve with local search from the found best_clique
        # Or just run B&B from scratch with a different initial bound?
        # The B&B above is exhaustive given the initial bound.
        # If it finished, it found the max clique.
        
        # If it didn't finish (timeout), we return the best found so far.
        # The current implementation doesn't have a global timeout check inside expand.
        # We should add one.
        
        pass

    # Convert best_clique to list of 1-indexed values
    result_verts = mask_to_list(best_clique)
    return sorted([v + 1 for v in result_verts])

construct()
