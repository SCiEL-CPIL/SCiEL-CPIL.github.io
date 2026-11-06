import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    n = 16
    num_edges = 120
    num_k4 = 1820
    
    # Precompute edge indices
    edge_idx = {}
    idx = 0
    for v1 in range(n):
        for v2 in range(v1+1, n):
            edge_idx[(v1, v2)] = idx
            idx += 1
            
    # Generate all K4s and their edge indices
    vertices = list(range(n))
    k4_list = list(itertools.combinations(vertices, 4))
    
    k4_edges = []
    for k4 in k4_list:
        v1, v2, v3, v4 = k4
        e1 = edge_idx[(v1, v2)]
        e2 = edge_idx[(v1, v3)]
        e3 = edge_idx[(v1, v4)]
        e4 = edge_idx[(v2, v3)]
        e5 = edge_idx[(v2, v4)]
        e6 = edge_idx[(v3, v4)]
        k4_edges.append([e1, e2, e3, e4, e5, e6])
        
    k4_edges = np.array(k4_edges) # shape (1820, 6)
    
    # Precompute affected K4s for each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for i in range(num_k4):
        edges_in_k4 = k4_edges[i]
        for e in edges_in_k4:
            edge_to_k4s[e].append(i)

    # Function to check if a K4 is bad (fewer than 5 distinct colors)
    # Input: 6 colors
    # Returns: True if bad, False if good
    def is_bad_k4(c1, c2, c3, c4, c5, c6):
        # Fast check for < 5 distinct colors among 6
        # If any color appears >= 2 times, and total distinct < 5
        # Actually, just checking len(set) is O(6).
        # Let's try a manual check to avoid set creation overhead?
        # c1==c2 or c1==c3 ... is not enough, need distinct count.
        # Example: 1,1,2,3,4,5 -> distinct 5. Good.
        # Example: 1,1,2,3,4,4 -> distinct 4. Bad.
        # So we need to count distinct.
        # For 6 items, a small unrolled loop or set is best.
        # Let's stick to set but optimize the call.
        return len((c1, c2, c3, c4, c5, c6)) < 5 # Wait, len(tuple) is length, not distinct.
        # Must use set.
        return len(set((c1, c2, c3, c4, c5, c6))) < 5

    # Optimize is_bad_k4 by inlining?
    # We will inline the logic in the main loop for speed.

    def run_sa(k, time_limit):
        # Initialize random coloring
        coloring = np.random.randint(0, k, size=num_edges)
        
        # Initial bad mask
        bad_mask = np.zeros(num_k4, dtype=bool)
        for i in range(num_k4):
            e = k4_edges[i]
            if len(set(coloring[e])) < 5:
                bad_mask[i] = True
        current_bad = np.sum(bad_mask)
        best_coloring = coloring.copy()
        best_bad = current_bad
        
        if current_bad == 0:
            return coloring, 0

        temp = 1.0
        cooling = 0.999
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            e_idx = np.random.randint(num_edges)
            old_color = coloring[e_idx]
            new_color = np.random.randint(k)
            if new_color == old_color:
                continue
            
            # Identify affected K4s
            affected = edge_to_k4s[e_idx]
            
            # Calculate delta
            # We need to re-evaluate status of affected K4s
            # Store old bad status for these K4s
            old_bad_in_affected = 0
            for i in affected:
                if bad_mask[i]:
                    old_bad_in_affected += 1
            
            # Apply new color
            coloring[e_idx] = new_color
            
            new_bad_in_affected = 0
            for i in affected:
                e = k4_edges[i]
                # Inline check for speed
                # Access elements directly
                c1 = coloring[e[0]]
                c2 = coloring[e[1]]
                c3 = coloring[e[2]]
                c4 = coloring[e[3]]
                c5 = coloring[e[4]]
                c6 = coloring[e[5]]
                
                # Check distinct count < 5
                # Using set is still probably fastest in Python for 6 items
                if len(set((c1, c2, c3, c4, c5, c6))) < 5:
                    new_bad_in_affected += 1
                    bad_mask[i] = True
                else:
                    bad_mask[i] = False
            
            delta = new_bad_in_affected - old_bad_in_affected
            
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                # Accept
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_coloring = coloring.copy()
                    if best_bad == 0:
                        return best_coloring, 0
            else:
                # Reject
                # Revert coloring
                coloring[e_idx] = old_color
                # Revert bad_mask
                for i in affected:
                    e = k4_edges[i]
                    c1 = coloring[e[0]]
                    c2 = coloring[e[1]]
                    c3 = coloring[e[2]]
                    c4 = coloring[e[3]]
                    c5 = coloring[e[4]]
                    c6 = coloring[e[5]]
                    if len(set((c1, c2, c3, c4, c5, c6))) < 5:
                        bad_mask[i] = True
                    else:
                        bad_mask[i] = False
            
            temp *= cooling
            if temp < 0.001:
                temp = 1.0

        return best_coloring, best_bad

    # Strategy:
    # 1. Try to find a solution with 13 colors (unlikely but fast check)
    # 2. Try 14
    # 3. Try 15
    # 4. Fallback to 17 (Seed)

    # To maximize chance, we should spend most time on the most likely successful k.
    # Given bounds 13-15, 15 is most likely to have a solution.
    # But 13 is the target for best score.
    # Let's try 13 for a short time, then 14, then 15.
    
    # Time allocation:
    # 13: 10s
    # 14: 30s
    # 15: 70s
    
    if time.time() < start_time + 10:
        sol, bad = run_sa(13, 9)
        if bad == 0:
            return sol.tolist()
            
    if time.time() < start_time + 40:
        sol, bad = run_sa(14, 29)
        if bad == 0:
            return sol.tolist()
            
    if time.time() < start_time + 110:
        sol, bad = run_sa(15, 69)
        if bad == 0:
            return sol.tolist()

    # Fallback: Seed solution (17 colours)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
