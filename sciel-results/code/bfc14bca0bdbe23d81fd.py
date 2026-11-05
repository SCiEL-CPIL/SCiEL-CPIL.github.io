import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for each K4
    # Edge (i, j) with i < j maps to index: i*(n-1) - i*(i+1)//2 + (j-i-1)
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)
        
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(edges)
                    
    num_k4s = len(k4_edges)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Function to calculate cost of a colouring
    # Cost is sum of (5 - distinct)^2 for each K4 with distinct < 5
    def calculate_cost(colors):
        total_cost = 0
        for edges in k4_edges:
            # Get the 6 colours
            c = [colors[e] for e in edges]
            # Count distinct
            distinct = len(set(c))
            if distinct < 5:
                total_cost += (5 - distinct) ** 2
        return total_cost

    # Function to check if valid
    def is_valid(colors):
        for edges in k4_edges:
            if len(set(colors[e] for e in edges)) < 5:
                return False
        return True

    # Local search
    def local_search(k, time_limit):
        start_ls = time.time()
        best_cost = float('inf')
        best_colors = None
        
        # Try multiple restarts
        while time.time() - start_ls < time_limit:
            # Random initialization
            colors = [random.randint(0, k - 1) for _ in range(num_edges)]
            cur_cost = calculate_cost(colors)
            
            if cur_cost == 0:
                return colors
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_colors = colors[:]
            
            # Hill Climbing
            no_improve_count = 0
            max_no_improve = 2000
            while no_improve_count < max_no_improve and time.time() - start_ls < time_limit:
                # Find a violating K4
                # Pick a random K4
                k4_idx = random.randint(0, num_k4s - 1)
                edges = k4_edges[k4_idx]
                c = [colors[e] for e in edges]
                distinct = len(set(c))
                
                if distinct >= 5:
                    # No violation, pick a random edge from the whole graph? 
                    # Or just pick a random edge and hope?
                    # Better: pick a random edge and try to improve
                    e_idx = random.randint(0, num_edges - 1)
                else:
                    # Pick a random edge from this violating K4
                    e_idx = random.choice(edges)
                
                old_color = colors[e_idx]
                best_new_color = old_color
                best_local_cost = cur_cost
                
                # Try all k colours for this edge
                # We need to calculate the cost change efficiently
                # Cost change = new_cost_of_affected_k4s - old_cost_of_affected_k4s
                
                # Get affected K4s
                affected = edge_to_k4s[e_idx]
                
                # Calculate old cost contribution of affected K4s
                old_contrib = 0
                for ak4_idx in affected:
                    ak4_edges = k4_edges[ak4_idx]
                    ac = [colors[e] for e in ak4_edges]
                    ad = len(set(ac))
                    if ad < 5:
                        old_contrib += (5 - ad) ** 2
                        
                # Try new colours
                for new_c in range(k):
                    if new_c == old_color:
                        continue
                    
                    # Temporarily set new colour
                    colors[e_idx] = new_c
                    
                    # Calculate new cost contribution of affected K4s
                    new_contrib = 0
                    for ak4_idx in affected:
                        ak4_edges = k4_edges[ak4_idx]
                        ac = [colors[e] for e in ak4_edges]
                        ad = len(set(ac))
                        if ad < 5:
                            new_contrib += (5 - ad) ** 2
                            
                    delta = new_contrib - old_contrib
                    if delta < best_local_cost - cur_cost:
                        # This is not quite right. 
                        # cur_cost is the total cost.
                        # We want to minimize cur_cost.
                        # new_total_cost = cur_cost - old_contrib + new_contrib
                        # delta_total = new_contrib - old_contrib
                        # If delta_total < 0, cost decreases.
                        # We want the minimum delta_total.
                        # Actually, we just need to find the new_c that minimizes new_contrib.
                        # Then new_cur_cost = cur_cost - old_contrib + min_new_contrib
                        # If new_cur_cost < cur_cost, accept.
                        pass

                # Let's simplify: just try all, track min new_contrib
                min_new_contrib = float('inf')
                best_c_to_use = old_color
                for new_c in range(k):
                    if new_c == old_color:
                        continue
                    colors[e_idx] = new_c
                    new_contrib = 0
                    for ak4_idx in affected:
                        ak4_edges = k4_edges[ak4_idx]
                        ac = [colors[e] for e in ak4_edges]
                        ad = len(set(ac))
                        if ad < 5:
                            new_contrib += (5 - ad) ** 2
                    if new_contrib < min_new_contrib:
                        min_new_contrib = new_contrib
                        best_c_to_use = new_c

                # Restore old colour for now, will set to best if improved
                colors[e_idx] = old_color
                
                # Check if improvement
                if min_new_contrib < old_contrib:
                    colors[e_idx] = best_c_to_use
                    cur_cost = cur_cost - old_contrib + min_new_contrib
                    no_improve_count = 0
                    
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_colors = colors[:]
                        
                    if cur_cost == 0:
                        return colors
                else:
                    no_improve_count += 1
                    
                    # If stuck for too long, random restart or random move
                    if no_improve_count > 100:
                        # Random move to escape local minimum
                        e_idx = random.randint(0, num_edges - 1)
                        old_c = colors[e_idx]
                        new_c = random.randint(0, k - 1)
                        colors[e_idx] = new_c
                        # Recalculate cost fully? Or just accept and continue?
                        # Recalculating fully is O(1820). Acceptable occasionally.
                        cur_cost = calculate_cost(colors)
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_colors = colors[:]
                        no_improve_count = 0

        return best_colors if best_cost == 0 else None

    # Try to find a solution with 13 colours
    # Time budget allocation:
    # 13 colours: 40s
    # 14 colours: 40s
    # 15 colours: 40s
    # Fallback: cyclic 17
    
    # Helper to return cyclic colouring
    def cyclic_colouring():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    # Attempt 13 colours
    if time.time() < deadline - 60:
        sol = local_search(13, 40)
        if sol:
            # Verify
            if is_valid(sol):
                return sol

    # Attempt 14 colours
    if time.time() < deadline - 30:
        sol = local_search(14, 25)
        if sol:
            if is_valid(sol):
                return sol

    # Attempt 15 colours
    if time.time() < deadline - 10:
        sol = local_search(15, 10)
        if sol:
            if is_valid(sol):
                return sol

    # Fallback to cyclic 17
    return cyclic_colouring()
