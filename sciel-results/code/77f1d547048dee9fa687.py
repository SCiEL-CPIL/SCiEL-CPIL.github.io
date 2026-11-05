import time
import random
import numpy as np

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)
        
    # Precompute K4 edge tuples
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
                    
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)
    num_k4s = len(k4_edges)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Convert to arrays for faster access
    edge_to_k4s_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]

    def calculate_penalties(colors):
        """
        Vectorized calculation of penalties for all K4s.
        Returns:
        penalties: np.array of shape (num_k4s,)
        """
        # Get colours for all K4s: shape (num_k4s, 6)
        k4_colors = colors[k4_edges_arr]
        # Sort each row
        sorted_colors = np.sort(k4_colors, axis=1)
        # Calculate differences
        diffs = np.diff(sorted_colors, axis=1)
        # Count distinct: 1 + number of non-zero diffs
        distinct = 1 + np.sum(diffs != 0, axis=1)
        # Calculate penalties: (5 - distinct)^2 if distinct < 5 else 0
        penalties = np.where(distinct < 5, (5 - distinct) ** 2, 0)
        return penalties

    def is_valid(colors):
        return np.sum(calculate_penalties(colors)) == 0

    def fallback():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    def local_search(k, time_limit):
        start_time = time.time()
        
        # Initialization
        # Try a few different initialization strategies
        best_solution = None
        best_cost = float('inf')
        
        # Strategy 1: Random
        # Strategy 2: 17-colour mapped to k
        init_17 = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        
        for restart in range(20): # 20 restarts
            if time.time() - start_time > time_limit:
                break
                
            if restart % 2 == 0:
                colors = np.random.randint(0, k, size=num_edges, dtype=np.int32)
            else:
                colors = init_17 % k
                
            k4_penalties = calculate_penalties(colors)
            cur_cost = np.sum(k4_penalties)
            
            if cur_cost == 0:
                return colors.tolist()
                
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_solution = colors.copy()
                
            # Local Search Loop
            no_improve_count = 0
            while time.time() - start_time < time_limit:
                # Check for valid
                if cur_cost == 0:
                    return colors.tolist()
                    
                # Find a violating K4
                violating_indices = np.where(k4_penalties > 0)[0]
                if len(violating_indices) == 0:
                    break # Should not happen if cur_cost > 0
                
                k4_idx = violating_indices[random.randint(0, len(violating_indices) - 1)]
                edges = k4_edges_arr[k4_idx]
                
                # Pick a random edge from this K4
                e_idx = edges[random.randint(0, 5)]
                old_c = colors[e_idx]
                
                # Try all colours for this edge
                best_new_c = old_c
                best_delta = 0
                best_new_penalties = None # Store updated penalties for the best move? 
                # Actually, we can just compute delta for each candidate.
                
                affected_k4s = edge_to_k4s_arr[e_idx]
                
                # Get current penalties for affected K4s
                old_affected_penalties = k4_penalties[affected_k4s]
                
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    # Temporarily change colour
                    colors[e_idx] = new_c
                    
                    # Calculate new penalties for affected K4s
                    # Get colours for affected K4s
                    affected_colors = colors[k4_edges_arr[affected_k4s]]
                    sorted_affected = np.sort(affected_colors, axis=1)
                    diffs_affected = np.diff(sorted_affected, axis=1)
                    distinct_affected = 1 + np.sum(diffs_affected != 0, axis=1)
                    new_affected_penalties = np.where(distinct_affected < 5, (5 - distinct_affected) ** 2, 0)
                    
                    delta = np.sum(new_affected_penalties) - np.sum(old_affected_penalties)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                        # We don't need to store new_affected_penalties unless we accept
                        # But if we accept, we need to update k4_penalties
                        # So we can just recalculate or store it.
                        # Storing is faster.
                        # But we only need it for the best one.
                        # Let's just store the best one's penalties.
                        # Note: new_affected_penalties is an array.
                        # We can't easily store it in a scalar variable.
                        # We can store the index of the best colour and recalculate?
                        # Or just store the array.
                        # Let's assume we store the array for the best candidate.
                        # But we are in a loop.
                        # Let's just keep track of best_delta and best_new_c.
                        # If we accept, we will recalculate the penalties for affected K4s?
                        # No, that defeats the purpose.
                        # We can store the `new_affected_penalties` for the best candidate.
                        # But `new_affected_penalties` is a numpy array.
                        # We can just do:
                        # if delta < best_delta:
                        #    best_delta = delta
                        #    best_new_c = new_c
                        #    best_new_affected_penalties = new_affected_penalties.copy()
                        
                        # Let's do that.
                        pass # Placeholder

                # Re-run the loop to capture the best penalties? 
                # Or just accept if best_delta < 0 and update.
                # To avoid double computation, let's restructure.
                
                # Actually, the loop above is fine. Let's just implement the storage properly.
                # We need to be careful with memory allocation.
                
                # Let's simplify: 
                # Just try the best colour found.
                # If best_delta < 0, accept.
                # To update k4_penalties, we need the new penalties for affected K4s.
                # We can just recalculate them for the accepted move.
                # Since we only accept if it improves, and we only do this occasionally,
                # the overhead is acceptable.
                
                if best_delta < 0:
                    colors[e_idx] = best_new_c
                    # Update k4_penalties for affected K4s
                    affected_colors = colors[k4_edges_arr[affected_k4s]]
                    sorted_affected = np.sort(affected_colors, axis=1)
                    diffs_affected = np.diff(sorted_affected, axis=1)
                    distinct_affected = 1 + np.sum(diffs_affected != 0, axis=1)
                    new_affected_penalties = np.where(distinct_affected < 5, (5 - distinct_affected) ** 2, 0)
                    
                    k4_penalties[affected_k4s] = new_affected_penalties
                    cur_cost += best_delta
                    no_improve_count = 0
                else:
                    # No improvement
                    no_improve_count += 1
                    if no_improve_count > 100:
                        # Random kick
                        for _ in range(5):
                            kick_e = random.randint(0, num_edges - 1)
                            colors[kick_e] = random.randint(0, k - 1)
                        
                        k4_penalties = calculate_penalties(colors)
                        cur_cost = np.sum(k4_penalties)
                        no_improve_count = 0
                        
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_solution = colors.copy()

        # End of restarts
        if best_solution is not None and is_valid(best_solution):
            return best_solution.tolist()
        return None

    time_start = time.time()
    time_limit_per_k = 30.0
    
    for k in [13, 14, 15]:
        if time.time() - time_start > 100: # Safety margin
            break
        res = local_search(k, time_limit_per_k)
        if res is not None:
            if is_valid(res):
                return res

    return fallback()
