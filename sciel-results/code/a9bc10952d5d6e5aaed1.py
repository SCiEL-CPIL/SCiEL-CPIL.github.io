"""Construct 5-colourings of K_n without monochromatic C4, maximising n."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0
    
    def full_check(n, colors):
        """Verify no colour class has a C4."""
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i][c] & adj[j][c]) >= 2:
                        return False
        return True

    def local_search(n, max_time=10.0):
        """
        Find a valid 5-coloring of K_n using simulated annealing / hill climbing.
        Returns None if no valid coloring found within time.
        """
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # colors[i] is the color of edge i
        # We maintain adj[c][i] as a set of neighbors
        # And common_count[c][i][j] is the number of common neighbors of i and j in color c
        
        def init_random():
            colors = [random.randint(0, 4) for _ in range(total_edges)]
            adj = [[set() for _ in range(n)] for _ in range(5)]
            common = [[[0]*n for _ in range(n)] for _ in range(5)]
            
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        # Count common neighbors
                        # This is O(n) per pair, so O(n^3) total
                        # For n=26, this is fast enough for initialization
                        common[c][i][j] = len(adj[c][i] & adj[c][j])
                        common[c][j][i] = common[c][i][j]
                        
            return colors, adj, common

        def calc_cost(common):
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        k = common[c][i][j]
                        if k >= 2:
                            cost += k * (k - 1) // 2
            return cost

        def flip_edge(idx, new_color):
            i, j = edges[idx]
            old_color = colors[idx]
            if old_color == new_color:
                return 0
            
            # Update cost delta
            # Removing (i,j) from old_color
            # Adding (i,j) to new_color
            
            # We need to update common counts for pairs involving i or j
            # and for the pair (i,j) itself in the respective colors.
            
            delta_cost = 0
            
            # 1. Handle removal from old_color
            oc = old_color
            # Update common[oc][i][k] and common[oc][j][k] for all k
            # If (j,k) is in oc, then i and k lose j as common neighbor?
            # No, common[oc][i][k] is count of m such that (i,m) in oc and (k,m) in oc.
            # When we remove (i,j) from oc:
            # - j is no longer a neighbor of i in oc.
            # - i is no longer a neighbor of j in oc.
            # - For any k, if (j,k) is in oc, then i and k no longer share j. So common[oc][i][k] -= 1.
            # - For any k, if (i,k) is in oc, then j and k no longer share i. So common[oc][j][k] -= 1.
            # - Also, common[oc][i][j] might change?
            #   common[oc][i][j] counts m such that (i,m) in oc and (j,m) in oc.
            #   Removing (i,j) doesn't change the set of m, so common[oc][i][j] stays same?
            #   Wait, i and j are no longer connected, but common neighbors are defined by existence of edges to m.
            #   The edge (i,j) itself is not a "common neighbor".
            #   So common[oc][i][j] is unchanged by removing the edge (i,j) itself, 
            #   UNLESS i or j was the common neighbor? No, common neighbor m is distinct.
            
            # Let's re-verify the update logic.
            # State: adj[oc][i] contains j. We remove j.
            # For each k in adj[oc][j] (which are neighbors of j in oc):
            #   i and k used to share j. Now they don't.
            #   So common[oc][i][k] -= 1.
            #   And common[oc][k][i] -= 1.
            # For each k in adj[oc][i] (which are neighbors of i in oc):
            #   j and k used to share i. Now they don't.
            #   So common[oc][j][k] -= 1.
            #   And common[oc][k][j] -= 1.
            
            # Note: If k is in both adj[oc][i] and adj[oc][j], then:
            #   From first loop: common[oc][i][k] -= 1
            #   From second loop: common[oc][j][k] -= 1 (and common[oc][k][j] -= 1)
            #   Does common[oc][i][j] change?
            #   common[oc][i][j] is count of m in adj[oc][i] \cap adj[oc][j].
            #   We are removing j from adj[oc][i] and i from adj[oc][j].
            #   This does not affect the intersection of the *other* neighbors.
            #   So common[oc][i][j] is unchanged.
            
            # We calculate cost change for affected pairs.
            # Affected pairs: (i,k) for k in adj[oc][j], (j,k) for k in adj[oc][i].
            # Plus (i,j) itself? No, common[i][j] unchanged.
            
            # To avoid double counting and errors, it's safer to:
            # 1. Calculate cost of affected pairs before.
            # 2. Apply updates to common and adj.
            # 3. Calculate cost of affected pairs after.
            # 4. Delta = after - before.
            
            affected_pairs = set()
            for k in adj[oc][j]:
                affected_pairs.add((min(i,k), max(i,k)))
            for k in adj[oc][i]:
                affected_pairs.add((min(j,k), max(j,k)))
            # Also (i,j) just in case?
            affected_pairs.add((i,j))
            
            cost_before = 0
            for u, v in affected_pairs:
                k_val = common[oc][u][v]
                if k_val >= 2:
                    cost_before += k_val * (k_val - 1) // 2
            
            # Apply removal
            # Remove j from adj[oc][i] and i from adj[oc][j]
            # But we need to update common counts first based on OLD adj sets
            
            # Save the sets to iterate over
            neighbors_of_j_in_oc = list(adj[oc][j])
            neighbors_of_i_in_oc = list(adj[oc][i])
            
            for k in neighbors_of_j_in_oc:
                if k == i: continue # i is in adj[oc][j], but we are removing (i,j)
                # i and k lose common neighbor j
                common[oc][i][k] -= 1
                common[oc][k][i] -= 1
            for k in neighbors_of_i_in_oc:
                if k == j: continue
                # j and k lose common neighbor i
                common[oc][j][k] -= 1
                common[oc][k][j] -= 1
                
            # Now remove the edge from adj
            adj[oc][i].remove(j)
            adj[oc][j].remove(i)
            
            cost_after = 0
            for u, v in affected_pairs:
                k_val = common[oc][u][v]
                if k_val >= 2:
                    cost_after += k_val * (k_val - 1) // 2
            delta_cost += cost_after - cost_before
            
            # 2. Handle addition to new_color
            nc = new_color
            # Add i to adj[nc][j] and j to adj[nc][i]
            
            affected_pairs_nc = set()
            for k in adj[nc][j]:
                affected_pairs_nc.add((min(i,k), max(i,k)))
            for k in adj[nc][i]:
                affected_pairs_nc.add((min(j,k), max(j,k)))
            affected_pairs_nc.add((i,j))
            
            cost_before_nc = 0
            for u, v in affected_pairs_nc:
                k_val = common[nc][u][v]
                if k_val >= 2:
                    cost_before_nc += k_val * (k_val - 1) // 2
            
            # Update common counts
            neighbors_of_j_in_nc = list(adj[nc][j])
            neighbors_of_i_in_nc = list(adj[nc][i])
            
            for k in neighbors_of_j_in_nc:
                if k == i: continue
                # i and k gain common neighbor j
                common[nc][i][k] += 1
                common[nc][k][i] += 1
            for k in neighbors_of_i_in_nc:
                if k == j: continue
                # j and k gain common neighbor i
                common[nc][j][k] += 1
                common[nc][k][j] += 1
                
            # Add edge to adj
            adj[nc][i].add(j)
            adj[nc][j].add(i)
            
            cost_after_nc = 0
            for u, v in affected_pairs_nc:
                k_val = common[nc][u][v]
                if k_val >= 2:
                    cost_after_nc += k_val * (k_val - 1) // 2
            delta_cost += cost_after_nc - cost_before_nc
            
            # Update colors array
            colors[idx] = new_color
            
            return delta_cost

        colors, adj, common = init_random()
        current_cost = calc_cost(common)
        best_cost = current_cost
        best_colors = colors[:]
        
        temperature = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() < start + max_time and temperature > min_temp:
            # Pick a random edge
            idx = random.randint(0, total_edges - 1)
            i, j = edges[idx]
            current_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == current_color:
                continue
            
            delta = flip_edge(idx, new_color)
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                # Accept
                pass # flip_edge already applied
                if current_cost + delta < best_cost:
                    best_cost = current_cost + delta
                    best_colors = colors[:]
                current_cost += delta
            else:
                # Reject: undo the flip
                flip_edge(idx, current_color)
            
            temperature *= cooling
            
        if best_cost == 0:
            return best_colors
        return None

    # Try to find valid coloring for increasing n
    # Start from known good n (11) and go up
    # Or just try 26 directly?
    # Local search is stochastic. Trying 26 directly is risky but high reward.
    # We can try 26, 25, 24...
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Allocate time for each n
    # If we have 110s, and we try n=26, 25, 24...
    # We should spend most time on the largest n.
    
    for n_try in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on n
        # Larger n gets more time
        time_budget = 10.0 if n_try >= 25 else 5.0
        
        res = local_search(n_try, max_time=time_budget)
        if res is not None:
            if full_check(n_try, res):
                best_n = n_try
                best_colors = res
                print(f"Found valid construction for n={n_try}")
                break # If we found a valid one for n, we are done? 
                # No, we want to maximize n. 
                # If we found n=26, we are done.
                # If we found n=25, we should try 26? 
                # The loop goes 26, 25... 
                # So if we fail 26, we try 25.
                # If we succeed at 25, we break? 
                # Yes, because we are iterating downwards.
                # So the first success is the maximum n.
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
