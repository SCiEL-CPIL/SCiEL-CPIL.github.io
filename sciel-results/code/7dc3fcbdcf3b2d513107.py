"""Construct 5-colourings of K_n without monochromatic C4, maximising n."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0

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

    def sa_search(n, max_time=15.0):
        """
        Simulated Annealing for finding a valid 5-coloring of K_n.
        Uses efficient delta computation for cost changes.
        """
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Build adjacency bitsets
        adj = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        # Compute common neighbor counts
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    cnt = bin(adj[c][i] & adj[c][j]).count('1')
                    common[c][i][j] = cnt
                    common[c][j][i] = cnt

        def calc_cost():
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        k = common[c][i][j]
                        if k >= 2:
                            cost += k * (k - 1) // 2
            return cost

        current_cost = calc_cost()
        best_cost = current_cost
        best_colors = colors[:]
        
        temperature = 10.0
        cooling = 0.999
        min_temp = 0.01
        
        t_end = start + max_time
        while time.time() < t_end and temperature > min_temp:
            idx = random.randint(0, total_edges - 1)
            i, j = edges[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            delta = 0
            
            # Remove edge (i,j) from old_c
            # Update common counts for pairs involving i or j
            # For each k in adj[old_c][i] (excluding j): common[old_c][j][k] -= 1
            # For each k in adj[old_c][j] (excluding i): common[old_c][i][k] -= 1
            
            # Calculate cost change for old_c
            # Pairs affected: (i,k) for k in N(i, old_c) and (j,k) for k in N(j, old_c)
            # Note: (i,j) itself is not a common-neighbor pair in the sense of C4 formation 
            # unless we consider the edge itself, but C4 is formed by 4 distinct vertices.
            # The cost function sums over pairs (u,v) the number of C4s containing edge (u,v)?
            # No, the cost function defined is: sum over all pairs (i,j) of C(common[i][j], 2).
            # This counts the number of C4s in the graph.
            
            # Let's compute delta_cost_old_c
            # Before removal:
            # For each k in N(i, old_c) \ {j}: common[old_c][i][k] is current value.
            # After removal:
            # common[old_c][i][k] decreases by 1 if k is in N(j, old_c) \ {i}?
            # Wait. common[old_c][i][k] = |N(i) ∩ N(k)|.
            # If we remove edge (i,j), then j is no longer in N(i).
            # So if j was in N(k), then j was in N(i) ∩ N(k). Now it is not.
            # So common[old_c][i][k] -= 1 for all k such that (j,k) is an edge in old_c (and k != i).
            # Similarly, common[old_c][j][k] -= 1 for all k such that (i,k) is an edge in old_c (and k != j).
            
            # Let's iterate over neighbors to update common and track cost delta.
            
            # Cost contribution of a pair (u,v) is C(common[u][v], 2).
            # When common[u][v] changes from x to x-1:
            # Delta = C(x-1, 2) - C(x, 2) = (x-1)(x-2)/2 - x(x-1)/2 = (x-1)/2 * (x-2-x) = -(x-1).
            # So if common decreases by 1, cost decreases by (old_common - 1).
            
            # We need to sum these deltas for all affected pairs.
            
            # Affected pairs for old_c:
            # 1. Pairs (i, k) where k is in N(j, old_c) \ {i}.
            #    common[old_c][i][k] decreases by 1.
            #    Delta_cost += -(common[old_c][i][k] - 1)
            # 2. Pairs (j, k) where k is in N(i, old_c) \ {j}.
            #    common[old_c][j][k] decreases by 1.
            #    Delta_cost += -(common[old_c][j][k] - 1)
            
            # Note: If k is in both N(i) and N(j) (excluding each other), then (i,k) and (j,k) are distinct pairs.
            # Also, is pair (i,j) affected? common[old_c][i][j] is not changed by removing edge (i,j).
            
            mask_j_old = adj[old_c][j] & ~(1 << i)
            k = 0
            while mask_j_old:
                if mask_j_old & 1:
                    # Pair (i, k)
                    old_val = common[old_c][i][k]
                    if old_val > 0: # Should be >= 1 since j was a common neighbor
                        delta -= (old_val - 1)
                    common[old_c][i][k] -= 1
                    common[old_c][k][i] -= 1
                mask_j_old >>= 1
                k += 1
                
            mask_i_old = adj[old_c][i] & ~(1 << j)
            k = 0
            while mask_i_old:
                if mask_i_old & 1:
                    # Pair (j, k)
                    old_val = common[old_c][j][k]
                    if old_val > 0:
                        delta -= (old_val - 1)
                    common[old_c][j][k] -= 1
                    common[old_c][k][j] -= 1
                mask_i_old >>= 1
                k += 1
                
            # Update adj for old_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            
            # Add edge (i,j) to new_c
            # common[new_c][i][k] increases by 1 for k in N(j, new_c) \ {i}
            # common[new_c][j][k] increases by 1 for k in N(i, new_c) \ {j}
            
            # Cost change when common increases from x to x+1:
            # Delta = C(x+1, 2) - C(x, 2) = (x+1)x/2 - x(x-1)/2 = x.
            # So if common increases by 1, cost increases by old_common.
            
            mask_j_new = adj[new_c][j] & ~(1 << i)
            k = 0
            while mask_j_new:
                if mask_j_new & 1:
                    # Pair (i, k)
                    old_val = common[new_c][i][k]
                    delta += old_val
                    common[new_c][i][k] += 1
                    common[new_c][k][i] += 1
                mask_j_new >>= 1
                k += 1
                
            mask_i_new = adj[new_c][i] & ~(1 << j)
            k = 0
            while mask_i_new:
                if mask_i_new & 1:
                    # Pair (j, k)
                    old_val = common[new_c][j][k]
                    delta += old_val
                    common[new_c][j][k] += 1
                    common[new_c][k][j] += 1
                mask_i_new >>= 1
                k += 1
                
            # Update adj for new_c
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_cost += delta
                colors[idx] = new_c
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                # Undo
                # Revert old_c
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
                
                mask_j_old = adj[old_c][j] & ~(1 << i)
                k = 0
                while mask_j_old:
                    if mask_j_old & 1:
                        common[old_c][i][k] += 1
                        common[old_c][k][i] += 1
                    mask_j_old >>= 1
                    k += 1
                    
                mask_i_old = adj[old_c][i] & ~(1 << j)
                k = 0
                while mask_i_old:
                    if mask_i_old & 1:
                        common[old_c][j][k] += 1
                        common[old_c][k][j] += 1
                    mask_i_old >>= 1
                    k += 1
                    
                # Revert new_c
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                
                mask_j_new = adj[new_c][j] | (1 << i)
                k = 0
                while mask_j_new:
                    if mask_j_new & 1:
                        if k != i:
                            common[new_c][i][k] -= 1
                            common[new_c][k][i] -= 1
                    mask_j_new >>= 1
                    k += 1
                    
                mask_i_new = adj[new_c][i] | (1 << j)
                k = 0
                while mask_i_new:
                    if mask_i_new & 1:
                        if k != j:
                            common[new_c][j][k] -= 1
                            common[new_c][k][j] -= 1
                    mask_i_new >>= 1
                    k += 1
                    
                # colors[idx] remains old_c
            
            temperature *= cooling
            
        if best_cost == 0:
            return best_colors
        return None

    # Try to find valid coloring for increasing n
    best_n = 11
    # Fallback coloring for n=11 (known to be valid)
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # We will try n=26 first, then 25, etc.
    for n_try in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on n
        if n_try == 26:
            restarts = 5
            time_per_restart = 15.0
        elif n_try == 25:
            restarts = 4
            time_per_restart = 12.0
        else:
            restarts = 3
            time_per_restart = 8.0
            
        found = False
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            res = sa_search(n_try, max_time=time_per_restart)
            if res is not None:
                if full_check(n_try, res):
                    best_n = n_try
                    best_colors = res
                    found = True
                    break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
