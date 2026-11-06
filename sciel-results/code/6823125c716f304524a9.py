import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Fallback: n=11 (Known valid: K11 decomposed into 5 Hamiltonian cycles)
    # Cycle i uses edges (j, j+i) mod 11 for j=0..10. 
    # For n=11, distance d in 1..5.
    # Edge (i, j) with i<j. dist = min(j-i, 11-(j-i)).
    # Color = dist - 1.
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    def check_validity(colors, n):
        """Fast verification for a complete coloring."""
        if not colors:
            return n <= 2
        if len(colors) != n * (n - 1) // 2:
            return False
        
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c >= 5:
                    return False
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        if common.bit_count() > 1:
                            return False
        return True

    def solve_backtracking(n, time_limit):
        """
        Solves the C4-free 5-coloring problem for K_n using backtracking with forward checking.
        Returns a list of colors if found, None otherwise.
        """
        # Edge ordering: Lexicographic (i, j) with i < j
        # We process edges in this order.
        # State: 
        #   colors: list of assigned colors (or -1)
        #   adj: list of 5 lists of bitmasks, adj[c][u] is bitmask of neighbors of u in color c
        
        colors = [-1] * (n * (n - 1) // 2)
        adj = [[0] * n for _ in range(5)]
        
        # Precompute edge indices
        edge_idx = {}
        inv_edge = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                inv_edge.append((i, j))
                idx += 1
        
        def get_common_count_c(u, v, c):
            """Count common neighbors of u and v in color c, excluding u and v themselves."""
            # Common neighbors are k such that (u,k) is c and (v,k) is c.
            # Using bitmasks:
            mask_u = adj[c][u]
            mask_v = adj[c][v]
            common = mask_u & mask_v
            # Exclude u and v from the mask (though they shouldn't be in each other's mask unless self-loops, which we don't have)
            # But wait, adj[c][u] contains neighbors. u is not a neighbor of u.
            # So common contains vertices k such that k is neighbor of u and v in color c.
            return common.bit_count()

        def is_valid_assignment(i, j, c):
            # Check A: Common neighbors of i and j in color c
            # Note: (i,j) is not yet assigned, so j is not in adj[c][i] and i is not in adj[c][j].
            cnt_ij = get_common_count_c(i, j, c)
            if cnt_ij > 1:
                return False
            
            # Check B: For all k, if (k,j) is c, then common(i,k) in c must be 0
            # Because j will become a common neighbor of i and k.
            # We need to ensure that before adding (i,j), i and k had 0 common neighbors in c.
            # If they had 1, adding j makes it 2 -> C4.
            # If they had 0, adding j makes it 1 -> OK.
            
            # Iterate over all k
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check if (k,j) is c
                # Need to check if edge (k,j) exists and is color c
                # k,j order: if k<j, edge is (k,j), else (j,k)
                if k < j:
                    e_idx = edge_idx[(k, j)]
                else:
                    e_idx = edge_idx[(j, k)]
                
                if colors[e_idx] == c:
                    # (k,j) is c. So j is a neighbor of k in color c.
                    # And we are making (i,j) c, so j is a neighbor of i in color c.
                    # So j is a common neighbor of i and k.
                    # We need total common neighbors of i and k in c to be <= 1.
                    # Currently (before adding i-j), common neighbors are get_common_count_c(i,k,c).
                    # After adding, it becomes common_count + 1 (for j).
                    # So we need common_count + 1 <= 1 => common_count <= 0.
                    if get_common_count_c(i, k, c) > 0:
                        return False
                
                # Check C: Symmetric
                # If (i,k) is c, then i becomes a common neighbor of j and k.
                if k < i:
                    e_idx_ik = edge_idx[(k, i)]
                else:
                    e_idx_ik = edge_idx[(i, k)]
                
                if colors[e_idx_ik] == c:
                    # (i,k) is c. So i is a neighbor of k in color c.
                    # And we are making (i,j) c, so i is a neighbor of j in color c.
                    # So i is a common neighbor of j and k.
                    # We need common neighbors of j and k in c to be <= 1.
                    # Currently get_common_count_c(j,k,c).
                    # After adding, it becomes common_count + 1 (for i).
                    # So we need common_count <= 0.
                    if get_common_count_c(j, k, c) > 0:
                        return False
            
            return True

        def backtrack(idx):
            if time.time() > deadline - 2.0:
                return False
            
            if idx == len(inv_edge):
                return True
            
            i, j = inv_edge[idx]
            
            # Try colors 0-4
            # Randomize order to avoid symmetry issues
            colors_to_try = [0, 1, 2, 3, 4]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                if is_valid_assignment(i, j, c):
                    # Assign
                    colors[idx] = c
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Unassign
                    colors[idx] = -1
                    adj[c][i] &= ~(1 << j)
                    adj[c][j] &= ~(1 << i)
            
            return False

        # Run backtracking
        # To handle time limits, we can run multiple random restarts with different seeds
        # or just one long run. Backtracking can get stuck.
        # Let's try a few restarts.
        
        # Actually, backtracking is deterministic if we don't shuffle.
        # Shuffling makes it randomized.
        
        # Let's run backtracking for a fixed time.
        # Since backtracking might be slow for larger n, we limit the depth or time.
        # But n=12, 13, 14 are small enough that if a solution exists, it should be found quickly
        # unless the search space is pathological.
        
        # Let's just call backtrack once. If it fails, it failed.
        # But randomized restarts help.
        
        # We'll wrap it in a loop with time checks.
        
        # Actually, the recursive backtrack doesn't check time at every step efficiently if the stack is deep.
        # But n is small.
        
        # Let's try one run.
        if backtrack(0):
            return colors
        
        return None

    # Try to find solutions for n > 11
    # Backtracking is exact. If it finds a solution, it's valid.
    # If it doesn't, it might be because of time or no solution.
    # We assume solutions exist for small n.
    
    # We will try n=12, 13, 14.
    # Backtracking is generally better than SA for small n if the constraint propagation is strong.
    # My constraint check is O(n^2). For n=14, edges=91. 91 * 196 ~ 18k ops per node.
    # Depth 91. 18k * 91 ~ 1.6M ops per path.
    # If branching factor is low (due to constraints), it should be fast.
    
    for n in range(12, 15):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few restarts with different random seeds for the shuffle
        found = False
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            random.seed(restart * 1000 + n) # Deterministic per restart
            
            res = solve_backtracking(n, deadline - 5.0)
            
            if res is not None:
                # Verify just in case
                if check_validity(res, n):
                    best_n = n
                    best_colors = res
                    found = True
                    break
        
        if found:
            continue # Try next n
        else:
            # If we failed for n, should we try n+1?
            # If n=12 fails, n=13 is likely to fail or take much longer.
            # But maybe n=12 is hard and n=13 is easy? Unlikely.
            # Let's break if we fail n=12 to save time for other strategies if needed?
            # No, we only have this strategy.
            # If backtracking fails n=12 in ~20s, it's probably not finding it.
            # Let's break.
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
