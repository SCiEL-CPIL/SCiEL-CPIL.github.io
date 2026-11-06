import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s/150s limits
    
    # Fallback: seed solution for n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors
    
    best_n, best_colors = get_seed_solution()
    
    # Helper: count violations using bitsets
    def count_violations(n, colors):
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    # Check if >= 2 bits set (popcount >= 2)
                    if common & (common - 1):
                        cost += 1
        return cost

    def solve_local_search(n, time_limit):
        """
        Local search with incremental cost updates.
        Returns a valid coloring list or None if not found within time_limit.
        """
        total_edges = n * (n - 1) // 2
        # Initialize random
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Build state for incremental updates
        # color_matrix[i][j]
        color_matrix = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        
        # neighbors[c][u] is a bitmask of neighbors of u in color c
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        # Precompute bit masks for vertices? Not strictly needed, 1<<v is fast.
        
        # Function to get cost from current state
        def get_cost():
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0:
                        continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost
        
        current_cost = get_cost()
        if current_cost == 0:
            return colors
        
        search_start = time.time()
        
        while time.time() - search_start < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = color_matrix[i][j]
            
            # Try all other colors
            best_new_color = -1
            best_cost = current_cost
            
            # We can optimize by only checking colors that might reduce cost,
            # but for now, check all 4 others.
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                # Calculate delta cost
                # Removing edge (i,j) from c_old
                # Adding edge (i,j) to c_new
                
                # The change in cost comes from pairs (u,v) where the common neighbor count changes.
                # A pair (u,v) is affected if u or v is i or j, AND the other vertex's neighborhood in the respective color changes.
                
                # Actually, the only pairs (u,v) whose common neighbor count in color c changes are those where
                # one of u,v is i or j, and the "other" vertex w is a neighbor of the "first" vertex in color c.
                
                # Let's derive the delta carefully.
                # Let N_c(x) be the set of neighbors of x in color c.
                # Cost = sum_{u<v} C(|N_c(u) ∩ N_c(v)|, 2)
                
                # Change:
                # 1. (i,j) removed from c_old.
                #    N_c_old(i) loses j. N_c_old(j) loses i.
                #    For any pair (u,v), the intersection size changes only if j was in the intersection (affecting pairs involving i) or i was in the intersection (affecting pairs involving j).
                #    Specifically:
                #    - Pairs (i, w) where w in N_c_old(j) (w != i,j): j is a common neighbor of i and w. So common count for (i,w) decreases by 1.
                #    - Pairs (j, w) where w in N_c_old(i) (w != i,j): i is a common neighbor of j and w. So common count for (j,w) decreases by 1.
                #    - Pair (i,j): common neighbors are N_c_old(i) ∩ N_c_old(j). Neither i nor j is in this set (no self-loops). So (i,j) count is UNCHANGED by removing the edge (i,j) itself?
                #      Wait. N_c_old(i) contains j. N_c_old(j) contains i.
                #      Intersection N_c_old(i) ∩ N_c_old(j) does NOT contain i or j.
                #      So yes, common count for (i,j) is unaffected by the status of edge (i,j).
                
                # 2. (i,j) added to c_new.
                #    N_c_new(i) gains j. N_c_new(j) gains i.
                #    - Pairs (i, w) where w in N_c_new(j) (w != i,j): j is a common neighbor of i and w. So common count for (i,w) increases by 1.
                #    - Pairs (j, w) where w in N_c_new(i) (w != i,j): i is a common neighbor of j and w. So common count for (j,w) increases by 1.
                
                # Delta calculation:
                # For a pair (u,v) with common count k, change to k-1: delta = - (k-1)
                # Change to k+1: delta = + k
                
                delta = 0
                
                # c_old removals
                # Pairs (i, w) for w in N_c_old(j) \ {i}
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_old][j] & (1 << w):
                        # w is in N_c_old(j)
                        # This means j is a common neighbor of i and w (since i is connected to j in c_old, and w is connected to j in c_old).
                        # Wait, is i connected to w in c_old? No, common neighbor j is connected to both.
                        # The pair is (i, w). The common neighbor is j.
                        # Current common count for (i, w) in c_old:
                        k = bin(neighbors[c_old][i] & neighbors[c_old][w]).count('1')
                        # This k includes j.
                        # After removal, k becomes k-1.
                        # Delta = - (k-1)
                        delta -= (k - 1)
                        
                    if neighbors[c_old][i] & (1 << w):
                        # w is in N_c_old(i)
                        # Pair (j, w). Common neighbor is i.
                        k = bin(neighbors[c_old][j] & neighbors[c_old][w]).count('1')
                        # k includes i.
                        delta -= (k - 1)
                
                # c_new additions
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_new][j] & (1 << w):
                        # w is in N_c_new(j)
                        # Pair (i, w). Common neighbor j will be added.
                        # Current common count for (i, w) in c_new:
                        k = bin(neighbors[c_new][i] & neighbors[c_new][w]).count('1')
                        # After addition, k becomes k+1.
                        # Delta = + k
                        delta += k
                        
                    if neighbors[c_new][i] & (1 << w):
                        # w is in N_c_new(i)
                        # Pair (j, w). Common neighbor i will be added.
                        k = bin(neighbors[c_new][j] & neighbors[c_new][w]).count('1')
                        delta += k
                        
                new_cost = current_cost + delta
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_new_color = c_new
            
            # Apply best move
            if best_new_color != -1:
                # Update state
                # Remove from c_old
                neighbors[c_old][i] &= ~(1 << j)
                neighbors[c_old][j] &= ~(1 << i)
                # Add to c_new
                neighbors[c_new][i] |= (1 << j)
                neighbors[c_new][j] |= (1 << i)
                
                color_matrix[i][j] = best_new_color
                color_matrix[j][i] = best_new_color
                
                # Update flat colors
                # Find index of (i,j)
                # idx = i*(2*n - i - 1)//2 + j - i - 1
                idx = i * (2 * n - i - 1) // 2 + j - i - 1
                colors[idx] = best_new_color
                
                current_cost = best_cost
                
                if current_cost == 0:
                    return colors
                    
            # Optional: Random restart or perturbation if stuck?
            # For now, simple hill climbing.
            
        return None

    # Strategy: Try to find valid colorings for n=26 down to 23.
    # Allocate time slices.
    # If we find n=26, return it.
    # If not, try 25, etc.
    
    # We should also keep the best solution found so far.
    # The seed is n=11.
    # If local search finds n=22, that's better than 11.
    
    # Let's try n=26 first with a generous time limit, then 25, etc.
    # But if 26 fails, we might have wasted time.
    # However, 26 is the max known.
    
    # Let's try n=24, 25, 26.
    # If we find a valid coloring for n, we update best.
    
    # To be safe, let's start from a lower n to ensure we get *some* improvement over 11 quickly, then try higher.
    # Or just try 26, 25, 24, 23 in order of preference, but stop if time is low.
    
    # Let's try 24 first as a balance? No, 26 is the target.
    # Let's try 26 with 30s, 25 with 30s, 24 with 30s, 23 with 20s.
    
    targets = [26, 25, 24, 23, 22]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        # Allocate remaining time proportionally or fixed?
        # Let's give each target 20 seconds.
        t_limit = min(20.0, deadline - time.time() - 2)
        if t_limit < 1.0:
            break
            
        sol = solve_local_search(n, t_limit)
        
        if sol is not None:
            # Verify
            if count_violations(n, sol) == 0:
                best_n = n
                best_colors = sol
                # If we found a valid coloring, we can break if n is high enough?
                # Or continue to try higher?
                # The list is descending, so if we find 26, we are done.
                # If we find 25, we should have tried 26 first.
                # So if we find a solution, we can return it immediately?
                # No, maybe 26 was skipped due to time? No, we iterate in order.
                # If we found 26, break.
                # If we found 25, it means 26 failed. So we keep 25.
                # We should break because we are going downwards.
                break
            else:
                # Invalid? Should not happen if solve_local_search returns 0 cost.
                pass
                
    # If we didn't find anything in the loop (e.g. all failed),
    # we still have the seed n=11.
    # But maybe we can try a quick search for n=22?
    # The current code structure returns the best found.
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
