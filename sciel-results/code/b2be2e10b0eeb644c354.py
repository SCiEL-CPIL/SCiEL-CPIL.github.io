import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def _count_c6_bitset(adj, n):
        """
        Count the number of monochromatic C6s in the graph defined by adj.
        Returns 0 if no C6 exists.
        adj[i] is an integer bitmask.
        """
        count = 0
        for v0 in range(n):
            mask_v1 = adj[v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                
                mask_v2 = adj[v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    
                    mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        
                        mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            
                            mask_v5 = adj[v4] & adj[v0]
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            
                            count += bin(mask_v5).count('1')
        return count

    def _is_valid(colors, n):
        """Check if the coloring is valid (no mono C6)."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
            
        for c in range(4):
            if _count_c6_bitset(adj[c], n) > 0:
                return False
        return True

    def _run_sa(n, time_limit):
        """
        Run Simulated Annealing to find a valid 4-coloring of K_n.
        Returns colors list if found, else None.
        """
        end_time = time.time() + time_limit
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize randomly
        colors = [random.randint(0, 3) for _ in range(E)]
        
        # Build initial adjacency
        adj = [[0] * n for _ in range(4)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
        current_energy = sum(_count_c6_bitset(adj[c], n) for c in range(4))
        
        if current_energy == 0:
            return colors
            
        T = 100.0  # Initial temperature
        cooling = 0.9995  # Cooling factor
        min_T = 0.01
        
        steps = 0
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Pick a random edge to flip
            idx = random.randint(0, E - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            
            # Choose a new color different from old_c
            new_c = random.randint(0, 3)
            while new_c == old_c:
                new_c = random.randint(0, 3)
                
            # Undo old edge
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            old_energy_c = _count_c6_bitset(adj[old_c], n)
            
            # Apply new edge
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            new_energy_c = _count_c6_bitset(adj[new_c], n)
            
            delta_E = (new_energy_c - old_energy_c)
            
            # Note: We only recalculated for the affected colors. 
            # The total energy change is exactly delta_E because only color old_c and new_c changed.
            
            if delta_E <= 0 or random.random() < pow(2.718281828, -delta_E / T):
                # Accept move
                colors[idx] = new_c
                current_energy += delta_E
                
                if current_energy == 0:
                    return colors
            else:
                # Reject move: Undo new edge, restore old edge
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                
            T *= cooling
            steps += 1
            
            # Periodic check to avoid infinite loops in tight time
            if steps % 10000 == 0 and time.time() > end_time:
                break
                
        return None

    best_n = 0
    best_colors = None

    # Try to find valid colorings for n=17 down to n=1
    # Allocate more time to larger n as they are harder but more valuable
    time_allocs = {17: 120, 16: 15, 15: 5, 14: 2, 13: 1, 12: 0.5, 11: 0.5, 10: 0.5, 9: 0.5, 8: 0.5, 7: 0.5, 6: 0.5, 5: 0.5, 4: 0.5, 3: 0.5, 2: 0.5, 1: 0.5}

    for n in range(17, 0, -1):
        if time.time() > deadline - 2:
            break
            
        t_alloc = time_allocs.get(n, 1.0)
        
        # For n=17, we might need multiple restarts if the first one fails
        # But SA is stochastic, so a single long run is often better than many short ones
        # However, if it gets stuck in a local minimum with high energy, restarting helps.
        # Let's do one long run first.
        
        sol = _run_sa(n, t_alloc)
        if sol is not None:
            # Verify just in case
            if _is_valid(sol, n):
                best_n = n
                best_colors = sol
                break # Found the max n we tried
        
        # If n=17 fails, we don't have time for many restarts.
        # If n=16 fails, try another quick restart?
        if n <= 16 and t_alloc > 1.0:
             # Maybe try a second short run?
             pass # Keeping it simple for now

    # Fallback
    if best_n == 0:
        n = 4
        best_colors = [0, 1, 2, 3, 0, 1] 
        best_n = n

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res)
