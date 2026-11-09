import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def solve_n(n, time_limit):
        """
        Attempts to find a valid 5-coloring of K_n with no C4 in any color class.
        Returns (colors, valid)
        """
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # State:
        # colors: list of length m, color of edge idx
        # neigh[c][v]: bitmask of neighbors of v in color c
        # k[c][u][v]: number of common neighbors of u and v in color c.
        # We only need to store k for u < v? Or use a 2D array for each color.
        
        # To save memory and time, we can compute k on the fly? No, O(n) update is better.
        # Let's use a list of 5 2D arrays (or lists of lists) for k.
        # k[c] is an n x n matrix.
        
        def init_state():
            colors = [random.randint(0, 4) for _ in range(m)]
            neigh = [[0] * n for _ in range(5)]
            k = [[[0] * n for _ in range(n)] for _ in range(5)]
            
            # Build initial state
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                neigh[c][u] |= (1 << v)
                neigh[c][v] |= (1 << u)
            
            # Calculate initial k
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        common = neigh[c][u] & neigh[c][v]
                        k[c][u][v] = popcount(common)
                        k[c][v][u] = k[c][u][v]
            
            return colors, neigh, k

        def calc_energy(k):
            e = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        if k[c][u][v] > 1:
                            e += k[c][u][v] - 1
            return e

        def try_move(u, v, c_old, c_new, colors, idx, neigh, k):
            """
            Returns delta_energy if we change edge (u,v) from c_old to c_new.
            Assumes c_old != c_new.
            """
            if c_old == c_new:
                return 0
            
            # 1. Calculate change in energy for removing (u,v) from c_old
            # 2. Calculate change in energy for adding (u,v) to c_new
            
            delta = 0
            
            # Remove from c_old
            # v is removed from N(u), u from N(v)
            # Update k[u][w] and k[v][w] for all w
            
            # Helper to get contribution of a pair (a,b) to energy
            # contrib(x) = max(0, x - 1)
            
            # We need to update k for pairs involving u or v.
            # Pairs (u, w) and (v, w) for all w != u, v.
            
            # Before update:
            # k_old_uw = k[c_old][u][w]
            # k_old_vw = k[c_old][v][w]
            
            # After update:
            # If v in N_c_old(w): k_new_uw = k_old_uw - 1
            # Else: k_new_uw = k_old_uw
            # Similarly for v.
            
            # Note: k[c_old][u][v] is also affected?
            # u and v are neighbors.
            # Common neighbors of u and v:
            # Before: N(u) \cap N(v). v is in N(u), u is in N(v).
            # v is in N(u). Is v in N(v)? No.
            # So v is NOT a common neighbor of u and v.
            # u is NOT a common neighbor of u and v.
            # So removing the edge (u,v) does NOT change the set of common neighbors of u and v?
            # Wait.
            # N(u) contains v. N(v) contains u.
            # Common neighbors = (N(u) \ {v}) \cap (N(v) \ {u})? No.
            # N(u) \cap N(v).
            # If x is in N(u) \cap N(v), then x is neighbor of both.
            # Removing edge (u,v) removes v from N(u) and u from N(v).
            # Does this remove any common neighbor?
            # Only if v was in N(v) (false) or u was in N(u) (false).
            # So the set of common neighbors of u and v DOES NOT CHANGE when we remove the edge between them.
            # So k[c_old][u][v] remains the same.
            
            # However, for w != u, v:
            # k[c_old][u][w] changes if v was in N(c_old)[w].
            # Because v was in N(c_old)[u].
            # If v in N(c_old)[w], then v was a common neighbor of u and w.
            # Now v is not in N(c_old)[u], so v is no longer a common neighbor.
            # So k[c_old][u][w] decreases by 1.
            
            # So:
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Update u-w
                if (neigh[c_old][w] >> v) & 1:
                    old_kw = k[c_old][u][w]
                    new_kw = old_kw - 1
                    delta -= max(0, old_kw - 1)
                    delta += max(0, new_kw - 1)
                    k[c_old][u][w] = new_kw
                    k[c_old][w][u] = new_kw
                else:
                    # No change in k[u][w]
                    pass
                
                # Update v-w
                if (neigh[c_old][w] >> u) & 1:
                    old_kw = k[c_old][v][w]
                    new_kw = old_kw - 1
                    delta -= max(0, old_kw - 1)
                    delta += max(0, new_kw - 1)
                    k[c_old][v][w] = new_kw
                    k[c_old][w][v] = new_kw
                else:
                    pass
            
            # Add to c_new
            # u is added to N(v), v to N(u)
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Update u-w
                if (neigh[c_new][w] >> v) & 1:
                    old_kw = k[c_new][u][w]
                    new_kw = old_kw + 1
                    delta -= max(0, old_kw - 1)
                    delta += max(0, new_kw - 1)
                    k[c_new][u][w] = new_kw
                    k[c_new][w][u] = new_kw
                else:
                    pass
                
                # Update v-w
                if (neigh[c_new][w] >> u) & 1:
                    old_kw = k[c_new][v][w]
                    new_kw = old_kw + 1
                    delta -= max(0, old_kw - 1)
                    delta += max(0, new_kw - 1)
                    k[c_new][v][w] = new_kw
                    k[c_new][w][v] = new_kw
                else:
                    pass
            
            # Apply the move to neigh and colors
            # Remove from c_old
            neigh[c_old][u] &= ~(1 << v)
            neigh[c_old][v] &= ~(1 << u)
            # Add to c_new
            neigh[c_new][u] |= (1 << v)
            neigh[c_new][v] |= (1 << u)
            colors[idx] = c_new
            
            return delta

        def revert_move(u, v, c_old, c_new, colors, idx, neigh, k):
            """
            Reverts the move. c_new is current, c_old is target.
            """
            # This is essentially the same as try_move(c_new, c_old) but we know the state.
            # For simplicity and safety, we might just keep a copy of the state?
            # No, that's too much memory.
            # We can just re-run the logic in reverse.
            # Or, since we only accept if delta < 0, we don't need to revert often if we use hill climbing.
            # But if we use Simulated Annealing, we need to revert.
            # Let's stick to Hill Climbing for now to keep it simple and fast.
            pass

        # Local Search
        best_energy = float('inf')
        best_colors = None
        
        t_start = time.time()
        while time.time() < t_start + time_limit:
            colors, neigh, k = init_state()
            energy = calc_energy(k)
            
            if energy == 0:
                return colors, True
            
            if energy < best_energy:
                best_energy = energy
                best_colors = colors[:]
                
            improved = True
            while improved and time.time() < t_start + time_limit:
                improved = False
                # Randomly pick an edge
                for _ in range(1000): # 1000 random attempts per pass
                    if time.time() > t_start + time_limit:
                        break
                    idx = random.randint(0, m - 1)
                    u, v = edges[idx]
                    c_old = colors[idx]
                    
                    best_delta = 0
                    best_c_new = -1
                    
                    for c_new in range(5):
                        if c_new == c_old:
                            continue
                        
                        # Calculate delta
                        # We need a function that calculates delta WITHOUT modifying state,
                        # or we modify and revert.
                        # Modifying and reverting is safer for correctness.
                        
                        # Save state of affected k values?
                        # Or just use a "dry run" calculation.
                        
                        # Let's implement a dry run calculation.
                        d = 0
                        # Remove from c_old
                        for w in range(n):
                            if w == u or w == v: continue
                            if (neigh[c_old][w] >> v) & 1:
                                old_kw = k[c_old][u][w]
                                new_kw = old_kw - 1
                                d -= max(0, old_kw - 1)
                                d += max(0, new_kw - 1)
                            if (neigh[c_old][w] >> u) & 1:
                                old_kw = k[c_old][v][w]
                                new_kw = old_kw - 1
                                d -= max(0, old_kw - 1)
                                d += max(0, new_kw - 1)
                        
                        # Add to c_new
                        for w in range(n):
                            if w == u or w == v: continue
                            if (neigh[c_new][w] >> v) & 1:
                                old_kw = k[c_new][u][w]
                                new_kw = old_kw + 1
                                d -= max(0, old_kw - 1)
                                d += max(0, new_kw - 1)
                            if (neigh[c_new][w] >> u) & 1:
                                old_kw = k[c_new][v][w]
                                new_kw = old_kw + 1
                                d -= max(0, old_kw - 1)
                                d += max(0, new_kw - 1)
                        
                        if d < best_delta:
                            best_delta = d
                            best_c_new = c_new
                    
                    if best_c_new != -1:
                        # Apply the move
                        u, v = edges[idx]
                        c_old = colors[idx]
                        c_new = best_c_new
                        
                        # Update k
                        for w in range(n):
                            if w == u or w == v: continue
                            if (neigh[c_old][w] >> v) & 1:
                                old_kw = k[c_old][u][w]
                                new_kw = old_kw - 1
                                k[c_old][u][w] = new_kw
                                k[c_old][w][u] = new_kw
                            if (neigh[c_old][w] >> u) & 1:
                                old_kw = k[c_old][v][w]
                                new_kw = old_kw - 1
                                k[c_old][v][w] = new_kw
                                k[c_old][w][v] = new_kw
                            
                            if (neigh[c_new][w] >> v) & 1:
                                old_kw = k[c_new][u][w]
                                new_kw = old_kw + 1
                                k[c_new][u][w] = new_kw
                                k[c_new][w][u] = new_kw
                            if (neigh[c_new][w] >> u) & 1:
                                old_kw = k[c_new][v][w]
                                new_kw = old_kw + 1
                                k[c_new][v][w] = new_kw
                                k[c_new][w][v] = new_kw
                        
                        # Update neigh
                        neigh[c_old][u] &= ~(1 << v)
                        neigh[c_old][v] &= ~(1 << u)
                        neigh[c_new][u] |= (1 << v)
                        neigh[c_new][v] |= (1 << u)
                        
                        colors[idx] = c_new
                        energy += best_delta
                        improved = True
                        
                        if energy == 0:
                            return colors, True

        if best_colors and calc_energy_from_colors(n, best_colors) == 0:
             return best_colors, True
        return best_colors, False

    def calc_energy_from_colors(n, colors):
        # Slow check for final verification
        m = n * (n - 1) // 2
        neigh = [set() for _ in range(5)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            neigh[c].add((u, v)) # This is not efficient for neighbor sets
            
        # Rebuild neighbor sets properly
        neigh_sets = [ [set() for _ in range(n)] for _ in range(5) ]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            neigh_sets[c][u].add(v)
            neigh_sets[c][v].add(u)
            
        e = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neigh_sets[c][u] & neigh_sets[c][v]
                    if len(common) > 1:
                        e += len(common) - 1
        return e

    # Try n=27
    if time.time() < deadline:
        colors27, valid27 = solve_n(27, 60.0)
        if valid27:
            return {'n': 27, 'colors': colors27}

    # Try n=26
    if time.time() < deadline:
        colors26, valid26 = solve_n(26, 45.0)
        if valid26:
            return {'n': 26, 'colors': colors26}

    # Fallback n=25
    if time.time() < deadline:
        colors25, valid25 = solve_n(25, 10.0)
        if valid25:
            return {'n': 25, 'colors': colors25}
            
    # Fallback n=20
    n20 = 20
    m20 = n20 * (n20 - 1) // 2
    colors20 = [0] * m20
    for idx in range(m20):
        i = idx // (n20 - 1) # Approx
        # Correct mapping
        k = 0
        for i in range(n20):
            if idx < k + n20 - 1 - i:
                j = i + 1 + (idx - k)
                break
            k += n20 - 1 - i
        d = (j - i) % n20
        colors20[idx] = d % 5
    if calc_energy_from_colors(n20, colors20) == 0:
        return {'n': n20, 'colors': colors20}
        
    return {'n': 1, 'colors': [0]}

