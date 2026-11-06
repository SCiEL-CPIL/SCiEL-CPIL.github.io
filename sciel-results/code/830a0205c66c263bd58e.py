import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 115.0

    # ---------- 1. Hard-coded valid colouring for n=17 (fallback guarantee) ----------
    def _hardcoded_17():
        n = 17
        E = n * (n - 1) // 2
        colors = [0] * E
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Try to build a valid 17-vertex coloring via a deterministic pattern
                # We'll use a random-looking but fixed seed to generate it
                colors[idx] = 0
                idx += 1
        # Actually, we can't guarantee a hardcoded pattern is valid.
        # Instead, we'll generate one on the fly using a fast local search if we have time,
        # but for the fallback, we return n=16 if 17 fails.
        # For now, return None to signal we need to search.
        return None

    # ---------- 2. C6 Detection ----------
    def _has_c6_bitset(adj, n):
        """Check if graph defined by adjacency masks contains a C6."""
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
                            
                            if mask_v5:
                                return True
        return False

    def _build_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def _count_c6s(colors, n):
        adj = _build_adj(colors, n)
        total = 0
        for c in range(4):
            # Simple count by checking each subset? No, use the detection logic modified to count.
            # Actually, for local search, we just need to know if it's 0 or not.
            # But for SA, we need a penalty score.
            # Let's just check validity for simplicity in the "greedy" phase, 
            # and use a proxy for SA? 
            # No, let's implement a fast "number of C6s" counter.
            # It's O(N^6) which is too slow for SA if we call it every step.
            # Alternative: Only check validity. If valid, score = 0. If not, score = 1?
            # That makes the landscape very steep (binary).
            # Better: Count "partial" C6s? Or just use validity.
            # Let's use validity for now. If we have time, we can optimize.
            if _has_c6_bitset(adj[c], n):
                return 1 # Just indicate invalid
        return 0

    # ---------- 3. Local Search / Simulated Annealing ----------
    def _local_search(n, time_limit):
        """
        Use Simulated Annealing or Greedy Local Search to find a valid 4-coloring of K_n.
        Objective: Minimize number of monochromatic C6s.
        Since counting is expensive, we use a heuristic:
        1. Start with random coloring.
        2. If invalid, pick a random C6 (if we can find one quickly) and flip a random edge in it?
        Or:
        1. Start with random coloring.
        2. While invalid:
           - Find a C6 in any color.
           - Change color of a random edge in that C6 to a different color.
           - Check if still valid.
        """
        deadline = time.time() + time_limit
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize with random coloring
        random.seed(42) # Deterministic start? Or random each time?
        colors = [random.randint(0, 3) for _ in range(E)]
        
        # Check initial validity
        if not _has_c6_bitset(_build_adj(colors, n), n):
            # Wait, _build_adj returns 4 graphs. We need to check all 4.
            adj = _build_adj(colors, n)
            valid = True
            for c in range(4):
                if _has_c6_bitset(adj[c], n):
                    valid = False
                    break
            if valid:
                return colors
        
        # Iterative improvement
        # We need a function to find a specific C6 to fix.
        def find_c6(adj, n, color):
            """Return a list of 6 vertices forming a C6 in color, or None."""
            for v0 in range(n):
                mask_v1 = adj[color][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    mask_v2 = adj[color][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        mask_v3 = adj[color][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            mask_v4 = adj[color][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                mask_v5 = adj[color][v4] & adj[color][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    v5 = (mask_v5 & -mask_v5).bit_length() - 1
                                    return [v0, v1, v2, v3, v4, v5]
            return None

        steps = 0
        while time.time() < deadline:
            adj = _build_adj(colors, n)
            found_c6 = None
            found_color = -1
            for c in range(4):
                c6 = find_c6(adj, n, c)
                if c6:
                    found_c6 = c6
                    found_color = c
                    break
            
            if not found_c6:
                return colors # Found valid coloring
            
            # We found a C6 in color found_color.
            # Strategy: Pick a random edge in this C6 and recolor it.
            # Edges in C6: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
            cycle_edges = [(found_c6[0], found_c6[1]),
                           (found_c6[1], found_c6[2]),
                           (found_c6[2], found_c6[3]),
                           (found_c6[3], found_c6[4]),
                           (found_c6[4], found_c6[5]),
                           (found_c6[5], found_c6[0])]
            
            # Pick a random edge from the cycle
            e_idx = random.randint(0, 5)
            u, v = cycle_edges[e_idx]
            
            # Find index of this edge in the colors array
            # Edge (u,v) where u < v
            if u > v:
                u, v = v, u
            # Index calculation: sum of (n-i) for i < u? 
            # Easier: precompute edge indices
            # But we can just search or use a formula.
            # Formula for index of (u,v) in lex order:
            # count edges (i,j) with i < u: sum_{k=0}^{u-1} (n-1-k)
            # count edges (i,v) with i < u? No, (i,j) with i=u and j<v?
            # Let's just do a linear search for now, it's O(E) which is fine for <136 edges.
            # Or precompute a dict.
            edge_to_idx = {}
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    edge_to_idx[(i, j)] = idx
                    idx += 1
            
            e_pos = edge_to_idx.get((u, v))
            if e_pos is None:
                e_pos = edge_to_idx.get((v, u))
            
            old_color = colors[e_pos]
            # Try changing to a different color
            new_color = (old_color + 1 + random.randint(0, 2)) % 4
            colors[e_pos] = new_color
            
            steps += 1
            if steps % 100 == 0 and time.time() > deadline:
                break

        return None

    # ---------- 4. Backtracking (Fallback / Exact) ----------
    def _backtrack_search(n, time_limit):
        deadline = time.time() + time_limit
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * E
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            # Symmetry breaking: first edge must be color 0
            if idx == 0:
                color_range = [0]
            else:
                color_range = [0, 1, 2, 3]
                
            for c in color_range:
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                # Check C6 in color c
                if not _has_c6_bitset(adj[c], n):
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
                
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                
            return False
        
        return colors if backtrack(0) else None

    # ---------- 5. Main Logic ----------
    # Try n=17 first
    # Strategy: 
    # 1. Quick local search (10s)
    # 2. If fails, try backtracking (30s) - might be too slow for 17
    # 3. If fails, drop to n=16
    
    best_n = 0
    best_colors = None
    
    # Priority: 17, 16, 15...
    # Time allocation:
    # n=17: 20s local search, 20s backtrack
    # n=16: 10s local search, 10s backtrack
    # n=15: 5s local search
    
    for n in [17, 16, 15, 14]:
        if time.time() > DEADLINE - 5:
            break
        
        # Try local search
        sol = _local_search(n, 10.0)
        if sol:
            # Verify
            adj = _build_adj(sol, n)
            valid = True
            for c in range(4):
                if _has_c6_bitset(adj[c], n):
                    valid = False
                    break
            if valid:
                best_n = n
                best_colors = sol
                break
        
        # Try backtracking (only for smaller n or if local search fails)
        if n <= 16 and time.time() < DEADLINE - 10:
            sol_bt = _backtrack_search(n, 5.0)
            if sol_bt:
                best_n = n
                best_colors = sol_bt
                break

    # Fallback if nothing found (should not happen for n<=14)
    if best_n == 0:
        # Hardcode a small valid solution?
        # n=4: K4 is C4-free? No, K4 has C4. But we need C6-free.
        # K4 is small enough that no C6 exists.
        # Any coloring of K4 is valid.
        n = 4
        E = 6
        best_colors = [0, 1, 2, 3, 0, 1]
        best_n = n

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res)
