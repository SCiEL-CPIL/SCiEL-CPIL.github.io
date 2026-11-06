"""
Direct construction of a 4-coloring of K_n with no monochromatic triangle.

Strategy:
  The seed gives a valid 4-coloring of K_32 (two disjoint Clebsch copies, inter-copy edges all color 3).
  We attempt to extend it to larger n by greedy + local-search addition of vertices, and if that
  fails we fall back to a random local-search from scratch for larger n.
  
  Key insight: The known record is n=50 (Chung). We try to reach n=48 or 50.
  
  Approach:
  1. Start with the K_32 coloring.
  2. For each new vertex v (32, 33, ...), try to assign colors to its edges to existing vertices
     such that no monochromatic triangle is formed. Use a backtracking/greedy method.
  3. If greedy fails for a vertex, use local search on the existing coloring to "make room".
  
  Time budget: ~110 seconds to leave safety margin.
"""

import time
import random
import itertools
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill
    
    random.seed(42)
    
    # --- Helper: check if adding a vertex with given colors creates a mono triangle ---
    # colors list is for the current n vertices. For a new vertex at index n,
    # we propose a list 'new_colors' of length n (color for edge (i, n) for i in 0..n-1).
    # A mono triangle (i, j, n) with color c exists if:
    #   colors[i*n - i*(i+1)//2 + (j-i-1)] == c  AND  new_colors[i] == c  AND  new_colors[j] == c
    # More precisely, for each pair (i, j) with i < j < n:
    #   if new_colors[i] == new_colors[j] == c, then edge (i,j) must NOT have color c.
    
    def edge_idx(i, j, n):
        # index of edge (i,j) in the list of n-vertex edges (lex order)
        # number of edges before vertex i: i*n - i*(i+1)//2
        # within vertex i's row, edge to j is at offset (j - i - 1)
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def check_new_vertex_valid(n, colors, new_colors):
        """Check if adding vertex n with edges given by new_colors (length n) creates no mono triangle."""
        for i in range(n):
            ci = new_colors[i]
            for j in range(i + 1, n):
                cj = new_colors[j]
                if ci == cj:
                    # edge (i,j) must not have this color
                    idx = edge_idx(i, j, n)
                    if colors[idx] == ci:
                        return False
        return True
    
    def try_greedy_add_vertex(n, colors):
        """Try to assign colors for a new vertex n. Returns list of n colors or None."""
        # For each new vertex, we need to assign colors[0..n-1] such that
        # for any i<j: if colors[i]==colors[j]==c then edge(i,j)!=c
        # Use backtracking with forward checking.
        
        assignment = [None] * n
        
        def backtrack(v):
            if time.time() > DEADLINE:
                return None
            if v == n:
                return assignment[:]
            
            for c in [0, 1, 2, 3]:
                assignment[v] = c
                # Check consistency with already-assigned vertices
                ok = True
                for i in range(v):
                    if assignment[i] == c:
                        # edge (i, v) must not be color c
                        idx = edge_idx(i, v, n)
                        if colors[idx] == c:
                            ok = False
                            break
                if not ok:
                    assignment[v] = None
                    continue
                # Also check: for any pair (i, v) with assignment[i] == c, we already checked edge(i,v).
                # But also: for pairs (i, j) where i<j<v, if assignment[i]==assignment[j]==c, edge(i,j)!=c.
                # This is already satisfied by the existing coloring (it was valid).
                # The only new constraints are involving v.
                result = backtrack(v + 1)
                if result is not None:
                    return result
                assignment[v] = None
            return None
        
        return backtrack(0)
    
    # --- Step 1: Build K_32 coloring (same as seed) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    n = 32
    
    # --- Step 2: Greedily add vertices 32, 33, ... up to some limit ---
    max_target = 50
    for target in range(32, max_target + 1):
        if time.time() > DEADLINE - 5:
            break
        new_colors = try_greedy_add_vertex(n, colors)
        if new_colors is not None:
            # Add the new vertex
            for i in range(n):
                colors.append(new_colors[i])
            n += 1
        else:
            # Greedy failed. Try local search to permute/relabel to make room.
            # Simple approach: try random perturbations of existing colors and retry.
            success = False
            for _ in range(200):
                if time.time() > DEADLINE - 5:
                    break
                # Try a random recoloring of a random edge
                ei = random.randint(0, len(colors) - 1)
                old = colors[ei]
                colors[ei] = (old + 1 + random.randint(0, 2)) % 4
                new_colors = try_greedy_add_vertex(n, colors)
                if new_colors is not None:
                    # Keep the perturbation and add vertex
                    for i in range(n):
                        colors.append(new_colors[i])
                    n += 1
                    success = True
                    break
                else:
                    colors[ei] = old  # revert
            if not success:
                break
    
    # --- Step 3: If we didn't reach 50, try local search from a random K_n ---
    if n < 50 and time.time() < DEADLINE - 10:
        # Try random initialization + local search for larger n
        for trial_n in [50, 48, 46, 44, 42, 40]:
            if trial_n <= n:
                continue
            if time.time() > DEADLINE - 5:
                break
            
            # Random 4-coloring
            rand_colors = [random.randint(0, 3) for _ in range(trial_n * (trial_n - 1) // 2)]
            
            # Local search: find mono triangles, recolor one edge randomly
            best_n = n
            for iteration in range(50000):
                if time.time() > DEADLINE - 5:
                    break
                
                # Find a monochromatic triangle
                found = -1
                found_color = -1
                found_i, found_j, found_k = -1, -1, -1
                
                # Sample random triangles
                for _ in range(20):
                    i, j, k = random.sample(range(trial_n), 3)
                    # Get colors of edges (i,j), (i,k), (j,k)
                    if i > j: i, j = j, i
                    if i > k: i, k = k, i
                    if j > k: j, k = k, j
                    
                    ij = edge_idx(i, j, trial_n)
                    ik = edge_idx(i, k, trial_n)
                    jk = edge_idx(j, k, trial_n)
                    
                    c_ij = rand_colors[ij]
                    c_ik = rand_colors[ik]
                    c_jk = rand_colors[jk]
                    
                    if c_ij == c_ik == c_jk:
                        found = ij
                        found_color = c_ij
                        found_i, found_j, found_k = i, j, k
                        break
                else:
                    # No triangle found in sample; likely valid or nearly valid
                    # Do a full check
                    valid = True
                    for i in range(trial_n):
                        for j in range(i + 1, trial_n):
                            c_ij = rand_colors[edge_idx(i, j, trial_n)]
                            for k in range(j + 1, trial_n):
                                c_ik = rand_colors[edge_idx(i, k, trial_n)]
                                c_jk = rand_colors[edge_idx(j, k, trial_n)]
                                if c_ij == c_ik == c_jk:
                                    valid = False
                                    found = edge_idx(i, j, trial_n)
                                    found_color = c_ij
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # We found a valid coloring of K_trial_n!
                        return {'n': trial_n, 'colors': rand_colors}
                    
                    if not valid and found < 0:
                        # Shouldn't happen, but safety
                        break
                
                if found < 0:
                    # No triangle found in sampling; do full check
                    valid = True
                    for i in range(trial_n):
                        for j in range(i + 1, trial_n):
                            c_ij = rand_colors[edge_idx(i, j, trial_n)]
                            for k in range(j + 1, trial_n):
                                c_ik = rand_colors[edge_idx(i, k, trial_n)]
                                c_jk = rand_colors[edge_idx(j, k, trial_n)]
                                if c_ij == c_ik == c_jk:
                                    valid = False
                                    found = edge_idx(i, j, trial_n)
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {'n': trial_n, 'colors': rand_colors}
                    if found < 0:
                        break
                
                # Recolor the found edge
                rand_colors[found] = (found_color + 1 + random.randint(0, 2)) % 4
            
            # Didn't find valid for this trial_n, continue
    
    # Return the best we have
    if time.time() > DEADLINE - 2:
        return {'n': n, 'colors': colors}
    
    return {'n': n, 'colors': colors}
