import time
import random
import numpy as np

def _get_index(n, i, j):
    """
    Get the flat index for edge (i, j) with i < j in a list of length n*(n-1)//2.
    """
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_paley(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _count_violations(colors, n):
    """
    Count the number of monochromatic K2,2,2 structures.
    Returns 0 if valid.
    """
    if n < 6:
        return 0
        
    # Build adjacency bitsets
    # adj[c][i] is a bitmask of neighbors of i in color c
    adj = [[0] * n for _ in range(2)]
    
    for i in range(n):
        for j in range(i + 1, n):
            idx = _get_index(n, i, j)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    violations = 0
    
    for c in range(2):
        for u in range(n):
            for v in range(u + 1, n):
                # Check if (u, v) is an edge of color c
                if not (adj[c][u] & (1 << v)):
                    continue
                    
                # Common neighbors of u and v in color c, excluding u and v
                W = adj[c][u] & adj[c][v] & ~(1 << u) & ~(1 << v)
                
                # Get list of vertices in W
                # We need to find a, b in W such that there are at least 2 common neighbors in W for a and b
                # i.e., |W & adj[c][a] & adj[c][b]| >= 2
                # This ensures we can pick c, d from that intersection to form the third pair.
                
                # Iterate over pairs in W
                # To be efficient, we can just iterate over all a, b in W with a < b
                w_list = [x for x in range(n) if (W & (1 << x))]
                lw = len(w_list)
                
                if lw < 4:
                    continue
                    
                for ia in range(lw):
                    a = w_list[ia]
                    for ib in range(ia + 1, lw):
                        b = w_list[ib]
                        # Common neighbors of a and b in color c, restricted to W
                        common = W & adj[c][a] & adj[c][b]
                        # Count bits
                        # bin(x).count('1') is fast enough for n<=31
                        if bin(common).count('1') >= 2:
                            violations += 1
                            
    return violations

def _is_valid(colors, n):
    return _count_violations(colors, n) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify starting coloring
    if not _is_valid(colors, n):
        # Should not happen
        return {'n': n, 'colors': colors}
        
    best_n = n
    best_colors = colors[:]
    
    # Try to extend to n+1, n+2 ...
    for target_n in range(n + 1, 32):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline:
                break
                
            # Initialize new vertex edges randomly
            new_colors = colors[:]
            v_new = target_n - 1
            for i in range(v_new):
                new_colors.append(random.randint(0, 1))
                
            # Local search
            cur_colors = new_colors[:]
            cur_violations = _count_violations(cur_colors, target_n)
            
            if cur_violations == 0:
                if _is_valid(cur_colors, target_n):
                    best_n = target_n
                    best_colors = cur_colors[:]
                    found = True
                    break
                # If count is 0, it should be valid. _is_valid calls count anyway.
                # But maybe count has false negatives? Unlikely with this logic.
                # Let's assume count==0 implies valid.
                best_n = target_n
                best_colors = cur_colors[:]
                found = True
                break
                
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try flipping each edge
                # To make it faster, we can prioritize edges connected to the new vertex?
                # Or just random order.
                indices = list(range(len(cur_colors)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() > deadline:
                        break
                        
                    old_val = cur_colors[idx]
                    cur_colors[idx] = 1 - old_val
                    
                    new_violations = _count_violations(cur_colors, target_n)
                    
                    if new_violations < cur_violations:
                        cur_violations = new_violations
                        improved = True
                        if cur_violations == 0:
                            break
                    else:
                        cur_colors[idx] = old_val
                        
                if cur_violations == 0:
                    if _is_valid(cur_colors, target_n):
                        best_n = target_n
                        best_colors = cur_colors[:]
                        found = True
                        break
                        
            if found:
                break
                
        if found:
            colors = best_colors[:]
            n = best_n
        else:
            # If we couldn't extend to target_n, we probably can't extend to larger n either
            # without a very different structure, but let's try one more step if time permits?
            # Actually, if n=30 fails, n=31 is extremely unlikely.
            break
            
    return {'n': best_n, 'colors': best_colors}
