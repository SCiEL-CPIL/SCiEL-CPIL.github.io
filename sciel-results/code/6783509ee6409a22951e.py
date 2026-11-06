import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
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
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _local_search(n, time_budget):
    """
    Attempt to find a valid 4-coloring of K_n with no monochromatic triangles.
    Returns the list of colors if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Build adjacency bitsets
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(4)]
        
        for idx in range(m):
            c = colors[idx]
            # Recover i, j from idx? 
            # It's faster to build adj during init and update during search.
            pass # We need a mapping from idx to (i,j) or iterate i,j to build adj.
        
        # Re-init adj properly
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Local search loop
        # We track the number of monochromatic triangles? 
        # Or just find one and fix it.
        # Finding one is O(E * bit_ops).
        
        while True:
            if time.time() - start_time > time_budget:
                return None
            
            # Find a monochromatic triangle
            tri_found = False
            for c in range(4):
                for i in range(n):
                    # Neighbors of i in color c, greater than i
                    # We only care about triangles (i, j, k) with i < j < k
                    # So for edge (i, j) with j > i, we check common neighbors k > j
                    neigh = adj[c][i] & ~((1 << (i + 1)) - 1) # neighbors > i
                    # Iterate over set bits in neigh
                    # To be efficient, we can iterate j > i
                    j_mask = neigh
                    while j_mask:
                        j_bit = j_mask & (-j_mask)
                        j = j_bit.bit_length() - 1
                        j_mask ^= j_bit
                        
                        # Check if (i, j) is color c. It is, because j is in adj[c][i].
                        # Check common neighbors k > j in color c
                        common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                        if common:
                            k = (common & (-common)).bit_length() - 1
                            # Found triangle (i, j, k) with color c
                            tri_found = True
                            
                            # Choose an edge to recolor. Let's pick (i, j).
                            # Determine best new color c'
                            best_c = -1
                            min_new_tris = float('inf')
                            
                            for new_c in range(4):
                                if new_c == c:
                                    continue
                                
                                # Count new triangles formed by (i, j) with color new_c
                                # These are vertices k such that (i, k) and (j, k) are color new_c
                                # Note: changing (i, j) to new_c might break existing triangles of color new_c?
                                # No, (i, j) was color c. So it wasn't part of any new_c triangle.
                                # So we only add triangles.
                                new_tris = (adj[new_c][i] & adj[new_c][j]).bit_count()
                                
                                # Optional: prefer colors that don't create triangles
                                if new_tris < min_new_tris:
                                    min_new_tris = new_tris
                                    best_c = new_c
                                    
                            if best_c == -1:
                                best_c = (c + 1) % 4 # Fallback, shouldn't happen as 3 options available
                            
                            # Recolor (i, j) from c to best_c
                            idx = _compute_index(n, i, j)
                            colors[idx] = best_c
                            
                            # Update adj
                            adj[c][i] &= ~(1 << j)
                            adj[c][j] &= ~(1 << i)
                            adj[best_c][i] |= (1 << j)
                            adj[best_c][j] |= (1 << i)
                            
                            break # Found a triangle and fixed it, restart search for next triangle
                        # End if common
                    # End while j_mask
                # End for i
                if tri_found:
                    break # Restart outer color loop? No, we fixed one triangle, we should restart the whole triangle search to see if new ones appeared or if it's clean.
            # End for c
            
            if not tri_found:
                return colors # Valid coloring found!

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        # Try to find valid colorings for n > 32
        # We try from 50 down to 33? Or 33 up to 50?
        # If we find a valid coloring for 50, that's best.
        # If not, maybe 49?
        # The search is hard. Let's try 50 first with a decent budget.
        # If it fails, try 43 (known bound) with a smaller budget?
        # Actually, if 50 is hard, 43 might be easier.
        
        # Strategy:
        # 1. Try n=50 with 60s budget.
        # 2. If fail, try n=43 with 30s budget.
        # 3. If fail, return n=32.
        
        # Note: The local search is stochastic. It might fail even if a solution exists.
        # But for n=32, it's guaranteed.
        
        targets = [50, 43, 40] # Try high values first
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            # Allocate time based on target
            # If we've used a lot of time, reduce budget
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 5:
                break
            
            # Give 50s for 50, 20s for 43
            budget = min(remaining - 5, 50 if target_n == 50 else 20)
            if budget < 2:
                break
                
            result = _local_search(target_n, budget)
            if result is not None:
                return {'n': target_n, 'colors': result}
                
        return best
    except Exception as e:
        # Fallback
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}
