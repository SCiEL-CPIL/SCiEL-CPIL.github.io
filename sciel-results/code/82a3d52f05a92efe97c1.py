import numpy as np
import random

def construct():
    def count_violations(colors, n, adj_masks):
        """
        Count the number of pairs (u, v) that are 'bad'.
        A pair (u, v) is bad for color c if edge (u,v) is color c and 
        the common neighborhood N_c(u) & N_c(v) contains a K_2,2 in color c.
        """
        violations = 0
        for u in range(n):
            for v in range(u + 1, n):
                # Check color 0
                if (adj_masks[0][u] >> v) & 1:
                    common = adj_masks[0][u] & adj_masks[0][v]
                    # Check if common contains K_2,2
                    # Get list of vertices in common
                    # For efficiency, we can just iterate bits
                    # But we need to check pairs a,b in common
                    # Let's extract indices
                    # This might be slow if done naively.
                    # Alternative: 
                    # A K_2,2 exists in common if there are a,b in common 
                    # such that (common & adj[a] & adj[b]) has >= 2 bits.
                    
                    # Iterate a in common
                    # To iterate bits in common:
                    temp_common = common
                    while temp_common:
                        a_bit = temp_common & -temp_common
                        a = a_bit.bit_length() - 1
                        temp_common ^= a_bit
                        
                        # Iterate b in common, b > a
                        temp_common_b = common & ~( (1 << (a+1)) - 1 ) # bits > a
                        # Actually just iterate all b in common, b != a
                        # Since we need distinct a,b, and symmetric, we can just pick any pair.
                        # Let's just iterate all b in common
                        temp_b = common
                        while temp_b:
                            b_bit = temp_b & -temp_b
                            b = b_bit.bit_length() - 1
                            temp_b ^= b_bit
                            if a == b:
                                continue
                            # Check if there are at least 2 common neighbors of a and b within common
                            # i.e., popcount(common & adj[0][a] & adj[0][b]) >= 2
                            shared = common & adj_masks[0][a] & adj_masks[0][b]
                            if shared & (shared - 1): # has at least 2 bits
                                violations += 1
                                break # Found one K_2,2 for this pair (u,v), no need to check more for this (u,v)
                        else:
                            continue
                        break
                    if violations > 0 and (violations % 2 == 1): # dummy check to break early? No, just count
                         pass
                # Check color 1
                if (adj_masks[1][u] >> v) & 1:
                    common = adj_masks[1][u] & adj_masks[1][v]
                    temp_common = common
                    found_k22 = False
                    while temp_common and not found_k22:
                        a_bit = temp_common & -temp_common
                        a = a_bit.bit_length() - 1
                        temp_common ^= a_bit
                        
                        temp_b = common
                        while temp_b and not found_k22:
                            b_bit = temp_b & -temp_b
                            b = b_bit.bit_length() - 1
                            temp_b ^= b_bit
                            if a == b:
                                continue
                            shared = common & adj_masks[1][a] & adj_masks[1][b]
                            if shared & (shared - 1):
                                violations += 1
                                found_k22 = True
        return violations

    def build_masks(colors, n):
        masks = [[0]*n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
                idx += 1
        return masks

    def local_search(n, max_iters=5000):
        # Initialize random
        colors = [random.choice([0, 1]) for _ in range(n*(n-1)//2)]
        masks = build_masks(colors, n)
        
        best_violations = count_violations(colors, n, masks)
        
        for it in range(max_iters):
            if best_violations == 0:
                return colors
            
            # Pick a random edge to flip
            # Better: Pick an edge involved in a violation?
            # For simplicity, random edge flip with acceptance if it improves or with prob p.
            # Let's try random edge flip.
            e_idx = random.randint(0, n*(n-1)//2 - 1)
            
            # Determine u, v from e_idx
            # Inverse of lexicographic order
            # idx = i*n - i*(i+1)/2 + (j-i-1) ?
            # Let's just precompute edge indices to (u,v)
            # For n=30, 435 edges.
            
            # Flip
            old_c = colors[e_idx]
            new_c = 1 - old_c
            colors[e_idx] = new_c
            
            # Update masks
            # Find u, v
            # We can store a list of (u,v) for each index
            # Or compute
            # i = 0
            # count = 0
            # for i in range(n):
            #   if count + (n-i-1) > e_idx:
            #      j = i + 1 + (e_idx - count)
            #      break
            #   count += n - i - 1
            
            # This is O(n). 30 iterations.
            # Let's precompute edge_map
            # But we are in a function, so we can compute once outside or inside.
            # Let's compute inside for clarity, it's fast.
            
            i = 0
            count = 0
            found_i = False
            for i in range(n):
                if count + (n - i - 1) > e_idx:
                    j = i + 1 + (e_idx - count)
                    found_i = True
                    break
                count += n - i - 1
            
            if not found_i:
                # Should not happen
                continue

            # Update masks
            masks[old_c][i] &= ~(1 << j)
            masks[old_c][j] &= ~(1 << i)
            masks[new_c][i] |= (1 << j)
            masks[new_c][j] |= (1 << i)
            
            new_violations = count_violations(colors, n, masks)
            
            if new_violations < best_violations:
                best_violations = new_violations
            else:
                # Revert
                colors[e_idx] = old_c
                masks[new_c][i] &= ~(1 << j)
                masks[new_c][j] &= ~(1 << i)
                masks[old_c][i] |= (1 << j)
                masks[old_c][j] |= (1 << i)
                # Optionally, accept with some probability (Simulated Annealing)
                # For now, greedy only.
        
        return colors

    # Try n=30
    random.seed(42)
    for restart in range(50):
        sol = local_search(30, max_iters=2000)
        # Verify
        masks = build_masks(sol, 30)
        if count_violations(sol, 30, masks) == 0:
            return {'n': 30, 'colors': sol}
            
    # Try n=29
    for restart in range(50):
        sol = local_search(29, max_iters=2000)
        masks = build_masks(sol, 29)
        if count_violations(sol, 29, masks) == 0:
            return {'n': 29, 'colors': sol}
            
    # Fallback to Paley 29
    p = 29
    squares = {x*x % p for x in range(1, p)}
    squares.discard(0)
    colors = [0 if (j-i) % p in squares else 1 for i in range(p) for j in range(i+1, p)]
    return {'n': p, 'colors': colors}
