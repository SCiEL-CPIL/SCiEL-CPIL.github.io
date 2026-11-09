import random
import time
import numpy as np

def construct():
    """
    Uses Simulated Annealing with bitmask-based C4 counting to find a valid 
    5-coloring of K_n without monochromatic C4s.
    """
    
    def count_c4_fast(n, colors):
        """
        Counts the number of monochromatic C4s (or excess common neighbors).
        colors: list of length n*(n-1)//2, values 0-4.
        """
        # Build neighbor bitmasks for each color
        # neighbors[c][u] is a bitmask of vertices v such that (u,v) has color c
        neighbors = [[0] * n for _ in range(5)]
        
        # Map edge index to (u, v)
        # We iterate through all pairs
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = colors[idx]
                # Set bit v in neighbors[c][u] and bit u in neighbors[c][v]
                neighbors[c][u] |= (1 << v)
                neighbors[c][v] |= (1 << u)
                idx += 1
        
        total_c4s = 0
        for c in range(5):
            for u in range(n):
                mask_u = neighbors[c][u]
                for v in range(u + 1, n):
                    # Check if v is a neighbor of u in color c
                    # (It must be, since we only consider edges, but wait, 
                    # the graph is complete, so every pair has a color.
                    # So for a fixed color c, the graph is a subgraph.
                    # We need to check pairs (u,v) that are BOTH adjacent in color c?
                    # No, a C4 is u-w-v-x-u.
                    # The condition "no two vertices with two common neighbours in one colour"
                    # means: For any u, v, the number of w such that (u,w) and (v,w) are color c must be < 2.
                    # So we check ALL pairs (u,v), not just edges of color c.
                    mask_v = neighbors[c][v]
                    common = mask_u & mask_v
                    # Count bits
                    # If common has >= 2 bits, it contributes to C4 count
                    # We count the number of C4s as sum of (common_count - 1) for common_count >= 2
                    # Or just count pairs with >=2 common neighbors.
                    # For optimization, we just want to minimize the number of violations.
                    if common:
                        # Popcount
                        # Python int has no popcount, but we can use bin().count('1')
                        # Or use a lookup table for bytes if needed, but n=26 fits in int.
                        cnt = bin(common).count('1')
                        if cnt >= 2:
                            total_c4s += (cnt - 1) # Penalty for each excess common neighbor
        return total_c4s

    def local_search_sa(n, max_time=5.0):
        """
        Simulated Annealing to find a valid coloring.
        """
        start_time = time.time()
        num_edges = n * (n - 1) // 2
        
        # Initial state: Random
        random.seed(42)
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Better initial state: Affine plane slope coloring (if n=26)
        # This provides structure.
        if n == 26:
            # 0-24 are (i,j) in Z5xZ5, 25 is infinity
            # Slope coloring
            for u in range(n):
                for v in range(u + 1, n):
                    if u == 25:
                        # Edge to infinity
                        # In affine plane, lines through infinity are the parallel classes.
                        # Color based on the direction of the line in the plane part?
                        # Let's just use random for now to keep it simple, 
                        # or use u % 5?
                        colors[u * (n - 1) - u * (u + 1) // 2 - 1 + (v - u - 1)] = u % 5 
                        # This indexing is tricky, let's stick to random init for simplicity 
                        # unless we have a verified good init.
                    else:
                        if v == 25:
                            # Edge from plane point to infinity
                            # Color based on x-coordinate?
                            colors[u * (n - 1) - u * (u + 1) // 2 - 1 + (v - u - 1)] = (u // 5) % 5
                        else:
                            # Both in plane
                            i1, j1 = divmod(u, 5)
                            i2, j2 = divmod(v, 5)
                            if i1 == i2:
                                c = 4 # Vertical
                            else:
                                # Slope m = (j2-j1)/(i2-i1) mod 5
                                di = (i2 - i1) % 5
                                dj = (j2 - j1) % 5
                                # Inverse of di mod 5
                                inv = pow(di, -1, 5)
                                m = (dj * inv) % 5
                                c = m
                            # Map edge to index
                            # Index calculation:
                            # k = 0
                            # for i in range(n):
                            #   for j in range(i+1, n):
                            #     edge (i,j) is index k
                            #     k++
                            # We need a fast index lookup.
                            pass # Too complex to inline without a helper, let's just use random init 
                                # or a simpler structured init.

        # Let's use a simpler structured init: Random but with some bias? 
        # No, random is fine with SA.
        
        current_cost = count_c4_fast(n, colors)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 10.0
        cooling = 0.999
        
        while time.time() - start_time < max_time:
            if current_cost == 0:
                break
                
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            old_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[idx] = new_color
            new_cost = count_c4_fast(n, colors)
            delta = new_cost - current_cost
            
            if delta <= 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    colors[idx] = old_color
            
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_colors, best_cost == 0

    # Try n from 26 down to 11
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        # Allocate time based on n
        # n=26 is hard, give it more time? 
        # Total time budget ~100s.
        # Try 26 with 20s, 25 with 10s, etc.
        time_budget = 20.0 if n >= 25 else 5.0
        
        colors, is_valid = local_search_sa(n, max_time=time_budget)
        
        if is_valid:
            return {'n': n, 'colors': colors}
            
    # Fallback: Return a known valid small instance
    # n=11 cyclic construction
    n = 11
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            d = min(d, n - d)
            colors.append(d - 1)
    return {'n': n, 'colors': colors}
