"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = set(x * x % p for x in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            c = 0 if (j - i) % p in squares else 1
            colors.append(c)
            adj[i][j] = c
            adj[j][i] = c
    return p, colors, adj

def _construct_30_vertex():
    """
    Try to find a 30-vertex coloring by extending Paley 29 with Simulated Annealing.
    """
    # 1. Get Paley 29
    p = 29
    squares = set(x * x % p for x in range(1, p))
    adj29 = np.zeros((29, 29), dtype=np.int8)
    for i in range(29):
        for j in range(i + 1, 29):
            c = 0 if (j - i) % p in squares else 1
            adj29[i][j] = c
            adj29[j][i] = c
    
    # Precompute neighbor lists for color 0 and 1
    # neigh[c][v] is a list of neighbors of v in color c
    neigh = [[[] for _ in range(29)] for _ in range(2)]
    for i in range(29):
        for j in range(29):
            if i != j:
                neigh[adj29[i][j]][i].append(j)
                
    # For fast K2,2 check in a subset of vertices, we can use bitmasks
    # neigh_mask[c][v] is a bitmask of neighbors of v in color c
    neigh_mask = [[0]*29 for _ in range(2)]
    for i in range(29):
        for j in range(29):
            if i != j:
                neigh_mask[adj29[i][j]][i] |= (1 << j)

    def _count_k222(mask1):
        """
        Count number of monochromatic K2,2,2 involving vertex 29.
        mask1: 29-bit integer. Bit j is 1 if edge (29, j) is color 1.
               Bit j is 0 if edge (29, j) is color 0.
        """
        count = 0
        # For each color c in {0, 1}
        for c in [0, 1]:
            # S_c is the set of vertices v such that (29, v) has color c
            if c == 1:
                S_c = mask1
            else:
                S_c = ((1 << 29) - 1) ^ mask1
            
            # For each potential 'a' vertex (the one in the same part as 29 in K2,2,2)
            for a in range(29):
                # Candidates for b1, b2, c1, c2 must be in S_c AND neighbors of a in color c
                # Let Cand = S_c & neigh_mask[c][a]
                Cand = S_c & neigh_mask[c][a]
                
                # We need to check if the subgraph induced by Cand in color c contains a K2,2
                # K2,2 means there exist b1, b2, c1, c2 in Cand such that
                # all edges between {b1,b2} and {c1,c2} are present in color c.
                # This is equivalent to: 
                # There exist b1, b2 in Cand such that (neigh_mask[c][b1] & neigh_mask[c][b2] & Cand) has at least 2 bits.
                # Those 2 bits are c1, c2.
                
                # Iterate over pairs b1, b2 in Cand
                # To be efficient, we can iterate over b1 in Cand, then b2 in Cand, b2 > b1
                # But Cand is a bitmask.
                # Let's extract list of vertices in Cand
                cand_list = [v for v in range(29) if (Cand >> v) & 1]
                
                # If len(cand_list) < 4, no K2,2
                if len(cand_list) < 4:
                    continue
                
                # Check for K2,2
                # For each pair b1, b2 in cand_list:
                #   common = neigh_mask[c][b1] & neigh_mask[c][b2] & Cand
                #   if bin(common).count('1') >= 2:
                #       count += 1 (or count the number of ways? We just need to know if > 0 for validity, 
                #       but for SA cost, counting is better for gradient)
                #       Actually, for SA, we want a smooth cost. Counting the number of K2,2s is good.
                
                for i in range(len(cand_list)):
                    b1 = cand_list[i]
                    for j in range(i + 1, len(cand_list)):
                        b2 = cand_list[j]
                        common = neigh_mask[c][b1] & neigh_mask[c][b2] & Cand
                        # Number of pairs c1, c2 in common
                        num_c = bin(common).count('1')
                        if num_c >= 2:
                            # Number of ways to choose 2 from num_c
                            count += num_c * (num_c - 1) // 2
                            
        return count

    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize mask randomly
    mask = 0
    for i in range(29):
        if random.randint(0, 1) == 1:
            mask |= (1 << i)
            
    current_cost = _count_k222(mask)
    best_mask = mask
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if best_cost == 0:
            break
            
        # Flip a random bit
        bit = random.randint(0, 28)
        new_mask = mask ^ (1 << bit)
        new_cost = _count_k222(new_mask)
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0.1 and random.random() < np.exp(-delta / T)):
            mask = new_mask
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = mask
                
        T *= cooling
        
    if best_cost == 0:
        # Construct the 30-vertex coloring
        colors = []
        # First, edges within 0..28
        for i in range(29):
            for j in range(i + 1, 29):
                colors.append(adj29[i][j])
        # Edges involving 29
        for j in range(29):
            if (best_mask >> j) & 1:
                colors.append(1)
            else:
                colors.append(0)
                
        return 30, colors

    return None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    # Try to find 30-vertex coloring
    result = _construct_30_vertex()
    if result is not None:
        n, colors = result
        return {'n': n, 'colors': colors}
        
    # Fallback to Paley 29
    n, colors, adj = _paley_coloring(29)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
