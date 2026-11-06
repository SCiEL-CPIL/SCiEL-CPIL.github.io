import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    This is a proven valid configuration for R(C5, K9) > 32.
    
    Design Family: Disjoint union of cliques.
    
    Optimality within Family:
    - Components must be K_k with k <= 4 to avoid C5 (K5 contains C5).
    - To maximize n for a fixed independence number alpha < 9, we should use 
      the largest possible cliques (K4) because they contribute more vertices 
      per unit of independence number (4 vertices for alpha=1) compared to 
      smaller cliques (e.g., K3 contributes 3 vertices for alpha=1).
    - Let m be the number of K4 components. alpha = m.
    - Constraint: m < 9 => m <= 8.
    - Max n = 8 * 4 = 32.
    - Any mix of K3 and K4 with total alpha <= 8 yields n <= 32.
      (e.g., 7 K4s and 1 K3: n=31, alpha=8).
    
    Therefore, n=32 is the maximum for this design family.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 32
    edges = []
    
    # 8 disjoint blocks of size 4
    for b in range(8):
        if time.time() > deadline:
            break
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        
    return {'n': n, 'edges': edges}
    # EVOLVE-BLOCK-END
