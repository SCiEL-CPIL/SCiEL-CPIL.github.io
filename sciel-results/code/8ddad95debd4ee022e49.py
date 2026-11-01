import numpy as np
import time
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points for a cluster of side 3 (19 hexagons)
    # Axial coordinates (q, r)
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance from origin
            dist = max(abs(q), abs(r), abs(-q-r))
            if dist <= 2:
                points.append((q, r))
                
    # Convert to Cartesian
    # u = (sqrt(3), 0), v = (sqrt(3)/2, 3/2)
    def to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    cart_points = [to_cartesian(q, r) for q, r in points]
    
    # We need to choose 12 points from these 19.
    # To minimize the bounding hexagon side S (aligned), we minimize the max projection.
    # Projection onto normal k: p_k = x * cos(60k) + y * sin(60k)
    # S is proportional to max(p_k) + 1 (vertex extension)
    
    # Since C(19,12) is 50388, we can brute force it quickly.
    best_S = float('inf')
    best_subset = None
    
    normals = []
    for k in range(6):
        angle = np.pi / 3 * k
        normals.append((np.cos(angle), np.sin(angle)))
        
    def calc_S(subset):
        max_proj = 0
        for p in subset:
            for nx, ny in normals:
                proj = p[0]*nx + p[1]*ny
                if proj > max_proj:
                    max_proj = proj
        # Apothem A = max_proj + 1 (approx, assuming aligned)
        # Actually, the vertex extends by 1 in the direction of the normal?
        # For a hexagon aligned with the normal, the max projection of its vertices
        # from its center is 1.
        # So A >= max_proj + 1.
        A = max_proj + 1
        S = 2 * A / np.sqrt(3)
        return S

    # Optimization: Symmetry suggests we should pick a symmetric subset.
    # But brute force is fast enough for 50k combos.
    # To be safe against time limits, we limit iterations or use symmetry.
    # 50k * 12 * 6 ops is ~3.6M ops. Python might take 1-2s. Acceptable.
    
    # We only need to check subsets of size 12.
    # To speed up, we can prune.
    
    # Let's just iterate.
    count = 0
    for combo in itertools.combinations(range(len(cart_points)), 12):
        subset = [cart_points[i] for i in combo]
        S = calc_S(subset)
        if S < best_S:
            best_S = S
            best_subset = subset
        count += 1
        # Safety break if taking too long (though 50k is small)
        if count > 100000: 
            break

    # Format result
    # inner: list of [x, y, angle]
    # center: (0,0)
    # side: best_S
    # angle: 0.0
    
    inner = []
    for x, y in best_subset:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
