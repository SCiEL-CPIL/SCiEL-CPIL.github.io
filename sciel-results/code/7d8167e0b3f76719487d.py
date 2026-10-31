import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def barycentric_to_cartesian(bary):
    """Convert (N,3) barycentric to (N,2) cartesian."""
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])

def min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    cart = barycentric_to_cartesian(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def hard_obj(bary_flat):
    """Objective to minimize: negative of min area."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -min_area_from_bary(bary_norm)

def soft_obj(bary_flat, beta=100.0):
    """Smooth objective for global search."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    
    cart = barycentric_to_cartesian(bary_norm)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    best_val = -np.inf
    best_bary = None
    
    def update_best(bary_candidate, val_candidate):
        nonlocal best_val, best_bary
        if val_candidate > best_val:
            best_val = val_candidate
            best_bary = bary_candidate.copy()
            
    def random_bary():
        s = np.sqrt(rng.random(N))
        t = rng.random(N)
        a = 1 - s
        b = s * t
        c = s * (1 - t)
        bary = np.column_stack([a, b, c])
        return bary.flatten()

    def hex_init():
        """
        Hexagonal lattice initialization.
        The known optimal solution for Heilbronn triangles often has hexagonal symmetry.
        We generate a hexagonal packing and pick 11 points.
        """
        points = []
        # Center
        points.append([1/3, 1/3, 1/3])
        
        # We want to place points roughly in a hexagonal grid.
        # In barycentric coordinates, a hexagonal grid is tricky.
        # Let's try to place points at specific barycentric positions that form a hexagon.
        # Or simply use a grid in Cartesian and convert, ensuring they are inside.
        
        # Actually, let's try a specific structured guess based on known literature or symmetry.
        # For N=11, a common configuration is the "hexagonal" one.
        # Let's try to construct a hexagon around the center.
        
        # Center
        pts = [[1/3, 1/3, 1/3]]
        
        # 6 points forming a hexagon around center
        # Distance from center in barycentric space?
        # Let's try small perturbations.
        # Or use Cartesian hexagon.
        
        # Let's just use a dense random search for init and rely on DE + Polish.
        # But a good init helps.
        
        # Try a 3x3 grid in Cartesian, mapped to barycentric, then clipped.
        # This is robust.
        cart_pts = []
        # 3x3 grid
        for i in range(3):
            for j in range(3):
                x = 0.25 + 0.25 * i
                y = 0.25 + 0.25 * j
                # Check if inside triangle: y <= H * (1 - x) is not quite right for this triangle
                # Triangle: (0,0), (1,0), (0.5, H)
                # Left edge: y = H * x
                # Right edge: y = H * (1 - x)
                # Bottom edge: y = 0
                if y <= H * x and y <= H * (1 - x) and y >= 0:
                    cart_pts.append([x, y])
        
        # If we have fewer than 11, add random ones
        while len(cart_pts) < N:
            x = rng.random()
            y = rng.random() * H * x * (1 - x) # Approx
            # Better random sampling in triangle
            s = np.sqrt(rng.random())
            t = rng.random()
            x = s * t
            y = H * s * (1 - t)
            cart_pts.append([x, y])
        
        cart_pts = np.array(cart_pts[:N])
        
        # Convert to barycentric
        barys = []
        for p in cart_pts:
            x, y = p
            c = y / H
            b = x - 0.5 * c
            a = 1.0 - b - c
            barys.append([a, b, c])
        
        bary = np.array(barys)
        # Normalize just in case
        bary = np.clip(bary, 0, None)
        row_sums = bary.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        bary = bary / row_sums
        
        return bary.flatten()

    # 1. Initial Guesses
    inits = []
    inits.append(hex_init())
    for _ in range(10):
        inits.append(random_bary())
    
    for bary0 in inits:
        val = -hard_obj(bary0)
        update_best(bary0, val)
    
    # 2. Differential Evolution (Global Search)
    # Use soft objective for smoother landscape
    if time.time() < deadline - 40:
        try:
            bounds = [(0, 1)] * (N * 3)
            de_result = differential_evolution(
                lambda x: soft_obj(x, beta=50.0),
                bounds=bounds,
                seed=42,
                maxiter=40,
                tol=1e-4,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            val_hard = -hard_obj(de_result.x)
            update_best(de_result.x, val_hard)
        except Exception:
            pass

    # 3. Local Polish Loop
    # Alternate between SLSQP (soft) and Nelder-Mead (hard)
    for i in range(15):
        if time.time() > deadline - 10:
            break
        
        # SLSQP with Soft Min
        if time.time() < deadline - 15:
            try:
                result = minimize(
                    lambda x: soft_obj(x, beta=200.0),
                    best_bary,
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 3),
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                val_hard = -hard_obj(result.x)
                update_best(result.x, val_hard)
            except Exception:
                pass
        
        # Nelder-Mead with Hard Min
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    hard_obj,
                    best_bary,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
                )
                val_hard = -hard_obj(result.x)
                update_best(result.x, val_hard)
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                hard_obj,
                best_bary,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            val_hard = -hard_obj(result.x)
            update_best(result.x, val_hard)
        except Exception:
            pass

    # Convert to cartesian
    bary_final = best_bary.reshape(N, 3)
    bary_final = np.clip(bary_final, 0, None)
    row_sums = bary_final.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_final = bary_final / row_sums
    cart = barycentric_to_cartesian(bary_final)
    
    return cart
