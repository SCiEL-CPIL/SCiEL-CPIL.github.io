import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    # 3-5-3 honeycomb arrangement of 11 unit hexagons
    centers = []
    # Middle row (5 hexagons) at y=0
    for k in range(-2, 3):
        centers.append((k * sqrt3, 0.0))
    # Top row (3 hexagons) at y=1.5
    for k in range(-1, 2):
        centers.append((k * sqrt3 + sqrt3 / 2, 1.5))
    # Bottom row (3 hexagons) at y=-1.5
    for k in range(-1, 2):
        centers.append((k * sqrt3 + sqrt3 / 2, -1.5))
    
    centers = np.array(centers)
    
    # Unit hexagon vertices (circumradius 1, vertices at 0, 60, 120, 180, 240, 300 degrees)
    hex_angles = np.arange(6) * np.pi / 3
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def h_unit(theta):
        """Support function of unit hexagon in direction theta."""
        d = np.array([np.cos(theta), np.sin(theta)])
        return float(np.max(hex_verts @ d))
    
    def side_length(alpha):
        """Compute required outer hexagon side length for rotation alpha.
        alpha is the angle of the first side normal of the enclosing hexagon."""
        apothem_factor = np.sqrt(3) / 2
        max_val = 0.0
        for k in range(6):
            n_theta = alpha + k * np.pi / 3
            n_hat = np.array([np.cos(n_theta), np.sin(n_theta)])
            h_centers = float(np.max(centers @ n_hat))
            h_hex = h_unit(n_theta)
            total = h_centers + h_hex
            if total > max_val:
                max_val = total
        return max_val / apothem_factor
    
    # Minimize over alpha in [0, pi/3) using grid + ternary search
    alphas = np.linspace(0, np.pi / 3, 3000)
    sides = np.array([side_length(a) for a in alphas])
    best_idx = int(np.argmin(sides))
    
    # Refine with ternary search
    lo = alphas[max(0, best_idx - 10)]
    hi = alphas[min(len(alphas) - 1, best_idx + 10)]
    for _ in range(300):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if side_length(m1) < side_length(m2):
            hi = m2
        else:
            lo = m1
    best_alpha = (lo + hi) / 2
    best_side = side_length(best_alpha)
    
    inner = []
    for (cx, cy) in centers:
        inner.append([float(cx), float(cy), 0.0])
    
    return inner, (0.0, 0.0), float(best_side), float(np.degrees(best_alpha))
# EVOLVE-BLOCK-END
