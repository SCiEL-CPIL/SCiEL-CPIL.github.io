# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon."""
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _min_enclosing_hex_side(inner_hexes):
    """
    Given 11 unit hexagons (each with center and angle=0), find the minimum
    side length of a regular hexagon (centered at origin, angle=0) that contains all.
    """
    # For a regular hexagon centered at origin with side length S and angle 0:
    # Vertices at S*(cos(60i), sin(60i)) for i=0..5
    # The constraint is that each vertex of each inner hex must be inside.
    # A point (px, py) is inside the outer hex if for each edge, the point is on the inner side.
    # Edge i goes from vertex i to vertex i+1. The inward normal for edge i points toward center.
    
    # For hex with vertices at angles 0,60,120,180,240,300:
    # Edge from v0 to v1: midpoint direction is 30 deg, inward normal is 30 deg
    # Actually, let's use the half-plane representation.
    # The hexagon is the intersection of 6 half-planes.
    # For a hex centered at origin with "radius" S (vertex distance), 
    # the apothem is S * cos(30 deg) = S * sqrt(3)/2
    # The half-plane for edge i (normal at angle 30+60i deg):
    #   x * cos(30+60i) + y * sin(30+60i) <= S * sqrt(3)/2
    
    # So for each inner vertex (px, py), we need:
    #   max over i of (px*cos(30+60i) + py*sin(30+60i)) <= S * sqrt(3)/2
    # => S >= max over all vertices and all i of (px*cos(30+60i) + py*sin(30+60i)) / (sqrt(3)/2)
    
    normals = []
    for i in range(6):
        a = np.radians(30 + 60 * i)
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)  # shape (6, 2)
    
    apothem_factor = np.sqrt(3) / 2  # S * this = apothem
    
    max_proj = 0
    for hex_data in inner_hexes:
        cx, cy, _ = hex_data
        verts = _hex_vertices(cx, cy, 1.0, 0)
        # For each vertex, compute max projection onto normals
        projs = verts @ normals.T  # shape (6, 6)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / apothem_factor

def _check_separation(inner_hexes, tol=1e-6):
    """Check that all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = inner_hexes[i], inner_hexes[j]
            vi = _hex_vertices(ci[0], ci[1], 1.0, ci[2])
            vj = _hex_vertices(cj[0], cj[1], 1.0, cj[2])
            
            # Check 12 edge normals (6 from each hex)
            separated = False
            all_normals = []
            for k in range(6):
                a = np.radians(ci[2]) + np.radians(30 + 60 * k)
                all_normals.append([np.cos(a), np.sin(a)])
            for k in range(6):
                a = np.radians(cj[2]) + np.radians(30 + 60 * k)
                all_normals.append([np.cos(a), np.sin(a)])
            
            for normal in all_normals:
                ni = vi @ normal
                nj = vj @ normal
                # Check if intervals [min(ni), max(ni)] and [min(nj), max(nj)] are apart
                if max(ni) < min(nj) - tol or max(nj) < min(ni) - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial arrangement: hexagonal cluster
    # Center hex + 6 around it + 4 more
    # For pointy-top hexagons (vertex at angle 0), neighbors are at:
    # distance sqrt(3) in directions 30, 90, 150, 210, 270, 330 degrees
    
    d = np.sqrt(3)  # center-to-center distance for touching hexagons
    
    # Hex grid positions for 11 hexagons (axial coordinates)
    # Using offset coordinates for pointy-top:
    # Row 0: 3 hexagons
    # Row 1: 5 hexagons  
    # Row 2: 3 hexagons
    # That's 11
    
    # Actually, let me use a symmetric arrangement:
    # Center + 6 surrounding + 4 more in a ring
    # Or: a 3-4-3 or similar pattern
    
    # Let me try: center at (0,0), 6 at distance d in 6 directions, 
    # then 4 more at distance 2*d in 4 of the 6 directions
    
    initial_positions = []
    # Center
    initial_positions.append([0.0, 0.0])
    # 6 surrounding
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        initial_positions.append([d * np.cos(angle), d * np.sin(angle)])
    # 4 more: at distance 2*d in directions 0, 60, 120, 180 (or symmetric)
    for k in range(0, 6, 2):  # every other direction
        angle = np.radians(30 + 60 * k)
        initial_positions.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    
    # That's 1 + 6 + 3 = 10. Need one more.
    # Add at 2*d in the remaining direction
    angle = np.radians(30 + 60 * 5)  # 330 deg... wait let me recount
    # Directions used for outer ring: 30, 90, 150, 210, 270, 330
    # I took every other: 30, 150, 270 (k=0,2,4)
    # Add at 90 or 210 or 330
    angle = np.radians(30 + 60 * 1)  # 90 deg
    initial_positions.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    
    initial_positions = np.array(initial_positions)
    
    def objective(params):
        """Minimize the enclosing hexagon side length."""
        positions = params.reshape(11, 2)
        inner = np.column_stack([positions, np.zeros(11)])
        return _min_enclosing_hex_side(inner)
    
    def constraint_sep(params):
        """All pairs must be separated."""
        positions = params.reshape(11, 2)
        inner = np.column_stack([positions, np.zeros(11)])
        return _check_separation(inner)
    
    # First, verify initial is valid
    initial_inner = np.column_stack([initial_positions, np.zeros(11)])
    initial_side = _min_enclosing_hex_side(initial_inner)
    initial_valid = _check_separation(initial_inner)
    
    best_inner = initial_inner.copy()
    best_side = initial_side
    best_valid = initial_valid
    
    if not initial_valid:
        # Try to fix by spreading out
        scale = 1.1
        while not _check_separation(initial_inner) and scale < 2.0:
            initial_positions = initial_positions * scale
            initial_inner = np.column_stack([initial_positions, np.zeros(11)])
            scale += 0.05
        best_inner = initial_inner.copy()
        best_side = _min_enclosing_hex_side(initial_inner)
        best_valid = _check_separation(initial_inner)
    
    # Try optimization
    if best_valid and time.time() < deadline - 60.0:
        try:
            # Use SLSQP with bounds
            bounds = [(-10, 10)] * 22
            
            res = minimize(
                objective,
                initial_positions.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success and time.time() < deadline - 30.0:
                opt_positions = res.x.reshape(11, 2)
                opt_inner = np.column_stack([opt_positions, np.zeros(11)])
                opt_side = _min_enclosing_hex_side(opt_inner)
                opt_valid = _check_separation(opt_inner)
                
                if opt_valid and opt_side < best_side:
                    best_inner = opt_inner
                    best_side = opt_side
                    best_valid = True
        except Exception:
            pass
    
    # Try multiple random restarts if time allows
    if time.time() < deadline - 120.0:
        np.random.seed(42)
        for restart in range(20):
            if time.time() > deadline - 60.0:
                break
            
            # Perturb the best solution
            perturbation = np.random.randn(11, 2) * 0.5
            trial_positions = best_inner[:, :2] + perturbation
            trial_inner = np.column_stack([trial_positions, np.zeros(11)])
            
            if not _check_separation(trial_inner):
                continue
            
            trial_side = _min_enclosing_hex_side(trial_inner)
            if trial_side < best_side:
                best_inner = trial_inner
                best_side = trial_side
                best_valid = True
    
    # Final result
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(best_side)
    
    return best_inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
