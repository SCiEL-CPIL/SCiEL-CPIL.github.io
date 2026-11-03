"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

def hexagon_containment(inner_points, inner_angles, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagons (side 1) are contained within the outer hexagon.
    inner_points: list of (x, y)
    inner_angles: list of rotation angles in radians
    outer_center: (cx, cy)
    outer_side: side length of outer hexagon
    outer_angle: rotation of outer hexagon in radians
    Returns: True if all contained, False otherwise.
    """
    cx, cy = outer_center
    s = outer_side
    theta = outer_angle
    
    # The 6 edge normals of the outer hexagon
    # For a regular hexagon with circumradius s, the inradius is s * sqrt(3)/2
    # Edge normals are at angles theta + k * pi/3 for k=0..5
    inradius = s * np.sqrt(3) / 2.0
    
    # For each inner hexagon, check if all its vertices are inside the outer hexagon
    for i, (px, py) in enumerate(inner_points):
        rot = inner_angles[i]
        # Vertices of unit hexagon centered at (px, py) with rotation rot
        # Circumradius of unit hexagon is 1
        for k in range(6):
            vx = px + np.cos(rot + k * np.pi / 3)
            vy = py + np.sin(rot + k * np.pi / 3)
            
            # Check if (vx, vy) is inside outer hexagon
            dx = vx - cx
            dy = vy - cy
            
            # For each of the 6 edges, check the half-plane constraint
            contained = True
            for m in range(6):
                normal_angle = theta + m * np.pi / 3
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                # The edge is at distance inradius from center along normal
                # Point is inside if dot((p - c), n) <= inradius
                dist = dx * nx + dy * ny
                if dist > inradius + 1e-9:
                    contained = False
                    break
            if not contained:
                return False
    return True

def objective(params, N):
    """
    Objective: minimize outer side length subject to containment.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, s, theta]
    Total: 3*N + 4
    """
    if N <= 0:
        return 1e6
    
    inner_x = params[0:N]
    inner_y = params[N:2*N]
    inner_a = params[2*N:3*N]
    cx = params[3*N]
    cy = params[3*N + 1]
    s = params[3*N + 2]
    theta = params[3*N + 3]
    
    if s < 1.0:
        return 1e6
    
    inner_points = list(zip(inner_x, inner_y))
    inner_angles = inner_a
    
    if hexagon_containment(inner_points, inner_angles, (cx, cy), s, theta):
        return s
    else:
        # Penalty for violation
        # Compute max violation
        max_viol = 0.0
        inradius = s * np.sqrt(3) / 2.0
        for i in range(N):
            px, py = inner_points[i]
            rot = inner_angles[i]
            for k in range(6):
                vx = px + np.cos(rot + k * np.pi / 3)
                vy = py + np.sin(rot + k * np.pi / 3)
                dx = vx - cx
                dy = vy - cy
                for m in range(6):
                    normal_angle = theta + m * np.pi / 3
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    dist = dx * nx + dy * ny
                    viol = dist - inradius
                    if viol > max_viol:
                        max_viol = viol
        return s + 10.0 * max(0, max_viol)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial guess: place hexagons in a hexagonal lattice pattern
    # For N hexagons, arrange in a compact hexagonal shape
    # Hexagonal lattice spacing: distance between centers of adjacent hexagons is sqrt(3)
    
    # Generate hexagonal lattice positions
    def hex_lattice_positions(n):
        """Generate positions for n hexagons in a hexagonal lattice."""
        positions = [(0.0, 0.0)]
        if n <= 1:
            return positions
        
        # Add rings around the center
        ring = 1
        while len(positions) < n:
            # Ring r has 6*r hexagons
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                # Position on ring
                angle = i * np.pi / (3 * ring)
                # Distance from center to ring r is r * sqrt(3)
                dist = ring * np.sqrt(3)
                x = dist * np.cos(angle)
                y = dist * np.sin(angle)
                positions.append((x, y))
            ring += 1
        
        return positions[:n]
    
    init_positions = hex_lattice_positions(N)
    
    # Initial parameters
    init_params = []
    for x, y in init_positions:
        init_params.extend([x, y, 0.0])
    # Outer hexagon: centered at origin, large enough to contain
    init_params.extend([0.0, 0.0, 10.0, 0.0])
    
    best_params = init_params.copy()
    best_obj = objective(best_params, N)
    
    # Try multiple local optimizations with different initializations
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            params = init_params.copy()
        else:
            # Perturb the initial solution
            params = [p + np.random.normal(0, 0.5) for p in init_params]
        
        # Use Nelder-Mead for local optimization
        result = minimize(
            lambda p: objective(p, N),
            params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
        
        if time.time() > deadline:
            break
    
    # Extract best solution
    inner_x = best_params[0:N]
    inner_y = best_params[N:2*N]
    inner_a_rad = best_params[2*N:3*N]
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    s = best_params[3*N + 2]
    theta_rad = best_params[3*N + 3]
    
    # Convert angles to degrees
    inner = [[float(x), float(y), float(np.degrees(a))] for x, y, a in zip(inner_x, inner_y, inner_a_rad)]
    
    return inner, (float(cx), float(cy)), float(s), float(np.degrees(theta_rad))

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_lattice_positions(n):
        positions = [(0.0, 0.0)]
        if n <= 1:
            return positions
        ring = 1
        while len(positions) < n:
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                angle = i * np.pi / (3 * ring)
                dist = ring * np.sqrt(3)
                x = dist * np.cos(angle)
                y = dist * np.sin(angle)
                positions.append((x, y))
            ring += 1
        return positions[:n]
    
    init_positions = hex_lattice_positions(N)
    init_params = []
    for x, y in init_positions:
        init_params.extend([x, y, 0.0])
    init_params.extend([0.0, 0.0, 10.0, 0.0])
    
    best_params = init_params.copy()
    best_obj = objective(best_params, N)
    
    n_attempts = 8
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            params = init_params.copy()
        else:
            params = [p + np.random.normal(0, 0.3) for p in init_params]
        
        result = minimize(
            lambda p: objective(p, N),
            params,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
        
        if time.time() > deadline:
            break
    
    inner_x = best_params[0:N]
    inner_y = best_params[N:2*N]
    inner_a_rad = best_params[2*N:3*N]
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    s = best_params[3*N + 2]
    theta_rad = best_params[3*N + 3]
    
    inner = [[float(x), float(y), float(np.degrees(a))] for x, y, a in zip(inner_x, inner_y, inner_a_rad)]
    
    return inner, (float(cx), float(cy)), float(s), float(np.degrees(theta_rad))
# EVOLVE-BLOCK-END
