import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 130.0  # Safety margin before 150s kill

def _hex_verts(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy), rotated by angle_deg degrees."""
    a = math.radians(angle_deg)
    cas, sas = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3
        x = cx + math.cos(theta)
        y = cy + math.sin(theta)
        verts.append((x, y))
    return verts

def _container_verts(cxo, cyo, S, angle_deg):
    """Return 6 vertices of outer hexagon with side S centered at (cxo,cyo), rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3
        x = cxo + S * math.sqrt(3) / 2 * math.cos(theta)
        y = cyo + S * math.sqrt(3) / 2 * math.sin(theta)
        verts.append((x, y))
    return verts

def _inside_hex(px, py, cxo, cyo, S, angle_deg):
    """Check if point (px,py) is inside the container hexagon."""
    dx = px - cxo
    dy = py - cyo
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    rx = dx * ca - dy * sa
    ry = dx * sa + dy * ca
    # In rotated frame, hexagon with side S: |rx| <= S*sqrt(3)/2, |ry| <= S*sqrt(3)/2, |rx+ry| <= S*sqrt(3)/2
    # Actually for a regular hexagon with flat sides, the condition is:
    # max over 6 directions of dot product <= S*sqrt(3)/2
    # Normal directions at angles 0, 60, 120, 180, 240, 300 degrees in the rotated frame
    for k in range(6):
        theta = k * math.pi / 3
        proj = rx * math.cos(theta) + ry * math.sin(theta)
        if proj > S * math.sqrt(3) / 2 + 1e-9:
            return False
    return True

def _hex_in_container(cx, cy, angle_deg, cxo, cyo, S, adeg):
    """Check if unit hexagon centered at (cx,cy) rotated by angle_deg is inside container."""
    verts = _hex_verts(cx, cy, angle_deg)
    for vx, vy in verts:
        if not _inside_hex(vx, vy, cxo, cyo, S, adeg):
            return False
    return True

def _hex_overlap(c1, a1, c2, a2):
    """Check if two unit hexagons overlap (centers distance check is approximate; use vertex-in-hex test)."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist >= 2.0:
        return False
    # More precise: check if any vertex of hex1 is inside hex2 or vice versa
    v1 = _hex_verts(*c1, a1)
    v2 = _hex_verts(*c2, a2)
    for vx, vy in v1:
        # Check if (vx,vy) is inside hex2
        dx2 = vx - c2[0]
        dy2 = vy - c2[1]
        a = math.radians(a2)
        ca, sa = math.cos(-a), math.sin(-a)
        rx = dx2 * ca - dy2 * sa
        ry = dx2 * sa + dy2 * ca
        inside = True
        for k in range(6):
            theta = k * math.pi / 3
            proj = rx * math.cos(theta) + ry * math.sin(theta)
            if proj > math.sqrt(3)/2 + 1e-9:
                inside = False
                break
        if inside:
            return True
    for vx, vy in v2:
        dx1 = vx - c1[0]
        dy1 = vy - c1[1]
        a = math.radians(a1)
        ca, sa = math.cos(-a), math.sin(-a)
        rx = dx1 * ca - dy1 * sa
        ry = dx1 * sa + dy1 * ca
        inside = True
        for k in range(6):
            theta = k * math.pi / 3
            proj = rx * math.cos(theta) + ry * math.sin(theta)
            if proj > math.sqrt(3)/2 + 1e-9:
                inside = False
                break
        if inside:
            return True
    return False

def construct():
    start_time = time.time()
    
    # Known good configuration: 1-6-5 star pattern
    # Center hex at origin, 6 around it, 5 more in outer ring
    # All hexagons rotated 30 degrees, container rotated 30 degrees
    # Side length 4.0
    
    # Let's try to find a better configuration via optimization
    
    # Initial guess: hexagonal lattice pattern
    # Place 12 hexagons in a compact arrangement
    # For a hexagonal packing, we can use a pattern like:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # Or a more compact 3-4-5 or similar
    
    # Let's use the 1-6-5 star as starting point
    def initial_config():
        """Generate initial 1-6-5 star configuration."""
        hexes = []
        # Center
        hexes.append((0.0, 0.0, 30.0))
        # 6 around center at distance sqrt(3)
        for k in range(6):
            theta = k * math.pi / 3
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            hexes.append((x, y, 30.0))
        # 5 more at distance 3.0, angles 30, 90, 150, 210, 270
        for k in range(5):
            theta = math.radians(30 + 60 * k)
            x = 3.0 * math.cos(theta)
            y = 3.0 * math.sin(theta)
            hexes.append((x, y, 30.0))
        return hexes
    
    def evaluate_config(hexes, container_center=(0.0, 0.0), container_side=4.0, container_angle=30.0):
        """Evaluate if configuration is valid and return side length or None."""
        # Check overlaps
        for i in range(len(hexes)):
            for j in range(i+1, len(hexes)):
                if _hex_overlap((hexes[i][0], hexes[i][1]), hexes[i][2],
                               (hexes[j][0], hexes[j][1]), hexes[j][2]):
                    return None
        # Check containment
        for hx, hy, ha in hexes:
            if not _hex_in_container(hx, hy, ha, container_center[0], container_center[1], 
                                     container_side, container_angle):
                return None
        return container_side
    
    # Start with known good
    hexes = initial_config()
    best_side = 4.0
    best_config = hexes
    
    # Try to improve by adjusting positions slightly
    # Use a simple local search
    
    # Parameters: x, y, angle for each hexagon
    # We'll keep the container fixed at center (0,0), angle 30, and optimize side length
    
    def make_params(hexes):
        """Flatten hexes to parameter vector."""
        params = []
        for hx, hy, ha in hexes:
            params.extend([hx, hy, ha])
        return np.array(params)
    
    def from_params(params):
        """Reconstruct hexes from parameter vector."""
        hexes = []
        for i in range(N):
            hx = params[3*i]
            hy = params[3*i+1]
            ha = params[3*i+2]
            hexes.append((hx, hy, ha))
        return hexes
    
    def objective(params):
        """Minimize the required container side length."""
        hexes = from_params(params)
        
        # Check overlaps
        for i in range(len(hexes)):
            for j in range(i+1, len(hexes)):
                if _hex_overlap((hexes[i][0], hexes[i][1]), hexes[i][2],
                               (hexes[j][0], hexes[j][1]), hexes[j][2]):
                    return 100.0  # Penalty for overlap
        
        # For each hexagon, compute the minimum container side needed
        # Container is centered at (0,0) with angle 30 degrees
        min_side = 0.0
        container_angle = 30.0
        for hx, hy, ha in hexes:
            verts = _hex_verts(hx, hy, ha)
            for vx, vy in verts:
                # Distance from origin to this vertex in the direction of each container side normal
                # The container side constraint: for each normal direction theta, 
                # vx*cos(theta) + vy*sin(theta) <= S*sqrt(3)/2
                # So S >= (vx*cos(theta) + vy*sin(theta)) / (sqrt(3)/2) for each theta
                for k in range(6):
                    theta = math.radians(container_angle) + k * math.pi / 3
                    proj = vx * math.cos(theta) + vy * math.sin(theta)
                    if proj > 0:  # Only positive projections matter
                        s_required = proj / (math.sqrt(3) / 2)
                        if s_required > min_side:
                            min_side = s_required
                # Also check negative projections (opposite side)
                for k in range(6):
                    theta = math.radians(container_angle) + k * math.pi / 3 + math.pi
                    proj = vx * math.cos(theta) + vy * math.sin(theta)
                    if proj > 0:
                        s_required = proj / (math.sqrt(3) / 2)
                        if s_required > min_side:
                            min_side = s_required
        
        return min_side
    
    # Initial parameters
    x0 = make_params(hexes)
    
    # Use scipy minimize with Nelder-Mead or BFGS
    # First try a simple local optimization
    
    def local_optimize(x0, max_iters=200):
        """Simple local optimization."""
        x = x0.copy()
        f = objective(x)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Try small perturbations
            best_x = x
            best_f = f
            
            for dim in range(len(x)):
                for step in [0.01, -0.01, 0.1, -0.1]:
                    x_new = x.copy()
                    x_new[dim] += step
                    f_new = objective(x_new)
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new
            
            if np.allclose(best_x, x):
                break
            x = best_x
            f = best_f
        
        return x, f
    
    # Run local optimization
    if time.time() < DEADLINE:
        x_opt, f_opt = local_optimize(x0, max_iters=100)
        if f_opt < best_side:
            best_side = f_opt
            best_config = from_params(x_opt)
    
    # Also try differential evolution for global search
    if time.time() < DEADLINE - 30:
        bounds = [(-5, 5), (-5, 5), (-30, 30)] * N
        try:
            result = differential_evolution(objective, bounds, seed=42, maxiter=50, 
                                            popsize=20, tol=1e-4, polish=False)
            if time.time() < DEADLINE and result.fun < best_side:
                best_side = result.fun
                best_config = from_params(result.x)
        except Exception:
            pass
    
    # Format the result
    inner = []
    for hx, hy, ha in best_config:
        inner.append([float(hx), float(hy), float(ha)])
    
    return inner, (0.0, 0.0), float(best_side), 30.0

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
