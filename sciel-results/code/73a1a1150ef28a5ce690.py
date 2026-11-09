import numpy as np
from scipy.optimize import minimize
import math

N = 11

def construct():
    # Outer hexagon is centered at (0,0) with orientation 0.
    # Inner hexagons are unit side length.
    
    # Precompute normals for outer hexagon containment check
    # Outer hexagon normals are at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))]
        for k in range(6)
    ])
    
    # Support function for a unit regular hexagon
    # h(phi) = (sqrt(3)/2) / cos(delta) where delta is angle to nearest normal
    # Normals of inner hexagon (local) are at 30, 90, ...
    inner_normals_local = np.array([
        [math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))]
        for k in range(6)
    ])
    
    def support_function(phi, theta):
        # phi is absolute angle
        # theta is rotation of the hexagon
        # We need the angle of phi relative to the hexagon's local frame
        # Local normals are at 30+60k.
        # Absolute normals are at 30+60k + theta.
        # So relative angle is phi - theta.
        # Find the closest local normal to (phi - theta).
        # The local normals are at 30, 90, ...
        # Let's just compute the distance to the 6 local normal angles.
        # Local normal angles: 30, 90, 150, 210, 270, 330.
        local_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0
        rel_angle = phi - theta
        # Find min absolute difference modulo 2pi
        # delta = min_k |rel_angle - (30+60k)pi/180|
        # Since the function is periodic with 60 deg (pi/3), we can map rel_angle to [0, pi/3)
        # and find the distance to the center of that sector?
        # The support function h(phi) for a regular hexagon with apothem a=sqrt(3)/2 is:
        # h(phi) = a / cos(phi - phi_k) where phi_k is the nearest normal angle.
        # The nearest normal angle to phi is one of the 6.
        # Due to symmetry, we can just take phi mod (pi/3) and adjust.
        # Actually, simpler:
        # delta = (phi - theta) % (math.pi/3)
        # if delta > math.pi/6: delta = math.pi/3 - delta
        # h = (math.sqrt(3)/2) / math.cos(delta)
        
        delta = (phi - theta) % (math.pi / 3.0)
        if delta > math.pi / 6.0:
            delta = math.pi / 3.0 - delta
        return (math.sqrt(3) / 2.0) / math.cos(delta)
        
    # We need to evaluate the max over phi.
    # The max of h_i(phi) + h_j(phi+pi) - d occurs when the "corners" align.
    # It is sufficient to check phi at the normal angles of the hexagons and midpoints.
    # There are 12 critical directions for the sum of two hexagons.
    # Let's sample 36 angles (every 10 degrees) to be safe.
    sample_phis = np.linspace(0, 2*math.pi, 36, endpoint=False)
    
    def objective(vars):
        S = vars[0]
        if S < 0:
            return 1e6
        # Unpack positions and angles
        # vars[1:1+3*N]
        inner = []
        for i in range(N):
            x = vars[1 + 3*i]
            y = vars[2 + 3*i]
            theta = vars[3 + 3*i]
            inner.append((x, y, theta))
            
        penalty = 0.0
        
        # 1. Containment constraints
        # Inner hexagon i must be inside outer hexagon (side S, center 0,0, rot 0)
        # Condition: for all phi, c_i . n_outer(phi) + h_inner(phi + theta_i) <= S * sqrt(3)/2
        # We check this for the 6 outer normals and maybe some intermediate ones?
        # Actually, the max of c . n + h_inner is convex. The max over phi is attained at the normals of the inner hexagon or outer?
        # To be safe, we check the 6 outer normals and the 6 inner normals (rotated).
        check_phis = np.concatenate([
            np.array([math.radians(30 + 60*k) for k in range(6)]),
            np.array([math.radians(30 + 60*k) for k in range(6)]) # Inner normals relative to world?
            # The inner support function h_inner(phi+theta) has peaks at phi+theta = 30+60k.
            # So phi = 30+60k - theta.
        ])
        
        # Better: Just sample the 12 critical points:
        # Outer normals: 30, 90, ...
        # Inner normals (world): 30+theta, 90+theta, ...
        # Let's just use the 36 sample angles for robustness, it's cheap.
        
        for i in range(N):
            x, y, theta = inner[i]
            c = np.array([x, y])
            max_violation = 0.0
            for phi in sample_phis:
                # Project center onto outer normal
                # We need the outer normal at angle phi?
                # No, the containment condition is:
                # The inner hexagon is inside the outer if the Minkowski sum of the inner hexagon and the negative outer hexagon is in the negative orthant?
                # Simpler: The maximum distance from the origin to the inner hexagon in any direction n must be <= the distance from origin to outer boundary in direction n.
                # dist_outer(n) = S * sqrt(3)/2 / cos(delta_outer) where delta_outer is angle to nearest outer normal.
                # dist_inner(n) = c . n + h_inner(n + theta).
                # We need dist_inner(n) <= dist_outer(n) for all n.
                
                # Calculate dist_outer for direction phi
                # Outer normals are at 30, 90...
                # delta_outer = phi mod 60 deg.
                delta_o = phi % (math.pi / 3.0)
                if delta_o > math.pi / 6.0:
                    delta_o = math.pi / 3.0 - delta_o
                dist_outer = (S * math.sqrt(3) / 2.0) / math.cos(delta_o)
                
                # Calculate dist_inner for direction phi
                # h_inner(phi + theta)
                h_val = support_function(phi + theta, 0.0) # support_function expects phi and theta. Here we want h at angle phi+theta for a hexagon at 0?
                # My support_function(phi, theta) calculates h for a hexagon rotated by theta at angle phi.
                # So h_inner(phi) for hexagon at theta is support_function(phi, theta).
                h_val = support_function(phi, theta)
                
                # Project center c onto direction phi
                c_proj = x * math.cos(phi) + y * math.sin(phi)
                
                dist_inner = c_proj + h_val
                
                violation = dist_inner - dist_outer
                if violation > max_violation:
                    max_violation = violation
                    
            if max_violation > 0:
                penalty += max_violation ** 2
                
        # 2. Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, th1 = inner[i]
                x2, y2, th2 = inner[j]
                d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                max_violation = 0.0
                for phi in sample_phis:
                    # Gap in direction phi from i to j
                    # dist from ci to boundary in dir phi: h1(phi)
                    # dist from cj to boundary in dir phi+pi: h2(phi+pi)
                    h1 = support_function(phi, th1)
                    h2 = support_function(phi + math.pi, th2)
                    required_d = h1 + h2
                    violation = required_d - d
                    if violation > max_violation:
                        max_violation = violation
                if max_violation > 0:
                    penalty += max_violation ** 2
                    
        return S + 10.0 * penalty

    # Initial Guess
    # Hexagonal lattice
    # Side length 1, distance between centers in lattice is sqrt(3)
    dx = math.sqrt(3)
    dy = 1.5
    
    init_vars = []
    init_vars.append(4.0) # S
    positions = []
    # Generate a cluster of 11 points
    # 1 center, 6 ring 1, 4 ring 2
    positions.append((0,0,0))
    for k in range(6):
        angle = math.pi/3 * k
        x = dx * math.cos(angle)
        y = dx * math.sin(angle)
        positions.append((x,y,0))
    # Add 4 more in the gaps of ring 1?
    # Ring 2 positions are at distance sqrt(3)*sqrt(3)=3?
    # Let's just place them roughly and let optimizer fix.
    positions.append((2*dx, 0, 0))
    positions.append((dx, 2*dy, 0))
    positions.append((-dx, 2*dy, 0))
    positions.append((-2*dx, 0, 0))
    
    for x, y, th in positions:
        init_vars.append(x)
        init_vars.append(y)
        init_vars.append(th)
        
    # Optimize
    best_result = None
    best_score = float('inf')
    
    # Run a few times with slight perturbations
    for attempt in range(5):
        vars_copy = init_vars.copy()
        if attempt > 0:
            # Add noise
            for i in range(1, len(vars_copy)):
                vars_copy[i] += (np.random.rand() - 0.5) * 0.1
                
        res = minimize(objective, vars_copy, method='Powell', options={'maxiter': 200, 'disp': False})
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
    # Extract best solution
    S = best_result[0]
    inner = []
    for i in range(N):
        x = best_result[1 + 3*i]
        y = best_result[2 + 3*i]
        angle_deg = math.degrees(best_result[3 + 3*i])
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
