import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12
EPS = 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at angles 0, 60, 120...
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint, False if overlapping.
        Uses Separating Axis Theorem (SAT) which is exact for convex polygons.
        """
        # Get all edges
        edges1 = []
        edges2 = []
        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            # Normal to edge
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Perpendicular
            nx = -dy
            ny = dx
            # Normalize (optional for SAT if we just check projection overlap, but good for robustness)
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 0:
                nx /= norm
                ny /= norm
            edges1.append((nx, ny))
            
            p3 = poly2[i]
            p4 = poly2[(i+1)%6]
            dx = p4[0] - p3[0]
            dy = p4[1] - p3[1]
            nx = -dy
            ny = dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 0:
                nx /= norm
                ny /= norm
            edges2.append((nx, ny))
        
        # Test all axes
        for nx, ny in edges1 + edges2:
            # Project poly1
            min1 = float('inf')
            max1 = float('-inf')
            for v in poly1:
                proj = v[0]*nx + v[1]*ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
            
            # Project poly2
            min2 = float('inf')
            max2 = float('-inf')
            for v in poly2:
                proj = v[0]*nx + v[1]*ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            
            # Check overlap
            if max1 < min2 + EPS or max2 < min1 + EPS:
                return True # Disjoint
        
        return False # Overlapping

    def objective(params):
        if time.time() > deadline - 5:
            return 1e6
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Check disjointness
        # To save time, only check if side is potentially good? No, penalty must be strict.
        # We can pre-compute polys
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(polys[i], polys[j]):
                    # Estimate violation depth for smoother gradient? 
                    # For DE, a simple large penalty often works if the feasible region is connected.
                    # Let's add a penalty proportional to the overlap depth if possible, or just fixed.
                    # Fixed penalty of 10.0 is usually enough if the objective scale is ~5-10.
                    penalty += 10.0
                if penalty > 100.0:
                    break
            if penalty > 100.0:
                break
        
        return side + penalty

    # Bounds for Differential Evolution
    # We expect the outer hexagon side to be around 5-8.
    # So centers should be within radius ~4.
    # Angles: 0 to 360.
    bounds = [(-4, 4)] * (2*N) + [(0, 360)] * N

    best_score = float('inf')
    best_positions = None
    best_angles = None

    # Initial best guess: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    # Generate a few initial candidates to seed DE or to have a fallback
    candidates = []
    
    # Candidate 1: Standard honeycomb 1+6+5
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 5 more in the next ring
    
    # Let's use a simple grid search for initial valid positions to help DE
    # Or just let DE do it. DE is global.
    
    # Run Differential Evolution
    # maxiter=100, popsize=20 might be too slow if check_disjoint is expensive.
    # check_disjoint is O(1) per pair, 66 pairs. ~400 ops.
    # 100 iter * 20 pop * 120 evals/iter? No, DE is 100 * 20 * 1 eval per individual?
    # DE generates 100*20 individuals.
    # 2000 evals * 400 ops = 800,000 ops. Very fast.
    # Let's try popsize=30, maxiter=200.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            tol=1e-6, 
            polish=True, 
            seed=42,
            init='sobol' # Better sampling
        )
        
        if result.success or result.fun < best_score:
            # Verify validity
            new_positions = result.x[:2*N].reshape(N, 2)
            new_angles = result.x[2*N:]
            
            polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                score = calc_min_outer_side(new_positions, new_angles, 0.0)
                if score < best_score:
                    best_score = score
                    best_positions = new_positions
                    best_angles = new_angles
    except Exception as e:
        pass

    # If DE failed or didn't find a valid solution, try a manual local refinement from a good seed
    if best_positions is None or best_score > 8.0:
        # Fallback: Use the honeycomb seed and Nelder-Mead
        # Construct a valid honeycomb seed
        # Center (0,0)
        # Neighbors at distance sqrt(3) at angles 0, 60, 120, 180, 240, 300
        # Wait, standard honeycomb:
        # If flat-top hexes, neighbors are at 0, 60...
        # If pointy-top (our case), flats are at 30, 90...
        # Neighbors are in direction of normals.
        # Normals at 30, 90, 150, 210, 270, 330.
        
        init_pos = np.zeros((N, 2))
        init_ang = np.zeros(N)
        
        # Center
        init_pos[0] = [0, 0]
        
        # 6 Neighbors
        for k in range(6):
            ang = math.radians(30 + k*60)
            init_pos[k+1] = [sqrt3 * math.cos(ang), sqrt3 * math.sin(ang)]
        
        # Next ring: 5 more?
        # The 7th position in a hexagonal packing of 12?
        # 1 center, 6 ring 1, 6 ring 2? Total 13.
        # We need 12. So 1 center, 6 ring 1, 5 from ring 2.
        # Ring 2 positions are at distance 2*sqrt(3)? No.
        # Distance to ring 2 center:
        # Vector sum of two ring 1 vectors.
        # v1 + v2. |v1|=|v2|=sqrt(3). Angle 60.
        # |v1+v2| = sqrt(3 + 3 + 2*3*cos(60)) = sqrt(6+3) = 3.
        # So ring 2 is at distance 3.
        
        # Let's pick 5 from the 12 possible ring 2 positions?
        # Or just place them randomly and refine?
        # Let's place them at distance 3, angles 0, 60, 120, 180, 240.
        for k in range(5):
            ang = math.radians(k*60)
            init_pos[k+7] = [3 * math.cos(ang), 3 * math.sin(ang)]
        
        # Check validity of this seed
        polys = [get_hex_verts(init_pos[i][0], init_pos[i][1], 1.0, 0.0) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(polys[i], polys[j]):
                    valid = False
                    break
            if not valid: break
        
        if valid:
            score = calc_min_outer_side(init_pos, init_ang, 0.0)
            if score < best_score:
                best_score = score
                best_positions = init_pos
                best_angles = init_ang
                
            # Refine with Nelder-Mead
            def obj_nm(params):
                if time.time() > deadline - 5:
                    return 1e6
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side = calc_min_outer_side(positions, angles, 0.0)
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j]):
                            penalty += 10.0
                    if penalty > 100: break
                return side + penalty

            x0 = np.concatenate([best_positions.flatten(), best_angles])
            try:
                from scipy.optimize import minimize
                res = minimize(obj_nm, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                if res.fun < best_score:
                    new_positions = res.x[:2*N].reshape(N, 2)
                    new_angles = res.x[2*N:]
                    polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            if not check_disjoint(polys[i], polys[j]):
                                valid = False
                                break
                    if not valid: break
                    if valid:
                        best_score = res.fun
                        best_positions = new_positions
                        best_angles = new_angles
            except Exception:
                pass

    # Final fallback if nothing worked
    if best_positions is None:
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        best_score = 10.0 # Arbitrary

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

# To ensure the function is available if imported
if __name__ == "__main__":
    construct()
