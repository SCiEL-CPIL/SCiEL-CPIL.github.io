"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import time
import math
import numpy as np

N = 11
DEADLINE = 115.0  # safety margin before 120s limit

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation in degrees."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + math.radians(60 * k)
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)

def _hexagon_contains_hexagon(outer_center, outer_side, outer_angle, inner_center, inner_side, inner_angle):
    """Check if inner hexagon is fully inside outer hexagon using separating axis theorem."""
    ov = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    iv = _hex_vertices(inner_center[0], inner_center[1], inner_side, inner_angle)
    
    # Edges of both hexagons give 12 axes
    axes = []
    for i in range(6):
        for vset in (ov, iv):
            e = vset[(i + 1) % 6] - vset[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for ax in axes:
        o_proj = np.dot(ov, ax)
        i_proj = np.dot(iv, ax)
        if o_proj.min() > i_proj.max() or o_proj.max() < i_proj.min():
            return False
    return True

def _hexagons_disjoint(h1_center, h1_angle, h2_center, h2_angle, side=1.0, tol=1e-9):
    """Check if two unit hexagons are disjoint (no interior overlap)."""
    v1 = _hex_vertices(h1_center[0], h1_center[1], side, h1_angle)
    v2 = _hex_vertices(h2_center[0], h2_center[1], side, h2_angle)
    
    axes = []
    for i in range(6):
        for vset in (v1, v2):
            e = vset[(i + 1) % 6] - vset[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for ax in axes:
        p1 = np.dot(v1, ax)
        p2 = np.dot(v2, ax)
        # Disjoint if separated by at least tol
        if p1.max() + tol < p2.min() or p2.max() + tol < p1.min():
            return True
    return False

def _max_radial_projection(vertices, center):
    """Max distance from center to any vertex projected onto 6 hexagon facet normals."""
    # For a regular hexagon centered at origin with side s, the inradius is s*sqrt(3)/2
    # The 6 facet normals are at angles 30, 90, 150, 210, 270, 330 degrees
    # Actually for a hexagon with vertices at angles 0,60,120,..., the facets are between consecutive vertices
    # Facet normals point outward at angles 30, 90, 150, 210, 270, 330
    max_proj = 0.0
    for k in range(6):
        angle = math.radians(30 + 60 * k)
        n = np.array([math.cos(angle), math.sin(angle)])
        rel = vertices - np.array(center)
        proj = np.max(np.dot(rel, n))
        max_proj = max(max_proj, proj)
    return max_proj

def _min_enclosing_hex_side(vertices, center):
    """Find the minimum side length of a regular hexagon (centered at given center, any rotation) 
    that encloses all vertices. Returns (side, angle_deg)."""
    # For a regular hexagon with inradius r (distance from center to facet), side = r * 2 / sqrt(3)
    # The constraint is that for each of 6 facet directions (at angles 30+60k + rotation),
    # all vertices must satisfy dot(v - center, n_k) <= r
    # So r >= max over vertices and directions... but we can rotate.
    # For a given rotation phi, r(phi) = max_k max_v dot(v-c, n_k(phi))
    # We want to minimize r(phi), then side = 2*r/sqrt(3)
    
    rel = vertices - np.array(center)
    best_r = float('inf')
    best_phi = 0.0
    # Search over rotation in fine steps
    for phi_deg in range(0, 30, 1):  # 30 degree periodicity, 1 degree steps
        phi = math.radians(phi_deg)
        r_max = 0.0
        for k in range(6):
            angle = math.radians(30 + 60 * k) + phi
            n = np.array([math.cos(angle), math.sin(angle)])
            proj = np.max(np.dot(rel, n))
            r_max = max(r_max, proj)
        if r_max < best_r:
            best_r = r_max
            best_phi = phi_deg
    # Refine around best
    for phi_deg in np.linspace(best_phi - 1, best_phi + 1, 101):
        phi = math.radians(phi_deg)
        r_max = 0.0
        for k in range(6):
            angle = math.radians(30 + 60 * k) + phi
            n = np.array([math.cos(angle), math.sin(angle)])
            proj = np.max(np.dot(rel, n))
            r_max = max(r_max, proj)
        if r_max < best_r:
            best_r = r_max
            best_phi = phi_deg % 30
    side = 2.0 * best_r / math.sqrt(3.0)
    return side, best_phi

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal lattice of 11 hexagons
    # Center + 6 in first ring + 4 in second ring (partial)
    # Actually, let's use a compact arrangement
    # For 11 hexagons, a good starting layout is:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # Or use hex lattice coordinates
    
    # Hex lattice: axial coordinates
    # 11 hexagons in a hex shape: center (0,0), ring 1 (6), ring 2 partial (4)
    # Or just 3 rows: 4-3-4 or 3-4-4
    
    # Let's use a 3-4-4 arrangement in hex lattice
    # Axial to pixel for pointy-top hex with side 1:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # But our hexagons have side 1, so adjacent centers are sqrt(3) apart (for flat-top)
    # or 1.5 apart vertically and sqrt(3)/2 horizontally for pointy-top
    
    # For flat-top hexagons with side 1:
    # width = 2, height = sqrt(3)
    # horizontal spacing = 1.5, vertical spacing = sqrt(3)
    # odd columns offset by sqrt(3)/2
    
    # Let me just place them in a simple grid pattern and optimize
    # Start with a roughly circular arrangement
    
    # Simple approach: place 11 hexagons in a 3x4 grid-like pattern
    # Then optimize positions and rotations
    
    # Initial positions (roughly hex-packed)
    sqrt3 = math.sqrt(3)
    # Flat-top hex: center to center distance for adjacent = sqrt(3) (horizontal) or 1.5 (vertical with offset)
    # Let's use: horizontal spacing = 1.5, vertical = sqrt(3)
    # For flat-top: 
    #   col 0: y = 0, sqrt(3), 2*sqrt(3)
    #   col 1: y = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2
    #   etc.
    
    # 11 hexagons: let's do 3 columns of 4, 4, 3 or similar
    # Actually, let's just start with a line and let optimization fix it
    # Better: start with a reasonable cluster
    
    # Hex lattice positions for 11 hexagons (axial coords):
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),  # center + ring 1 = 7
    # (2,0), (1,1), (0,2), (-2,1)  # 4 from ring 2 = 11 total
    # Convert axial to Cartesian for flat-top hex with side 1:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    axial_coords = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 1)
    ]
    
    inner_positions = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        inner_positions.append((x, y))
    
    # All start with 0 rotation
    inner_angles = [0.0] * N
    
    # Optimize using Nelder-Mead on all positions and angles
    # Parameters: for each hexagon, (x, y, angle) = 3*N params
    # Plus outer hexagon rotation (1 param) - but we'll compute optimal outer for given inner
    
    # Actually, the outer hexagon center and rotation can be computed optimally given inner positions
    # For fixed inner positions, find min enclosing regular hexagon
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return float('inf')
        
        # Extract positions and angles
        pos = []
        ang = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            pos.append((x, y))
            ang.append(a)
        
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                if not _hexagons_disjoint(pos[i], ang[i], pos[j], ang[j]):
                    # Penalty for overlap
                    # Compute penetration depth approximately
                    penalty = 0.0
                    v1 = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
                    v2 = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                    # Check all vertex-edge pairs
                    for vi in v1:
                        for k in range(6):
                            e1 = v2[(k+1)%6] - v2[k]
                            n = np.array([-e1[1], e1[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-12:
                                n = n / norm
                                d = np.dot(vi - v2[k], n)
                                if d < 0:
                                    penalty += -d
                    for vi in v2:
                        for k in range(6):
                            e1 = v1[(k+1)%6] - v1[k]
                            n = np.array([-e1[1], e1[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-12:
                                n = n / norm
                                d = np.dot(vi - v1[k], n)
                                if d < 0:
                                    penalty += -d
                    return 10.0 * penalty + 100.0  # Large penalty
        
        # Compute min enclosing hexagon
        all_vertices = []
        for i in range(N):
            v = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
            all_vertices.append(v)
        
        # Find center as centroid of all vertices
        all_v = np.vstack(all_vertices)
        center = np.mean(all_v, axis=0)
        
        side, angle = _min_enclosing_hex_side(all_v, center)
        
        # Verify containment
        for i in range(N):
            if not _hexagon_contains_hexagon((center[0], center[1]), side, angle, 
                                             pos[i], 1.0, ang[i]):
                return side + 10.0  # Penalty if not contained
        
        return side
    
    # Initial params
    params0 = []
    for i in range(N):
        params0.append(inner_positions[i][0])
        params0.append(inner_positions[i][1])
        params0.append(inner_angles[i])
    params0 = np.array(params0, dtype=float)
    
    best_params = params0.copy()
    best_score = objective(params0)
    
    # Nelder-Mead optimization with restarts
    from scipy.optimize import minimize
    
    restarts = 3
    for restart in range(restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if restart == 0:
            p = params0.copy()
        else:
            # Perturb initial
            p = params0 + 0.5 * np.random.randn(3 * N)
        
        try:
            result = minimize(
                objective, p, method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final refinement with tighter tolerance
    if time.time() - start_time < DEADLINE - 10:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final solution
    pos = []
    ang = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        a = best_params[3*i + 2]
        pos.append((x, y))
        ang.append(a)
    
    # Compute final enclosing hexagon
    all_vertices = []
    for i in range(N):
        v = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
        all_vertices.append(v)
    all_v = np.vstack(all_vertices)
    center = np.mean(all_v, axis=0)
    side, angle = _min_enclosing_hex_side(all_v, center)
    
    # Build inner list: (x, y, angle_degrees)
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    
    return inner, (center[0], center[1]), side, angle
