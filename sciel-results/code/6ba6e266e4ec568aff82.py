# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _solve_topology_6_4():
    """Solve the contact graph: center at origin, 6 ring hexagons at distance 2
    from center (at angles 30, 90, ..., 330 deg), 4 outer hexagons each tangent
    to two adjacent ring hexagons (distance 2 from each).
    
    The 4 outer hexagons sit in the 'valleys' between ring hexagons.
    For outer hex k (k=0..3), it's tangent to ring hex (2k+1) and ring hex (2k+2) mod 6.
    Actually, let's use a simpler topology: 4 outer hexagons at angles 0, 60, 120, 180
    (the 'peaks'), each tangent to the two nearest ring hexagons.
    
    Ring hexagons at angles: 30, 90, 150, 210, 270, 330 deg (60 deg apart, offset 30)
    Outer hexagons at angles: 0, 120, 240, 0... no, let's use 4 at 0, 60, 120, 180.
    
    Actually, the standard 11-hexagon packing: center + 6 around + 4 more.
    Let me solve it properly: outer hexagons tangent to pairs of adjacent ring hexagons.
    """
    # Ring positions (fixed): at distance 2 from origin, angles 30+60k
    ring_angles = [30 + 60 * k for k in range(6)]
    ring_pos = np.array([[2.0 * np.cos(np.radians(a)), 2.0 * np.sin(np.radians(a))] for a in ring_angles])
    
    # 4 outer hexagons. Each is tangent to 2 adjacent ring hexagons.
    # Topology: outer k is tangent to ring[2k] and ring[2k+1] for k=0,1,2,3? 
    # That gives 4 pairs: (0,1), (2,3), (4,5), and... we need 4 more.
    # Let's try: outer hexagons tangent to (ring[0],ring[1]), (ring[2],ring[3]), (ring[4],ring[5]), and one more?
    # Better: use the known optimal topology for 11 hexagons.
    
    # Alternative: 4 outer hexagons at the "valley" positions between ring hexagons.
    # Valley angles: 0, 60, 120, 180, 240, 300 (6 valleys). Pick 4.
    # By symmetry, pick valleys at 0, 120, 240, and... we need 4. 
    # Let's try valleys at 0, 60, 120, 180 (4 of the 6).
    
    # For each outer hex at valley angle, it should be tangent to the two adjacent ring hexagons.
    # Valley at angle 0 is between ring at 330 and ring at 30.
    # Valley at angle 60 is between ring at 30 and ring at 90.
    # etc.
    
    valley_angles = [0, 60, 120, 180]
    # For valley at angle va, adjacent ring hexagons are at va-30 and va+30 (mod 360)
    # So outer hex must be at distance 2 from both ring[va-30] and ring[va+30].
    
    def equations(vars):
        eqs = []
        for k, va in enumerate(valley_angles):
            ox, oy = vars[2*k], vars[2*k+1]
            # Adjacent ring hexagons
            a1 = np.radians(va - 30)
            a2 = np.radians(va + 30)
            r1 = np.array([2.0 * np.cos(a1), 2.0 * np.sin(a1)])
            r2 = np.array([2.0 * np.cos(a2), 2.0 * np.sin(a2)])
            # Distance to r1 = 2
            eqs.append(np.sqrt((ox - r1[0])**2 + (oy - r1[1])**2) - 2.0)
            # Distance to r2 = 2
            eqs.append(np.sqrt((ox - r2[0])**2 + (oy - r2[1])**2) - 2.0)
        return eqs
    
    # Initial guess: along the valley direction, at some radius
    guess = []
    for va in valley_angles:
        a = np.radians(va)
        guess.extend([3.5 * np.cos(a), 3.5 * np.sin(a)])
    
    try:
        sol, info, ier, msg = fsolve(equations, guess, full_output=True)
        if ier != 1:
            return None
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        angles[0] = 0.0
        for k in range(6):
            positions[k+1] = ring_pos[k]
            angles[k+1] = 0.0
        for k in range(4):
            positions[7+k] = sol[2*k:2*k+2]
            angles[7+k] = 0.0
        
        if not _check_separation_fast(positions, angles):
            return None
        return positions, angles
    except Exception:
        return None

def _solve_topology_6_4_alt():
    """Alternative topology: 4 outer hexagons at valleys 0, 120, 240, and 300.
    Or try different valley selections."""
    valley_options = [
        [0, 60, 120, 180],
        [0, 120, 240, 300],
        [0, 60, 180, 240],
        [30, 90, 150, 210],
        [30, 90, 210, 270],
        [60, 120, 180, 240],
    ]
    ring_angles = [30 + 60 * k for k in range(6)]
    ring_pos = np.array([[2.0 * np.cos(np.radians(a)), 2.0 * np.sin(np.radians(a))] for a in ring_angles])
    
    best_result = None
    best_side = float('inf')
    
    for valley_angles in valley_options:
        def equations(vars, va_list=valley_angles):
            eqs = []
            for k, va in enumerate(va_list):
                ox, oy = vars[2*k], vars[2*k+1]
                a1 = np.radians(va - 30)
                a2 = np.radians(va + 30)
                r1 = np.array([2.0 * np.cos(a1), 2.0 * np.sin(a1)])
                r2 = np.array([2.0 * np.cos(a2), 2.0 * np.sin(a2)])
                eqs.append(np.sqrt((ox - r1[0])**2 + (oy - r1[1])**2) - 2.0)
                eqs.append(np.sqrt((ox - r2[0])**2 + (oy - r2[1])**2) - 2.0)
            return eqs
        
        guess = []
        for va in valley_angles:
            a = np.radians(va)
            guess.extend([3.5 * np.cos(a), 3.5 * np.sin(a)])
        
        try:
            sol, info, ier, msg = fsolve(equations, guess, full_output=True)
            if ier != 1:
                continue
            positions = np.zeros((11, 2))
            angles = np.zeros(11)
            positions[0] = [0.0, 0.0]
            for k in range(6):
                positions[k+1] = ring_pos[k]
            for k in range(4):
                positions[7+k] = sol[2*k:2*k+2]
            
            if not _check_separation_fast(positions, angles):
                continue
            side = _min_enclosing_side(positions, angles)
            if side < best_side:
                best_side = side
                best_result = (positions.copy(), angles.copy())
        except Exception:
            continue
    
    return best_result

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0

    best_positions = None
    best_angles = None
    best_side = float('inf')

    # Phase 1: Solve topology 6+4 (valleys 0,60,120,180)
    res = _solve_topology_6_4()
    if res is not None:
        positions, angles = res
        side = _min_enclosing_side(positions, angles)
        if side < best_side:
            best_side = side
            best_positions = positions.copy()
            best_angles = angles.copy()

    # Phase 2: Try alternative valley topologies
    if time.time() < deadline - 100:
        res2 = _solve_topology_6_4_alt()
        if res2 is not None:
            positions, angles = res2
            side = _min_enclosing_side(positions, angles)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()

    # Fallback: simple symmetric arrangement
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)

    # Phase 3: Global rotation optimization (1-DOF)
    if time.time() < deadline - 30:
        def obj_rot(theta):
            c = np.cos(np.radians(theta))
            s = np.sin(np.radians(theta))
            rot_mat = np.array([[c, -s], [s, c]])
            pos_rot = best_positions @ rot_mat.T
            angles_rot = (best_angles + theta) % 360
            return _min_enclosing_side(pos_rot, angles_rot)

        best_rot = 0.0
        best_rot_side = best_side
        for t in np.linspace(0, 60, 241):
            s_t = obj_rot(t)
            if s_t < best_rot_side:
                best_rot_side = s_t
                best_rot = t
        try:
            res_rot = minimize(obj_rot, [best_rot], method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-11})
            if res_rot.fun < best_rot_side:
                best_rot = res_rot.x[0]
                best_rot_side = res_rot.fun
        except Exception:
            pass
        if best_rot_side < best_side:
            best_side = best_rot_side
            c = np.cos(np.radians(best_rot))
            s = np.sin(np.radians(best_rot))
            rot_mat = np.array([[c, -s], [s, c]])
            best_positions = best_positions @ rot_mat.T
            best_angles = (best_angles + best_rot) % 360

    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)

    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001

    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)

    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
