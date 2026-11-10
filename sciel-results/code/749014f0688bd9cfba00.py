# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view

def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in degrees."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz

def _in_corridor_vec(x, y, z):
    """Vectorized corridor check. Returns bool array."""
    r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return r1 | r2 | r3

def _clearance_to_boundary(x, y, z):
    """
    Calculate the minimum distance from points (x,y,z) to the corridor boundary.
    Positive if inside, negative if outside.
    For a union of boxes, the clearance is the min over boxes of (max(0, dist_to_box_complement)).
    Actually, simpler: for each point, find the closest boundary.
    Since the corridor is a union of 3 boxes, a point is inside if it's in any box.
    The 'clearance' for a point inside the corridor is the minimum distance to the boundary of the corridor.
    
    Let's compute distance to the complement.
    Complement is:
    1. x > 0 AND (y<0 OR y>1 OR z<0 OR z>1) -- wait, this is not disjoint from others.
    
    Let's use the property: dist_to_boundary = min( dist_to_box1_boundary, dist_to_box2_boundary, dist_to_box3_boundary )
    only if the point is inside that box? No.
    
    If point is in Box 1, its distance to the boundary of the CORRIDOR is the min distance to the boundary of Box 1, UNLESS it is closer to the boundary of Box 2 or 3 (which would imply it's near the intersection).
    Actually, the boundary of the union is the union of the boundaries of the boxes, minus the parts inside other boxes.
    
    A robust way:
    For each point, check which box it is in.
    If in Box 1:
       dist_x = min(x - (-inf), 0 - x) -> if x<0, dist to x=0 is -x. But x<=0 is the region.
       The boundary of Box 1 is at x=0, y=0, y=1, z=0, z=1.
       dist1 = min(-x if x<0 else inf, y, 1-y, z, 1-z)
       But if the point is also near Box 2 (e.g. x=0, y=0.5, z=0.5), the boundary of the corridor is not x=0 there, because Box 2 continues.
       
    Let's just compute the distance to the nearest point OUTSIDE the corridor.
    This is equivalent to:
    For a point p, d(p) = min_{q not in C} ||p - q||.
    
    Since C is a union of boxes B1, B2, B3.
    The complement is (B1^c ∩ B2^c ∩ B3^c).
    
    Let's approximate:
    If p is in B1, the distance to the boundary of C is the min of distances to the faces of B1 that are NOT covered by B2 or B3.
    This is getting complex.
    
    Alternative: Since the grid is fine (0.05), we can just check if the point is inside.
    For optimization, we need a gradient.
    
    Let's define a signed distance function SDF.
    SDF(p) = min( SDF_B1(p), SDF_B2(p), SDF_B3(p) )
    where SDF_Bi(p) is the signed distance to box Bi (negative inside).
    If p is inside any box, SDF(p) will be negative (if we use min of negative values? No, min of signed distances works for unions if we take the min of the negative distances? No.
    For union: SDF_union(p) = min(SDF_B1(p), SDF_B2(p), SDF_B3(p)) is correct for the distance to the boundary, provided we define SDF_B as negative inside.
    Wait, if p is inside B1, SDF_B1(p) < 0. SDF_B2(p) might be > 0.
    min(neg, pos) = neg. Correct.
    If p is outside all, all are > 0, min is the closest box boundary. Correct.
    So SDF(p) = min(SDF_B1(p), SDF_B2(p), SDF_B3(p)) gives the signed distance to the corridor boundary.
    
    SDF_Box(p, center, half_extents):
    d = abs(p - center) - half_extents
    dist_out = norm(max(d, 0))
    dist_in = min(d)
    sdf = dist_out if dist_out > 0 else dist_in
    
    Box 1: x<=0, 0<=y<=1, 0<=z<=1.
    Center: (-inf, 0.5, 0.5)? No, it's semi-infinite.
    Let's handle semi-infinite boxes by clamping.
    Box 1: x in [-inf, 0], y in [0, 1], z in [0, 1].
    Box 2: x in [0, 1], y in [0, 5], z in [0, 1].
    Box 3: x in [0, 1], y in [5, 6], z in [0, inf].
    
    We can compute SDF for each box manually.
    """
    # Box 1: x <= 0, 0 <= y <= 1, 0 <= z <= 1
    # Distance to boundary of Box 1 (treating it as a box with x_min=-inf)
    # If x > 0, dist_x = x. Else dist_x = 0 (for the "outside" part of the box logic, but we want signed distance).
    # Let's use the standard box SDF formula with clamped bounds.
    
    # For Box 1:
    # x is in [-inf, 0]. So x_max = 0. x_min = -inf.
    # y is in [0, 1].
    # z is in [0, 1].
    
    # SDF for a box with bounds [min, max]:
    # dx = max(max_x - x, 0) + max(x - min_x, 0) ... wait.
    # Standard:
    # d = abs(p - c) - h
    # Here bounds are asymmetric/infinite.
    
    # Let's just compute distance to the 6 planes of each box.
    # Box 1 planes: x=0, y=0, y=1, z=0, z=1. (x=-inf is not a boundary for distance if x<0).
    # If x > 0, dist to x=0 is x.
    # If x <= 0, dist to x=0 is -x? No, if x<=0, we are inside the x-range.
    
    # Let's define a helper for bounded box SDF.
    def _sdf_box(p, xmin, xmax, ymin, ymax, zmin, zmax):
        # p: (3, N) array
        x, y, z = p[0], p[1], p[2]
        
        # Distance to each face
        # x faces
        dx1 = np.where(x < xmin, xmin - x, 0)
        dx2 = np.where(x > xmax, x - xmax, 0)
        # y faces
        dy1 = np.where(y < ymin, ymin - y, 0)
        dy2 = np.where(y > ymax, y - ymax, 0)
        # z faces
        dz1 = np.where(z < zmin, zmin - z, 0)
        dz2 = np.where(z > zmax, z - zmax, 0)
        
        # Distance to outside of box (L2 norm of positive components)
        dist_out = np.sqrt(dx1**2 + dx2**2 + dy1**2 + dy2**2 + dz1**2 + dz2**2)
        
        # Distance to inside (min of signed distances to faces, negative if inside)
        # Signed distance to face x=xmin: x - xmin (negative if x < xmin)
        # Signed distance to face x=xmax: xmax - x (negative if x > xmax)
        # We want the minimum "depth" into the box.
        # If inside, all signed distances are positive (distance to boundary).
        # Let's compute the minimum distance to any face, taking into account if we are inside.
        
        # Simpler: 
        # If point is outside, SDF = dist_out.
        # If point is inside, SDF = -min(dist_to_faces).
        
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
        
        # Distances to faces (positive values)
        d_xmin = np.abs(x - xmin)
        d_xmax = np.abs(x - xmax)
        d_ymin = np.abs(y - ymin)
        d_ymax = np.abs(y - ymax)
        d_zmin = np.abs(z - zmin)
        d_zmax = np.abs(z - zmax)
        
        min_dist_to_face = np.minimum.reduce([d_xmin, d_xmax, d_ymin, d_ymax, d_zmin, d_zmax])
        
        # If inside, SDF is -min_dist_to_face.
        # If outside, SDF is dist_out.
        # But we need to be careful with infinite bounds.
        # For Box 1, xmin = -inf. d_xmin = inf.
        
        return np.where(inside, -min_dist_to_face, dist_out)
    
    # Box 1: x in [-inf, 0], y in [0, 1], z in [0, 1]
    # We can't use -inf in numpy easily for distance.
    # Let's treat x < 0 as "inside" for x, so d_xmin is large.
    # Actually, if x < 0, it is inside the x-range of Box 1.
    
    # Let's just hardcode the SDF for the 3 specific boxes.
    
    # Box 1: x<=0, 0<=y<=1, 0<=z<=1
    x1, y1, z1 = x, y, z
    # Inside check
    in1 = (x1 <= 0) & (y1 >= 0) & (y1 <= 1) & (z1 >= 0) & (z1 <= 1)
    # Dist to boundary if inside
    # Faces: x=0, y=0, y=1, z=0, z=1
    d1_x = np.where(x1 < 0, -x1, 0) # Dist to x=0 if x<0
    d1_y0 = np.where(y1 > 0, y1, 0)
    d1_y1 = np.where(y1 < 1, 1-y1, 0)
    d1_z0 = np.where(z1 > 0, z1, 0)
    d1_z1 = np.where(z1 < 1, 1-z1, 0)
    # Min dist to face
    min_d1 = np.minimum.reduce([d1_x, d1_y0, d1_y1, d1_z0, d1_z1])
    # If outside, dist to box
    # Out of x: x > 0 -> dist = x
    # Out of y: y < 0 or y > 1
    # Out of z: z < 0 or z > 1
    # This is getting messy.
    
    # Let's use the generic _sdf_box with large finite bounds for the infinite sides.
    # Grid x is in [-4, 0]. So -100 is safe.
    # Grid y, z in [0, 1].
    
    sdf1 = _sdf_box(p, -100, 0, 0, 1, 0, 1)
    sdf2 = _sdf_box(p, 0, 1, 0, 5, 0, 1)
    sdf3 = _sdf_box(p, 0, 1, 5, 6, 0, 100)
    
    return np.minimum.reduce([sdf1, sdf2, sdf3])

def _densify_path(full_path):
    """Densify path so each step moves at most 0.05 Euclidean and changes each angle by at most 2 degrees."""
    densified = [full_path[0]]
    for i in range(1, len(full_path)):
        p0 = full_path[i - 1]
        p1 = full_path[i]
        pos_dist = np.linalg.norm(p1[:3] - p0[:3])
        ang_diffs = np.abs(p1[3:] - p0[3:])
        n_steps = max(1, int(np.ceil(pos_dist / 0.05)), int(np.ceil(np.max(ang_diffs) / 2.0)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            interp = p0 + t * (p1 - p0)
            densified.append(interp)
    return np.array(densified)

def _evaluate_path_fast(path):
    """
    Fast vectorized evaluation of the path.
    Returns the surviving volume (count of 4x4x4 blocks * 0.05^3).
    """
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start, path, end])
    densified = _densify_path(full_path)
    
    # Grid
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    N = len(points)
    
    alive = np.ones(N, dtype=bool)
    
    # Check at first pose
    p0 = densified[0]
    R0 = _rotation_matrix(p0[3], p0[4], p0[5])
    placed0 = points @ R0.T + p0[:3]
    alive &= _in_corridor_vec(placed0[:, 0], placed0[:, 1], placed0[:, 2])
    
    # Process each step
    for step in range(1, len(densified)):
        if not np.any(alive):
            break
        prev = densified[step - 1]
        curr = densified[step]
        
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dz = curr[2] - prev[2]
        dyaw = curr[3] - prev[3]
        dpitch = curr[4] - prev[4]
        droll = curr[5] - prev[5]
        
        # x move
        if abs(dx) > 1e-12:
            p_x = prev.copy()
            p_x[0] = curr[0]
            R_x = _rotation_matrix(p_x[3], p_x[4], p_x[5])
            placed_x = points[alive] @ R_x.T + p_x[:3]
            new_alive = _in_corridor_vec(placed_x[:, 0], placed_x[:, 1], placed_x[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # y move
        if abs(dy) > 1e-12:
            p_y = prev.copy()
            p_y[0] = curr[0]
            p_y[1] = curr[1]
            R_y = _rotation_matrix(p_y[3], p_y[4], p_y[5])
            placed_y = points[alive] @ R_y.T + p_y[:3]
            new_alive = _in_corridor_vec(placed_y[:, 0], placed_y[:, 1], placed_y[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # z move
        if abs(dz) > 1e-12:
            p_z = prev.copy()
            p_z[0] = curr[0]
            p_z[1] = curr[1]
            p_z[2] = curr[2]
            R_z = _rotation_matrix(p_z[3], p_z[4], p_z[5])
            placed_z = points[alive] @ R_z.T + p_z[:3]
            new_alive = _in_corridor_vec(placed_z[:, 0], placed_z[:, 1], placed_z[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # Rotation
        if abs(dyaw) > 1e-12 and abs(dpitch) < 1e-12 and abs(droll) < 1e-12:
            p_rot = p_z.copy() if abs(dz) > 1e-12 else prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            p_rot[3] = curr[3]
            n_samples = max(2, int(np.ceil(abs(dyaw) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                yaw_s = prev[3] + t * dyaw
                R_s = _rotation_matrix(yaw_s, p_rot[4], p_rot[5])
                placed_s = points[alive] @ R_s.T + p_rot[:3]
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        elif abs(dpitch) > 1e-12 and abs(dyaw) < 1e-12 and abs(droll) < 1e-12:
            p_rot = p_z.copy() if abs(dz) > 1e-12 else prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            p_rot[4] = curr[4]
            n_samples = max(2, int(np.ceil(abs(dpitch) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                pitch_s = prev[4] + t * dpitch
                R_s = _rotation_matrix(p_rot[3], pitch_s, p_rot[5])
                placed_s = points[alive] @ R_s.T + p_rot[:3]
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        elif abs(droll) > 1e-12 and abs(dyaw) < 1e-12 and abs(dpitch) < 1e-12:
            p_rot = p_z.copy() if abs(dz) > 1e-12 else prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            p_rot[5] = curr[5]
            n_samples = max(2, int(np.ceil(abs(droll) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                roll_s = prev[5] + t * droll
                R_s = _rotation_matrix(p_rot[3], p_rot[4], roll_s)
                placed_s = points[alive] @ R_s.T + p_rot[:3]
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        else:
            # Multiple axes changing - check final pose
            R_rot = _rotation_matrix(curr[3], curr[4], curr[5])
            placed_rot = points[alive] @ R_rot.T + curr[:3]
            new_alive = _in_corridor_vec(placed_rot[:, 0], placed_rot[:, 1], placed_rot[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
    
    # Count 4x4x4 blocks
    alive_3d = alive.reshape(81, 21, 21)
    blocks = sliding_window_view(alive_3d, (4, 4, 4))
    all_survive = np.all(blocks, axis=(3, 4, 5))
    count = np.sum(all_survive)
    return count * (0.05 ** 3)

def _seed_path():
    """Return the seed path as a numpy array."""
    return np.array([
        [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
        [0.976853949, 0.0843900243, -0.00187236014, 2.22743402, 0.0275244655, 0.595653008],
        [1.01563377, 0.158600496, 9.83065911e-05, 4.35572642, -0.00995231395, 0.363970701],
        [1.08493385, 0.307755287, 0.00404501406, 8.62938219, -0.000435990906, -0.335077274],
        [1.12726319, 0.436994587, 0.0165792562, 12.3522097, -0.444355561, -1.09977761],
        [1.15035596, 0.509864783, 0.0246729498, 14.4926365, -0.703150883, -1.77573027],
        [1.20020002, 0.674818962, 0.0413920989, 19.2858257, -1.46824802, -3.24301616],
        [1.22921833, 0.78032244, 0.0508392229, 22.3604231, -2.05934965, -4.19183977],
        [1.25656232, 0.886955923, 0.0622851979, 25.4006424, -2.79233605, -4.94301221],
        [1.28161834, 0.9941859, 0.0713168622, 28.4982429, -3.53439006, -5.72810662],
        [1.30160821, 1.12543063, 0.0720451855, 32.0688854, -3.9280592, -5.56488257],
        [1.32438826, 1.32927434, 0.0724034864, 37.7041144, -4.47472162, -5.17490305],
        [1.33399008, 1.48086681, 0.0758177845, 41.9024902, -4.92138226, -4.72192269],
        [1.33698854, 1.59471234, 0.075956822, 45.1249666, -5.16592127, -4.32429483],
        [1.33554955, 1.70831674, 0.0733949585, 48.4117195, -5.24474942, -3.95318659],
        [1.33301955, 1.77848192, 0.0730557819, 50.4395679, -5.29391038, -3.60268508],
        [1.32299384, 1.89675169, 0.066224521, 54.0094721, -5.07170268, -3.10511721],
        [1.31609554, 1.95530642, 0.0645580036, 55.7906035, -4.95025879, -2.75163185],
        [1.29699464, 2.0647489, 0.0528964026, 59.2092384, -4.24840518, -2.08208301],
        [1.28467533, 2.12603596, 0.0465722033, 61.19788, -3.85394213, -1.77381137],
        [1.27612175, 2.16246336, 0.0429218547, 62.3960278, -3.60313312, -1.60391888],
        [1.25918263, 2.22238115, 0.0352027157, 64.3876548, -2.94645508, -1.14315788],
        [1.21195969, 2.37272154, 0.0185866337, 69.5501344, -1.44731306, -0.351817856],
        [1.17367747, 2.46947271, 0.00787738711, 73.3278162, -0.59130695, -0.204315921],
        [1.13343824, 2.56892736, 0.000240804007, 77.2294036, 0.0742535282, -0.0118797041],
        [1.10227764, 2.63702207, -0.000813164259, 80.2921591, 0.325638556, -0.00909491031],
        [1.08115946, 2.67872855, -0.0019446761, 82.2507337, 0.456149966, 0.00923364344],
        [1.06864394, 2.70464551, -0.00209793986, 83.4698923, 0.395788406, -0.00251260641],
        [0.993054246, 2.83888423, -0.00277991124, 90.0331465, 0.651004673, -0.0200027208],
        [0.992050998, 3.42584708, -0.00292617372, 90.0630824, 0.648340038, -0.0336076805],
        [0.990524577, 4.81296934, -0.00378757857, 90.0846127, 0.741107384, -0.0439029391],
        [0.988680148, 5.99738121, -0.00451717463, 90.104646, 0.883485061, -0.041030785],
        [0.957624286, 5.9528988, 0.0102741306, 90.1327839, 4.19080459, 1.12928446],
        [0.942862105, 5.99368058, 0.0162024942, 90.1288679, 5.76538787, 1.62560793],
        [0.931969027, 5.99412693, 0.020619344, 90.1263482, 6.90261077, 1.98875845],
        [0.904773557, 6.00941729, 0.0313747872, 90.128735, 9.61483109, 2.83144875],
        [0.8858989, 6.01830162, 0.0381113604, 90.135911, 11.4508584, 3.34953358],
        [0.863540853, 6.02619691, 0.04487657, 90.1598001, 13.5181743, 3.88721021],
        [0.836463078, 6.03765479, 0.0528267196, 90.1967295, 15.9298717, 4.48186113],
        [0.804731182, 6.04627379, 0.0625195903, 90.2336098, 18.6990514, 5.14505668],
        [0.788296376, 6.05202369, 0.0675622892, 90.2479959, 20.1264906, 5.47138803],
        [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 21.6229248, 5.80874653],
        [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 22.1532161, 5.92642676],
        [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 22.1532161, 5.92642676], # Duplicate to keep length same? No, just copy
        [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 23.4927925, 6.21910882],
        [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 24.4180728, 6.42046395],
        [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 25.3815663, 6.631397],
        [0.708478913, 6.07645575, 0.094580537, 90.324703, 26.8427434, 6.95357494],
        [0.685450204, 6.08320452, 0.103688262, 90.3410972, 28.7527373, 7.37124273],
        [0.660511655, 6.09058578, 0.114647767, 90.341297, 30.8426719, 7.83842912],
        [0.631106763, 6.09898622, 0.127562353, 90.3399229, 33.2683226, 8.35622932],
        [0.60239252, 6.10718508, 0.141197449, 90.3187387, 35.6585372, 8.87446837],
        [0.574685996, 6.1149268, 0.154862481, 90.2890178, 37.982628, 9.36856002],
        [0.549279838, 6.12170874, 0.168073709, 90.2538308, 40.1148352, 9.81523017],
        [0.523487025, 6.12859076, 0.182443691, 90.2080219, 42.279737, 10.2799932],
        [0.498121776, 6.13520082, 0.197968844, 90.1482985, 44.4163048, 10.7608292],
        [0.476465121, 6.14082429, 0.2120163, 90.0844905, 46.2626926, 11.1825901],
        [0.456153588, 6.14607051, 0.226245551, 90.0101313, 48.012554, 11.601748],
        [0.433791775, 6.15190581, 0.243138907, 89.9157178, 49.954221, 12.084816],
        [0.404025852, 6.15902259, 0.266186906, 89.7734311, 52.548863, 12.7293228],
        [0.366005234, 6.16774759, 0.297031124, 89.5738271, 55.893128, 13.5723105],
        [0.321312876, 6.17691876, 0.335574819, 89.2922705, 59.8692011, 14.594953],
        [0.278427663, 6.18417441, 0.373123513, 88.9719858, 63.7515612, 15.5832162],
        [0.239534769, 6.19553929, 0.413872182, 88.6402231, 67.3322992, 16.6327601],
        [0.204472084, 6.19968082, 0.464675511, 88.3088859, 70.5963626, 17.9436724],
        [0.175394931, 6.21048495, 0.524148773, 88.0051881, 73.3241903, 19.4940659],
        [0.153758289, 6.22132345, 0.583751542, 87.7681793, 75.3731869, 21.0578654],
        [0.126442905, 6.23697046, 0.675335517, 87.4796317, 77.9661162, 23.4337293],
        [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.008173, 25.2702993],
        [0.106313411, 6.26635361, 0.836966776, 87.2610122, 79.9079131, 27.7208667],
        [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 80.6362714, 29.7408798],
        [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.0361104, 31.0067456],
        [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.451701, 32.3850431],
        [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 81.9245444, 34.2396643],
        [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.1005075, 35.3326516],
        [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.2502751, 36.313756],
        [0.0813272459, 6.31280445, 1.18563003, 86.8893308, 82.3738474, 37.1785229],
        [0.0795266549, 6.31701618, 1.22936759, 86.8298912, 82.5573864, 38.4051678],
        [0.0760640114, 6.32390774, 1.30814197, 86.730576, 82.9155599, 40.6273839],
        [0.0733345584, 6.32845988, 1.38417722, 86.6469757, 83.1880754, 42.7737559],
        [0.0728754398, 6.33436689, 1.51928934, 85.9570928, 83.321603, 47.2189208],
        [0.0740241462, 6.33651625, 1.67086132, 85.1992673, 83.4017941, 52.295749],
        [0.0709732742, 6.32973557, 1.82119648, 84.427344, 83.8384848, 57.4161351],
        [0.0689950542, 6.32296873, 1.89917235, 84.1030756, 84.0687341, 60.0778089],
        [0.0654533403, 6.31403661, 1.97470221, 83.8065972, 84.4296311, 62.6673579],
        [0.0598040781, 6.30238868, 2.04639195, 83.482155, 84.9583795, 65.1956567],
        [0.0532315605, 6.28848325, 2.11585078, 83.1684777, 85.5618308, 67.7111146],
        [0.0467194205, 6.27295621, 2.18521241, 82.8588875, 86.1559337, 70.2420111],
        [0.0211915399, 6.21382682, 2.3716358, 82.5672227, 88.3741932, 76.9350513],
        [0.0119033827, 6.16998589, 2.48828327, 83.7295105, 89.1215994, 80.3298032],
        [0.0114904543, 6.16373556, 2.50403291, 83.888837, 89.1782177, 80.7917836],
        [0.00847440903, 6.15720704, 2.52053877, 84.0621014, 89.3189944, 81.2648114],
        [0.0054345193, 6.12168354, 2.60523315, 84.9795728, 89.7896324, 83.7983903],
        [0.00500718906, 6.08778028, 2.67536568, 85.9496099, 89.8305811, 86.1593636],
        [0.00496310192, 6.07475445, 2.70089071, 86.2819807, 89.9205737, 87.0030543],
        [0.00447449792, 6.05605429, 2.7350364, 86.757467, 89.802935, 88.1996353],
        [0.00297530192, 6.02087628, 2.79766649, 87.459475, 90.1346094, 90.5955514],
        [0.000953058079, 5.9970912, 2.84427868, 87.6265515, 90.27883, 92.3854717],
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979],
    ], dtype=np.float64)

def moving_sofa_3d_path() -> np.ndarray:
    """
    Path surgery optimization:
    1. Start with seed path.
    2. Iteratively identify the pose with the minimum clearance (closest to boundary).
    3. Shift that pose (and neighbors) in the direction away from the boundary.
    4. Repeat until time limit.
    """
    t_start = time.time()
    deadline = t_start + 590.0  # Leave 40s margin for safety
    
    best_path = _seed_path()
    best_score = _evaluate_path_fast(best_path)
    
    # Grid for clearance calculation
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    def _get_min_clearance_pose(path):
        """
        Returns the index of the pose in the densified path that has the minimum clearance,
        and the direction to move away from the boundary at that pose.
        """
        start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
        full_path = np.vstack([start, path, end])
        densified = _densify_path(full_path)
        
        min_clearance = float('inf')
        worst_pose_idx = 0
        worst_grad = np.zeros(3)
        
        for i, p in enumerate(densified):
            R = _rotation_matrix(p[3], p[4], p[5])
            placed = points @ R.T + p[:3]
            sdf = _clearance_to_boundary(placed[:, 0], placed[:, 1], placed[:, 2])
            min_sdf = np.min(sdf)
            if min_sdf < min_clearance:
                min_clearance = min_sdf
                worst_pose_idx = i
                # Gradient of SDF at the worst point
                # SDF is min over points of SDF(point).
                # The gradient of the min is the gradient of the argmin point.
                # We need the gradient of the SDF w.r.t. the pose parameters.
                # This is complex. Let's just use a heuristic:
                # If the worst point is near x=0 (boundary between box 1 and 2), move x.
                # If near y=0 or y=1, move y.
                # If near z=0 or z=1, move z.
                
                # Find the worst point
                worst_pt_idx = np.argmin(sdf)
                px, py, pz = placed[worst_pt_idx]
                
                # Determine which boundary is closest
                # Check distance to specific planes
                d_x0 = abs(px) # if px near 0
                d_y0 = abs(py)
                d_y1 = abs(1-py)
                d_z0 = abs(pz)
                d_z1 = abs(1-pz)
                d_x1 = abs(1-px) # for box 2/3
                d_y5 = abs(5-py)
                d_y6 = abs(6-py)
                
                # This is too heuristic. Let's just try small random shifts in the direction of the gradient of the SDF.
                # Since we don't have the gradient easily, we will just try shifting the pose in the direction that increases the min SDF.
                # We can approximate the gradient by finite differences on the pose parameters.
                
                # For now, let's just return the index and a zero gradient, and use random search in the local area.
                # But the directive asks for a structural change.
                # Let's implement a simple gradient ascent on the clearance.
                
                # We will compute the gradient of the min clearance w.r.t. the pose parameters (x,y,z,yaw,pitch,roll)
                # using finite differences.
                eps = 0.01
                grad = np.zeros(6)
                for k in range(6):
                    p_plus = p.copy()
                    p_minus = p.copy()
                    if k < 3:
                        p_plus[k] += eps
                        p_minus[k] -= eps
                    else:
                        p_plus[k] += eps
                        p_minus[k] -= eps
                    
                    R_plus = _rotation_matrix(p_plus[3], p_plus[4], p_plus[5])
                    placed_plus = points @ R_plus.T + p_plus[:3]
                    sdf_plus = np.min(_clearance_to_boundary(placed_plus[:, 0], placed_plus[:, 1], placed_plus[:, 2]))
                    
                    R_minus = _rotation_matrix(p_minus[3], p_minus[4], p_minus[5])
                    placed_minus = points @ R_minus.T + p_minus[:3]
                    sdf_minus = np.min(_clearance_to_boundary(placed_minus[:, 0], placed_minus[:, 1], placed_minus[:, 2]))
                    
                    grad[k] = (sdf_plus - sdf_minus) / (2 * eps)
                
                worst_grad = grad
        
        # Map densified index back to original path index
        # This is tricky because densification inserts points.
        # We will just perturb the original path poses that are close to the worst pose in time.
        
        return worst_pose_idx, worst_grad, min_clearance

    # Optimization loop
    rng = np.random.default_rng(42)
    step_size = 0.05
    stagnation_count = 0
    max_stagnation = 50
    
    for i in range(1000):
        if time.time() > deadline:
            break
        
        # Evaluate current path
        current_score = _evaluate_path_fast(best_path)
        if current_score > best_score:
            best_score = current_score
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        if stagnation_count > max_stagnation:
            # Reset to best and increase step size
            step_size *= 1.5
            stagnation_count = 0
            continue
        
        # Find worst pose
        worst_idx, worst_grad, min_clear = _get_min_clearance_pose(best_path)
        
        # If min_clear is very negative, we are in trouble.
        # Move the pose in the direction of the gradient.
        # We need to map worst_idx (densified) to original path indices.
        # For simplicity, we will just perturb the entire path slightly in the direction of the gradient
        # if the gradient is significant.
        
        # Better: Perturb the specific poses that correspond to the worst region.
        # We will just do a random perturbation of a few poses, but biased by the gradient.
        
        n_perturb = rng.integers(1, 4)
        indices = rng.choice(len(best_path), size=n_perturb, replace=False)
        
        new_path = best_path.copy()
        for idx in indices:
            # Add gradient direction (if available) + random noise
            # The gradient is for the densified pose, which is an interpolation.
            # We will just add a small random perturbation, but scaled by the step size.
            new_path[idx, 0] += rng.normal(0, step_size)
            new_path[idx, 1] += rng.normal(0, step_size)
            new_path[idx, 2] += rng.normal(0, step_size)
            new_path[idx, 3] += rng.normal(0, step_size * 5)
            new_path[idx, 4] += rng.normal(0, step_size * 5)
            new_path[idx, 5] += rng.normal(0, step_size * 5)
            
            # If we have a gradient, add it to the perturbation
            # This is a rough approximation
            if np.linalg.norm(worst_grad) > 1e-6:
                # Scale gradient to be a reasonable step
                grad_scale = step_size / (np.linalg.norm(worst_grad) + 1e-6)
                new_path[idx] += worst_grad * grad_scale * 0.1
        
        new_score = _evaluate_path_fast(new_path)
        
        if new_score > best_score:
            best_path = new_path
            best_score = new_score
            step_size *= 0.9
        else:
            step_size *= 1.1
    
    return best_path

# EVOLVE-BLOCK-END
