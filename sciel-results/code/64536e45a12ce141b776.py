import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # 11 hexagons in a flower pattern (cluster at reference orientation)
    eps = 1e-12
    centers_ref = np.array([[0.0, 0.0]])

    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers_ref = np.vstack([centers_ref, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers_ref = np.vstack([centers_ref, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers_ref)

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    # Pre-compute all 360 vertices of the rigid cluster in reference orientation
    all_vertices_ref = []
    for i in range(n_hex):
        for v in base_vertices:
            all_vertices_ref.append(v + centers_ref[i])
    all_vertices_ref = np.array(all_vertices_ref)  # shape (360, 2)

    def compute_side_for_relative_angle(delta):
        """
        Compute the minimum enclosing hexagon side length for a given
        relative angle between cluster and container.
        
        delta = cluster_angle - container_angle.
        We set container_angle = 0, so cluster_angle = delta.
        
        The cluster vertices are rotated by delta, and we check against
        container normals at angles 30 + 60k degrees.
        """
        c, s = np.cos(delta), np.sin(delta)
        rot = np.array([[c, -s], [s, c]])
        rotated_vertices = all_vertices_ref @ rot.T  # (360, 2)

        # Container normals: at angles 30 + k*60 degrees (container at angle 0)
        normal_angles = np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)

        # Project all vertices onto all normals
        dots = rotated_vertices @ normals.T  # (360, 6)

        max_proj = np.max(np.abs(dots))
        return 2.0 * max_proj / np.sqrt(3)

    # Phase 1: Coarse scan of relative angle
    # Due to 6-fold symmetry, the function has period pi/6
    n_coarse = 2000
    angles = np.linspace(0, np.pi / 6, n_coarse, endpoint=False)
    
    best_side = np.inf
    best_delta = 0.0
    
    for delta in angles:
        side = compute_side_for_relative_angle(delta)
        if side < best_side:
            best_side = side
            best_delta = delta

    # Phase 2: Golden-section refinement
    a = 0.0
    b = np.pi / 6
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = compute_side_for_relative_angle(c)
    fd = compute_side_for_relative_angle(d)
    
    for _ in range(80):
        if time.time() > deadline - 30:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = compute_side_for_relative_angle(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = compute_side_for_relative_angle(d)
        
        if abs(b - a) < 1e-15:
            break
    
    best_delta = (a + b) / 2
    best_side = compute_side_for_relative_angle(best_delta)

    # Phase 3: Finer refinement
    a = best_delta - 1e-8
    b = best_delta + 1e-8
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = compute_side_for_relative_angle(c)
    fd = compute_side_for_relative_angle(d)
    
    for _ in range(60):
        if time.time() > deadline - 20:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = compute_side_for_relative_angle(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = compute_side_for_relative_angle(d)
        
        if abs(b - a) < 1e-16:
            break
    
    best_delta = (a + b) / 2
    best_side = compute_side_for_relative_angle(best_delta)

    # Phase 4: Rigorous verification
    c_rot, s_rot = np.cos(best_delta), np.sin(best_delta)
    rot = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
    rotated_vertices = all_vertices_ref @ rot.T
    
    normal_angles = np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = rotated_vertices @ normals.T
    max_proj = np.max(np.abs(dots))
    verified_side = 2.0 * max_proj / np.sqrt(3)
    
    side = verified_side + 1e-9

    # Build inner hexagon list
    # Rotate all centers by best_delta
    inner = []
    for i in range(n_hex):
        cx, cy = centers_ref[i]
        rx = cx * c_rot - cy * s_rot
        ry = cx * s_rot + cy * c_rot
        inner.append([rx, ry, np.rad2deg(best_delta)])

    return inner, (0.0, 0.0), side, 0.0  # Container at 0 degrees (flat-top)

if __name__ == "__main__":
    construct()
