import math
import time
import random
import numpy as np

N = 11
SQRT3 = math.sqrt(3)
DEADLINE_MARGIN = 10.0  # seconds before hard kill


def _hex_apothem(s):
    return s * SQRT3 / 2.0


def _hex_support(nx, ny, s):
    """Support function of regular hexagon with circumradius s in direction (nx,ny).
    Hexagon vertices at angles 0, 60, 120, 180, 240, 300 deg (pointy-right).
    Edges have outward normals at 30, 90, 150, 210, 270, 330 deg.
    Support = s * max over 6 vertices of dot(vertex_unit, dir)
    = s * max(cos(a_i - theta) for i in 0..5) where theta = atan2(ny,nx)
    Equivalent: s * (cos(theta mod 60) + sin(theta mod 60)*sqrt(3)) / (sqrt(3)) ... 
    Actually for regular hex with circumradius s:
    h(theta) = s * cos(theta - k*60) for theta in [k*60 - 30, k*60 + 30]
    So h(theta) = s * cos(theta mod 60 - 30) ... let me just compute directly.
    """
    theta = math.atan2(ny, nx)
    # Normalize to [0, 2pi)
    theta = theta % (2 * math.pi)
    # The hexagon has 6-fold symmetry, period = pi/3
    # Within one sector [0, pi/3], the supporting vertex is at angle 0
    # h(theta) = s * cos(theta) for theta in [-pi/6, pi/6] relative to vertex
    # General: fold theta into [0, pi/3], then h = s*cos(theta_folded)
    # where theta_folded is angle from nearest vertex
    # Vertex angles: 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
    # For theta in [0, pi/6]: nearest vertex at 0, h = s*cos(theta)
    # For theta in [pi/6, pi/2]: nearest vertex at pi/3, h = s*cos(theta - pi/3)
    # etc.
    # Fold: t = theta mod (pi/3), then if t > pi/6: t = pi/3 - t
    t = theta % (math.pi / 3.0)
    if t > math.pi / 6.0:
        t = math.pi / 3.0 - t
    return s * math.cos(t)


def _min_enclosing_hex_side(centers):
    """Given N center points, find minimum side length S and rotation angle phi
    of a regular hexagon (circumradius S) that contains all points.
    Returns (S, phi_deg).
    
    For a given rotation phi, the required circumradius is:
    S(phi) = max_i [max_j h_i_j(phi)]
    where for each point i and each of 6 edge-normals j,
    h_i_j = dot(p_i, n_j) / h(n_j) ... 
    
    Actually: point p is inside hex(center=0, circumradius=S, rotation=phi) iff
    for all 6 edge normals n_k(phi): dot(p, n_k) <= S * h_edge(n_k)
    where h_edge is the apothem = S * sqrt(3)/2.
    
    Wait, let me re-derive. Regular hexagon with circumradius R (vertex distance = R = S):
    Apothem a = R * sqrt(3)/2.
    The 6 edge outward unit normals are at angles phi + 30, phi + 90, phi + 150, phi + 210, phi + 270, phi + 330 degrees.
    Point p is inside iff for all k: dot(p, n_k) <= a = R * sqrt(3)/2.
    So R >= max_k [ dot(p, n_k) / (sqrt(3)/2) ] for each p.
    And R must satisfy this for ALL p.
    So R(phi) = max_p max_k [ dot(p, n_k(phi)) / (sqrt(3)/2) ]
             = (2/sqrt(3)) * max_p max_k dot(p, n_k(phi))
    
    But we also need dot(p, n_k) >= -a for the opposite edges, which is automatically
    satisfied if we center the hex at the centroid... actually no, we can choose the center too.
    
    Better: For a given phi, the minimum R hexagon containing all points has its center
    at some optimal position. This is a 2D linear programming problem for each phi.
    
    For simplicity and speed, I'll fix the hexagon center at the centroid of the points
    and just optimize phi. This is an approximation but often near-optimal for symmetric packings.
    
    Actually, let's do it properly: for a given phi, we need to find center c and R such that
    for all p_i and all k: -a <= dot(p_i - c, n_k) <= a
    where a = R * sqrt(3)/2.
    This means: for all i,k: dot(p_i, n_k) - dot(c, n_k) <= a
    and: dot(c, n_k) - dot(p_i, n_k) <= a
    So: max_i dot(p_i, n_k) - min_i dot(p_i, n_k) <= 2a = R*sqrt(3)
    for all k.
    And the center c can be chosen as: dot(c, n_k) = (max_i dot(p_i,n_k) + min_i dot(p_i,n_k))/2
    for each k. But c must be consistent across all k.
    
    For 2D with 6 directions, this is feasible iff the constraints are consistent.
    The minimum R for given phi is:
    R(phi) = max_k [ (max_i dot(p_i, n_k(phi)) - min_i dot(p_i, n_k(phi))) / sqrt(3) ]
    
    This is exact! The center is determined by the midpoints.
    """
    pts = np.array(centers)  # (N, 2)
    
    def R_for_phi(phi):
        # 6 edge normals
        base_angles = np.radians([30, 90, 150, 210, 270, 330]) + phi
        normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])  # (6,2)
        # Project all points onto each normal
        projs = pts @ normals.T  # (N, 6)
        ranges = projs.max(axis=0) - projs.min(axis=0)  # (6,)
        return ranges.max() / SQRT3
    
    # Optimize over phi in [0, 60 degrees) by 60-fold symmetry
    # Use golden section search
    a_phi = 0.0
    b_phi = math.pi / 3.0  # 60 degrees
    gr = (math.sqrt(5) - 1) / 2.0  # golden ratio conjugate
    
    c_phi = b_phi - gr * (b_phi - a_phi)
    d_phi = a_phi + gr * (b_phi - a_phi)
    f_c = R_for_phi(c_phi)
    f_d = R_for_phi(d_phi)
    
    for _ in range(60):
        if f_c < f_d:
            b_phi = d_phi
            d_phi = c_phi
            f_d = f_c
            c_phi = b_phi - gr * (b_phi - a_phi)
            f_c = R_for_phi(c_phi)
        else:
            a_phi = c_phi
            c_phi = d_phi
            f_c = f_d
            d_phi = a_phi + gr * (b_phi - a_phi)
            f_d = R_for_phi(d_phi)
    
    best_phi = (a_phi + b_phi) / 2.0
    best_R = R_for_phi(best_phi)
    return best_R, math.degrees(best_phi)


def _reconstruct_from_adjacency(adj_matrix, base_spacing):
    """Given an adjacency matrix (N x N boolean) and base spacing,
    reconstruct positions using force-directed relaxation.
    Adjacent hexagons should be at distance 2*apothem = sqrt(3) apart (touching).
    Non-adjacent should be >= sqrt(3) apart.
    Returns list of (x, y) centers.
    """
    # Initialize with random positions in a disk
    np.random.seed(42)
    positions = np.zeros((N, 2))
    for i in range(N):
        r = base_spacing * math.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * math.pi)
        positions[i] = [r * math.cos(theta), r * math.sin(theta)]
    
    target_dist = SQRT3  # distance between centers of touching unit hexagons
    iterations = 200
    step = 0.1
    
    for it in range(iterations):
        forces = np.zeros((N, 2))
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-9:
                    dx, dy = 0.01, 0.01
                    dist = math.sqrt(dx * dx + dy * dy)
                ux, uy = dx / dist, dy / dist
                
                if adj_matrix[i][j]:
                    # Want distance = target_dist
                    f = (dist - target_dist)
                else:
                    # Want distance >= target_dist, push apart if too close
                    if dist < target_dist:
                        f = (dist - target_dist) * 2.0  # stronger repulsion
                    else:
                        f = 0.0
                
                forces[i][0] -= f * ux
                forces[i][1] -= f * uy
                forces[j][0] += f * ux
                forces[j][1] += f * uy
        
        positions += step * forces
        step *= 0.995
    
    return positions


def _evaluate_adjacency(adj_matrix):
    """Given adjacency, reconstruct positions, compute min enclosing hex side.
    Also check validity (no overlaps). Returns (side_length, valid, positions).
    """
    positions = _reconstruct_from_adjacency(adj_matrix, 2.0)
    
    # Check validity: all pairwise distances >= sqrt(3) - epsilon
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[j][0] - positions[i][0]
            dy = positions[j][1] - positions[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < SQRT3 - 0.01:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return float('inf'), False, positions
    
    side, angle = _min_enclosing_hex_side(positions.tolist())
    return side, True, positions


def _adjacency_to_list(adj_matrix):
    """Convert NxN bool matrix to sorted list of (i,j) edges."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj_matrix[i][j]:
                edges.append((i, j))
    return edges


def _list_to_adjacency(edges):
    mat = [[False] * N for _ in range(N)]
    for i, j in edges:
        mat[i][j] = True
        mat[j][i] = True
    return mat


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    random.seed(42)
    np.random.seed(42)
    
    # Initial adjacency: hexagonal lattice structure for 11 hexagons
    # Center + ring 1 (6) + 4 from ring 2
    # Hex grid coordinates: use axial (q, r)
    # Center: (0,0)
    # Ring 1: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # Ring 2: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,1)
    # Pick 4 from ring 2: (2,0), (0,-2), (-2,0), (1,1)
    
    # Axial to pixel: x = 1.5*q, y = sqrt(3)*(r + q/2)
    # But for unit hexagons, center-to-center distance for neighbors is sqrt(3).
    # In axial coords, neighbors are 1 step apart.
    # Pixel: x = 1.5*q, y = sqrt(3)*(r + 0.5*q)
    # Distance between (0,0) and (1,0): x=1.5, y=sqrt(3)/2, dist = sqrt(2.25+0.75)=sqrt(3). Good.
    
    hex_coords = [
        (0, 0),       # 0: center
        (1, 0),       # 1
        (1, -1),      # 2
        (0, -1),      # 3
        (-1, 0),      # 4
        (-1, 1),      # 5
        (0, 1),       # 6
        (2, 0),       # 7: ring 2
        (0, -2),      # 8: ring 2
        (-2, 0),      # 9: ring 2
        (1, 1),       # 10: ring 2
    ]
    
    # Build initial adjacency from hex grid (neighbors in axial coords)
    # Neighbors of (q,r): (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    neighbor_deltas = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
    coord_to_idx = {c: i for i, c in enumerate(hex_coords)}
    
    adj = [[False] * N for _ in range(N)]
    for i, (q, r) in enumerate(hex_coords):
        for dq, dr in neighbor_deltas:
            j_coord = (q + dq, r + dr)
            if j_coord in coord_to_idx:
                j = coord_to_idx[j_coord]
                adj[i][j] = True
                adj[j][i] = True
    
    # Evaluate initial
    best_side, best_valid, best_positions = _evaluate_adjacency(adj)
    if not best_valid:
        best_side = float('inf')
    
    best_adj = [row[:] for row in adj]
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.997
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_edges_total = len(all_edges)  # 55
    
    current_adj = [row[:] for row in adj]
    current_side = best_side
    current_positions = best_positions
    
    eval_count = 0
    max_evals = 3000  # limit evaluations to stay within time
    
    while time.time() < deadline and eval_count < max_evals:
        # Propose: flip a random edge
        edge = random.choice(all_edges)
        i, j = edge
        current_adj[i][j] = not current_adj[i][j]
        current_adj[j][i] = current_adj[i][j]
        
        new_side, new_valid, new_positions = _evaluate_adjacency(current_adj)
        eval_count += 1
        
        if not new_valid:
            new_side = float('inf')
        
        if new_side < current_side:
            # Accept
            current_side = new_side
            current_positions = new_positions
            if new_side < best_side:
                best_side = new_side
                best_adj = [row[:] for row in current_adj]
                best_positions = new_positions
        else:
            # Metropolis criterion
            delta = new_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_side = new_side
                current_positions = new_positions
            else:
                # Reject: revert
                current_adj[i][j] = not current_adj[i][j]
                current_adj[j][i] = not current_adj[j][i]
        
        T = max(T_min, T * cooling)
        
        # Periodic check
        if eval_count % 500 == 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 20:
                break
    
    # Final: use best found
    if best_side == float('inf') or not best_valid:
        # Fallback: use initial lattice
        best_side, best_valid, best_positions = _evaluate_adjacency(best_adj)
        if not best_valid:
            # Absolute fallback: spread out
            best_positions = np.array([[0,0],[2,0],[1,2],[-1,2],[-2,0],[-1,-2],[1,-2],[3,0],[0,3],[-3,0],[0,-3]], dtype=float)
            best_side = 6.0
    
    # Compute final outer hexagon
    side, angle_deg = _min_enclosing_hex_side(best_positions.tolist())
    
    # Center of outer hex: centroid of inner positions
    cx = float(np.mean(best_positions[:, 0]))
    cy = float(np.mean(best_positions[:, 1]))
    
    # Build inner list (all angles 0 for simplicity, since we only optimized positions)
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), 0.0])
    
    outer_center = (cx, cy)
    outer_side = side
    outer_angle = angle_deg
    
    return inner, outer_center, outer_side, outer_angle


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Time: {time.time()-t0:.1f}s")
    print(f"Outer side: {result[2]:.4f}")
    print(f"Outer center: {result[1]}")
    print(f"Outer angle: {result[3]:.2f}")
    for i, h in enumerate(result[0]):
        print(f"  Inner {i}: ({h[0]:.3f}, {h[1]:.3f}, {h[2]:.1f}°)")
