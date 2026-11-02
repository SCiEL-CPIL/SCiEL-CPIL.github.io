import math
import time
import numpy as np
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Parameters for Simulated Annealing
    # State vector:
    # [x1, y1, a1, ..., xN, yN, aN, cx, cy, side, oang]
    # Total dimension: 3*N + 4 = 37 for N=11
    
    dim = 3 * N + 4
    
    # Initial state:
    # Place inner hexagons in a rough hexagonal cluster around origin
    # This is a good starting point for N=11 (center + 6 neighbors + 4 more)
    # Standard hex grid spacing for touching hexagons is sqrt(3) in "flat" directions
    # but let's just use a loose initial configuration to avoid immediate heavy penalties.
    
    def init_state():
        state = np.zeros(dim)
        # Inner hexagons
        # Center
        state[0] = 0.0
        state[1] = 0.0
        state[2] = 0.0
        
        # Neighbors in hex grid (distance sqrt(3) ~ 1.732)
        # Directions: 0, 60, 120, 180, 240, 300
        dist = math.sqrt(3) * 1.1 # Slightly separated to start
        for i in range(1, 7):
            angle = i * math.pi / 3
            idx = i * 3
            state[idx] = dist * math.cos(angle)
            state[idx+1] = dist * math.sin(angle)
            state[idx+2] = 0.0
            
        # Remaining 4 hexagons (indices 7, 8, 9, 10)
        # Place them in the second ring roughly
        # Second ring distance is 2*dist? No, 2*sqrt(3) is too far.
        # Let's just scatter them randomly in a circle of radius 3
        for i in range(7, N):
            angle = random.uniform(0, 2*math.pi)
            r = random.uniform(2.0, 3.5)
            idx = i * 3
            state[idx] = r * math.cos(angle)
            state[idx+1] = r * math.sin(angle)
            state[idx+2] = random.uniform(0, 60) # Random initial rotation
        
        # Outer hexagon
        # Center at centroid of inner
        cx = np.mean(state[0:3*N:3])
        cy = np.mean(state[1:3*N:3])
        # Side: large enough to contain everything initially
        max_r = max(math.sqrt(state[i*3]**2 + state[i*3+1]**2) for i in range(N))
        
        state[3*N] = cx
        state[3*N+1] = cy
        state[3*N+2] = max_r + 2.0 # Side length
        state[3*N+3] = 0.0 # Outer angle
        
        return state

    def calculate_penalty(state):
        """
        Calculates the total penalty for a given state.
        Penalty = Pairwise Collision Penalty + Outer Containment Penalty + Side Length Penalty
        """
        penalty = 0.0
        
        # Extract inner hexagons
        inner_pos = []
        inner_angles = []
        for i in range(N):
            x = state[i*3]
            y = state[i*3+1]
            a = state[i*3+2]
            inner_pos.append((x, y))
            inner_angles.append(a)
            
        # Extract outer hexagon
        ocx = state[3*N]
        ocy = state[3*N+1]
        oside = state[3*N+2]
        oang = state[3*N+3]
        
        # 1. Pairwise Collision Penalty
        # For two convex polygons A and B, they are disjoint iff 
        # 0 is not in the Minkowski sum of A and -B.
        # Equivalently, if the vector d = c_B - c_A is outside the Minkowski sum of A and B (reflected).
        # For regular hexagons, we can use the support function.
        # Two hexagons overlap if for ALL directions u, h_A(u) + h_B(-u) < |d|? No.
        # They overlap if d is inside the Minkowski sum of A and -B.
        # Let's use a simpler geometric check for regular hexagons.
        # Two regular hexagons of side 1 with centers d apart and relative angle delta.
        # The minimum distance to separate them depends on the orientation.
        # A robust differentiable penalty:
        # For each pair, calculate the minimum distance between boundaries? Expensive.
        # Alternative: Check if centers are too close.
        # The "exclusion zone" of a hexagon relative to another is its Minkowski sum with the other.
        # For aligned hexagons, Minkowski sum is a hexagon of side 2.
        # With rotation, it's more complex.
        
        # Let's use a vertex-based penetration depth approximation or a simplified support function check.
        # A good proxy: Check if the distance between centers is less than the sum of apothems in the direction of the center vector.
        # But this fails for vertex-vertex contacts.
        
        # Better approach for differentiable penalty:
        # Calculate the maximum "overlap" in 3 principal directions? No, hexagons have 6 normals.
        
        # Let's use the fact that two convex polygons intersect if there is no separating axis.
        # For regular hexagons, the candidate separating axes are the normals of the sides of both hexagons.
        # Total 12 axes (6 from A, 6 from B).
        # For each axis, project both hexagons. If projections overlap for all axes, they intersect.
        # Penalty = sum of squared overlaps for all axes where they overlap.
        
        # Precompute cos/sin for inner angles
        # This is still O(N^2 * 12) per evaluation. N=11 -> 55 pairs * 12 = 660 ops. Very fast.
        
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ai = inner_pos[i][0], inner_pos[i][1], inner_angles[i]
                xj, yj, aj = inner_pos[j][0], inner_pos[j][1], inner_angles[j]
                
                dx = xj - xi
                dy = yj - yi
                
                # Axes to check:
                # Normals of hex i: ai + k*pi/3 + pi/2 (k=0..5)
                # Normals of hex j: aj + k*pi/3 + pi/2 (k=0..5)
                # Actually, normals are at angles: angle + pi/6 + k*pi/3?
                # Vertex at 0: (1,0). Normal to side (1,0)-(0.5, sqrt(3)/2) is at 30 deg?
                # Side vector: (-0.5, sqrt(3)/2). Normal: (sqrt(3)/2, 0.5) -> 30 deg.
                # So normals are at angle + 30, 90, 150, 210, 270, 330.
                
                max_overlap = 0.0
                for k in range(6):
                    # Axis from hex i
                    theta_i = math.radians(ai) + math.pi/6 + k * math.pi/3
                    # Axis from hex j
                    theta_j = math.radians(aj) + math.pi/6 + k * math.pi/3
                    
                    # We can just check the 6 axes of i and 6 of j.
                    # To save time, check 6 axes of i. Due to symmetry, if they separate on any of i's axes or j's axes, they are separated.
                    # If they overlap on all 12, they intersect.
                    # Let's check 6 axes of i first.
                    
                    nx_i = math.cos(theta_i)
                    ny_i = math.sin(theta_i)
                    
                    # Project centers
                    proj_ci = 0.0
                    proj_cj = dx * nx_i + dy * ny_i
                    
                    # Project extents
                    # Extent of a unit hexagon along normal at angle phi:
                    # It is the max dot product of vertices with the normal.
                    # For a regular hexagon, the support function h(u) is:
                    # h(u) = cos(pi/6) * |u|? No.
                    # h(u) = 1 if u is vertex direction, sqrt(3)/2 if u is normal direction.
                    # General formula: h(u) = max_{v in V} dot(v, u).
                    # For unit hex, vertices at 0, 60, 120...
                    # Let's calculate h for hex i and hex j along this axis.
                    # Since they are regular, h_i(u) = h_j(u) if we account for rotation?
                    # No, h depends on the orientation of the hex relative to u.
                    # h_hex(u, angle) = h_unit(u - angle)
                    
                    # We can precompute h for the 6 normal directions of a unit hex.
                    # But u varies.
                    # Let's just compute the 6 vertices of i and j? No, too slow.
                    
                    # Analytical support function for regular hexagon side 1:
                    # h(theta) = (2/3) * (1 + cos(2 * (theta - angle)))? No.
                    # Let's use the fact that h(theta) is piecewise cosine.
                    # In the sector between two adjacent normal directions (width 60 deg),
                    # the support is determined by the vertex in the middle.
                    # Actually, simpler:
                    # The width of a regular hexagon in any direction is 2 * h(direction).
                    # h(theta) = cos(pi/6) / cos((theta - nearest_normal) mod pi/3)?
                    # Let's stick to the vertex calculation for accuracy, it's only 6 dots.
                    
                    # Vertices of unit hex at angle 0:
                    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
                    # To get vertices for angle a, rotate.
                    # Instead of rotating vertices, rotate the normal?
                    # dot(R(v), u) = dot(v, R(-u)).
                    # So h_hex(u, a) = h_unit(R(-a) u).
                    
                    # Let's compute h_unit for a vector w.
                    # w = (wx, wy).
                    # h_unit(w) = max( wx, 0.5wx + sqrt(3)/2 wy, -0.5wx + sqrt(3)/2 wy, -wx, -0.5wx - sqrt(3)/2 wy, 0.5wx - sqrt(3)/2 wy )
                    
                    # For hex i:
                    # Rotate normal by -ai
                    c_i = math.cos(-math.radians(ai))
                    s_i = math.sin(-math.radians(ai))
                    wx_i = nx_i * c_i - ny_i * s_i
                    wy_i = nx_i * s_i + ny_i * c_i
                    
                    # Calculate h_i
                    # Using the 6 vertices
                    # v1=(1,0), v2=(.5, .866), v3=(-.5, .866), v4=(-1,0), v5=(-.5, -.866), v6=(.5, -.866)
                    h_i = max(
                        wx_i,
                        0.5*wx_i + 0.8660254*wy_i,
                        -0.5*wx_i + 0.8660254*wy_i,
                        -wx_i,
                        -0.5*wx_i - 0.8660254*wy_i,
                        0.5*wx_i - 0.8660254*wy_i
                    )
                    
                    # For hex j:
                    c_j = math.cos(-math.radians(aj))
                    s_j = math.sin(-math.radians(aj))
                    wx_j = nx_i * c_j - ny_i * s_j
                    wy_j = nx_i * s_j + ny_i * c_j
                    
                    h_j = max(
                        wx_j,
                        0.5*wx_j + 0.8660254*wy_j,
                        -0.5*wx_j + 0.8660254*wy_j,
                        -wx_j,
                        -0.5*wx_j - 0.8660254*wy_j,
                        0.5*wx_j - 0.8660254*wy_j
                    )
                    
                    # Overlap in this direction
                    # Distance between projections of centers is proj_cj - proj_ci (since ci is at 0 in local frame)
                    # Wait, proj_ci is 0 if we treat ci as origin?
                    # Yes, we computed proj_cj as dot(dx, dy, n). proj_ci is 0.
                    # So the gap is (proj_cj - h_j) - (0 + h_i)?
                    # No.
                    # Interval for i: [-h_i, h_i]
                    # Interval for j: [proj_cj - h_j, proj_cj + h_j]
                    # They overlap if proj_cj + h_j > -h_i AND proj_cj - h_j < h_i
                    # i.e., |proj_cj| < h_i + h_j
                    overlap = (h_i + h_j) - abs(proj_cj)
                    if overlap > max_overlap:
                        max_overlap = overlap
                    
                    # Optimization: If we find a separating axis (overlap < 0), we can break?
                    # No, we need the SUM of overlaps for the penalty to be smooth.
                    # But if they are separated, penalty is 0 for this pair.
                    # If we just take the max overlap, it's a valid measure of "how deep" they are.
                    # Using max overlap is smoother than sum?
                    # Actually, if they are separated, max_overlap is negative.
                    # We want penalty 0 if separated.
                    # So penalty = max(0, max_overlap)^2.
                    
                # We only checked 6 axes of hex i. Is that sufficient?
                # The separating axis theorem says we must check all normals of both polygons.
                # If they are separated by a normal of hex j, we missed it.
                # However, for regular hexagons, the set of normals is dense enough?
                # No, we must check j's normals too.
                # Let's add a check for j's normals if max_overlap > 0.
                if max_overlap > 0:
                    for k in range(6):
                        theta_j = math.radians(aj) + math.pi/6 + k * math.pi/3
                        nx_j = math.cos(theta_j)
                        ny_j = math.sin(theta_j)
                        
                        proj_cj = dx * nx_j + dy * ny_j
                        
                        c_i = math.cos(-math.radians(ai))
                        s_i = math.sin(-math.radians(ai))
                        wx_i = nx_j * c_i - ny_j * s_i
                        wy_i = nx_j * s_i + ny_j * c_i
                        h_i = max(
                            wx_i, 0.5*wx_i + 0.8660254*wy_i, -0.5*wx_i + 0.8660254*wy_i,
                            -wx_i, -0.5*wx_i - 0.8660254*wy_i, 0.5*wx_i - 0.8660254*wy_i
                        )
                        
                        c_j = math.cos(-math.radians(aj))
                        s_j = math.sin(-math.radians(aj))
                        wx_j = nx_j * c_j - ny_j * s_j
                        wy_j = nx_j * s_j + ny_j * c_j
                        h_j = max(
                            wx_j, 0.5*wx_j + 0.8660254*wy_j, -0.5*wx_j + 0.8660254*wy_j,
                            -wx_j, -0.5*wx_j - 0.8660254*wy_j, 0.5*wx_j - 0.8660254*wy_j
                        )
                        
                        overlap = (h_i + h_j) - abs(proj_cj)
                        if overlap > max_overlap:
                            max_overlap = overlap
                
                if max_overlap > 0:
                    penalty += max_overlap ** 2
        
        # 2. Outer Containment Penalty
        # All inner vertices must be inside the outer hexagon.
        # Outer hexagon: center (ocx, ocy), side oside, angle oang.
        # Apothem a_out = oside * sqrt(3) / 2.
        
        # For each inner hex, check its 6 vertices.
        # If a vertex is outside, add penalty.
        # "Outside" means projection on any normal > apothem.
        
        # We can vectorize this slightly or just loop.
        # N=11, 6 verts = 66 vertices.
        
        # Precompute outer normals
        # oang is in degrees
        oang_rad = math.radians(oang)
        # Normals at oang + 30, 90, 150...
        # Actually, if angle is defined as rotation of the hexagon,
        # the normals are at oang + pi/6 + k*pi/3.
        
        n_out = []
        for k in range(3): # 3 unique normal directions (opposite are covered by abs)
            theta = oang_rad + math.pi/6 + k * math.pi/3
            n_out.append((math.cos(theta), math.sin(theta)))
        
        apothem = oside * math.sqrt(3) / 2.0
        
        for i in range(N):
            xi, yi = inner_pos[i]
            ai = inner_angles[i]
            ai_rad = math.radians(ai)
            
            # Vertices of inner hex i
            # (1,0) rotated by ai, etc.
            # We can generate the 6 vertices
            for v_idx in range(6):
                theta_v = ai_rad + v_idx * math.pi / 3
                vx = xi + math.cos(theta_v)
                vy = yi + math.sin(theta_v)
                
                # Check against outer hex
                dx = vx - ocx
                dy = vy - ocy
                
                for nx, ny in n_out:
                    proj = dx * nx + dy * ny
                    if proj > apothem:
                        penalty += (proj - apothem) ** 2
                    elif proj < -apothem:
                        penalty += (-apothem - proj) ** 2
        
        # 3. Side Length Penalty
        # We want to minimize oside.
        # Add a small penalty for oside to encourage shrinking.
        # But if containment is not satisfied, this will be dominated by containment penalty.
        penalty += 0.01 * oside
        
        return penalty

    # Simulated Annealing
    best_state = init_state()
    best_penalty = calculate_penalty(best_state)
    
    current_state = best_state.copy()
    current_penalty = best_penalty
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.999
    
    # Estimate max steps
    # 145s / (time per step)
    # Time per step: penalty calc + neighbor gen + accept.
    # Penalty calc is ~1ms?
    # Let's run for as long as possible.
    
    while time.time() < deadline:
        # Generate neighbor
        # Perturb a random subset of variables
        # Or perturb all variables with small noise?
        # Perturbing all is better for global search.
        # Scale: 
        # Positions: 0.5
        # Angles: 10 deg
        # Outer center: 0.5
        # Outer side: 0.1
        # Outer angle: 10 deg
        
        new_state = current_state.copy()
        
        # Add noise
        # Scale factors
        scale = np.zeros(dim)
        for i in range(N):
            scale[i*3] = 0.5
            scale[i*3+1] = 0.5
            scale[i*3+2] = 10.0
        scale[3*N] = 0.5
        scale[3*N+1] = 0.5
        scale[3*N+2] = 0.2
        scale[3*N+3] = 10.0
        
        noise = np.random.normal(0, 1, dim) * scale * T
        # As T decreases, noise decreases.
        # But we want noise to be proportional to T?
        # Standard SA: x_new = x_old + delta, where delta is drawn from distribution with width proportional to T.
        # Or just fixed width and accept based on T?
        # Let's use fixed width and adjust acceptance.
        # Or scale width by T? If T is small, we want small steps.
        # Let's scale width by T.
        
        new_state += noise
        
        new_penalty = calculate_penalty(new_state)
        
        # Acceptance
        if new_penalty < current_penalty:
            current_state = new_state
            current_penalty = new_penalty
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_state = current_state.copy()
        else:
            # Metropolis criterion
            # delta = new_penalty - current_penalty > 0
            # Accept with prob exp(-delta / T)
            delta = new_penalty - current_penalty
            if delta < 1e-10:
                accept = True
            else:
                # Prevent overflow in exp
                if delta / T > 100:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_state = new_state
                current_penalty = new_penalty
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Reheat?
            # For now, just keep going with small T.
            
    # Final result
    # Extract best_state
    inner_rows = []
    for i in range(N):
        x = best_state[i*3]
        y = best_state[i*3+1]
        a = best_state[i*3+2]
        inner_rows.append([x, y, a])
    
    ocx = best_state[3*N]
    ocy = best_state[3*N+1]
    oside = best_state[3*N+2]
    oang = best_state[3*N+3]
    
    # Ensure oside is positive
    if oside < 0:
        oside = 0.0
        
    return inner_rows, (ocx, ocy), oside, oang

if __name__ == "__main__":
    r = construct()
    print(r)
