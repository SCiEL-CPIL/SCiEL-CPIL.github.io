import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def min_distance_polygons(A, B):
        """
        Approximate minimum distance between two convex polygons.
        Returns negative value if they overlap (penalty).
        We check distance between all vertices of A to edges of B, and vice versa.
        For 6x6 vertices, this is 36 edge-vertex pairs + 36 vertex-vertex pairs.
        """
        min_dist = float('inf')
        
        # Vertex of A to Edge of B
        for va in A:
            for i in range(6):
                vb1 = B[i]
                vb2 = B[(i+1)%6]
                # Project va onto line vb1-vb2
                vab = vb2 - vb1
                len_sq = np.dot(vab, vab)
                if len_sq < 1e-9:
                    continue
                t = np.dot(va - vb1, vab) / len_sq
                t = max(0, min(1, t))
                closest = vb1 + t * vab
                dist = np.linalg.norm(va - closest)
                if dist < min_dist:
                    min_dist = dist
        
        # Vertex of B to Edge of A
        for vb in B:
            for i in range(6):
                va1 = A[i]
                va2 = A[(i+1)%6]
                vab = va2 - va1
                len_sq = np.dot(vab, vab)
                if len_sq < 1e-9:
                    continue
                t = np.dot(vb - va1, vab) / len_sq
                t = max(0, min(1, t))
                closest = va1 + t * vab
                dist = np.linalg.norm(vb - closest)
                if dist < min_dist:
                    min_dist = dist
        
        # If they overlap, min_dist will be 0.
        # To detect overlap depth, we'd need SAT. 
        # For penalty, we can use a large penalty if min_dist < epsilon.
        # However, for optimization, we want a gradient.
        # A common trick: if min_dist < 0.1, apply penalty.
        # But min_dist is always >= 0 for disjoint.
        # We need to detect if they are NOT disjoint.
        
        # SAT check for overlap
        overlap = False
        for i in range(3):
            e1 = A[(i+1)%6] - A[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
                proj1 = np.dot(A, n1)
                proj2 = np.dot(B, n1)
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    continue # Separating axis found, no overlap
            
            e2 = B[(i+1)%6] - B[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                proj1 = np.dot(A, n2)
                proj2 = np.dot(B, n2)
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    continue
        
        # If we didn't find a separating axis, they overlap.
        # How much? We can estimate penetration depth as -min_dist (if we had signed dist).
        # Since min_dist is 0 on boundary, we can use a small negative value as penalty.
        # Better: return a penalty proportional to how "deep" they are.
        # For simplicity in objective, if overlap, return large penalty.
        # To make it smooth, we can use the minimum distance from vertices to opposite edges if inside.
        # This is complex. Let's stick to: if overlap, penalty = 10.0 + 10.0 * (1.0 - min_dist) # min_dist is 0
        # Actually, if they overlap, min_dist is 0.
        # We can use the minimum of the SAT gaps (which are negative if overlapping).
        
        # Recalculate SAT with gap
        max_gap = -float('inf')
        for i in range(3):
            e1 = A[(i+1)%6] - A[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
                proj1 = np.dot(A, n1)
                proj2 = np.dot(B, n1)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
            
            e2 = B[(i+1)%6] - B[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                proj1 = np.dot(A, n2)
                proj2 = np.dot(B, n2)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
        
        if max_gap < 0:
            return max_gap # Negative indicates overlap depth
        else:
            return max_gap # Positive indicates separation

    def get_gap(v1, v2):
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    def is_valid(centers, angles):
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    return False
        return True

    def compute_side(centers, angles):
        max_proj = 0.0
        normal_angles = [30, 90, 150, 210, 270, 330]
        
        for i in range(N):
            verts = hex_vertices(centers[i][0], centers[i][1], angles[i])
            for deg in normal_angles:
                a = math.radians(deg)
                n = np.array([math.cos(a), math.sin(a)])
                proj = np.dot(verts, n)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        centers = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            centers.append((x, y))
            angles.append(a)
        
        side = compute_side(centers, angles)
        
        # Penalty for overlap
        penalty = 0.0
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += -gap * 10.0 # Heavy penalty for overlap
        
        return side + penalty

    # Topologies to try
    topologies = []
    
    # 1. Honeycomb Cluster
    size = 1.0
    sq3 = math.sqrt(3)
    def axial_to_xy(q, r, s):
        x = s * sq3 * (q + r/2.0)
        y = s * 1.5 * r
        return x, y
    
    centers0 = [(0.0, 0.0)]
    angles0 = [0.0]
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in neighbors:
        x, y = axial_to_xy(q, r, size)
        centers0.append((x, y))
        angles0.append(0.0)
    extra = [(2,0), (1,1), (-1,2), (-2,0), (0,-2)]
    for q, r in extra:
        x, y = axial_to_xy(q, r, size)
        centers0.append((x, y))
        angles0.append(0.0)
    
    params0 = []
    for i in range(N):
        params0.extend([centers0[i][0], centers0[i][1], angles0[i]])
    topologies.append(params0)

    # 2. Ring 12
    params_ring = []
    for k in range(12):
        theta = math.radians(k * 30.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        params_ring.extend([x, y, 0.0])
    topologies.append(params_ring)

    # 3. Two Rings (6+6)
    params_2ring = []
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = 1.5 * math.cos(theta)
        y = 1.5 * math.sin(theta)
        params_2ring.extend([x, y, 0.0])
    for k in range(6):
        theta = math.radians(k * 60.0 + 30.0)
        x = 3.0 * math.cos(theta)
        y = 3.0 * math.sin(theta)
        params_2ring.extend([x, y, 0.0])
    topologies.append(params_2ring)

    best_side = float('inf')
    best_params = None

    for init_params in topologies:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-6})
            if res.success:
                # Validate and compute side
                centers = []
                angles = []
                for i in range(N):
                    centers.append((res.x[3*i], res.x[3*i+1]))
                    angles.append(res.x[3*i+2])
                
                if is_valid(centers, angles):
                    side = compute_side(centers, angles)
                    if side < best_side:
                        best_side = side
                        best_params = res.x.copy()
        except Exception:
            pass

    # If no valid solution found, fallback to initial honeycomb
    if best_params is None:
        best_params = topologies[0]
        centers = []
        angles = []
        for i in range(N):
            centers.append((best_params[3*i], best_params[3*i+1]))
            angles.append(best_params[3*i+2])
        best_side = compute_side(centers, angles)
        
        # If still invalid, spread them out
        if not is_valid(centers, angles):
            centers = []
            angles = []
            for i in range(N):
                a = i * 2 * math.pi / N
                centers.append((10.0 * math.cos(a), 10.0 * math.sin(a)))
                angles.append(0.0)
            best_side = compute_side(centers, angles)
            best_params = []
            for i in range(N):
                best_params.extend([centers[i][0], centers[i][1], angles[i]])

    # Final Polish: Run a short SA to tighten
    T = 0.1
    T_min = 0.001
    alpha = 0.95
    current_params = best_params.copy()
    current_side = best_side
    
    while time.time() < deadline - 5 and T > T_min:
        idx = np.random.randint(0, N)
        new_params = current_params.copy()
        
        if np.random.random() < 0.7:
            new_params[3*idx] += np.random.normal(0, 0.05)
            new_params[3*idx+1] += np.random.normal(0, 0.05)
        else:
            new_params[3*idx+2] += np.random.normal(0, 0.05)
            
        centers = []
        angles = []
        for i in range(N):
            centers.append((new_params[3*i], new_params[3*i+1]))
            angles.append(new_params[3*i+2])
            
        if is_valid(centers, angles):
            new_side = compute_side(centers, angles)
            delta = new_side - current_side
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_params = new_params
                current_side = new_side
                best_side = new_side
                best_params = new_params
        T *= alpha

    # Construct final output
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
        
    final_side = compute_side(
        [(best_params[3*i], best_params[3*i+1]) for i in range(N)],
        [best_params[3*i+2] for i in range(N)]
    )
    
    # Add safety margin
    final_side = final_side * 1.01 + 1e-3

    return inner, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    construct()
