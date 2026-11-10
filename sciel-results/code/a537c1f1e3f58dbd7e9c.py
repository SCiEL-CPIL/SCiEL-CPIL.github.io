# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def objective(params):
        """Negative of squared min/max ratio using exact min/max."""
        pts = params.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        pts = pts / norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_matrix = np.sqrt(dist_sq)
        
        mask = ~np.eye(n, dtype=bool)
        dists = dist_matrix[mask]
        
        if len(dists) == 0:
            return 0.0
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 0.0
            
        ratio = d_min / d_max
        return -(ratio ** 2)

    def polish_ratio(points, iterations=30000, lr=0.001):
        """
        Polish to maximize (d_min/d_max)^2 with adaptive step size.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)
        best_local_score = calculate_score(points)
        stall_count = 0
        max_stall = 5000

        for i in range(iterations):
            if time.time() - start_time > timeout - 15:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(dist_sq)
            
            mask = ~np.eye(n_pts, dtype=bool)
            dists = dist_matrix[mask]
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-12:
                break
                
            # Find min and max pairs
            min_pairs = np.argwhere(dist_matrix <= d_min + 1e-10)
            max_pairs = np.argwhere(dist_matrix >= d_max - 1e-10)
            
            # Build gradient for ratio improvement
            grad = np.zeros_like(points)
            
            min_weight = 1.0 / d_max
            max_weight = d_min / (d_max ** 2)
            
            for idx in min_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += min_weight * unit
                    grad[j_idx] -= min_weight * unit
            
            for idx in max_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] -= max_weight * unit
                    grad[j_idx] += max_weight * unit
            
            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            grad = grad / grad_norm
            
            # Adaptive learning rate with backtracking
            step_lr = lr
            improved = False
            for attempt in range(10):
                new_points = project_to_sphere(points + step_lr * grad)
                new_score = calculate_score(new_points)
                if new_score > best_local_score + 1e-12:
                    points = new_points
                    best_local_score = new_score
                    improved = True
                    break
                step_lr *= 0.5
            
            if not improved:
                stall_count += 1
                if stall_count > max_stall:
                    break
            else:
                stall_count = 0
                lr = min(lr * 1.01, 0.01)

        return points

    def fibonacci_sphere(n_pts, R=1.0):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts * R

    def dodecahedron_based(n_pts):
        """Select 14 points from dodecahedron vertices for good initial spread."""
        phi = (1 + np.sqrt(5)) / 2
        # Dodecahedron vertices (20 vertices)
        verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Greedy selection: pick points to maximize minimum distance
        selected = [0]  # Start with first vertex
        while len(selected) < n_pts:
            best_idx = -1
            best_min_dist = -1
            for idx in range(len(verts)):
                if idx in selected:
                    continue
                # Min distance to already selected points
                min_d = min(np.linalg.norm(verts[idx] - verts[s]) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
        
        return verts[selected]

    def icosahedron_based(n_pts):
        """Start from icosahedron vertices and add points optimally."""
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        if n_pts <= 12:
            return verts[:n_pts]
        
        pts = verts.copy()
        remaining = n_pts - 12
        
        for _ in range(remaining):
            best_pos = None
            best_min_dist = -1
            np.random.seed(42 + len(pts))
            for _ in range(3000):
                candidate = np.random.randn(3)
                candidate = candidate / np.linalg.norm(candidate)
                dists = np.linalg.norm(pts - candidate, axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = candidate
            if best_pos is not None:
                pts = np.vstack([pts, best_pos])
        
        return pts[:n_pts]

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(dodecahedron_based(n))
    inits.append(icosahedron_based(n))

    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    for seed in [777, 888, 999, 1111]:
        np.random.seed(seed)
        inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.2)

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break

        pts = project_to_sphere(init.copy())
        s_init = calculate_score(pts)
        if s_init > best_score:
            best_score = s_init
            best_points = pts.copy()

        # Polish the ratio
        if time.time() - start_time < timeout - 20:
            pts_polished = polish_ratio(pts, iterations=25000, lr=0.001)
            s_polished = calculate_score(pts_polished)
            if s_polished > best_score:
                best_score = s_polished
                best_points = pts_polished.copy()

    # Final refinement on best points
    if best_points is not None and time.time() - start_time < timeout - 10:
        pts_final = polish_ratio(best_points, iterations=20000, lr=0.0005)
        s_final = calculate_score(pts_final)
        if s_final > best_score:
            best_score = s_final
            best_points = pts_final.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    best_points = project_to_sphere(best_points)
    return best_points

# EVOLVE-BLOCK-END
