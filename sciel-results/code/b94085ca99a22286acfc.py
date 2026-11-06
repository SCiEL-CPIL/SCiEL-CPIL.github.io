import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-14, 1.0, norms)
        return pts / norms

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_precise_radius(points):
        try:
            hull = ConvexHull(points)
        except Exception:
            return 100.0 # Invalid or degenerate
            
        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1,e1), 2*np.dot(e1,e2)],
                          [2*np.dot(e1,e2), 2*np.dot(e2,e2)]])
            b = np.array([np.dot(e1,e1), np.dot(e2,e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0]*e1 + params[1]*e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            if ang > max_rad:
                max_rad = ang
        return np.degrees(max_rad)

    def get_facet_gradients(points, top_k=10):
        """
        Returns a gradient vector (N, 3) and the list of worst facets.
        The gradient points in the direction that reduces the covering radius.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.zeros_like(points), []
            
        facets = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1,e1), 2*np.dot(e1,e2)],
                          [2*np.dot(e1,e2), 2*np.dot(e2,e2)]])
            b = np.array([np.dot(e1,e1), np.dot(e2,e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0]*e1 + params[1]*e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            facets.append((ang, tuple(int(x) for x in simplex), cc))
            
        if not facets:
            return np.zeros_like(points), []
            
        facets.sort(key=lambda x: x[0], reverse=True)
        worst = facets[:top_k]
        
        grad = np.zeros_like(points)
        for ang, simplex, cc in worst:
            # Weight by how much this facet is "worst"
            # Using a smooth weight: 1 if ang is close to max, else less
            # Here we just sum unit vectors
            for idx in simplex:
                cur = points[idx]
                to_cc = cc - cur
                radial = np.dot(to_cc, cur) * cur
                tangent = to_cc - radial
                nt = np.linalg.norm(tangent)
                if nt > 1e-12:
                    grad[idx] += tangent / nt
                    
        return grad, worst

    def lloyd_iteration(points, test_points, step_size=1.0):
        dots = np.clip(test_points @ points.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        n_pts = len(points)
        sums = np.zeros((n_pts, 3))
        np.add.at(sums, nearest, test_points)
        counts = np.bincount(nearest, minlength=n_pts)
        counts_safe = np.where(counts == 0, 1, counts)
        centroids = sums / counts_safe[:, np.newaxis]
        centroids = normalize(centroids)
        empty = counts == 0
        if np.any(empty):
            centroids[empty] = points[empty]
        result = (1 - step_size) * points + step_size * centroids
        return normalize(result)

    # Pre-compute Lloyd test grid
    lloyd_test = fibonacci_sphere(100000)
    rng = np.random.RandomState(42)

    best_points = None
    best_score = float('inf')

    # Main Loop: Multiple restarts with SA
    num_restarts = 0
    max_restarts = 30 # Adjust based on time

    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        
        # 1. Initialization
        if num_restarts == 1:
            points = fibonacci_sphere(N)
        elif num_restarts == 2:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.1
            points = normalize(points)
        else:
            if best_points is not None and rng.random() < 0.5:
                # Perturb best
                points = best_points + rng.standard_normal((N, 3)) * 0.15
            else:
                # Random start
                points = rng.standard_normal((N, 3))
            points = normalize(points)

        # 2. Lloyd's for quick convergence
        step_size = 0.5
        for _ in range(30):
            if time.time() > deadline: break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.95
            if step_size < 0.05: step_size = 0.05

        # 3. Simulated Annealing / Gradient Descent
        current_score = compute_precise_radius(points)
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()

        temperature = 0.5 # Initial "noise" scale
        lr = 0.05 # Initial learning rate
        
        for sa_iter in range(100): # Max SA iterations per restart
            if time.time() > deadline - 5: break
            
            # Get gradient
            grad, _ = get_facet_gradients(points, top_k=10)
            
            # Normalize gradient
            gn = np.linalg.norm(grad)
            if gn > 1e-10:
                grad = grad / gn
            
            # Propose move: Gradient step + Noise
            noise = rng.standard_normal((N, 3)) * temperature
            move = lr * grad + noise
            new_points = points + move
            new_points = normalize(new_points)
            
            # Evaluate
            new_score = compute_precise_radius(new_points)
            
            # Acceptance
            delta = new_score - current_score
            if delta < 0:
                # Better
                points = new_points
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Worse, accept with prob exp(-delta/T)
                # Note: delta is in degrees. T is in degrees.
                prob = np.exp(-delta / max(temperature, 1e-6))
                if rng.random() < prob:
                    points = new_points
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = points.copy()
            
            # Cooling
            temperature *= 0.98
            lr *= 0.99

    return best_points
