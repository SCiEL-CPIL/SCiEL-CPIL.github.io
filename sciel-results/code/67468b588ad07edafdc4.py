import numpy as np
import scipy.spatial as spat
import time
import itertools

N = 14

def _hull_area(points):
    """Calculate area of convex hull."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    n = len(points)
    i, j, k = np.array(list(itertools.combinations(range(n), 3))).T
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n = N
    idx_arr = np.array(list(itertools.combinations(range(n), 3)))
    i_idx = idx_arr[:, 0]
    j_idx = idx_arr[:, 1]
    k_idx = idx_arr[:, 2]

    def fast_score(points):
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas) / hull_area

    def _build_from_angles(angles):
        """
        Given a sequence of exterior angles (turning angles) that sum to 2*pi,
        construct a closed convex polygon by integrating edge directions.
        Edge directions are the cumulative sum of turning angles.
        Edge lengths are chosen to close the polygon.
        """
        # Normalize angles to sum to 2*pi
        s = np.sum(angles)
        if s < 1e-10:
            return None
        angles = angles * (2 * np.pi / s)

        # Edge directions: cumulative sum of turning angles
        # Start with direction 0
        edge_dirs = np.cumsum(angles)
        # Edge direction for edge i is the angle after i-1 turns
        # Actually, the direction of edge i is the sum of the first i turning angles
        # But for a closed polygon, the last edge direction + last turn = 2*pi
        # Let's just use: dir_i = sum(angles[0:i]) for i in range(n)
        # Wait, that's not quite right. Let me think.
        # If we have turning angles theta_0, theta_1, ..., theta_{n-1}
        # Edge 0 has direction 0
        # Edge 1 has direction theta_0
        # Edge 2 has direction theta_0 + theta_1
        # ...
        # Edge i has direction sum(theta_0, ..., theta_{i-1})
        # So edge_dirs[i] = cumsum(angles[:i])
        # For i=0, edge_dirs[0] = 0
        
        edge_dirs = np.zeros(n)
        for i in range(1, n):
            edge_dirs[i] = np.sum(angles[:i])

        # Edge vectors in direction space
        ex = np.cos(edge_dirs)
        ey = np.sin(edge_dirs)

        # We need to find edge lengths l_0, ..., l_{n-1} >= 0 such that
        # sum(l_i * ex_i) = 0 and sum(l_i * ey_i) = 0
        # This is a 2-constraint system with n unknowns.
        # We can use a linear programming approach or just pick the first n-2
        # lengths freely and solve for the last 2.
        # Actually, let's use a simpler approach: pick lengths proportional to
        # some heuristic and then solve the closure constraints.
        
        # Use the approach: set all lengths to 1, compute the closure error,
        # then solve for the correction.
        # Actually, let's use a direct linear solve.
        # We have n equations (2 vector equations = 2 scalar) and n unknowns.
        # This is underdetermined. We can add n-2 constraints to make it determined.
        # A natural choice: set the first n-2 lengths to 1 (or some value) and solve
        # for the last 2.
        
        # But this can give negative lengths. Let's try a different approach.
        
        # Use the fact that for a convex polygon, all edge lengths should be positive.
        # We can use linear programming, but let's try a simpler approach first.
        
        # Set all edge lengths to 1 and see if the polygon closes approximately.
        # If not, we need to adjust.
        
        # Actually, let's just use a random set of positive edge lengths and see
        # if we can close the polygon. If not, try again.
        
        # Better approach: use the "edge vector closure" constraint.
        # Given directions, find positive lengths that close the polygon.
        # This is equivalent to finding a point in the positive orthant of R^n
        # that maps to 0 under the linear map defined by the edge directions.
        
        # For simplicity, let's just use the following heuristic:
        # 1. Start with all edge lengths = 1
        # 2. Compute the closure error
        # 3. Adjust the lengths to minimize the error (using least squares with
        #    non-negativity constraints)
        
        A = np.column_stack((ex, ey))  # shape (n, 2)
        
        # Solve for lengths that close the polygon: A @ lengths = 0
        # with lengths >= 0
        # This is a homogeneous system, so the only solution is lengths = 0.
        # We need to find a non-trivial solution.
        
        # Actually, the system is underdetermined (n > 2), so there's a
        # (n-2)-dimensional null space. We want a point in the null space
        # that is in the positive orthant.
        
        # Compute the null space of A
        try:
            U, S, Vt = np.linalg.svd(A)
            # Null space is the last n-2 columns of V
            # Vt has shape (n, n), so V = Vt.T
            # The null space basis is Vt[n-2:, :].T
            null_basis = Vt[n-2:, :].T  # shape (n, n-2)
        except:
            return None

        # We want a positive combination of the null space basis vectors
        # Try random combinations
        best_lengths = None
        best_error = np.inf
        
        for _ in range(100):
            coeffs = np.random.rand(n - 2)
            coeffs = coeffs / np.sum(coeffs) if np.sum(coeffs) > 0 else np.ones(n-2) / (n-2)
            lengths = null_basis @ coeffs
            if np.min(lengths) < -1e-8:
                continue
            lengths = np.maximum(lengths, 0)
            # Check closure
            closure = A @ lengths
            err = np.linalg.norm(closure)
            if err < best_error:
                best_error = err
                best_lengths = lengths

        if best_lengths is None or best_error > 1.0:
            # Fallback: use a regular polygon
            return None

        # Construct the polygon
        edge_vecs = best_lengths[:, np.newaxis] * np.column_stack((ex, ey))
        vertices = np.cumsum(edge_vecs, axis=0)
        # The last vertex should be close to 0 (closure)
        # Shift so that the centroid is at the origin
        vertices = vertices - np.mean(vertices, axis=0)
        
        return vertices

    # Strategy: optimize the turning angles (exterior angles) of a convex polygon
    # The turning angles must be positive and sum to 2*pi
    # We'll parameterize them as exp(theta_i) and optimize theta_i
    
    def angles_to_points(theta):
        """Convert log-turning-angles to polygon vertices."""
        angles = np.exp(theta)
        return _build_from_angles(angles)

    def objective(x):
        """Objective for optimization: negative of soft-min score."""
        points = angles_to_points(x)
        if points is None or len(points) != n:
            return 1e10
        
        # Center the points
        points = points - np.mean(points, axis=0)
        
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 1e10
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min
        beta = 50.0
        a = -beta * areas
        m = np.max(a)
        try:
            soft_min_val = -1.0 / beta * (m + np.log(np.sum(np.exp(a - m))))
        except:
            return 1e10
        
        return -soft_min_val / hull_area

    # Initial candidates
    candidates = []

    # 1. Regular 14-gon: all turning angles = 2*pi/14
    reg_angles = np.full(N, 2 * np.pi / N)
    reg_theta = np.log(reg_angles)
    candidates.append(reg_theta.copy())

    # 2. Perturbed regular polygon
    np.random.seed(42)
    for _ in range(20):
        noise = np.random.normal(0, 0.3, N)
        theta = reg_theta + noise
        candidates.append(theta.copy())

    # 3. Two-ring structure approximated by varying angles
    # Create a pattern where every other angle is different
    for ratio in [0.5, 0.7, 0.8, 1.2]:
        angles = np.full(N, 2 * np.pi / N)
        # Make odd indices have larger angles (sharper turns)
        angles[::2] *= ratio
        angles[1::2] *= (1 + (ratio - 1) * 0.5)
        # Normalize
        angles = angles * (2 * np.pi / np.sum(angles))
        if np.min(angles) > 1e-6:
            candidates.append(np.log(angles).copy())

    best_points = None
    best_score = 0.0

    # Evaluate initial candidates
    for theta in candidates:
        if time.time() > deadline - 10:
            break
        pts = angles_to_points(theta)
        if pts is not None:
            pts = pts - np.mean(pts, axis=0)
            s = fast_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

    # Optimize from top candidates
    scored_candidates = []
    for theta in candidates:
        pts = angles_to_points(theta)
        if pts is not None:
            pts = pts - np.mean(pts, axis=0)
            s = fast_score(pts)
            scored_candidates.append((s, theta))
        else:
            scored_candidates.append((0.0, theta))
    
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_candidates = [theta for s, theta in scored_candidates[:8]]

    for x0 in top_candidates:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-8})
            if res.success:
                pts_opt = angles_to_points(res.x)
                if pts_opt is not None:
                    pts_opt = pts_opt - np.mean(pts_opt, axis=0)
                    s_opt = fast_score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt.copy()
        except Exception:
            pass

    # Final refinement with higher beta
    if time.time() < deadline - 25 and best_points is not None:
        def objective_sharp(x):
            points = angles_to_points(x)
            if points is None or len(points) != n:
                return 1e10
            points = points - np.mean(points, axis=0)
            hull_area = _hull_area(points)
            if hull_area < 1e-10:
                return 1e10
            pi = points[i_idx]
            pj = points[j_idx]
            pk = points[k_idx]
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            beta = 200.0
            a = -beta * areas
            m = np.max(a)
            try:
                soft_min_val = -1.0 / beta * (m + np.log(np.sum(np.exp(a - m))))
            except:
                return 1e10
            return -soft_min_val / hull_area

        # Find the theta that gave best_points
        # Re-derive from best_points by computing edge directions
        if best_points is not None:
            # Compute turning angles from best_points
            best_pts = best_points - np.mean(best_points, axis=0)
            edges = np.diff(np.vstack((best_pts, best_pts[0])), axis=0)
            edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
            # Turning angles
            turns = np.diff(np.vstack((edge_angles, edge_angles[0] + 2*np.pi)), axis=0)
            turns = np.mod(turns, 2*np.pi)
            # Make sure they're positive and sum to 2*pi
            if np.min(turns) > 1e-8 and abs(np.sum(turns) - 2*np.pi) < 1e-4:
                theta_best = np.log(turns)
                try:
                    res = minimize(objective_sharp, theta_best, method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-9})
                    if res.success:
                        pts_final = angles_to_points(res.x)
                        if pts_final is not None:
                            pts_final = pts_final - np.mean(pts_final, axis=0)
                            s_final = fast_score(pts_final)
                            if s_final > best_score:
                                best_score = s_final
                                best_points = pts_final.copy()
                except:
                    pass

    if best_points is None:
        # Fallback: regular 14-gon
        angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))

    return best_points
