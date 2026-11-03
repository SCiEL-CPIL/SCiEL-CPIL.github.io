"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms

def _covering_radius(X):
    """
    Compute the covering radius: max over all points y on sphere of min distance to X.
    This is equivalent to the largest circumradius among facets of the convex hull,
    but we compute it directly by checking the "worst" points.
    
    The covering radius is determined by the vertices of the Voronoi cells on the sphere.
    Specifically, it's the maximum over all y on the sphere of the angular distance to the nearest point in X.
    This maximum occurs at either:
    1. A point in X itself (distance 0) - not the max
    2. A point equidistant to 2 or more points in X (on a Voronoi edge or vertex)
    
    For practical computation, we can use the fact that the covering radius equals
    the maximum circumradius of the spherical Voronoi cells.
    
    A simpler approach: the covering radius is the smallest r such that every point on the sphere
    is within distance r of some point in X. This is equivalent to finding the maximum
    over all y of min_i d(y, x_i).
    
    We compute this by checking the "farthest" points, which are the centers of the largest
    empty caps. These correspond to the vertices of the spherical Voronoi diagram.
    
    For a robust computation, we can use the fact that the covering radius is the maximum
    circumradius among all facets of the convex hull of X (when X is in general position).
    But a more direct approach: sample the sphere densely and find the max of min-distance.
    
    However, for a differentiable optimization, we need a smooth approximation.
    """
    # For the exact covering radius, we need to find the maximum over y on S^2 of min_i ||y - x_i|| (angular)
    # This is a non-smooth function. For optimization, we'll use a soft-max approximation.
    # But for evaluation, we need the true value.
    
    # The covering radius is the smallest r such that union of caps of radius r covers the sphere.
    # Equivalently, it's the maximum over y of min_i arccos(y . x_i).
    
    # To compute this exactly, we note that the maximum of min_i d(y, x_i) over y on S^2
    # occurs at a point that is equidistant to at least 3 points of X (a Voronoi vertex on the sphere),
    # or at a point equidistant to 2 points (on a Voronoi edge), or at a point in X.
    
    # A practical approach: the covering radius equals the maximum circumradius of the facets
    # of the convex hull of X. Each facet (triangle) of the convex hull corresponds to a region
    # of the sphere, and the circumradius of that facet gives the max distance from points in that
    # region to the nearest vertex.
    
    # But computing the convex hull and circumradii is complex. Instead, we can use a dense sampling
    # approach for evaluation, but for optimization we need something differentiable.
    
    # Let's compute the exact covering radius using a more careful method.
    # The covering radius is the maximum over all y on S^2 of the minimum angular distance to X.
    # This maximum is achieved at a point y* that is a vertex of the spherical Voronoi diagram of X,
    # i.e., y* is equidistant to at least 3 points of X.
    
    # For now, let's use a dense grid to estimate the covering radius for evaluation purposes.
    # This is not perfect but gives a good estimate.
    
    # Actually, let's use a better approach: the covering radius is the smallest r such that
    # every point on the sphere is within r of some point in X. We can binary search on r and
    # check coverage, but that's slow.
    
    # For the optimization, we'll use a soft-max surrogate. For the final evaluation,
    # we'll compute the true covering radius more carefully.
    
    # Let's compute it by finding the worst-case point. The worst-case point is one that
    # maximizes min_i d(y, x_i). We can find this by checking all Voronoi vertices.
    
    # A Voronoi vertex on the sphere is a point y such that y . x_i = y . x_j = y . x_k for some
    # triple (i, j, k), and y . x_l <= y . x_i for all l. The angular distance is arccos(y . x_i).
    
    # There are C(N, 3) = C(50, 3) = 19600 triples. For each, we can check if the corresponding
    # Voronoi vertex is valid.
    
    # For a triple (i, j, k), the Voronoi vertex y satisfies:
    # y . x_i = y . x_j = y . x_k = c
    # This means y is in the intersection of three great circles, which gives two antipodal points.
    # We need to find which one (if any) is a valid Voronoi vertex.
    
    # The point y is proportional to the cross product of (x_i - x_j) and (x_i - x_k)? No.
    # Actually, y . x_i = y . x_j means y . (x_i - x_j) = 0, so y is perpendicular to (x_i - x_j).
    # Similarly y is perpendicular to (x_i - x_k). So y is parallel to (x_i - x_j) x (x_i - x_k).
    
    # Let's compute this for all triples and find the maximum distance.
    
    X = _project_to_sphere(X)
    
    # For each triple, compute the candidate Voronoi vertex
    max_dist = 0.0
    n = len(X)
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # y is parallel to (X[i] - X[j]) x (X[i] - X[k])
                v1 = X[i] - X[j]
                v2 = X[i] - X[k]
                y = np.cross(v1, v2)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-10:
                    continue
                y = y / y_norm
                
                # Check if y is a valid Voronoi vertex: y . X[i] >= y . X[l] for all l
                # Actually, we need y . X[i] = y . X[j] = y . X[k] (which should hold by construction)
                # and y . X[l] <= y . X[i] for all l.
                
                dots = X @ y
                # The distance from y to the nearest point is arccos(max_l dots[l])
                # Wait, no. The angular distance from y to X[l] is arccos(y . X[l]).
                # The nearest point has the largest dot product.
                max_dot = np.max(dots)
                # The angular distance is arccos(max_dot)
                # But we need to check that i, j, k are among the nearest points.
                # Actually, for y to be a Voronoi vertex for (i, j, k), we need:
                # dots[i] = dots[j] = dots[k] = max_dot (or at least >= all others)
                
                # Check if dots[i], dots[j], dots[k] are all close to max_dot
                if (abs(dots[i] - max_dot) < 1e-6 and 
                    abs(dots[j] - max_dot) < 1e-6 and 
                    abs(dots[k] - max_dot) < 1e-6):
                    dist = np.arccos(np.clip(max_dot, -1.0, 1.0))
                    if dist > max_dist:
                        max_dist = dist
    
    # Also check the antipodal point
    # Actually, the cross product gives one direction; the antipodal point is -y.
    # We should check both.
    # Let me redo this more carefully.
    
    max_dist = 0.0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = X[i] - X[j]
                v2 = X[i] - X[k]
                y = np.cross(v1, v2)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-10:
                    continue
                y = y / y_norm
                
                for sign in [1.0, -1.0]:
                    yc = sign * y
                    dots = X @ yc
                    max_dot = np.max(dots)
                    if (abs(dots[i] - max_dot) < 1e-6 and 
                        abs(dots[j] - max_dot) < 1e-6 and 
                        abs(dots[k] - max_dot) < 1e-6):
                        dist = np.arccos(np.clip(max_dot, -1.0, 1.0))
                        if dist > max_dist:
                            max_dist = dist
    
    # Convert to degrees
    return np.degrees(max_dist)

def _soft_max_covering_radius(X, beta=100.0):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    For a point y on the sphere, min_i d(y, x_i) is non-smooth.
    We approximate max_y min_i d(y, x_i) with a smooth surrogate.
    
    Actually, let's use a different approach. The covering radius is:
    max_{y in S^2} min_i d(y, x_i)
    
    We can reformulate this as:
    min_{t} t  s.t.  d(y, x_i) <= t  for all y in S^2, i = 1..N
    
    But this is still complex. Let's use a different smooth surrogate.
    
    Alternative: use the fact that the covering radius is related to the maximum
    circumradius of the convex hull facets. For a triangle with vertices a, b, c on the sphere,
    the circumradius is related to the angles of the triangle.
    
    For a spherical triangle with side lengths a, b, c (angular distances), the circumradius R
    satisfies: sin(R) = sin(a)sin(b)sin(c) / (2 sin(s-a) sin(s-b) sin(s-c)) ... this is getting complex.
    
    Let's use a simpler approach: define the objective as the maximum over all pairs of points
    of some function, or use a grid-based approach.
    
    Actually, for L-BFGS-B, we need a differentiable objective. Let's use:
    f(X) = log(sum_i exp(beta * d_i)) / beta
    where d_i is the distance from some reference point to the nearest X_j.
    But this doesn't directly give the covering radius.
    
    Let me think differently. The covering radius is:
    R = max_{y in S^2} min_{i} arccos(y . x_i)
    
    Let g(y) = min_i arccos(y . x_i). We want max_y g(y).
    
    g(y) is non-smooth. We can approximate it with a soft-min:
    g_soft(y) = -1/beta * log(sum_i exp(-beta * arccos(y . x_i)))
    
    Then R_soft = max_y g_soft(y).
    
    But we still need to maximize over y, which is another optimization.
    
    Alternative: use the fact that the covering radius equals the maximum circumradius
    of the spherical Voronoi cells. Each cell is a spherical polygon, and its circumradius
    can be computed from its vertices.
    
    For a practical optimization, let's use a different smooth objective that correlates
    with the covering radius. For example, the maximum over all pairs (i, j) of the
    circumradius of the triangle formed by x_i, x_j, and some reference point.
    
    Actually, let's use a simpler approach: optimize the minimum pairwise distance
    (packing) as a proxy, since good packings tend to have good covering properties.
    But this isn't ideal.
    
    Let me use a grid-based approach for the objective: sample many points on the sphere,
    and for each, compute the distance to the nearest x_i. Then take the maximum.
    Use a soft-max to make it differentiable.
    """
    X = _project_to_sphere(X)
    
    # Create a grid of test points on the sphere
    # Use a Fibonacci spiral with many points
    M = 2000  # number of test points
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each Y to each X
    # dots[m, i] = Y[m] . X[i]
    dots = Y @ X.T  # shape (M, N)
    
    # Distance from Y[m] to nearest X[i] is arccos(max_i dots[m, i])
    max_dots = np.max(dots, axis=1)  # shape (M,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))  # shape (M,)
    
    # The covering radius is approximately max(distances)
    # Use soft-max to make it differentiable
    beta = 50.0
    # soft_max = 1/beta * log(sum(exp(beta * distances)))
    # For numerical stability:
    max_d = np.max(distances)
    soft_max = max_d + np.log(np.sum(np.exp(beta * (distances - max_d)))) / beta
    
    return soft_max

def _gradient_of_soft_max_covering_radius(X, beta=50.0):
    """
    Compute the gradient of the soft-max covering radius objective.
    """
    X = _project_to_sphere(X)
    
    M = 2000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = Y @ X.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))  # (M,)
    
    max_d = np.max(distances)
    exp_terms = np.exp(beta * (distances - max_d))  # (M,)
    sum_exp = np.sum(exp_terms)
    
    # soft_max = max_d + (1/beta) * log(sum_exp)
    # d(soft_max)/d(distances[m]) = beta * exp_terms[m] / sum_exp
    weights = beta * exp_terms / sum_exp  # (M,)
    
    # d(distances[m])/d(dots[m, i]) = -1 / sqrt(1 - dots[m, i]^2) for the i that achieves max
    # But we need the gradient with respect to X, which is more complex.
    
    # Let's use numerical gradient instead for simplicity and robustness.
    # Actually, let's use a simpler objective that's easier to differentiate.
    
    # Alternative: use the sum of squared distances to the nearest point, or similar.
    
    # For now, let's use scipy's numerical gradient via finite differences.
    return None  # We'll use numerical gradients

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_score = _covering_radius(best_X)
    
    # Try to improve using local optimization
    # Use a simple gradient descent with numerical gradients
    
    def objective_flat(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        return _soft_max_covering_radius(X, beta=50.0)
    
    # Use L-BFGS-B with numerical gradients
    try:
        result = minimize(
            objective_flat,
            X.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < deadline:
            X_opt = _project_to_sphere(result.x.reshape(N, 3))
            score_opt = _covering_radius(X_opt)
            if score_opt < best_score:
                best_X = X_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Try multiple restarts with different initializations
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > deadline - 20:
            break
        
        # Random initialization
        X_rand = np.random.randn(N, 3)
        X_rand = _project_to_sphere(X_rand)
        
        try:
            result = minimize(
                objective_flat,
                X_rand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            if time.time() - start_time > deadline - 10:
                break
                
            X_opt = _project_to_sphere(result.x.reshape(N, 3))
            score_opt = _covering_radius(X_opt)
            if score_opt < best_score:
                best_X = X_opt
                best_score = score_opt
        except Exception:
            pass
    
    return _project_to_sphere(best_X)

# EVOLVE-BLOCK-END
