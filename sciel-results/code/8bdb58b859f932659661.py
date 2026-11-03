"""
Strategy: Topological Spectral Optimization via Delaunay Graph Laplacian.

The current program uses continuous gradient descent (L-BFGS-B) on a soft-min
approximation of the packing score. This approach is sensitive to local minima
in the distance landscape.

This new program reformulates the problem using graph theory:
1. Construct a Delaunay triangulation (or k-Nearest Neighbors graph) of the points.
2. Define the Graph Laplacian L = D - A.
3. The algebraic connectivity (Fiedler value, the second smallest eigenvalue of L)
   measures how well the graph is connected. A higher Fiedler value implies a more
   rigid, uniform structure where no "loose" parts can be easily compressed.
4. We maximize the Fiedler value. This acts as a global structural regularizer
   that is distinct from local pairwise distance maximization.
5. We combine this with a penalty for boundary violations to keep points inside.
6. We use L-BFGS-B to optimize the coordinates with respect to this spectral objective.
7. We perform multiple restarts with different initial lattice rotations to escape
   topological traps.

Why this is different:
- Gradient methods on distance minimize local "stress" but can allow global
  distortions (e.g., a cluster of points moving together).
- Spectral methods enforce global connectivity and uniformity. Maximizing
  algebraic connectivity ensures the "skeleton" of the packing is robust,
  which often correlates with a more uniform distribution of gaps, potentially
  leading to a higher minimum gap.
"""

import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

N = 300
TIME_LIMIT = 115.0  # Safety margin for 150s kill switch

def _hex_init(rotation=0.0, jitter=0.0):
    """
    Generate a hexagonal lattice initialization.
    Returns an array of shape (N, 2).
    """
    # Estimate spacing. Best r ~ 0.053. Spacing ~ 0.106.
    # We want to fit 300 points.
    spacing = 0.105
    
    # Generate points in a grid large enough to cover the unit circle
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Rotate points
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
        
    # Add jitter if requested
    if jitter > 0.0:
        points += np.random.normal(0, jitter, points.shape)
        
    # Filter points strictly inside unit circle (with small margin)
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 0.99
    points = points[mask]
    radii = radii[mask]
    
    # Sort by radius (closest to center first) and take N
    idx = np.argsort(radii)
    points = points[idx]
    
    if len(points) < N:
        # Fallback: if we don't have enough points, pad with random points
        theta = np.random.uniform(0, 2 * np.pi, N - len(points))
        rho = np.sqrt(np.random.uniform(0, 0.5, N - len(points)))
        pad = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        points = np.vstack([points, pad])
    else:
        points = points[:N]
        
    return points

def _compute_spectral_objective_and_grad(points, lambda_penalty=10.0):
    """
    Compute the objective (negative Fiedler value + boundary penalty) and its gradient.
    
    Objective:
    f(x) = -lambda_2(L) + lambda_penalty * sum(max(0, |x_i| - 1)^2)
    
    We want to maximize lambda_2, so we minimize -lambda_2.
    The boundary penalty ensures points stay inside the unit circle.
    
    Gradient:
    The gradient of the Fiedler value with respect to the coordinates is derived
    from the eigenvector v_2 corresponding to lambda_2.
    d lambda_2 / d x = -2 * v_2^T (dL/dx) v_2
    
    The Laplacian L is built from the adjacency matrix A of the Delaunay graph.
    L = D - A, where D is the degree matrix.
    
    For a Delaunay triangulation, the edges are fixed topologically during a small
    perturbation (as long as points don't cross edges). We assume the topology is
    stable during the optimization step.
    
    The derivative of the Laplacian w.r.t. coordinates is non-zero only for the
    terms that depend on coordinates. However, in a standard Delaunay graph, the
    connectivity is topological (0 or 1), not weighted by distance. 
    If we use a weighted Delaunay graph (e.g., weights = 1/d^2), the gradient is
    easier to compute and more sensitive to local distances.
    
    Let's use a weighted graph where weight w_ij = 1 / d_ij^2 for connected pairs.
    Then L_ij = -w_ij for i!=j, L_ii = sum_j w_ij.
    
    d L_ij / d x_i = - d w_ij / d x_i = - d (1/d^2) / d x_i = - (-2/d^3) * (x_i - x_j)/d = 2 (x_i - x_j) / d^4
    d L_ii / d x_i = sum_{j in N(i)} d w_ij / d x_i = sum_{j in N(i)} 2 (x_i - x_j) / d_ij^4
    
    The gradient of the quadratic form v^T L v is:
    grad_f = -2 * (L v)_i * (d L / d x_i) ... this is complex.
    
    Actually, for a fixed topology, the gradient of lambda_2 with respect to node positions
    is related to the "Laplacian strain energy".
    
    A simpler and more robust approach:
    Use the standard unweighted Delaunay graph.
    The Fiedler value is a measure of connectivity.
    To get a gradient, we can use the fact that for a graph with fixed topology,
    the derivative of eigenvalues is tricky if the graph changes.
    
    Alternative: Use a Force-Directed Layout with a Spectral Regularizer.
    Or, simpler: Use the "Laplacian Energy" E = sum_{i,j} w_ij ||x_i - x_j||^2.
    Minimizing this energy spreads points out.
    Maximizing the Fiedler value is equivalent to maximizing the "stiffness".
    
    Let's use a simpler proxy: The "Variance of the Laplacian Eigenvalues" or just
    maximizing the second eigenvalue of a weighted Laplacian where weights are
    inverse distance squared.
    
    Let's define the objective as:
    Minimize: -lambda_2(L_weighted) + Boundary_Penalty
    
    We will compute the gradient numerically or use an analytical approximation.
    Given the complexity of analytical gradients for spectral objectives with changing
    topology, we will use a finite difference approximation for the gradient if
    analytical is too risky, OR we can use a "Spectral Force" heuristic.
    
    However, for 300 points, we can afford to compute the eigenvalue decomposition
    of a 300x300 matrix quickly.
    
    Let's try a different, more robust spectral objective:
    Maximize the sum of the smallest k eigenvalues? No.
    
    Let's stick to the Fiedler value but compute the gradient via the eigenvector.
    If the topology is fixed, the gradient of lambda_2 with respect to the
    Laplacian entries is (v_2 v_2^T).
    The gradient with respect to coordinates requires dL/dx.
    
    For a weighted graph with w_ij = 1/d_ij^2:
    L is a function of x.
    
    Let's implement a simplified version:
    1. Compute Delaunay.
    2. Build weighted Laplacian with w_ij = 1/d_ij^2.
    3. Compute lambda_2 and v_2.
    4. Approximate gradient: The force on node i is proportional to (L v_2)_i 
       in the direction that increases connectivity? 
       
    Actually, a known result is that the gradient of the Fiedler value with respect
    to the graph weights is v_2 v_2^T.
    Since weights depend on positions, we can use the chain rule.
    
    For simplicity and stability, let's use a "Spectral Energy" minimization:
    Minimize E = v_2^T L v_2 (which is lambda_2) ? No, we want to maximize it.
    
    Let's try: Minimize -lambda_2.
    
    Gradient approximation:
    The change in lambda_2 is approximated by -2 * v_2^T (delta L) v_2.
    delta L comes from delta x.
    
    This is getting too complex to derive perfectly in the time budget.
    Let's use a "Hybrid" approach:
    1. Use the spectral objective to guide a local search or to weight the forces.
    2. OR, simply use the Delaunay graph to define a "Rigid Body" constraint.
    
    Let's pivot to a simpler, distinct strategy: **Topological Constraint Optimization**.
    Use the Delaunay triangulation to identify "bad" edges (long edges) and "good" edges.
    Apply forces to shorten long edges and lengthen short edges, but weighted by
    the spectral centrality of the nodes.
    
    Actually, let's implement the **Spectral Clustering** based objective:
    Minimize the "Cut" of the graph? No.
    
    Let's go with **Maximizing the Algebraic Connectivity** using a numerical gradient
    for the objective, but using the analytical gradient for the boundary penalty.
    Since N=300, a numerical gradient (central difference) would require 600 evaluations
    of the eigenvalue solver. Each eigenvalue solver for 300x300 sparse matrix is ~ms.
    600 * 5ms = 3s per gradient step. Too slow for L-BFGS-B (which needs ~100-200 steps).
    
    So we need an analytical gradient or a very fast objective.
    
    Let's use the **Laplacian Energy** of the second eigenvector.
    E_2 = v_2^T L v_2 = lambda_2.
    
    If we fix the topology, the gradient of lambda_2 w.r.t. weights w_ij is (v_2(i) - v_2(j))^2.
    The weights w_ij = 1/d_ij^2.
    d w_ij / d x_i = 2 (x_i - x_j) / d_ij^4.
    
    So, grad_x_i (lambda_2) = sum_{j in N(i)} (v_2(i) - v_2(j))^2 * d w_ij / d x_i
    = sum_{j in N(i)} (v_2(i) - v_2(j))^2 * 2 (x_i - x_j) / d_ij^4.
    
    This is a valid analytical gradient assuming fixed topology!
    
    Objective to minimize: -lambda_2 + Penalty.
    Gradient: -grad_lambda_2 + grad_Penalty.
    """
    points = points.copy()
    
    # 1. Compute Delaunay Triangulation
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback if triangulation fails (e.g., collinear points)
        return 0.0, np.zeros_like(points)
    
    # Get edges from triangles
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                u, v = simplex[i], simplex[j]
                if u > v:
                    u, v = v, u
                edges.add((u, v))
    
    edges = list(edges)
    if len(edges) == 0:
        return 0.0, np.zeros_like(points)
    
    # Build adjacency lists and weights
    # We use w_ij = 1 / d_ij^2
    # L is sparse
    
    row = []
    col = []
    data = []
    
    # Accumulate degrees
    degrees = np.zeros(N)
    
    # First pass: calculate distances and weights
    edge_u = np.array([e[0] for e in edges])
    edge_v = np.array([e[1] for e in edges])
    
    diffs = points[edge_u] - points[edge_v]
    dists_sq = np.sum(diffs**2, axis=1)
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    weights = 1.0 / dists_sq
    
    # Build Laplacian
    # L_ii = sum_j w_ij
    # L_ij = -w_ij
    
    # Diagonal
    for i in range(N):
        mask_u = (edge_u == i)
        mask_v = (edge_v == i)
        deg = np.sum(weights[mask_u]) + np.sum(weights[mask_v])
        degrees[i] = deg
        row.append(i)
        col.append(i)
        data.append(deg)
        
    # Off-diagonal
    for k in range(len(edges)):
        u, v = edges[k]
        w = weights[k]
        row.append(u)
        col.append(v)
        data.append(-w)
        row.append(v)
        col.append(u)
        data.append(-w)
        
    L = csr_matrix((data, (row, col)), shape=(N, N))
    
    # 2. Compute Eigenvalues and Eigenvectors
    # We want the second smallest eigenvalue (Fiedler value)
    # L is symmetric positive semi-definite.
    # eigsh finds eigenvalues in a given range.
    # We want the smallest ones.
    try:
        # k=2 to get lambda_0 (0) and lambda_1 (Fiedler)
        # Actually, eigsh returns eigenvalues in ascending order if which='SA'
        eigvals, eigvecs = eigsh(L, k=2, which='SA')
        # Sort to be sure
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        lambda_2 = eigvals[1]
        v_2 = eigvecs[:, 1]
    except Exception:
        return 0.0, np.zeros_like(points)
    
    # 3. Compute Gradient of lambda_2 w.r.t. coordinates
    # grad_x_i = sum_{j in N(i)} (v_2(i) - v_2(j))^2 * 2 (x_i - x_j) / d_ij^4
    
    grad_lambda = np.zeros_like(points)
    
    for k in range(len(edges)):
        u, v = edges[k]
        w = weights[k]
        # w = 1/d^2, so 1/d^4 = w^2
        # d w / d x_u = 2 (x_u - x_v) / d^4 = 2 * w^2 * (x_u - x_v)
        
        diff = points[u] - points[v]
        w_sq = w * w
        
        # Term for u
        val_u = (v_2[u] - v_2[v])**2 * 2 * w_sq * diff
        # Term for v
        val_v = (v_2[v] - v_2[u])**2 * 2 * w_sq * (points[v] - points[u])
        
        grad_lambda[u] += val_u
        grad_lambda[v] += val_v
        
    # 4. Boundary Penalty
    # P = sum (max(0, |x_i| - 1)^2)
    radii = np.linalg.norm(points, axis=1)
    # Only points outside contribute
    outside_mask = radii > 1.0
    penalty = np.sum(np.maximum(radii - 1.0, 0.0)**2)
    
    # Gradient of penalty
    # d/dx_i ( (|x_i|-1)^2 ) = 2 (|x_i|-1) * x_i / |x_i|
    grad_penalty = np.zeros_like(points)
    if np.any(outside_mask):
        r_out = radii[outside_mask]
        p_out = points[outside_mask]
        # factor = 2 * (r - 1) / r
        factors = 2.0 * (r_out - 1.0) / r_out
        grad_penalty[outside_mask] = p_out * factors[:, np.newaxis]
        
    # 5. Combine
    # Objective: -lambda_2 + lambda_p * penalty
    obj = -lambda_2 + 10.0 * penalty
    
    # Gradient: -grad_lambda + 10.0 * grad_penalty
    grad = -grad_lambda + 10.0 * grad_penalty
    
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    
    # Define a few different initializations to restart from
    inits = [
        _hex_init(rotation=0.0, jitter=0.0),
        _hex_init(rotation=np.pi/6, jitter=0.0),
        _hex_init(rotation=0.0, jitter=0.01),
        _hex_init(rotation=np.pi/3, jitter=0.01),
        _hex_init(rotation=np.pi/12, jitter=0.005),
    ]
    
    time_per_run = 20.0
    
    for i, init_points in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
            
        x0 = init_points.flatten()
        
        def f_and_g(x):
            pts = x.reshape(N, 2)
            # Check for NaN/Inf
            if not np.all(np.isfinite(pts)):
                return 1e10, np.zeros_like(x)
            try:
                obj, grad = _compute_spectral_objective_and_grad(pts)
                if not np.isfinite(obj):
                    return 1e10, np.zeros_like(x)
                return obj, grad
            except Exception:
                return 1e10, np.zeros_like(x)
                
        try:
            res = opt.minimize(
                f_and_g,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 150,
                    'ftol': 1e-8,
                    'gtol': 1e-5,
                    'maxls': 20
                }
            )
        except Exception:
            continue
            
        x_opt = res.x.reshape(N, 2)
        
        # Calculate true score (hard min)
        diffs = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.linalg.norm(x_opt, axis=1)
        min_bound_dist = np.min(1.0 - radii)
        
        score = min(min_pair_dist / 2.0, min_bound_dist)
        
        if score > best_score:
            best_score = score
            best_points = x_opt.copy()
            
        if time.time() - start_time > TIME_LIMIT - 10.0:
            break
            
    # Final validity check
    if best_points is None:
        best_points = inits[0]
        
    radii = np.linalg.norm(best_points, axis=1)
    valid = np.isfinite(best_points).all(axis=1) & (radii < 1.0)
    if not np.all(valid):
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        best_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(pts, axis=1)
    min_bound_dist = np.min(1.0 - radii)
    score = min(min_pair_dist / 2.0, min_bound_dist)
    print(f"Score: {score}")
