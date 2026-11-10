import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _points_to_gram(points: np.ndarray) -> np.ndarray:
    """Compute the Gram matrix (inner products) for points on the unit sphere."""
    # Points are assumed to be on the unit sphere
    # G = P @ P.T
    return points @ points.T

def _gram_to_points(G: np.ndarray) -> np.ndarray:
    """Reconstruct points from a valid Gram matrix."""
    try:
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        # Keep only positive eigenvalues
        eigvals = np.clip(eigvals, 0, None)
        # Reconstruct
        # P = eigvecs @ diag(sqrt(eigvals))
        P = eigvecs @ np.diag(np.sqrt(eigvals))
        # Project back to sphere? The Gram matrix of unit vectors must have 1s on diagonal.
        # Due to numerical errors, diagonal might not be exactly 1.
        # Let's normalize the rows of P to ensure they are unit vectors.
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P = P / norms
        return P
    except Exception:
        return None

def _distances_from_gram(G: np.ndarray) -> np.ndarray:
    """Compute pairwise distances from Gram matrix."""
    # d_ij^2 = ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 p_i . p_j = 2 - 2 G_ij (for unit vectors)
    # We need to ensure diagonal is 1.
    # Let's assume G is a valid Gram matrix of unit vectors.
    D_sq = 2.0 - 2.0 * G
    # Set diagonal to 0
    np.fill_diagonal(D_sq, 0.0)
    # Due to numerical errors, D_sq might be slightly negative
    D_sq = np.clip(D_sq, 0, None)
    D = np.sqrt(D_sq)
    # Extract upper triangle
    return D[np.triu_indices_from(D, k=1)]

def _objective_gram(G_flat: np.ndarray, beta: float = 20.0) -> float:
    """
    Objective function for optimizing the Gram matrix.
    We want to maximize (d_min / d_max)^2.
    Equivalent to minimizing -log((d_min / d_max)^2) = 2*(log(d_max) - log(d_min)).
    Or simply minimize - (d_min / d_max)^2.
    Using a smooth approximation for d_min and d_max.
    """
    n = 14
    G = G_flat.reshape(n, n)
    
    # Ensure symmetry
    G = (G + G.T) / 2.0
    
    # Ensure diagonal is 1 (project if necessary? Or just enforce via penalty?)
    # For optimization stability, we can add a penalty for G_ii != 1
    diag_diff = np.diag(G) - 1.0
    penalty = np.sum(diag_diff**2)
    
    # Check positive semi-definiteness
    eigvals = np.linalg.eigvalsh(G)
    if np.min(eigvals) < -1e-6:
        return 1e10 + penalty
    
    dists = _distances_from_gram(G)
    
    if len(dists) == 0:
        return 1e10 + penalty
        
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 1e10 + penalty
        
    # Smooth min/max
    # Soft min: -log(sum(exp(-beta*d))) / beta
    # Soft max: log(sum(exp(beta*d))) / beta
    # To avoid overflow, shift by min/max
    shifted_min = -beta * (dists - d_min)
    shifted_min = np.clip(shifted_min, -700, 700)
    sum_exp_min = np.sum(np.exp(shifted_min))
    d_min_soft = d_min - np.log(sum_exp_min) / beta
    
    shifted_max = beta * (dists - d_max)
    shifted_max = np.clip(shifted_max, -700, 700)
    sum_exp_max = np.sum(np.exp(shifted_max))
    d_max_soft = d_max + np.log(sum_exp_max) / beta
    
    if d_max_soft < 1e-10:
        return 1e10 + penalty
        
    ratio = d_min_soft / d_max_soft
    return -ratio * ratio + penalty

def _refine_gram(G: np.ndarray, start_time: float, deadline: float) -> np.ndarray:
    """Refine the Gram matrix using L-BFGS-B."""
    if time.time() > deadline - 5:
        return G
        
    n = 14
    G_flat = G.flatten()
    
    try:
        result = minimize(
            _objective_gram,
            G_flat,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if result.success:
            G_refined = result.x.reshape(n, n)
            # Symmetrize
            G_refined = (G_refined + G_refined.T) / 2.0
            # Fix diagonal
            np.fill_diagonal(G_refined, 1.0)
            return G_refined
    except Exception:
        pass
        
    return G

def _generate_initial_grams() -> list:
    """Generate initial Gram matrices based on symmetric configurations."""
    grammars = []
    
    # 1. Icosahedron + 2 poles (12 + 2 = 14)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    pts = np.vstack([ico, poles])
    G = _points_to_gram(pts)
    grammars.append(G)
    
    # 2. Two octahedra rotated? (8+8=16, too many). 
    # Try 7 antipodal pairs.
    # Random symmetric start
    rng = np.random.RandomState(42)
    for _ in range(10):
        # Generate 7 random points, add their antipodes
        pts_half = rng.randn(7, 3)
        pts_half /= np.linalg.norm(pts_half, axis=1, keepdims=True)
        pts = np.vstack([pts_half, -pts_half])
        G = _points_to_gram(pts)
        grammars.append(G)
        
    return grammars

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_metric = -1.0
    
    initial_grams = _generate_initial_grams()
    
    # Simulated Annealing / Random Perturbation on Gram Matrix
    # We will iterate through the initial grams and refine them.
    # Also, we can do a "hill climbing" in the Gram space.
    
    # To keep it fast and robust, we will just refine each initial gram with L-BFGS-B.
    # If we have time, we can do multiple restarts.
    
    for i, G in enumerate(initial_grams):
        if time.time() > deadline - 10:
            break
            
        # Refine
        G_refined = _refine_gram(G, start_time, deadline)
        
        # Reconstruct points
        P = _gram_to_points(G_refined)
        if P is None:
            continue
            
        metric = _evaluate(P)
        
        if metric > best_metric:
            best_metric = metric
            best_points = P.copy()
            
        # If we have significant time left, try to perturb and refine again
        if time.time() < deadline - 60 and i < 5:
            # Generate a few perturbations of the current best
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                # Perturb G slightly
                G_pert = G_refined + 0.01 * np.random.randn(14, 14)
                G_pert = (G_pert + G_pert.T) / 2.0
                np.fill_diagonal(G_pert, 1.0)
                
                G_pert_refined = _refine_gram(G_pert, start_time, deadline)
                P_pert = _gram_to_points(G_pert_refined)
                if P_pert is not None:
                    m_pert = _evaluate(P_pert)
                    if m_pert > best_metric:
                        best_metric = m_pert
                        best_points = P_pert.copy()
                        G_refined = G_pert_refined # Update for next perturbation base

    # Fallback
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico /= np.linalg.norm(ico, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        
    # Final check
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico /= np.linalg.norm(ico, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
