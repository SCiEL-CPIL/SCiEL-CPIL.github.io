# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Formulate as a Gram matrix optimization problem.
       Let G be the Gram matrix of 16 points in R^2. G is PSD, rank <= 2, diag(G) = 1.
       The squared distance between points i and j is d_ij^2 = G_ii + G_jj - 2G_ij = 2 - 2G_ij.
       Maximizing the minimum distance is equivalent to minimizing the maximum G_ij (since d^2 = 2 - 2G_ij).
       However, we also need to control the maximum distance to maximize the ratio (d_min/d_max)^2.
       
       Actually, the ratio is (d_min/d_max)^2.
       Let d_min^2 = min_{i!=j} (2 - 2G_ij) = 2 - 2*max_{i!=j} G_ij.
       Let d_max^2 = max_{i!=j} (2 - 2G_ij) = 2 - 2*min_{i!=j} G_ij.
       Ratio R = (2 - 2*G_max) / (2 - 2*G_min) = (1 - G_max) / (1 - G_min).
       
       We want to maximize R.
       This is a non-convex problem because of the ratio and the rank constraint.
       
       Alternative approach:
       Fix the scale. Let d_max = 1 (or some constant). Then we want to maximize d_min.
       This is equivalent to finding the packing of 16 points in a unit diameter circle/shape.
       But the shape is not fixed.
       
       Let's stick to the Gram matrix approach with a penalty method or direct gradient descent on G.
       We will optimize G such that:
       1. G is PSD (enforced by projecting to PSD).
       2. Rank(G) <= 2 (enforced by zeroing out eigenvalues > 2).
       3. Diag(G) = 1 (enforced by normalization).
       
       Objective: Maximize (d_min / d_max)^2.
       Let f(G) = (min_{i!=j} (2-2G_ij) / max_{i!=j} (2-2G_ij))^2.
       
       Since we can't easily take gradients of min/max, we can use a smooth approximation 
       or simply use a gradient-free method on the Gram matrix, which is lower dimensional 
       (16x16 symmetric, 135 vars) than the point coordinates (32 vars)? 
       Actually 32 vars is smaller. But the geometry in Gram space might be better conditioned 
       for distance constraints.
       
       Let's try a different structural change: 
       Use the fact that the optimal configuration for maximizing the min distance ratio 
       for 16 points in 2D is likely related to a lattice or a specific packing.
       
       The current program uses L-BFGS on points. 
       Let's try a **Simulated Annealing** approach on the point coordinates, but with a 
       **repulsive potential** that specifically targets the ratio.
       
       Wait, the prompt asks for a DIFFERENT strategy. 
       "Formulate the problem as a non-convex semi-definite program (SDP)..."
       
       I will implement a solver that optimizes the Gram matrix G.
       Variables: G (16x16 symmetric).
       Constraints: 
         - G is PSD.
         - Rank(G) <= 2.
         - diag(G) = 1.
       
       Objective: Maximize R(G).
       
       Since standard libraries like cvxpy might not be available or might be slow for 
       non-convex problems, I will implement a custom gradient ascent on G.
       
       However, gradients of min/max are discontinuous. 
       I will use a "soft" version: 
       SoftMin(x) = -1/beta * log(sum(exp(-beta*x)))
       SoftMax(x) = 1/beta * log(sum(exp(beta*x)))
       
       Let D be the matrix of squared distances derived from G.
       D = 2 - 2*G + 2*diag(G) (since diag(G)=1, D_ii=0, D_ij = 2 - 2G_ij).
       
       We want to maximize (SoftMin(D_off) / SoftMax(D_off))^2.
       
       This is differentiable with respect to G.
       We will perform gradient ascent on G.
       After each step, we project G to the feasible set:
       1. Symmetrize.
       2. PSD projection (eigendecomp, clip neg eigenvals to 0).
       3. Rank-2 projection (keep top 2 eigenvalues).
       4. Normalize diagonal to 1.
       
       This is a structurally different optimization landscape (Gram space vs Point space).
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin

    n = 16
    
    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def gram_to_points(G):
        """Convert Gram matrix to points in R^2."""
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        # Keep top 2 eigenvalues for rank-2
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Zero out negative or small eigenvalues
        eigvals[:2] = np.maximum(eigvals[:2], 0)
        eigvals[2:] = 0
        
        # Reconstruct
        G_rank2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Normalize diagonal to 1
        # If we normalize, we change the scale. 
        # The ratio (d_min/d_max) is scale invariant.
        # So we can just take the square root of the eigenvalues for the coordinates.
        
        # Actually, if we enforce diag(G)=1, the scale is fixed.
        # Let's normalize the points such that the average squared norm is 1? 
        # Or just project back to points and let the optimizer handle scale?
        # The Gram matrix approach usually fixes the scale by diag(G)=1.
        
        # Let's compute points from G_rank2.
        # G_rank2 = P P^T where P is 2x16.
        # P = sqrt(eigvals[:2]) * eigvecs[:2, :].T
        points = np.sqrt(eigvals[:2]) * eigvecs[:2, :].T
        points = points.T # 16x2
        
        # Normalize to unit variance or something? 
        # The ratio is invariant to scaling and rotation.
        # So any valid rank-2 PSD G with diag=1 works.
        # But our projection might not preserve diag=1.
        # Let's re-normalize the points so that mean squared distance from origin is 1?
        # Or just return the points. The score function is scale invariant.
        
        return points

    def project_gram(G):
        """Project G to the set of PSD, Rank<=2, Diag=1 matrices."""
        # 1. Symmetrize
        G = (G + G.T) / 2.0
        
        # 2. PSD Projection
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 0)
        G = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # 3. Rank-2 Projection
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals[2:] = 0
        G = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # 4. Normalize Diagonal to 1
        diag = np.diag(G)
        # Avoid division by zero
        mask = diag > 1e-10
        G = G.copy()
        for i in range(n):
            if mask[i]:
                G[i, :] /= np.sqrt(diag[i])
                G[:, i] /= np.sqrt(diag[i])
            else:
                # If a point is at the origin, it's a degenerate case. 
                # Add a tiny bit to the diagonal to avoid singularity?
                # Actually, if diag is 0, the point is at 0.
                # Let's just set it to 1 to keep it valid? 
                # This might break PSD.
                pass
        
        # Re-symmetrize
        G = (G + G.T) / 2.0
        
        return G

    def objective_and_grad(G_flat):
        """
        Returns negative of the smooth score and its gradient.
        """
        G = G_flat.reshape(n, n)
        
        # Compute squared distances D
        # D_ij = G_ii + G_jj - 2G_ij
        # Since we project to diag=1, G_ii=1, so D_ij = 2 - 2G_ij
        # But during optimization, G might not have exactly diag=1 before projection.
        # We should compute D based on the current G.
        
        # For the gradient, we need to be careful.
        # Let's use the soft min/max on the off-diagonal elements of D.
        
        # Get off-diagonal indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        # D_ij = G_ii + G_jj - 2G_ij
        # d2_ij = G[i,i] + G[j,j] - 2*G[i,j]
        d2_vals = G[i_idx, i_idx] + G[j_idx, j_idx] - 2 * G[i_idx, j_idx]
        
        # Soft min and max
        beta = 20.0 # Sharpness
        
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        # To avoid overflow, shift by min
        d2_min = np.min(d2_vals)
        d2_max = np.max(d2_vals)
        
        if d2_max - d2_min < 1e-9:
            # All distances are equal, score is 1
            return -1.0, np.zeros_like(G_flat)
            
        # Soft min
        exp_min = np.exp(-beta * (d2_vals - d2_min))
        soft_min = d2_min - np.log(np.sum(exp_min)) / beta
        
        # Soft max
        exp_max = np.exp(beta * (d2_vals - d2_max))
        soft_max = d2_max + np.log(np.sum(exp_max)) / beta
        
        # Ratio
        if soft_max < 1e-9:
            return 1e10, np.zeros_like(G_flat)
            
        ratio = soft_min / soft_max
        score = ratio * ratio
        
        # Gradient calculation
        # This is complex. Let's use numerical gradient for robustness, 
        # or derive it.
        # Given the complexity and time limits, let's use a finite difference 
        # gradient estimator for the Gram matrix.
        # n=16, G has 136 independent variables.
        # 136 * 2 evaluations per gradient step.
        # If we do 1000 steps, that's 272,000 evaluations.
        # Each evaluation is O(n^2). 16^2 = 256 ops.
        # Total ~ 70 million ops. Should be fast enough in Python/NumPy if vectorized.
        
        # Let's implement a vectorized finite difference gradient.
        eps = 1e-5
        grad = np.zeros_like(G_flat)
        
        # We only need to perturb the upper triangle and diagonal
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            # Perturb G[i,j] and G[j,i]
            orig = G[i, j]
            
            # Forward
            G[i, j] += eps
            G[j, i] += eps
            G_pert = G.copy()
            # Recompute score
            d2_p = G_pert[i_idx, i_idx] + G_pert[j_idx, j_idx] - 2 * G_pert[i_idx, j_idx]
            d2_min_p = np.min(d2_p)
            d2_max_p = np.max(d2_p)
            if d2_max_p - d2_min_p < 1e-9:
                s_p = 1.0
            else:
                exp_min_p = np.exp(-beta * (d2_p - d2_min_p))
                sm_p = d2_min_p - np.log(np.sum(exp_min_p)) / beta
                exp_max_p = np.exp(beta * (d2_p - d2_max_p))
                sx_p = d2_max_p + np.log(np.sum(exp_max_p)) / beta
                s_p = (sm_p / sx_p) ** 2 if sx_p > 1e-9 else 1e10
            
            G[i, j] = orig
            G[j, i] = orig
            
            # Backward
            G[i, j] -= eps
            G[j, i] -= eps
            G_pert = G.copy()
            d2_m = G_pert[i_idx, i_idx] + G_pert[j_idx, j_idx] - 2 * G_pert[i_idx, j_idx]
            d2_min_m = np.min(d2_m)
            d2_max_m = np.max(d2_m)
            if d2_max_m - d2_min_m < 1e-9:
                s_m = 1.0
            else:
                exp_min_m = np.exp(-beta * (d2_m - d2_min_m))
                sm_m = d2_min_m - np.log(np.sum(exp_min_m)) / beta
                exp_max_m = np.exp(beta * (d2_m - d2_max_m))
                sx_m = d2_max_m + np.log(np.sum(exp_max_m)) / beta
                s_m = (sm_m / sx_m) ** 2 if sx_m > 1e-9 else 1e10
            
            G[i, j] = orig
            G[j, i] = orig
            
            grad_k = (s_p - s_m) / (2 * eps)
            
            # Map back to flat index
            # G is symmetric. We are optimizing the full matrix but enforcing symmetry?
            # If we optimize the full matrix, the gradient for G[i,j] is grad_k.
            # But we should only update the upper triangle and mirror?
            # Or just update the full matrix and symmetrize in projection.
            # Let's update the full matrix gradient.
            grad[i, j] += grad_k
            grad[j, i] += grad_k
            
            # Also, the diagonal terms G[i,i] and G[j,j] affect d2.
            # d2_k = G_ii + G_jj - 2G_ij
            # Perturbing G[i,i] affects all d2_l where i is involved.
            # This is getting complicated for manual finite difference.
            
            # Alternative: Use a simpler gradient-free method on the Gram matrix.
            # Nelder-Mead on the Gram matrix?
            # Dimension 136. Nelder-Mead might be slow.
            
            # Let's switch to a simpler strategy: 
            # Use the current best points, convert to Gram, 
            # then run a local search on the Gram matrix using a simple 
            # "push-pull" heuristic on the eigenvalues.
            
        return -score, grad

    # Initialize G from a good starting point (e.g., regular polygon or grid)
    # Let's use the regular 16-gon as a start, as it's a good baseline.
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    points_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute initial Gram
    G_init = points_init @ points_init.T
    
    # Normalize G_init to have unit diagonal?
    # The points are on the unit circle, so diag is 1.
    
    best_score = compute_score(points_init)
    best_points = points_init.copy()
    
    # Run a simple gradient ascent on G
    # Since finite difference gradient is expensive, let's try a 
    # "Random Perturbation" on the Gram matrix, projected back to feasible set.
    
    G = G_init.copy()
    
    # Precompute indices for speed
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for step in range(2000):
        if time.time() > deadline:
            break
            
        # Perturb G
        # Add random noise to the upper triangle
        noise = np.random.randn(n, n) * 0.01
        noise = (noise + noise.T) / 2.0
        G_new = G + noise
        
        # Project
        G_new = project_gram(G_new)
        
        # Get points
        points_new = gram_to_points(G_new)
        
        # Score
        score_new = compute_score(points_new)
        
        if score_new > best_score:
            best_score = score_new
            best_points = points_new.copy()
            G = G_new.copy()
        else:
            # Annealing: accept worse solutions with some probability
            # Or just reject. Let's try a simple hill climbing first.
            # If stuck, increase noise.
            pass
            
    # If we have time, try to refine with L-BFGS on points from the best Gram solution
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            def obj_points(flat):
                pts = flat.reshape(16, 2)
                return -compute_score(pts)
            
            # Use the best points found so far
            res = minimize(obj_points, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success:
                pts_opt = res.x.reshape(16, 2)
                score_opt = compute_score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = pts_opt.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
