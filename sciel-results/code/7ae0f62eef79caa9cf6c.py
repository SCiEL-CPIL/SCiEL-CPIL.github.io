"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def objective_gram(G_flat):
        """
        SDP relaxation objective: minimize ratio of max to min squared distance
        subject to G being PSD, rank <= d, and trace normalized.
        """
        G = G_flat.reshape(n, n)
        G = (G + G.T) / 2
        
        # Eigenvalue decomposition for PSD and rank constraints
        eigvals = np.linalg.eigvalsh(G)
        
        # PSD penalty: penalize negative eigenvalues
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: eigenvalues beyond rank d should be zero
        rank_penalty = np.sum(eigvals[d:] ** 2)
        
        # Compute squared distances
        d_sq = get_distance_vector(G)
        
        # If distances are invalid (negative), return large penalty
        if np.any(d_sq < -1e-8):
            return 1e10 + psd_penalty + rank_penalty
        
        # Clip to positive for log computation
        d_sq_pos = np.maximum(d_sq, 1e-10)
        
        # Use soft-min and soft-max with temperature
        T = 0.05
        # Soft max: T * log(sum(exp(d_sq/T)))
        max_d_sq = T * np.log(np.sum(np.exp(d_sq_pos / T)))
        # Soft min: -T * log(sum(exp(-d_sq/T)))
        min_d_sq = -T * np.log(np.sum(np.exp(-d_sq_pos / T)))
        
        if min_d_sq < 1e-10:
            return 1e10 + psd_penalty + rank_penalty
        
        ratio = max_d_sq / min_d_sq
        
        # Scale normalization: fix trace(G) = n to avoid trivial scaling
        trace_G = np.trace(G)
        scale_penalty = (trace_G - float(n)) ** 2
        
        # Weighted objective
        w_rank = 100.0
        w_psd = 100.0
        w_scale = 10.0
        
        return ratio + w_rank * rank_penalty + w_psd * psd_penalty + w_scale * scale_penalty

    def extract_points_from_gram(G):
        """Extract d-dimensional points from Gram matrix G."""
        G = (G + G.T) / 2
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep top d eigenvectors with positive eigenvalues
        V = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))
        return V

    def objective_coords(x):
        """Direct coordinate optimization objective."""
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        T = 0.05
        max_d_sq = T * np.log(np.sum(np.exp(d_sq / T)))
        min_d_sq = -T * np.log(np.sum(np.exp(-d_sq / T)))
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    # Generate multiple initializations for the Gram matrix
    np.random.seed(42)
    initializations = []
    
    # 1. Random rank-d Gram matrix
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    initializations.append(G_rand.flatten())
    
    # 2. Regular simplex in d dimensions (projected to n points)
    # For n points in d=3, use a configuration close to regular simplex
    # Start with random and let optimization work
    V_simplex = np.random.randn(n, d)
    V_simplex -= V_simplex.mean(axis=0)  # Center
    G_simplex = V_simplex @ V_simplex.T
    initializations.append(G_simplex.flatten())
    
    # 3. Another random initialization with different seed
    np.random.seed(123)
    V_rand2 = np.random.randn(n, d) * 0.5
    G_rand2 = V_rand2 @ V_rand2.T
    initializations.append(G_rand2.flatten())
    
    # 4. Initialize with points on a sphere (uniform distribution)
    phi = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, np.pi, n)
    V_sphere = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    G_sphere = V_sphere @ V_sphere.T
    initializations.append(G_sphere.flatten())
    
    best_points = None
    best_score = 1e10
    
    for G0_flat in initializations:
        try:
            # Optimize the Gram matrix
            res_gram = minimize(objective_gram, G0_flat, method='L-BFGS-B',
                               options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if not res_gram.success and res_gram.fun > 1e9:
                continue
                
            G_opt = res_gram.x.reshape(n, n)
            G_opt = (G_opt + G_opt.T) / 2
            
            # Extract points
            points = extract_points_from_gram(G_opt)
            
            # Refine in coordinate space
            res_coords = minimize(objective_coords, points.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res_coords.success or res_coords.fun < 1e9:
                refined_points = res_coords.x.reshape(n, d)
                score = _compute_score(refined_points)
                
                if score < best_score:
                    best_score = score
                    best_points = refined_points
                    
        except Exception as e:
            continue
    
    # If no valid solution found, return random points
    if best_points is None:
        return np.random.randn(n, d)
    
    return best_points

# EVOLVE-BLOCK-END
