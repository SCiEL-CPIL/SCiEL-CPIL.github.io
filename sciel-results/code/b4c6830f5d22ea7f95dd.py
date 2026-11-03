import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    # 1. Initialization: Hexagonal lattice
    # We generate a grid of points and select the N closest to the origin
    # to ensure a good starting packing.
    
    # Estimate spacing
    # Area of unit circle = pi. Area per circle ~ pi * (0.05)^2 ~ 0.008.
    # 300 circles * 0.008 = 2.4. This is larger than pi=3.14? No, 300 * pi * r^2 <= pi => r <= 1/sqrt(300) ~ 0.057.
    # So r ~ 0.053 is close to theoretical limit.
    
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by radius and take first N
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98

    # Ensure points are numpy array of shape (N, 2)
    x0 = points.flatten()
    
    # 2. Optimization
    # We define the objective and gradient functions
    
    def get_points(x):
        return x.reshape(N, 2)

    def compute_distances(points):
        # Returns list of (index_i, index_j, value, grad_i, grad_j) for pairs
        # And (index_i, value, grad_i) for boundaries
        # For efficiency, we compute all pairwise distances in a matrix
        
        # Pairwise distances
        # dist[i, j] = |c_i - c_j|
        # We need gradients for softmin.
        
        # Let's compute the softmin objective and gradient directly in vectorized form if possible,
        # or loop over pairs for gradient accumulation.
        # N=300, N^2=90k. Looping in Python is slow. Vectorize!
        
        # Compute all pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal and upper triangle
        # We only care about i < j
        iu = np.triu_indices(N, k=1)
        d_ij = dists[iu]
        
        # Boundary distances
        d_b = 1.0 - np.linalg.norm(points, axis=1)
        
        # Combine all values
        # v_k are the values to take softmin of
        # v = [d_ij / 2, d_b]
        v = np.concatenate([d_ij / 2.0, d_b])
        
        return v, diff, dists, d_ij, d_b, iu

    def objective_and_grad(x, beta):
        points = get_points(x)
        v, diff, dists, d_ij, d_b, iu = compute_distances(points)
        
        # Softmin calculation
        # To avoid overflow/underflow, subtract min(v)
        v_min = np.min(v)
        exp_v = np.exp(-beta * (v - v_min))
        S = np.sum(exp_v)
        weights = exp_v / S
        
        softmin_val = v_min - (1.0 / beta) * np.log(S)
        
        # Gradient of softmin wrt x
        # grad_softmin = sum_k weights[k] * grad(v_k)
        
        # Initialize gradient array
        grad_x = np.zeros_like(x)
        grad_points = grad_x.reshape(N, 2)
        
        # Gradients for pairwise distances
        # v_k = d_ij / 2
        # grad_{c_i} (d_ij / 2) = (c_i - c_j) / (2 * d_ij)
        # grad_{c_j} (d_ij / 2) = (c_j - c_i) / (2 * d_ij)
        
        # We have weights for d_ij part: weights[:N*(N-1)/2]
        w_pairs = weights[:len(d_ij)]
        
        # We need to accumulate gradients.
        # diff is (N, N, 2). diff[i, j] = c_i - c_j.
        # d_ij corresponds to diff[i, j] for i<j.
        
        # Create a matrix for weights, same shape as diff
        w_mat = np.zeros((N, N))
        w_mat[iu] = w_pairs
        # Symmetry: w_mat[j, i] should also be w_pairs? 
        # The softmin sum is over unique pairs i<j.
        # So the term is only in the upper triangle.
        # However, the gradient with respect to c_i involves all pairs (i, j) where j != i.
        # If we sum over i<j, for a fixed i, we sum over j>i and j<i.
        
        # Let's just use the full symmetric weight matrix for the gradient calculation.
        # The term in the sum is w_{ij} * grad_{c_i} (d_{ij}/2).
        # grad_{c_i} (d_{ij}/2) = (c_i - c_j) / (2 d_{ij}).
        # Note that w_{ij} is the weight of the pair (i,j).
        # Since the objective is sum over i<j w_{ij} (d_{ij}/2), 
        # the gradient wrt c_i is sum_{j!=i} w_{min(i,j), max(i,j)} * (c_i - c_j) / (2 d_{ij}).
        
        # So we can create a symmetric weight matrix W where W[i,j] = w_{ij} if i<j else w_{ji} (which is same value).
        w_sym = w_mat + w_mat.T
        
        # Avoid division by zero
        # d_safe = dists + 1e-9
        d_safe = np.where(dists < 1e-9, 1e-9, dists)
        
        # grad_points[i] = sum_j w_sym[i,j] * (c_i - c_j) / (2 * d_safe[i,j])
        # = sum_j w_sym[i,j] * diff[i,j] / (2 * d_safe[i,j])
        
        # diff is (N, N, 2). d_safe is (N, N).
        # We want to compute sum over axis 1.
        
        # Factor = w_sym / (2 * d_safe)
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(factor := w_sym / (2.0 * d_safe), 0.0)
        
        # grad_points[i] = sum_j factor[i,j] * diff[i,j]
        # np.einsum('ij,ijk->ik', factor, diff)
        grad_points = np.einsum('ij,ijk->ik', factor, diff)
        
        # Gradients for boundary distances
        # v_k = d_b = 1 - |c_i|
        # grad_{c_i} d_b = - c_i / |c_i|
        
        w_bound = weights[len(d_ij):]
        # d_b is (N,)
        # norms = np.linalg.norm(points, axis=1)
        # Avoid div by zero
        norms = np.linalg.norm(points, axis=1)
        norms_safe = np.where(norms < 1e-9, 1e-9, norms)
        
        # grad_bound_points = - w_bound[:, np.newaxis] * points / norms_safe[:, np.newaxis]
        grad_bound_points = - (w_bound[:, np.newaxis] * points) / norms_safe[:, np.newaxis]
        
        total_grad_points = grad_points + grad_bound_points
        
        # Objective is -softmin
        obj_val = -softmin_val
        grad_val = -total_grad_points.flatten()
        
        return obj_val, grad_val

    best_x = x0
    best_score = -1e9
    
    # Calculate initial score
    def calc_score(x):
        p = get_points(x)
        # min pairwise / 2
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        # min boundary
        min_bound = np.min(1.0 - np.linalg.norm(p, axis=1))
        return min(min_pair, min_bound)
    
    best_score = calc_score(best_x)
    
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Beta schedule
    betas = [10, 50, 200, 1000, 5000]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Run optimization for this beta
        # Limit time per beta? Or just let it run and check globally?
        # L-BFGS-B has maxiter. Let's set maxiter to something reasonable.
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_x,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 500}
            )
            
            if res.success or res.fun < -best_score: # Wait, obj is -softmin, so lower is better.
                # Check actual score
                new_score = calc_score(res.x)
                if new_score > best_score:
                    best_score = new_score
                    best_x = res.x.copy()
                    current_x = res.x.copy()
                    
        except Exception as e:
            pass # Keep current best
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_x.reshape(N, 2)
