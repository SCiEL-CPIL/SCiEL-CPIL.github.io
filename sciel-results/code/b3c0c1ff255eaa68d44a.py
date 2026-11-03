import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    def _pairwise_distances(points):
        """Compute all pairwise distances between N points."""
        n = len(points)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist = np.sqrt(np.dot(diff, diff))
                d[i, j] = dist
                d[j, i] = dist
        return d
    
    def _score(points):
        """Compute the packing score (minimum radius)."""
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Distance to boundary
        dist_to_center = np.linalg.norm(points, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 0.0
        
        # Minimum distance between any two centers
        d = _pairwise_distances(points)
        min_dist = np.min(d[np.triu_indices(N, k=1)])
        
        r1 = min_dist / 2.0
        r2 = 1.0 - max_dist
        
        return min(r1, r2)
    
    def _softmin_logsumexp(arr, beta):
        """Soft-min using log-sum-exp for smoothness."""
        beta = max(beta, 1e-6)
        scaled = -beta * arr
        max_val = np.max(scaled)
        if max_val < -100:
            return -np.min(arr)
        shifted = scaled - max_val
        log_sum = np.log(np.sum(np.exp(shifted)))
        return -(max_val + log_sum) / beta
    
    def _smooth_objective(params, beta=50.0):
        """
        Smooth surrogate for the packing objective.
        Uses soft-min to make the objective differentiable.
        params: flattened array of 2*N coordinates
        Returns: negative smooth score (to minimize)
        """
        points = params.reshape(N, 2)
        
        # Check bounds
        if np.any(np.abs(points) > 1.5):
            return 100.0
        
        # Pairwise distances
        d = _pairwise_distances(points)
        upper_tri = d[np.triu_indices(N, k=1)]
        
        # Boundary distances
        dist_to_center = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - dist_to_center
        
        # Combine all constraints into one array
        # We want: min(min_pairwise_dist/2, min_boundary_dist)
        # Use soft-min over all relevant distances
        all_dists = np.concatenate([upper_tri / 2.0, boundary_dists])
        
        # Soft-min gives smooth approximation of min
        soft_min = _softmin_logsumexp(all_dists, beta)
        
        return -soft_min
    
    def _ring_initialization():
        """Generate initial points with ring structure."""
        points = np.zeros((N, 2))
        
        # Ring 0: Center
        points[0] = [0, 0]
        
        # Ring 1: 6 points
        R1 = 0.25
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            points[1 + i] = [R1 * np.cos(angles1[i]), R1 * np.sin(angles1[i])]
            
        # Ring 2: 12 points
        R2 = 0.50
        angles2 = np.linspace(0, 2*np.pi, 12, endpoint=False) + np.pi/12
        for i in range(12):
            points[7 + i] = [R2 * np.cos(angles2[i]), R2 * np.sin(angles2[i])]
            
        # Ring 3: 18 points
        R3 = 0.75
        angles3 = np.linspace(0, 2*np.pi, 18, endpoint=False) + np.pi/18
        for i in range(18):
            points[19 + i] = [R3 * np.cos(angles3[i]), R3 * np.sin(angles3[i])]
            
        # Ring 4: 13 points
        R4 = 0.95
        angles4 = np.linspace(0, 2*np.pi, 13, endpoint=False) + np.pi/13
        for i in range(13):
            points[37 + i] = [R4 * np.cos(angles4[i]), R4 * np.sin(angles4[i])]
            
        return points
    
    def _force_directed_optimize(points, max_iter=5000, T0=0.01, T_min=1e-6, cooling=0.999):
        """
        Force-directed layout optimization with simulated annealing.
        Minimizes repulsive potential to find densest configuration.
        """
        n = len(points)
        best_points = points.copy()
        best_score = _score(points)
        T = T0
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Compute forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1, n):
                    diff = points[i] - points[j]
                    dist = np.sqrt(np.dot(diff, diff))
                    if dist < 1e-10:
                        # Random small perturbation
                        diff = np.random.randn(2) * 0.001
                        dist = np.sqrt(np.dot(diff, diff))
                    
                    # Repulsive force (inverse square)
                    if dist < 0.3:
                        f = 1.0 / (dist * dist + 1e-10)
                    else:
                        f = 0.1 / (dist * dist + 1e-10)
                    
                    force = f * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Boundary attraction (keep points inside)
            for i in range(n):
                dist_to_center = np.linalg.norm(points[i])
                if dist_to_center > 0.9:
                    # Pull back toward center
                    pull = (dist_to_center - 0.9) * 0.5
                    forces[i] -= pull * points[i] / (dist_to_center + 1e-10)
            
            # Propose new position with noise
            step_size = T
            new_points = points + forces * 0.01 + np.random.randn(n, 2) * step_size
            
            # Evaluate
            new_score = _score(new_points)
            
            if new_score >= best_score:
                points = new_points
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            else:
                # Accept with probability exp((new_score - best_score) / T)
                delta = new_score - best_score
                if delta < 0:
                    accept_prob = np.exp(delta / max(T, 1e-10))
                    if np.random.random() < accept_prob:
                        points = new_points
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_points, best_score
    
    def _local_refine(points, max_iter=2000):
        """
        Local refinement using gradient-based optimization.
        Uses smooth surrogate with increasing beta for precision.
        """
        params = points.flatten()
        
        # Start with moderate beta, increase for final polish
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 20:
                break
            
            try:
                result = minimize(
                    _smooth_objective, 
                    params, 
                    args=(beta,),
                    method='L-BFGS-B', 
                    options={'maxiter': max_iter // len(betas), 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if result.fun < -_score(points):
                    new_points = result.x.reshape(N, 2)
                    if _score(new_points) > _score(points):
                        points = new_points
            except Exception:
                pass
        
        return points
    
    # Main optimization loop
    best_points = None
    best_score = 0.0
    
    # Phase 1: Ring-structured initialization
    points = _ring_initialization()
    
    # Phase 2: Force-directed optimization with multiple restarts
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        # Add small perturbation for diversity
        if restart > 0:
            points = points + np.random.randn(N, 2) * 0.02
            # Keep inside bounds
            norms = np.linalg.norm(points, axis=1)
            mask = norms > 0.98
            points[mask] *= 0.98 / norms[mask, np.newaxis]
        
        # Force-directed optimization
        fd_points, fd_score = _force_directed_optimize(
            points, 
            max_iter=2000, 
            T0=0.005, 
            T_min=1e-7, 
            cooling=0.9995
        )
        
        # Local refinement
        refined_points = _local_refine(fd_points, max_iter=1000)
        refined_score = _score(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
        
        # Use best as starting point for next restart
        points = best_points.copy()
    
    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 25:
        # Very tight local refinement
        final_points = _local_refine(best_points, max_iter=5000)
        final_score = _score(final_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Validate and return
    if best_points is None or len(best_points) != N or not np.all(np.isfinite(best_points)):
        # Emergency fallback: simple hexagonal lattice
        points = _ring_initialization()
        best_points = points
        best_score = _score(points)
    
    # Final validation: ensure all points are inside unit circle
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms >= 1.0):
        # Scale down
        scale = 0.999 / np.max(norms)
        best_points *= scale
    
    return best_points
# EVOLVE-BLOCK-END
