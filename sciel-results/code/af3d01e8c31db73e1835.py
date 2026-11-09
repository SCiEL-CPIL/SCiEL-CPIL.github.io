"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _radius(pts):
    """Compute the valid packing radius for a set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return 0.0
    # Compute minimum distance between pairs
    min_pair = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair:
                min_pair = d
    # Compute minimum distance to boundary
    min_bound = np.inf
    for i in range(N):
        for k in range(2):
            d = 0.5 - abs(pts[i, k])
            if d < min_bound:
                min_bound = d
    r = min(min_pair / 2.0, min_bound)
    return r

def _score(pts):
    return _radius(pts)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Start with a grid layout
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best = grid[:N].copy()
    best_score = _score(best)
    
    # Multi-start local optimization using gradient-free approach
    # We'll use a simple repulsion-based relaxation with boundary constraints
    
    def _relax(pts, n_iter=20000, learning_rate=0.001):
        """Relax points to maximize minimum distance using simple repulsion."""
        pts = pts.copy()
        lr = learning_rate
        
        for it in range(n_iter):
            # Compute forces
            forces = np.zeros_like(pts)
            min_dist = np.inf
            
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-10:
                        # Random perturbation for coincident points
                        diff = np.array([np.random.randn(), np.random.randn()])
                        d = np.linalg.norm(diff)
                    
                    # Repulsion force (stronger when close)
                    if d < 0.2:
                        f = diff / d * (0.2 - d)
                        forces[i] += f
                        forces[j] -= f
                    
                    if d < min_dist:
                        min_dist = d
                
                # Boundary repulsion
                for k in range(2):
                    if pts[i, k] < -0.4:
                        forces[i, k] += (-0.4 - pts[i, k]) * 0.5
                    elif pts[i, k] > 0.4:
                        forces[i, k] -= (pts[i, k] - 0.4) * 0.5
            
            # Update positions
            pts += lr * forces
            
            # Project back into square
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
            
            # Check time
            if time.time() - start_time > deadline - 5:
                break
            
            # Early stopping if converged
            if min_dist > 0.14 and it > 5000:
                break
        
        return pts
    
    # Multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate initial configuration
        if restart == 0:
            pts = best.copy()
        elif restart == 1:
            # Grid with slight perturbation
            pts = grid[:N].copy()
            pts += np.random.randn(N, 2) * 0.01
            pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
            pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        else:
            # Random initialization
            pts = np.random.uniform(-0.45, 0.45, (N, 2))
        
        # Relax
        pts = _relax(pts, n_iter=30000, learning_rate=0.0005)
        
        # Evaluate
        score = _score(pts)
        if score > best_score:
            best_score = score
            best = pts.copy()
    
    # Final polish with smaller steps
    if time.time() - start_time < deadline - 20:
        best = _relax(best, n_iter=50000, learning_rate=0.0001)
        final_score = _score(best)
        if final_score > best_score:
            best_score = final_score
    
    return best
# EVOLVE-BLOCK-END
