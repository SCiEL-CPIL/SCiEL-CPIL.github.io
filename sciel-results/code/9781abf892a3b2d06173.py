import numpy as np
import time
from scipy.optimize import minimize
import scipy.spatial.distance as spd

N = 50

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # 1. Langevin Dynamics Phase
    # We simulate a system of N particles repelling each other and attracted to the center.
    # The potential is V = sum(1/d^12) for pairs + sum(1/(1-r)^12) for walls.
    # This drives the system to a state where distances are maximized.
    
    # Parameters
    dt = 0.005
    gamma = 0.1  # Friction
    k = 1.0      # Spring constant for centering (weak)
    T_init = 0.5
    T_final = 0.01
    steps_per_temp = 500
    
    # Initialize random positions
    # Ensure they are inside the unit circle
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9 # Keep away from edge initially
    pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    vel = np.zeros((N, 2))
    
    best_pos = pos.copy()
    best_score = _calc_score(pos)
    
    # Annealing schedule
    n_anneal_steps = 10
    temps = np.linspace(T_init, T_final, n_anneal_steps)
    
    for T in temps:
        if time.time() > deadline - 30: # Leave 30s for polish
            break
        for _ in range(steps_per_temp):
            if time.time() > deadline - 30:
                break
            # Calculate Forces
            forces = np.zeros((N, 2))
            
            # Pairwise repulsion
            # Using squared distances for efficiency, but we need gradients of 1/d^12
            # F_ij = -dV/dx = -d/dx (1/d^12) = 24 * (x_i - x_j) / d^14
            # We compute this for all pairs
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pos[i] - pos[j]
                    d2 = np.dot(diff, diff)
                    if d2 < 1e-12:
                        # Random kick if overlapping exactly
                        diff = np.random.randn(2) * 0.01
                        d2 = np.dot(diff, diff)
                    d4 = d2 * d2
                    d6 = d4 * d2
                    # Force magnitude factor: 24 / d^14 = 24 / (d2^7)
                    # d2^7 = d2 * d2^3 = d2 * (d2*d2*d2)
                    d2_7 = d2 * d2 * d2 * d2 * d2 * d2 * d2
                    if d2_7 < 1e-12: continue
                    f_mag = 24.0 / d2_7
                    f_vec = f_mag * diff
                    forces[i] += f_vec
                    forces[j] -= f_vec
            
            # Wall attraction (Harmonic) and Repulsion (Hard wall penalty)
            for i in range(N):
                r = np.linalg.norm(pos[i])
                # Harmonic attraction to center
                forces[i] -= k * pos[i]
                
                # Hard wall repulsion: V = 1/(1-r)^12 for r > 0.5
                if r > 0.5:
                    denom = 1.0 - r
                    if denom < 1e-6:
                        denom = 1e-6
                    # d/dr (1/(1-r)^12) = 12 / (1-r)^13
                    # Force is -dV/dr * (x/r)
                    f_wall = 12.0 / (denom ** 13)
                    forces[i] -= f_wall * (pos[i] / r)
            
            # Langevin Update
            # v_new = (1 - gamma) v + F dt + sqrt(2 gamma T dt) * noise
            noise = np.random.randn(N, 2)
            vel = (1 - gamma) * vel + forces * dt + np.sqrt(2 * gamma * T * dt) * noise
            pos = pos + vel * dt
            
            # Check validity and score
            # If any point goes out, project back or just let potential handle it
            # The potential should keep them in, but let's be safe
            dists = np.linalg.norm(pos, axis=1)
            mask = dists > 0.99
            if np.any(mask):
                pos[mask] = pos[mask] * (0.99 / dists[mask])
                vel[mask] = 0
            
            # Calculate score (min distance)
            # This is O(N^2), which is fine for N=50
            current_score = _calc_score(pos)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                
    # 2. Polish Phase
    # Use the best position found by Langevin as the starting point for local optimization.
    # We minimize a soft-min surrogate to maximize the minimum distance.
    
    if time.time() < deadline - 5:
        best_pos = _polish(best_pos, time_limit=deadline - 5)
        # Final validity check
        if _is_valid(best_pos):
            return best_pos
        else:
            # Fallback to a valid configuration if polish broke it
            # Try to repair by scaling down
            scale = 0.9
            while scale > 0.1:
                test_pos = best_pos * scale
                if _is_valid(test_pos):
                    return test_pos
                scale -= 0.05
    
    return best_pos

def _calc_score(pos):
    """Calculate the score: min(min_pair_dist, min_wall_dist)"""
    min_dist = float('inf')
    # Pairwise distances
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                min_dist = d
    # Wall distances
    for i in range(len(pos)):
        r = np.linalg.norm(pos[i])
        wall_dist = 1.0 - r
        if wall_dist < min_dist:
            min_dist = wall_dist
    # The radius is half the min distance between centers
    # But the score definition in the prompt is:
    # "Score: the largest common radius r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))"
    # So the score IS the radius r.
    return min_dist / 2.0

def _is_valid(pos):
    """Check if configuration is valid (inside unit circle, non-overlapping for a small r)"""
    # For validity, we just need to ensure no center is outside and no centers coincide.
    # The score calculation handles the "non-overlapping" for the specific r.
    # But the contract says: "A centre on or outside the container, coincident centres... make the candidate invalid."
    # So we must ensure r > 0 and centers are strictly inside.
    for p in pos:
        if np.linalg.norm(p) >= 1.0:
            return False
    min_d = float('inf')
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 1e-6:
                return False
            if d < min_d:
                min_d = d
    return min_d > 1e-6

def _soft_min_loss(pos, epsilon):
    """
    Loss function for minimization.
    We want to maximize min(d_ij, 1-r_i).
    Equivalent to minimizing -min(...).
    Surrogate: - (1/epsilon) * log( sum exp(-epsilon * d) )
    """
    N = len(pos)
    sum_exp = 0.0
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            # Clamp d to avoid overflow in exp
            d_clamped = max(d, 1e-6)
            sum_exp += np.exp(-epsilon * d_clamped)
    # Wall
    for i in range(N):
        r = np.linalg.norm(pos[i])
        d_wall = 1.0 - r
        d_wall_clamped = max(d_wall, 1e-6)
        sum_exp += np.exp(-epsilon * d_wall_clamped)
    
    if sum_exp == 0:
        return 0.0
    loss = -(1.0 / epsilon) * np.log(sum_exp)
    return loss

def _soft_min_grad(pos, epsilon):
    """
    Gradient of the soft-min loss.
    d/dx [ -1/epsilon * log(sum exp(-epsilon d_k)) ]
    = -1/epsilon * 1/sum * sum ( -epsilon * exp(-epsilon d_k) * d(d_k)/dx )
    = sum ( exp(-epsilon d_k) / sum_exp ) * d(d_k)/dx
    """
    N = len(pos)
    weights = np.zeros(N * (N-1)//2 + N)
    sum_exp = 0.0
    
    # Calculate all distances and exps first
    dists = []
    exps = []
    
    # Pairs
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            d_clamped = max(d, 1e-6)
            e = np.exp(-epsilon * d_clamped)
            dists.append((i, j, d_clamped))
            exps.append(e)
            sum_exp += e
            
    # Walls
    for i in range(N):
        r = np.linalg.norm(pos[i])
        d_wall = 1.0 - r
        d_wall_clamped = max(d_wall, 1e-6)
        e = np.exp(-epsilon * d_wall_clamped)
        dists.append((i, -1, d_wall_clamped)) # -1 indicates wall
        exps.append(e)
        sum_exp += e
        
    if sum_exp < 1e-12:
        return np.zeros_like(pos)
        
    grad = np.zeros_like(pos)
    
    for idx, (i, j, d) in enumerate(dists):
        w = exps[idx] / sum_exp
        if j == -1:
            # Wall gradient
            # d/dx (1 - ||x||) = -x/||x||
            r = np.linalg.norm(pos[i])
            if r > 1e-6:
                grad[i] -= w * (pos[i] / r)
        else:
            # Pair gradient
            # d/dx_i (||x_i - x_j||) = (x_i - x_j) / d
            diff = pos[i] - pos[j]
            if d > 1e-6:
                unit_diff = diff / d
                grad[i] += w * unit_diff
                grad[j] -= w * unit_diff
                
    return grad

def _polish(pos, time_limit):
    """
    Refine the configuration using L-BFGS-B on the soft-min surrogate.
    """
    start_time = time.time()
    current_pos = pos.copy()
    
    # Start with a moderate epsilon and decrease it
    epsilons = [10.0, 50.0, 100.0, 500.0]
    
    for eps in epsilons:
        if time.time() > time_limit - 2:
            break
            
        def loss_wrapper(x):
            x = x.reshape(N, 2)
            return _soft_min_loss(x, eps)
            
        def grad_wrapper(x):
            x = x.reshape(N, 2)
            return _soft_min_grad(x, eps).flatten()
            
        # Use L-BFGS-B
        try:
            res = minimize(
                loss_wrapper, 
                current_pos.flatten(), 
                jac=grad_wrapper, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success or res.status == 0:
                current_pos = res.x.reshape(N, 2)
            else:
                # Even if not fully converged, the result might be better
                # But check validity
                if _is_valid(res.x.reshape(N, 2)):
                    current_pos = res.x.reshape(N, 2)
        except Exception:
            pass
            
    return current_pos

# EVOLVE-BLOCK-END
