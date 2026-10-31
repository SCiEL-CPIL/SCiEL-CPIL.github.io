"""
Strategy: Multi-start Simulated Annealing with Hard Objective.
Instead of smooth approximations and gradients, we use stochastic moves
(translation, rotation, scaling, individual point jitter) and accept
worse solutions based on temperature to escape local minima.
We run multiple independent SA chains in parallel (sequentially in Python
but logically distinct) and keep the best result.
"""
import numpy as np
import time
import random

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Fixed seed for reproducibility in evaluation, but random in actual run?
    # The task implies we should find a good configuration.
    # Using random seed for each run allows exploring different landscapes.
    rng = np.random.default_rng()
    
    def compute_score(pts):
        """Calculate (d_max / d_min)^2"""
        # Compute pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = sq_dists[iu]
        
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-12:
            return float('inf')
            
        # Ratio of distances is sqrt(d_max_sq/d_min_sq)
        # Squared ratio is d_max_sq/d_min_sq
        return d_max_sq / d_min_sq

    def generate_initial_config(index):
        """Generate diverse initial configurations"""
        if index == 0:
            # 4x4 Grid
            x_coords = np.arange(4) - 1.5
            y_coords = np.arange(4) - 1.5
            xx, yy = np.meshgrid(x_coords, y_coords)
            pts = np.column_stack([xx.ravel(), yy.ravel()])
        elif index == 1:
            # Hexagonal lattice approximation (4x4 hex)
            pts = []
            for i in range(4):
                for j in range(4):
                    x = i + 0.5 * (j % 2)
                    y = j * np.sqrt(3) / 2
                    pts.append([x, y])
            pts = np.array(pts)
        elif index == 2:
            # Concentric rings
            pts = []
            # 6 points on inner ring
            for i in range(6):
                angle = 2 * np.pi * i / 6
                pts.append([np.cos(angle) * 0.5, np.sin(angle) * 0.5])
            # 10 points on outer ring
            for i in range(10):
                angle = 2 * np.pi * i / 10
                pts.append([np.cos(angle) * 1.5, np.sin(angle) * 1.5])
            pts = np.array(pts)
        elif index == 3:
            # Random uniform
            pts = rng.uniform(-2, 2, (N, 2))
        else:
            # Random with some structure
            pts = rng.normal(0, 1, (N, 2))
            
        # Normalize to unit norm to start in a standard scale
        pts = pts - np.mean(pts, axis=0)
        norm = np.linalg.norm(pts)
        if norm > 1e-9:
            pts = pts / norm
        return pts

    def sa_optimize(pts_init, time_budget):
        """Run Simulated Annealing for a specific time budget"""
        current_pts = pts_init.copy()
        current_score = compute_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # SA Parameters
        T_start = 1.0
        T_end = 1e-6
        # We don't know exact steps, so we use time-based cooling
        # We will adjust step size based on dimension
        
        t_start = time.time()
        elapsed = 0.0
        steps = 0
        
        while time.time() - t_start < time_budget:
            # Cooling schedule: exponential decay
            # Estimate total steps? No, just use a slow decay based on time
            # A common heuristic is T = T_start * (T_end/T_start)^(t/t_total)
            # But we don't know t_total exactly. 
            # Let's use a fixed number of iterations per second and decay per iteration.
            # Or simpler: decay T every 100 steps.
            
            # Generate a random move
            move_type = rng.integers(0, 4)
            new_pts = current_pts.copy()
            
            if move_type == 0:
                # Jitter one point
                idx = rng.integers(0, N)
                step_size = 0.1 * (1.0 + 0.5 * T_start) # Scale with temp? No, keep simple
                new_pts[idx] += rng.normal(0, 0.1, 2)
            elif move_type == 1:
                # Rotate all points slightly
                angle = rng.normal(0, 0.1)
                c, s = np.cos(angle), np.sin(angle)
                rot = np.array([[c, -s], [s, c]])
                new_pts = new_pts @ rot.T
            elif move_type == 2:
                # Scale all points slightly
                scale = 1.0 + rng.normal(0, 0.05)
                new_pts = new_pts * scale
            else:
                # Translate all points (less useful due to normalization, but keeps structure)
                # Actually, translation is removed by normalization in score? No, score is translation invariant.
                # So translation is useless. Let's do "stretch" instead.
                stretch = 1.0 + rng.normal(0, 0.05)
                new_pts[:, 0] *= stretch
                # Re-center to keep mean 0
                new_pts -= np.mean(new_pts, axis=0)
                
            # Normalize new points to unit norm to prevent drift
            new_pts -= np.mean(new_pts, axis=0)
            norm = np.linalg.norm(new_pts)
            if norm < 1e-9:
                continue
            new_pts = new_pts / norm
            
            new_score = compute_score(new_pts)
            
            # Acceptance criterion
            if new_score < current_score:
                current_pts = new_pts
                current_score = new_score
                if new_score < best_score:
                    best_pts = new_pts.copy()
                    best_score = new_score
            else:
                # Accept with probability exp(-(new_score - current_score) / T)
                # Note: Scores can be large, so delta might be large.
                # We need to scale temperature appropriately.
                # If score is ~10, T=1 might be too high or too low.
                # Let's normalize the score difference.
                delta = new_score - current_score
                # If delta is huge, probability is 0.
                # We want T to be comparable to typical score changes.
                # Let's adapt T based on current score magnitude?
                # Simpler: Just use a fixed T schedule and hope it works.
                # Or, use a relative temperature: T_rel = T / current_score
                # prob = exp(-delta / (T * current_score)) ?
                
                # Let's try standard SA with T starting at 0.1 and decaying.
                # The score is a ratio. A change of 0.1 in ratio is significant.
                prob = np.exp(-delta / T_start)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
            
            # Update temperature
            # Decay T every 50 steps
            if steps % 50 == 0:
                T_start *= 0.95
                if T_start < T_end:
                    T_start = T_end
                    
            steps += 1
            
            # Safety check for time
            if time.time() > deadline - 2.0:
                break
                
        return best_pts, best_score

    # Multi-start strategy
    num_restarts = 8
    # Allocate time to each restart, leaving some buffer
    # Total available time ~ 140s.
    # Let's give each restart 15s. 8 * 15 = 120s.
    time_per_restart = 15.0
    
    global_best_pts = None
    global_best_score = float('inf')
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial config
        pts_init = generate_initial_config(i)
        
        # Run SA
        try:
            res_pts, res_score = sa_optimize(pts_init, time_per_restart)
            if res_score < global_best_score:
                global_best_score = res_score
                global_best_pts = res_pts
        except Exception as e:
            pass
            
        # If we find a very good score, we might want to stop early?
        # Theoretical minimum for 16 points in 2D is not 1. 
        # Square lattice 4x4:
        # d_min = 1, d_max = sqrt(3^2+3^2)=sqrt(18)=4.24. Ratio^2 = 18.
        # If we get close to 10-15, it's good.
        # Let's just run all restarts unless time is tight.

    if global_best_pts is None:
        # Fallback to grid
        x_coords = np.arange(4) - 1.5
        y_coords = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x_coords, y_coords)
        global_best_pts = np.column_stack([xx.ravel(), yy.ravel()])

    return global_best_pts

# EVOLVE-BLOCK-END
