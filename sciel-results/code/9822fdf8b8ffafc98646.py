"""Rewritten program for sphere covering with 130 points."""
import time
import numpy as np
import math

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _greedy_init(n, seed=42):
    """Greedy farthest-point sampling on the sphere."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    # Start with a random point
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, n):
        # Compute min distance from all existing points to a dense sample grid
        # Use a moderate grid for speed
        S = 3000
        samples = _fibonacci_sphere(S)
        dots = samples @ pts[:i].T  # (S, i)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        worst_idx = np.argmax(min_dists)
        pts[i] = samples[worst_idx]
    
    return _normalize(pts)

def _local_search(P, samples, max_iters=5000, time_limit=None, start_time=None, rng=None):
    """Efficient local search: move each point toward the worst-covered region."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    S = len(samples)
    current_max_dist = np.inf
    no_improve = 0
    max_no_improve = 150
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Compute min distance from each sample to nearest point
        dots = samples @ P.T  # (S, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        current_max_dist = np.max(min_dists)
        
        # Find the worst sample
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        # Direction: move toward worst_sample
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            continue
        
        grad_dir /= grad_norm
        
        # Add small random perturbation to escape local minima
        noise_scale = 0.1 * (1.0 + it / max_iters)
        random_dir = rng.standard_normal(3)
        random_dir -= np.dot(random_dir, p_old) * p_old
        rn = np.linalg.norm(random_dir)
        if rn > 1e-8:
            random_dir /= rn
        else:
            random_dir = np.zeros(3)
        
        search_dir = grad_dir + noise_scale * random_dir
        search_dir /= np.linalg.norm(search_dir)
        
        # Try several step sizes, pick the best
        best_new_max = current_max_dist
        best_p = p_old.copy()
        best_step = 0.0
        
        for step_size in [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]:
            new_p = p_old + step_size * search_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            # Recompute only the max (fast)
            dots_temp = samples @ P.T
            dots_temp = np.clip(dots_temp, -1.0, 1.0)
            dists_temp = np.arccos(dots_temp)
            min_dists_temp = np.min(dists_temp, axis=1)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max_dist = best_new_max
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return P, current_max_dist

def _local_search_all_points(P, samples, max_rounds=200, time_limit=None, rng=None):
    """Local search that cycles through all points, moving each toward its worst region."""
    if rng is None:
        rng = np.random.default_rng(456)
    
    S = len(samples)
    current_max_dist = np.inf
    no_improve = 0
    max_no_improve = 50
    
    for rnd in range(max_rounds):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Compute min distance from each sample to nearest point
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        current_max_dist = np.max(min_dists)
        
        # For each point, find the worst sample assigned to it
        # and move it toward that sample
        improved = False
        for p_idx in range(N):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Find samples whose nearest point is p_idx
            mask = nearest_idx == p_idx
            if not np.any(mask):
                continue
            
            # Find the worst (farthest) sample among those assigned to p_idx
            sample_dists = min_dists[mask]
            worst_local = np.argmax(sample_dists)
            worst_global = np.where(mask)[0][worst_local]
            worst_sample = samples[worst_global]
            
            p_old = P[p_idx].copy()
            proj = np.dot(p_old, worst_sample)
            grad_dir = worst_sample - proj * p_old
            grad_norm = np.linalg.norm(grad_dir)
            
            if grad_norm < 1e-8:
                continue
            
            grad_dir /= grad_norm
            
            # Small random perturbation
            random_dir = rng.standard_normal(3)
            random_dir -= np.dot(random_dir, p_old) * p_old
            rn = np.linalg.norm(random_dir)
            if rn > 1e-8:
                random_dir /= rn
            else:
                random_dir = np.zeros(3)
            
            search_dir = grad_dir + 0.05 * random_dir
            search_dir /= np.linalg.norm(search_dir)
            
            # Try step sizes
            best_new_max = current_max_dist
            best_p = p_old.copy()
            
            for step_size in [0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]:
                new_p = p_old + step_size * search_dir
                new_p = _normalize(new_p.reshape(1, 3))[0]
                
                P[p_idx] = new_p
                dots_temp = samples @ P.T
                dots_temp = np.clip(dots_temp, -1.0, 1.0)
                dists_temp = np.arccos(dots_temp)
                min_dists_temp = np.min(dists_temp, axis=1)
                new_max = np.max(min_dists_temp)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_p = new_p.copy()
            
            if best_p is not p_old and np.any(best_p != p_old):
                P[p_idx] = best_p
                if best_new_max < current_max_dist:
                    current_max_dist = best_new_max
                    improved = True
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        else:
            no_improve = 0
    
    return P, current_max_dist

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit, leave 40s buffer)
    
    best_P = None
    best_score = 180.0
    
    # Use a moderate sample set for local search (balance accuracy vs speed)
    S_search = 8000
    samples_search = _fibonacci_sphere(S_search)
    
    # Multiple restarts with different initializations
    n_restarts = 6
    rng_master = np.random.default_rng(789)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            P = _greedy_init(N, seed=42)
        elif restart == 1:
            P = _greedy_init(N, seed=123)
        elif restart == 2:
            P = _fibonacci_sphere(N, seed_noise=0.02, seed=456)
        elif restart == 3:
            P = _fibonacci_sphere(N, seed_noise=0.05, seed=789)
        elif restart == 4:
            P = _greedy_init(N, seed=321)
        else:
            P = _fibonacci_sphere(N, seed_noise=0.03, seed=111)
        
        # Random rotation for diversity
        M = rng_master.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        d = np.sign(np.linalg.det(Q))
        Q[:, -1] *= d
        P = P @ Q.T
        P = _normalize(P)
        
        # Phase 1: Fast local search focusing on worst point
        rng_ls = np.random.default_rng(1000 + restart)
        P, score1 = _local_search(P, samples_search, max_iters=2000, 
                                   time_limit=deadline - 15.0, rng=rng_ls)
        
        if score1 < best_score:
            best_score = score1
            best_P = P.copy()
        
        if time.time() > deadline - 15.0:
            break
        
        # Phase 2: All-points local search for refinement
        rng_ls2 = np.random.default_rng(2000 + restart)
        P, score2 = _local_search_all_points(P, samples_search, max_rounds=100,
                                              time_limit=deadline - 10.0, rng=rng_ls2)
        
        if score2 < best_score:
            best_score = score2
            best_P = P.copy()
        
        if time.time() > deadline - 10.0:
            break
    
    # Final evaluation with higher resolution
    if best_P is not None:
        S_eval = 20000
        samples_eval = _fibonacci_sphere(S_eval)
        dots = samples_eval @ best_P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        final_score = np.max(min_dists)
    else:
        best_P = _fibonacci_sphere(N, seed_noise=0.01)
        final_score = 180.0
    
    return _normalize(best_P)
