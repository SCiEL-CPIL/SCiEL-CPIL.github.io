import numpy as np
import time
import random

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _score(pts):
    """
    Compute the maximum radius for the given points.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pairwise = np.sqrt(np.min(dists_sq))
    
    # Boundary
    dist_to_face = 0.5 - np.abs(pts)
    min_boundary = np.min(dist_to_face)
    
    return min(min_pairwise / 2.0, min_boundary)

def _fbb_placement(n=50):
    """
    Place spheres using the Best-Fit Bounding Box method.
    This places each new sphere in the corner of the current bounding box
    that is farthest from all existing spheres, maximizing the minimum distance.
    """
    pts = np.zeros((n, 3))
    
    # Start with a point near the center, maybe slightly offset to break symmetry
    pts[0] = np.array([0.0, 0.0, 0.0])
    
    for i in range(1, n):
        # Calculate distance from candidate position to all existing spheres
        # We want to maximize the minimum distance to existing spheres
        # AND keep it within the cube [-0.5, 0.5]^3
        
        # Heuristic: Try the 8 corners of the current "available space" 
        # or simply optimize a small set of candidate positions.
        # A more robust FBB: 
        # 1. Compute bounding box of existing points expanded by current min distance?
        # Simpler: Random restarts within FBB logic or just greedy farthest point.
        
        # Let's use a greedy "Farthest Point" approach restricted to the cube.
        # To make it efficient, we can sample a grid or use a few random candidates 
        # and pick the one with max min-dist.
        
        best_pos = None
        best_min_dist = -1.0
        
        # Sample candidates: 
        # 1. Center of cube
        # 2. Corners of the cube (adjusted for radius, but we don't know r yet)
        # 3. Random points
        
        # For FBB specifically, we often look at the "bounding box" of the current set.
        # Let's just use a robust random search for the next point, 
        # evaluating min distance to existing points.
        
        # To be faster, we can use a coarse grid first, then refine?
        # Or just 100 random candidates per step.
        
        candidates = np.random.uniform(-0.5, 0.5, size=(200, 3))
        
        # Add some structured candidates
        structured = np.array([
            [0.4, 0.4, 0.4], [-0.4, 0.4, 0.4], [0.4, -0.4, 0.4], [-0.4, -0.4, 0.4],
            [0.4, 0.4, -0.4], [-0.4, 0.4, -0.4], [0.4, -0.4, -0.4], [-0.4, -0.4, -0.4],
            [0.0, 0.4, 0.4], [0.0, -0.4, 0.4], [0.4, 0.0, 0.4], [-0.4, 0.0, 0.4],
            [0.0, 0.4, -0.4], [0.0, -0.4, -0.4], [0.4, 0.0, -0.4], [-0.4, 0.0, -0.4],
            [0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, -0.4, 0.0],
            [0.0, 0.0, 0.4], [0.0, 0.0, -0.4]
        ])
        candidates = np.vstack([candidates, structured])
        
        # Calculate min distance from each candidate to existing points
        # candidates: (M, 3), pts[:i]: (i, 3)
        diff = candidates[:, np.newaxis, :] - pts[:i][np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2) # (M, i)
        min_dists = np.min(dists_sq, axis=1) # (M,)
        
        # Also consider boundary distance? 
        # The FBB method usually maximizes distance to existing spheres.
        # The radius is determined by the global min.
        # Maximizing local min-dist is a good heuristic.
        
        best_idx = np.argmax(min_dists)
        pts[i] = candidates[best_idx]
        
    return pts

def _simulated_annealing(pts, start_time, deadline):
    """
    Perform Simulated Annealing on the point configuration.
    Objective: Maximize r.
    Equivalent to: Minimize "energy" E = max(0, R_target - r) or similar.
    Better: Use r directly. Accept moves that increase r.
    Accept moves that decrease r with probability exp(-(r_new - r_old)/T).
    """
    current_pts = pts.copy()
    current_r = _score(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    T = 0.01
    T_min = 0.0001
    cooling_rate = 0.999
    
    n_steps = 0
    step_size = 0.05
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with a slightly perturbed best solution or reset T
            # Let's do a random restart from best
            current_pts = best_pts + np.random.normal(0, 0.01, size=(N, 3))
            current_pts = np.clip(current_pts, -0.5, 0.5)
            current_r = _score(current_pts)
            T = 0.01
            step_size = 0.05
            continue
            
        # Generate a move: perturb one or two points
        num_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
        indices = np.random.choice(N, size=num_perturb, replace=False)
        
        old_pts = current_pts[indices].copy()
        
        # Random perturbation
        noise = np.random.normal(0, step_size, size=(num_perturb, 3))
        current_pts[indices] += noise
        current_pts[indices] = np.clip(current_pts[indices], -0.5, 0.5)
        
        new_r = _score(current_pts)
        
        if new_r >= current_r:
            # Accept
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = current_pts.copy()
        else:
            # Metropolis criterion
            delta = current_r - new_r
            if delta > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_r = new_r
                else:
                    # Reject
                    current_pts[indices] = old_pts
        
        T *= cooling_rate
        step_size *= 0.9995
        
        n_steps += 1
        # Check time every 1000 steps to avoid overhead
        if n_steps % 1000 == 0 and time.time() > deadline:
            break
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Generate a good initial configuration using FBB
    # Run FBB a few times with different random seeds to pick the best
    fbb_candidates = []
    for i in range(5):
        np.random.seed(42 + i)
        pts = _fbb_placement(N)
        r = _score(pts)
        fbb_candidates.append((r, pts))
        
    # Sort by score
    fbb_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_r, best_pts = fbb_candidates[0]
    
    # Phase 2: Simulated Annealing from the best FBB solution
    # Use remaining time
    remaining_time = deadline - time.time()
    
    if remaining_time > 5:
        sa_pts, sa_r = _simulated_annealing(best_pts, time.time(), deadline)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts
            
    # Final verification
    if not np.all(np.isfinite(best_pts)) or np.any(best_pts < -0.5) or np.any(best_pts > 0.5):
        # Fallback to a simple grid if something went wrong
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
        
    return best_pts

# Keep the interface compatible if needed, though the prompt says keep names/signatures of seed program.
# Seed program had: construct()
# It also had helper functions, but we only need to expose construct.

# For debugging/verification
if __name__ == "__main__":
    pts = construct()
    r = _score(pts)
    print(f"Radius: {r}")
    print(f"Points shape: {pts.shape}")
