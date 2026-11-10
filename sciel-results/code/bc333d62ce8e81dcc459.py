"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize with a simple cubic grid (valid starting point)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Helper to compute score (radius)
    def compute_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        # c is in [-0.5, 0.5]
        # Distance to nearest wall
        # min_k (0.5 - |c_ik|)
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    best_centers = centers.copy()
    best_radius = compute_radius(best_centers)
    
    # Simulated Annealing parameters
    # Start with a relatively large step size
    step_size = 0.05
    cooling_rate = 0.9995
    max_step_size = 0.1
    
    # Precompute indices for efficiency? No, N=100 is small enough for vectorized ops per step.
    
    while time.time() < deadline:
        # Check if we should stop cooling or if step size is too small
        if step_size < 1e-5:
            break
            
        # Pick a random sphere
        i = np.random.randint(N)
        
        # Generate a random displacement
        # Use Gaussian noise scaled by step_size
        delta = np.random.normal(0, step_size, size=3)
        
        new_center = centers[i].copy()
        new_center += delta
        
        # Keep within bounds? 
        # The score function penalizes going out of bounds via wall distance.
        # However, if a center goes far outside, the radius becomes negative.
        # We can just let the score function handle it, but for stability, 
        # it's better to keep centers within [-0.5, 0.5] or slightly larger?
        # The problem states centers must be inside. 
        # "A centre on or outside the cube ... make the candidate invalid."
        # So we MUST ensure centers stay in [-0.5, 0.5].
        
        # Clip to bounds? If we clip, the gradient is lost at the boundary.
        # Better to reject moves that go out of bounds immediately, or penalize heavily.
        # Since the score returns a value that makes the candidate "invalid" if outside,
        # and the evaluator rejects invalid candidates, we must ensure the FINAL result is valid.
        # During search, if we allow invalid states, we might get stuck in a bad region.
        # Let's restrict the move to stay within bounds.
        
        if np.any(new_center > 0.5) or np.any(new_center < -0.5):
            # Reject move
            continue
            
        # Calculate new radius
        # To speed up, we can check only the affected sphere i
        # 1. Distance to other spheres
        # 2. Distance to walls
        
        # Old radius was min(min_pair, min_wall)
        # New radius for sphere i:
        # min( min_{j!=i} |c_i - c_j| / 2, min_k (0.5 - |c_ik|) )
        
        # But the global radius is the MIN over ALL spheres.
        # Moving sphere i can only DECREASE the global radius if it gets closer to a neighbor or wall.
        # It can only INCREASE the global radius if the previous minimum was determined by sphere i 
        # and we move it away from that constraint.
        
        # Efficient check:
        # 1. Compute min distance from new c_i to all other c_j
        diffs = centers - new_center
        diffs[i] = np.inf # Exclude self
        dists = np.linalg.norm(diffs, axis=1)
        min_pair_i = np.min(dists) / 2.0
        
        # 2. Compute min wall distance for new c_i
        wall_dists_i = 0.5 - np.abs(new_center)
        min_wall_i = np.min(wall_dists_i)
        
        # The new global radius is min(
        #    min( min_pair_i, min_wall_i ), 
        #    min_radius_of_other_spheres
        # )
        # We need to know min_radius_of_other_spheres.
        # Let R_others = min( min_{j!=i} (min_{k!=j} |c_j-c_k|/2, min_wall_j) )
        # This is expensive to maintain dynamically without a data structure.
        # Given N=100, O(N^2) per step is 10,000 ops. 
        # 10,000 ops * 100,000 steps = 1e9 ops. Might be too slow in Python/NumPy loop.
        
        # Let's estimate: 1 NumPy vector operation of size 100 is fast.
        # But doing it in a Python while loop 100,000 times might take seconds.
        # 150s limit.
        # Let's try to optimize.
        
        # Alternative: Just recompute full radius every few steps? 
        # Or use the approximation that if min_pair_i < best_radius and min_wall_i < best_radius, 
        # the move is definitely worse (if best_radius was the global min).
        # Actually, if new_local_radius < best_radius, the new global radius <= new_local_radius < best_radius.
        # So if new_local_radius < best_radius, it's a worse move (unless best_radius was determined by i and we fix it? No, if new_local < best, global drops).
        # Wait, if best_radius was determined by sphere i, and we move i, the new global radius is min(new_local_i, R_others).
        # If new_local_i > best_radius, it could be better if R_others > new_local_i.
        
        # Let's just compute the full radius for candidate moves. 
        # N=100. 
        # Vectorized full radius calculation:
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # This creates a 100x100x3 array. 30,000 floats.
        # Norm: 100x100.
        # This is very fast in NumPy.
        
        # Let's do 10,000 steps first to see timing.
        
        # Create a temporary copy to test? No, modify in place and revert if rejected.
        old_c = centers[i].copy()
        centers[i] = new_center
        
        new_radius = compute_radius(centers)
        
        # Acceptance criterion
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = centers.copy()
            # Accept always
        else:
            # Simulated annealing: accept with probability exp((new - old) / T)
            # Here "energy" is -radius. So delta_E = -(new_radius - best_radius) = best_radius - new_radius
            # We want to accept worse moves (lower radius) to escape local minima.
            # T is temperature.
            # Let's use a simple temperature schedule based on time or iteration.
            # Since we don't have iteration count easily accessible in this structure without a counter, 
            # let's use step_size as a proxy for temperature.
            # Or just use a fixed low probability of accepting worse moves?
            # Better: T = step_size * 0.1 ?
            # Let's use a simple Metropolis criterion with T decaying.
            
            # We need a temperature variable.
            # Let's restart the loop structure to include T.
            
            # For now, simple hill climbing with occasional random restarts?
            # Or just accept if new_radius > best_radius - epsilon?
            
            # Let's stick to simple hill climbing first, it's robust.
            # If new_radius < best_radius, reject.
            centers[i] = old_c
            
        # Cool down step size
        step_size *= cooling_rate
        
        # Periodically check time
        if time.time() > deadline:
            break

    # Final check: ensure best_centers is valid
    # The compute_radius function doesn't check validity, just returns a score.
    # But we constrained moves to stay in bounds.
    # Initial grid is in bounds.
    # So best_centers should be valid.
    
    return best_centers
# EVOLVE-BLOCK-END
