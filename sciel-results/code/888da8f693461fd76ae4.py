"""Improved circle packing for N=300 in unit circle using Simulated Annealing."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_ring_init():
        """
        Generate an initial configuration using concentric rings.
        This approximates the hexagonal packing structure.
        """
        pts = []
        # Center point
        pts.append([0.0, 0.0])
        
        # Estimate optimal radius and spacing
        # Best known r ~ 0.053. Diameter d ~ 0.106.
        # Hex packing density is pi / (2 * sqrt(3)) * d^2 / Area?
        # Let's just use a radial spacing of d * 0.866 (hex row spacing)
        # and angular spacing determined by circumference.
        
        d_est = 0.106
        row_spacing = d_est * np.sqrt(3) / 2.0
        
        current_radius = 0.0
        ring_idx = 0
        while len(pts) < N:
            if ring_idx == 0:
                # Already added center
                pass
            else:
                # Determine how many points fit on this ring
                # Circumference = 2 * pi * r
                # Angular distance between centers should be >= d_est
                # max_points = floor(2 * pi * r / d_est)
                # Ensure at least 6 for symmetry if possible
                max_pts = int(2 * np.pi * current_radius / d_est)
                if max_pts < 6:
                    max_pts = 6 if current_radius > d_est else 1 # Handle small rings
                # We might want to limit to ensure we don't overshoot N too much
                # But we can just take min(max_pts, N - len(pts))
                
                num_points = min(max_pts, N - len(pts))
                
                # Add points on this ring
                angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                # Offset angle for alternating rings to mimic hex packing
                offset = np.pi / num_points if ring_idx % 2 == 1 else 0.0
                angles += offset
                
                for ang in angles:
                    x = current_radius * np.cos(ang)
                    y = current_radius * np.sin(ang)
                    pts.append([x, y])
            
            current_radius += row_spacing
            ring_idx += 1
            
            # Safety break
            if current_radius > 1.0:
                break
                
        if len(pts) < N:
            # If we ran out of space in the simple ring model, add random points inside
            while len(pts) < N:
                x = np.random.uniform(-0.9, 0.9)
                y = np.random.uniform(-0.9, 0.9)
                if np.hypot(x, y) < 0.9:
                    pts.append([x, y])
        
        return np.array(pts[:N])

    def anneal(pts, deadline):
        """
        Simulated Annealing optimization.
        Moves one point at a time.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Initial temperature
        T = 0.05
        # Cooling rate
        alpha = 0.9995
        # Min temperature
        T_min = 1e-6
        
        # Precompute some constants?
        # We need to be fast.
        
        while time.time() < deadline and T > T_min:
            # Pick a random index
            idx = np.random.randint(N)
            orig_pt = pts[idx].copy()
            
            # Propose a move
            # Step size scales with T
            step_size = T * 2.0
            delta = np.random.randn(2) * step_size
            new_pt = orig_pt + delta
            
            # Project inside unit circle
            r = np.hypot(new_pt[0], new_pt[1])
            if r > 1.0:
                new_pt = new_pt / r
            
            # Calculate new score
            # We only need to update distances involving idx
            # But compute_score is O(N^2).
            # For N=300, N^2 = 90,000 ops. 
            # If we do 10,000 iterations, that's 900M ops. Might be too slow in Python/NumPy overhead?
            # Let's optimize score calculation for single point move.
            
            # Old score components:
            # min_pair_old, min_boundary_old
            # We don't store them, so we just recompute full score.
            # To speed up, we can check if the move is clearly bad.
            # 1. Check boundary: 1 - |new_pt|
            # 2. Check nearest neighbor to new_pt among all other points.
            
            # Distance from new_pt to all others
            diff = pts - new_pt
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[idx] = np.inf
            min_dist_to_others = np.min(dists)
            
            # Boundary distance
            new_boundary = 1.0 - np.hypot(new_pt[0], new_pt[1])
            
            # The new score is min( global_min_pair_except_idx, min_dist_to_others/2, new_boundary, global_min_boundary_except_idx )
            # This requires knowing the global mins excluding idx.
            # This is complex. Let's stick to full recompute but optimize the vectorization.
            
            # Actually, let's just do full recompute. NumPy is fast.
            # 300x300 distance matrix is small.
            
            pts[idx] = new_pt
            new_score = compute_score(pts)
            
            # Acceptance
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
            else:
                # Calculate energy difference
                # We want to maximize score.
                # Delta E = best_score - new_score (if new is worse)
                # Or just use current score?
                # Standard SA: accept if exp(-(E_new - E_old)/T) > rand
                # Here "Energy" is -Score.
                # E_new = -new_score, E_old = -current_score (which is effectively best_score if we track it, or just the score of the current state)
                # Let's use the score of the *current* state (before move) vs new state.
                # We need to store current_score.
                
                # Let's refactor to store current_score.
                # For now, simple hill climbing with random restarts? No, SA is better.
                # Let's assume current_score is the score of pts before the move.
                # We didn't store it. Let's just use best_score as reference for "good".
                # If new_score < best_score, it's a "bad" move.
                # Probability = exp((new_score - best_score) / T)
                # Note: new_score - best_score is negative.
                
                delta_e = new_score - best_score
                if delta_e < 0:
                    prob = np.exp(delta_e / T)
                    if np.random.rand() > prob:
                        # Reject
                        pts[idx] = orig_pt
                else:
                    # Accept (improvement or equal)
                    pass
            
            T *= alpha
            
        return best_pts, best_score

    # Main execution
    # 1. Initialize
    pts = generate_ring_init()
    
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # 2. Annealing
    # Run multiple rounds with different seeds/restarts if time permits
    # But SA should handle exploration.
    
    # We can run SA for a fixed amount of time
    pts, score = anneal(pts, deadline - 5)
    
    if score > best_score:
        best_score = score
        best_pts = pts
        
    # 3. Final Local Search (Hill Climbing)
    # Do a quick pass of strict local search to polish
    T = 1e-6 # Very cold, effectively hill climbing
    pts, score = anneal(best_pts, deadline)
    
    if score > best_score:
        best_score = score
        best_pts = pts

    # Final check
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
        
    return best_pts
# EVOLVE-BLOCK-END
