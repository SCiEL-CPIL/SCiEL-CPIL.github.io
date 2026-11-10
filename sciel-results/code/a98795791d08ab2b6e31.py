import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0
    
    def calculate_score(centers):
        # Calculate min distance between pairs and to boundary
        # Vectorized
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
        return min(min_pair, min_boundary)

    def calculate_score_fast(centers):
        """Faster score using only distances, no sqrt where possible for comparison."""
        # For acceptance decisions, we can work with squared distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_sq = np.min(dists_sq)
        norms_sq = np.sum(centers**2, axis=1)
        min_boundary_sq = np.min((1.0 - np.sqrt(norms_sq))**2)
        # This isn't quite right for min - let's just use the full calculation
        # but cache the distance matrix
        dists = np.sqrt(dists_sq)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.sqrt(norms_sq))
        return min(min_pair, min_boundary), dists

    def run_optimized_sa(initial_centers, time_budget):
        centers = initial_centers.copy()
        current_score = calculate_score(centers)
        best_score = current_score
        best_centers = centers.copy()
        
        T = 0.01
        cooling_rate = 0.999
        step_size = 0.02
        
        no_improve_count = 0
        end_time = start_time + time_budget
        
        # Precompute distance matrix once per iteration for reuse
        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break
                
            # Calculate forces/directions based on current configuration
            # Find nearest neighbor for each point to determine repulsion direction
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Nearest neighbor index
            nn_idx = np.argmin(dists, axis=1)
            
            # Direction to move: away from nearest neighbor, and towards center if close to boundary
            moves = np.zeros((N, 2))
            for i in range(N):
                j = nn_idx[i]
                # Vector from i to j
                vec_to_nn = centers[j] - centers[i]
                # Move away from j
                norm_vec = dists[i, j]
                if norm_vec > 1e-9:
                    moves[i] -= vec_to_nn / norm_vec
                
                # Boundary repulsion
                dist_to_center = np.linalg.norm(centers[i])
                if dist_to_center > 0.9:
                    # Push towards center
                    if dist_to_center > 1e-9:
                        moves[i] -= centers[i] / dist_to_center * (dist_to_center - 0.9)
            
            # Normalize moves to have consistent magnitude
            move_norms = np.linalg.norm(moves, axis=1)
            for i in range(N):
                if move_norms[i] > 1e-9:
                    moves[i] /= move_norms[i]
            
            # Select a random point to move
            idx = np.random.randint(N)
            
            # Combine directed move with random noise
            noise = np.random.normal(0, step_size * 0.5, 2)
            new_point = centers[idx] + moves[idx] * step_size + noise
            
            # Keep inside circle
            norm = np.linalg.norm(new_point)
            if norm > 0.99:
                new_point = new_point / norm * 0.99
                
            old_point = centers[idx].copy()
            centers[idx] = new_point
            
            # Fast score update: only recompute distances involving idx
            new_dists_to_idx = np.linalg.norm(centers - centers[idx], axis=1)
            new_norm_to_boundary = 1.0 - np.linalg.norm(centers[idx])
            
            # Get min pair distance (excluding self)
            min_pair_dist = np.min(new_dists_to_idx[new_dists_to_idx > 1e-9]) / 2.0 if np.any(new_dists_to_idx > 1e-9) else 1.0
            
            # Get overall min pair distance
            # We need the full min, so use cached dists for non-idx pairs
            other_dists = dists.copy()
            other_dists[idx, :] = np.inf
            other_dists[:, idx] = np.inf
            min_other_pair = np.min(other_dists) / 2.0 if np.any(other_dists < np.inf) else 1.0
            
            min_pair = min(min_pair_dist, min_other_pair)
            min_boundary = min(new_norm_to_boundary, np.min(1.0 - np.linalg.norm(centers, axis=1)))
            
            # Recalculate full boundary min efficiently
            norms = np.linalg.norm(centers, axis=1)
            min_boundary = np.min(1.0 - norms)
            
            new_score = min(min_pair, min_boundary)
            
            delta_E = new_score - current_score
            
            if delta_E > 0:
                accept = True
            else:
                if T > 1e-8:
                    p = np.exp(delta_E / T)
                    accept = (np.random.rand() < p)
                else:
                    accept = False
            
            if accept:
                current_score = new_score
                # Update cached dists
                diff_new = centers[idx] - centers
                dists[idx, :] = np.linalg.norm(diff_new, axis=1)
                dists[:, idx] = dists[idx, :]
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                centers[idx] = old_point
                no_improve_count += 1
            
            # Cooling
            T *= cooling_rate
            
            # Reheating logic
            if no_improve_count > 1000:
                T = 0.05
                step_size = 0.05
                no_improve_count = 0
            elif no_improve_count > 500:
                T = max(T, 0.02)
                step_size = max(step_size, 0.03)
            
            # Adapt step size
            if no_improve_count < 100:
                step_size = max(0.001, step_size * 0.99)
            else:
                step_size = min(0.1, step_size * 1.01)
                
            if T < 1e-8:
                T = 0.001 # Floor

        return best_centers, best_score

    def polish(centers, time_budget):
        """Final high-precision polish: deterministic local search with very small steps."""
        centers = centers.copy()
        current_score = calculate_score(centers)
        best_score = current_score
        best_centers = centers.copy()
        
        end_time = start_time + time_budget
        step_size = 0.001
        no_improve = 0
        
        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break
            
            # Find the pair with minimum distance
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Find the closest pair
            i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = dists[i_min, j_min]
            
            # Also check boundary
            norms = np.linalg.norm(centers, axis=1)
            min_boundary_idx = np.argmin(1.0 - norms)
            min_boundary_val = 1.0 - norms[min_boundary_idx]
            
            # Determine which constraint is binding
            if min_dist / 2.0 < min_boundary_val:
                # Pair constraint binding: move both points apart
                vec = centers[j_min] - centers[i_min]
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-9:
                    direction = vec / norm_vec
                    # Move i away from j, j away from i
                    move_i = -direction * step_size
                    move_j = direction * step_size
                else:
                    move_i = np.array([step_size, 0])
                    move_j = np.array([-step_size, 0])
                
                old_i = centers[i_min].copy()
                old_j = centers[j_min].copy()
                
                centers[i_min] += move_i
                centers[j_min] += move_j
                
                # Ensure inside container
                for k in [i_min, j_min]:
                    n = np.linalg.norm(centers[k])
                    if n > 1.0:
                        centers[k] = centers[k] / n * 0.9999
            else:
                # Boundary constraint binding: move that point towards center
                idx = min_boundary_idx
                vec = centers[idx]
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-9:
                    move = -vec / norm_vec * step_size
                else:
                    move = np.array([step_size, 0])
                
                old = centers[idx].copy()
                centers[idx] += move
                
                n = np.linalg.norm(centers[idx])
                if n > 1.0:
                    centers[idx] = centers[idx] / n * 0.9999
            
            new_score = calculate_score(centers)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                if min_dist / 2.0 < min_boundary_val:
                    centers[i_min] = old_i
                    centers[j_min] = old_j
                else:
                    centers[idx] = old
                no_improve += 1
            
            # Decrease step size slowly
            step_size *= 0.9995
            if step_size < 1e-6:
                break
            
            # Early stop if stuck
            if no_improve > 5000:
                break
        
        return best_centers, best_score

    # Initialization: Hexagonal Grid
    # Generate a dense hex grid and pick the N closest to origin
    spacing = 0.2
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k+1):
        for i in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.95:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= N:
        # Sort by distance to origin
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        init_centers = points[sorted_indices[:N]]
    else:
        init_centers = np.random.uniform(-0.8, 0.8, (N, 2))
        
    # Run the optimized SA with most of the time
    sa_time = max_time - 15.0  # Reserve 15s for polish
    best_centers, best_score = run_optimized_sa(init_centers, sa_time)
    
    # Final polish pass
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 2.0:
        polished_centers, polished_score = polish(best_centers, remaining_time - 1.0)
        if polished_score >= best_score:
            best_centers = polished_centers
            best_score = polished_score
    
    # Final check
    if best_centers is None or len(best_centers) != N:
        best_centers = np.random.uniform(-0.9, 0.9, (N, 2))
        
    # Ensure validity
    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_centers[i] = best_centers[i] / norms[i] * 0.9999
            
    return best_centers
