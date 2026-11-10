import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0

    def calculate_score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
        return min(min_pair, min_boundary)

    def optimize_packing(initial_centers, time_budget):
        centers = initial_centers.copy()
        best_score = calculate_score(centers)
        best_centers = centers.copy()
        
        # Parameters
        step_size = 0.05
        shake_magnitude = 0.05
        shake_interval = 500
        iteration = 0
        end_time = start_time + time_budget
        
        # Pre-allocate
        diff = np.zeros((N, N, 2))
        dists = np.zeros((N, N))
        
        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break
            
            # 1. Calculate distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # 2. Find minimum distance and boundary distance
            min_pair_dist = np.min(dists)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            
            norms = np.linalg.norm(centers, axis=1)
            min_boundary_dist = np.min(1.0 - norms)
            min_boundary_idx = np.argmin(1.0 - norms)
            
            current_score = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
            
            # 3. Determine active constraints and compute gradient
            # We want to maximize the minimum distance.
            # Subgradient: move points to increase the distances of the "tightest" constraints.
            
            grad = np.zeros((N, 2))
            
            # Pair constraints: if dist < threshold, push apart
            # Use a soft threshold to handle multiple close pairs
            threshold = min_pair_dist + 0.01 
            for i in range(N):
                for j in range(i+1, N):
                    if dists[i, j] < threshold:
                        vec = centers[i] - centers[j]
                        norm = np.linalg.norm(vec)
                        if norm > 1e-9:
                            # Weight by how close it is to the minimum
                            weight = (threshold - dists[i, j])
                            grad[i] += (vec / norm) * weight
                            grad[j] -= (vec / norm) * weight
            
            # Boundary constraints: if 1 - norm < threshold, pull in
            boundary_threshold = min_boundary_dist + 0.01
            for i in range(N):
                dist_to_boundary = 1.0 - norms[i]
                if dist_to_boundary < boundary_threshold:
                    weight = (boundary_threshold - dist_to_boundary)
                    # Move towards origin
                    if norms[i] > 1e-9:
                        grad[i] -= (centers[i] / norms[i]) * weight
            
            # 4. Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                grad /= grad_norm
            else:
                # If no gradient (perfect packing or stuck), shake
                grad = np.random.normal(0, 1, (N, 2))
                grad /= np.linalg.norm(grad)
            
            # 5. Update centers
            centers += grad * step_size
            
            # 6. Enforce boundary strictly (project back inside if necessary)
            new_norms = np.linalg.norm(centers, axis=1)
            for i in range(N):
                if new_norms[i] > 0.99:
                    centers[i] = centers[i] / new_norms[i] * 0.99
            
            # 7. Shake to escape local minima
            iteration += 1
            if iteration % shake_interval == 0:
                # Random perturbation
                centers += np.random.normal(0, shake_magnitude, (N, 2))
                new_norms = np.linalg.norm(centers, axis=1)
                for i in range(N):
                    if new_norms[i] > 0.99:
                        centers[i] = centers[i] / new_norms[i] * 0.99
                # Reduce step size slightly after shake? Or keep constant.
                # step_size = max(0.01, step_size * 0.9)
            
            # Adaptive step size: if score improved, keep or increase; else decrease
            # This is a simple heuristic, proper line search is expensive.
            
        return best_centers, best_score

    def hex_lattice_init():
        spacing = 0.18
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists)
            return points[sorted_indices[:N]]
        else:
            return np.random.uniform(-0.8, 0.8, (N, 2))

    def random_init():
        centers = np.random.uniform(-0.9, 0.9, (N, 2))
        norms = np.linalg.norm(centers, axis=1)
        mask = norms > 0.99
        centers[mask] = centers[mask] / norms[mask, np.newaxis] * 0.99
        return centers

    # Run multiple starts
    best_centers = None
    best_score = -1.0
    
    num_starts = 5
    time_per_start = (max_time - 10.0) / num_starts
    
    for s in range(num_starts):
        if time.time() > start_time + max_time - 5.0:
            break
        
        if s == 0:
            init = hex_lattice_init()
        else:
            init = random_init()
            
        centers, score = optimize_packing(init, time_per_start)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # Final validation and cleanup
    if best_centers is None:
        best_centers = random_init()
        
    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_centers[i] = best_centers[i] / norms[i] * 0.99
            
    return best_centers
