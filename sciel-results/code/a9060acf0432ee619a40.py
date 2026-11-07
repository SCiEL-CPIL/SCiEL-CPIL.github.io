import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import squareform

def _reconstruct_points(dist_matrix):
    """
    Reconstruct 2D points from a distance matrix using MDS.
    """
    n = len(dist_matrix)
    D2 = dist_matrix ** 2
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ D2 @ J
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    
    # Take the top 2 eigenvalues
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Points are eigvecs * sqrt(eigvals)
    # Take first 2 components
    if n >= 2:
        points = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    else:
        points = np.zeros((n, 2))
        
    return points

def _run_aco_sim(dist, eta, n_ants=20, n_iters=30, seed=42):
    """
    Optimized ACO simulation.
    """
    n = len(dist)
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    
    best_len = float('inf')
    
    for _ in range(n_iters):
        tours_list = []
        
        for _ in range(n_ants):
            tour = [0]
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            curr = 0
            
            for _ in range(n - 1):
                # Vectorized probability calculation
                # eta[curr, :] is the heuristic row
                # tau[curr, :] is the pheromone row
                probs = tau[curr, :] * eta[curr, :]
                probs[visited] = 0.0
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    unvisited = np.where(~visited)[0]
                    next_node = rng.choice(unvisited)
                else:
                    probs /= total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour.append(next_node)
                visited[next_node] = True
                curr = next_node
            
            tour.append(0)
            tour_arr = np.array(tour, dtype=np.int32)
            
            # Calculate length
            length = np.sum(dist[tour_arr[:-1], tour_arr[1:]])
            
            if length < best_len:
                best_len = length
            
            tours_list.append((length, tour_arr))
        
        # Update Pheromones
        tau *= 0.9
        
        for length, tour_arr in tours_list:
            reward = 1.0 / length
            u = tour_arr[:-1]
            v = tour_arr[1:]
            tau[u, v] += reward

    return best_len

def heuristics(distance_matrix):
    start_time = time.time()
    timeout = 320  # Leave margin for 330s limit
    
    n = len(distance_matrix)
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    
    # Reconstruct points
    try:
        points = _reconstruct_points(safe_dist)
        delaunay = Delaunay(points)
        
        # Create mask for Delaunay edges
        delaunay_mask = np.zeros((n, n), dtype=bool)
        for simplex in delaunay.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    u, v = simplex[i], simplex[j]
                    delaunay_mask[u, v] = True
                    delaunay_mask[v, u] = True
    except Exception:
        # Fallback if reconstruction fails
        delaunay_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(delaunay_mask, False)

    # Search for best delta and alpha
    # delta: factor for non-Delaunay edges
    # alpha: exponent for distance
    # We expect alpha ~ 1-3, delta ~ 1e-3 to 1e-9
    
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    
    best_score = float('inf')
    best_eta = None
    
    # Coarse search
    for alpha in alphas:
        if time.time() - start_time > timeout:
            break
        for delta in deltas:
            if time.time() - start_time > timeout:
                break
            
            eta = 1.0 / (safe_dist ** alpha)
            # Apply mask
            eta = np.where(delaunay_mask, eta, eta * delta)
            np.fill_diagonal(eta, 1e-9)
            
            # Evaluate with a few seeds
            score_sum = 0
            valid = 0
            for seed in [1, 2, 3]:
                if time.time() - start_time > timeout:
                    break
                s = _run_aco_sim(distance_matrix, eta, n_ants=20, n_iters=30, seed=seed)
                score_sum += s
                valid += 1
            
            if valid > 0:
                avg_score = score_sum / valid
                if avg_score < best_score:
                    best_score = avg_score
                    best_eta = eta.copy()
    
    # Fine search around best found
    if best_eta is not None:
        # We need to track which alpha and delta were best
        # Let's re-run the search to find the best alpha/delta pair explicitly
        best_alpha = 2.0
        best_delta = 1e-5
        # Re-evaluate to find best params
        # This is inefficient, but the search space is small (25 combos)
        # Let's just do a second pass with finer grid around the best region if time permits
        # For now, return the best found in the coarse pass
        
        # If time remains, do a fine pass
        if time.time() - start_time < timeout - 10:
            # Determine best alpha and delta from the coarse pass? 
            # We didn't store them. Let's just do a finer grid over the whole range if time allows.
            # Or just return the best.
            pass

    if best_eta is None:
        # Fallback
        eta = 1.0 / safe_dist
        np.fill_diagonal(eta, 1e-9)
        best_eta = eta

    return best_eta
