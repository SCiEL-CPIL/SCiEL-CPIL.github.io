"""
Optimized heuristics for ACO CVRP using a Node-Potential Field.

Strategy:
- Replace pairwise edge-attraction with a node-potential model.
- Each node j has a "demand pressure" computed from the spatial density of
  other customers within a Gaussian kernel, weighted by their demands.
- The heuristic for edge (i, j) is:
    eta[i, j] = (1 / d[i,j]) * pressure[j] * exp(-beta * dist(j, centroid))
  where:
    - pressure[j] = sum_k demand[k] * exp(-gamma * dist(j, k)^2)
      (captures local demand clustering around j)
    - exp(-beta * dist(j, centroid)) biases toward the center of remaining
      demand, encouraging ants to load balanced routes that cover dense regions.
- This structural change moves from a purely distance-based rank to a
  demand-aware spatial potential field, giving the pheromone a richer
  landscape to exploit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.

    Returns:
        numpy.ndarray: Heuristic matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)

    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    eps = 1e-9

    # Safety: if coords or demands have wrong shape, fall back to inverse distance
    if coords.shape != (n, 2) or dem.shape != (n,):
        eta = 1.0 / (d + eps)
        mask = ~np.eye(n, dtype=bool)
        m = np.mean(eta[mask])
        if m > 0 and np.isfinite(m):
            eta = eta / m
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta

    # --- Step 1: Compute pairwise Euclidean distances from coordinates ---
    # d should already be the distance matrix, but we compute from coords
    # to ensure consistency and to get a clean matrix for kernel operations.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + eps)

    # Use the provided distance matrix if it looks valid, otherwise use computed
    # (they should be the same, but we prefer the input)
    if np.allclose(dist, d, atol=1e-6, rtol=1e-6):
        pass  # use d
    else:
        # If mismatched, use computed distances for internal calculations
        # but still return a matrix of shape (n,n)
        pass

    # --- Step 2: Compute node pressure (demand-weighted spatial density) ---
    # pressure[j] = sum over k != j of demand[k] * exp(-gamma * dist[j,k]^2)
    # gamma controls the kernel width. For unit square, a reasonable gamma ~ 20-50.
    gamma = 30.0

    # dist^2 matrix
    dist_sq = dist ** 2

    # Zero out diagonal for the kernel (no self-interaction)
    kernel = np.exp(-gamma * dist_sq)
    np.fill_diagonal(kernel, 0.0)

    # pressure[j] = sum_k kernel[j,k] * demand[k]
    pressure = kernel @ dem  # shape (n,)

    # Ensure positive pressure; nodes with no nearby demand get a small floor
    pressure = np.maximum(pressure, 0.1)

    # --- Step 3: Compute centroid of customers (excluding depot) ---
    # Customers are nodes 1..n-1
    customer_coords = coords[1:]
    customer_demands = dem[1:]
    total_demand = customer_demands.sum()

    if total_demand > 0:
        centroid = np.average(customer_coords, weights=customer_demands, axis=0)
    else:
        centroid = customer_coords.mean(axis=0)

    # Distance from each node to the centroid
    dist_to_centroid = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))

    # --- Step 4: Build the heuristic matrix ---
    # eta[i,j] = (1 / d[i,j]) * pressure[j] * exp(-beta * dist_to_centroid[j])
    # beta controls how strongly we bias toward the centroid.
    beta = 2.0

    centroid_bias = np.exp(-beta * dist_to_centroid)  # shape (n,)

    # Inverse distance matrix (with eps to avoid division by zero on diagonal)
    inv_dist = 1.0 / (d + eps)

    # eta[i,j] = inv_dist[i,j] * pressure[j] * centroid_bias[j]
    # pressure[j] and centroid_bias[j] are per-column (destination node)
    eta = inv_dist * pressure[np.newaxis, :] * centroid_bias[np.newaxis, :]

    # --- Step 5: Normalize ---
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])

    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        # Fallback to simple inverse distance
        eta = 1.0 / (d + eps)
        mask = ~np.eye(n, dtype=bool)
        m = np.mean(eta[mask])
        if m > 0 and np.isfinite(m):
            eta = eta / m

    # --- Step 6: Sanitize ---
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9

    # Deadline guard
    if time.time() > deadline:
        pass  # already computed; nothing more to do

    return eta
# EVOLVE-BLOCK-END
