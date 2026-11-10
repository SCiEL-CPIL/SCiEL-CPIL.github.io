"""
Structural Change: Geometric Angular Sweep Heuristic for CVRP.
Instead of savings-based optimization, this heuristic models the routing decision
as a dynamic angle-matching task. It enforces spatial clustering by sweeping
around the depot in a consistent direction, selecting the next node that minimizes
angular deviation from the current position.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None
_depot_coords = None
_customer_angles = None
_angle_cache = {} # cache angles per instance if needed, but usually distance matrix is constant per instance run? 
# Actually, distance_matrix is passed in every call. We can't easily cache coordinates unless we assume 
# the matrix is Euclidean. The problem statement says "distance_matrix", which might be arbitrary.
# However, for "geometric angular sweep", we need angles. 
# If the distance matrix is not Euclidean, we cannot compute true angles.
# But the prompt says "generated 50-customer instances", which usually implies Euclidean coordinates.
# Let's assume we can derive coordinates or that the matrix allows angle estimation.
# A safer bet for "geometric" without coordinates: 
# We can use the distance matrix to estimate angles via the Law of Cosines relative to the depot.
# angle(depot, current, next) can be estimated.
# But "sweep" usually implies a global ordering.
# Let's try to reconstruct coordinates from the distance matrix using MDS (Multidimensional Scaling) 
# or simply assume the matrix is Euclidean and try to extract coordinates? 
# No, that's too heavy and might fail.
# Alternative: Use the "angular" penalty from the parent but make it the PRIMARY driver, not a penalty on savings.
# The parent did: score = savings - weight * angle.
# The new approach: score = -angle (minimize angle deviation) + small savings tie-breaker?
# Or simply: Pick the node that continues the sweep.
# To define a sweep without coordinates, we can define a "direction" based on the last move.
# However, a true sweep requires a global reference frame (the depot).
# Let's estimate the angle of each node relative to the depot using the distances.
# We need a reference axis. We can pick the node with the maximum distance from the depot as the reference?
# Or better: Use the Law of Cosines to find the angle between (depot->current) and (depot->next).
# This is local.
# To make it a "sweep", we need a consistent orientation.
# Let's assume the distance matrix is Euclidean. We can compute the angle of each node i relative to the depot
# if we had coordinates. Without coordinates, we can't get a global angle.
# BUT, we can use a trick: 
# If we assume the matrix is Euclidean, we can perform a lightweight MDS or t-SNE? No, too slow.
# Let's stick to the local angle calculation but enforce a "sweep" behavior by tracking the cumulative angle?
# Actually, the most robust "structural" change that fits "geometric angular sweep" without coordinates
# is to use the **Savings** but re-interpret it as **Angular Clustering**.
# Wait, the prompt says "Represent the routing decision as a geometric angular sweep... minimizing angular deviation...".
# If I cannot get global angles, I can use the local angle at the depot between the current node and candidate node.
# The parent already did this.
# To make it a *sweep*, I should pick the candidate that has the smallest positive angle relative to the current node's vector, 
# assuming a consistent rotation direction.
# How to determine rotation direction? 
# We can calculate the signed area (cross product) if we had 2D coords.
# Without coords, we can't determine "left" or "right" turn uniquely.
# However, we can use the following heuristic:
# 1. Calculate the angle $\alpha$ at the depot between vector $D \to C$ and $D \to J$ for all feasible $J$.
# 2. This gives a magnitude in $[0, \pi]$.
# 3. To enforce a sweep, we need to know if we are turning "in" or "out".
# Let's look at the distances. If $d(D,J)$ is increasing, we are moving out.
# This is getting complicated without coordinates.
# 
# Let's reconsider the "distance_matrix". In many CVRP benchmarks (like OR-Library), coordinates are provided.
# Here, only `distance_matrix` is provided.
# Is it possible the `distance_matrix` is actually a Euclidean distance matrix derived from coordinates?
# Yes, "generated 50-customer instances" strongly suggests standard Euclidean instances.
# We can recover the coordinates (up to rotation/translation) using Classical MDS (CMDS).
# CMDS is $O(N^3)$ for $N=50$, which is very fast ($125,000$ ops).
# We can cache the coordinates!
# 
# Plan:
# 1. On the first call, if we detect a new distance matrix (or just on the first call per instance), 
#    compute the 2D coordinates using CMDS.
# 2. Calculate the polar angle of each customer relative to the depot (which will be at the origin or near it).
# 3. Implement a true Angular Sweep:
#    - Maintain a "current sweep angle" or simply pick the next node in the sorted angle list that is feasible and "ahead" of the current angle.
#    - If we wrap around or hit an infeasible node that blocks the path, handle capacity.
# 
# Let's implement CMDS.
# D is the distance matrix.
# B = -0.5 * J D^2 J, where J = I - (1/n) * 11^T.
# Eigenvalue decomposition of B. Take the top 2 eigenvectors.
# 
# This is a significant structural change: from local greedy savings to a global geometric sweep based on recovered coordinates.

import numpy as np

_mds_coords = None
_mds_instance_id = None
_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None

def _compute_mds_coords(dist_matrix):
    """
    Recovers 2D coordinates from a Euclidean distance matrix using Classical MDS.
    """
    D2 = dist_matrix ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort by eigenvalue in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take the top 2 eigenvectors
    # Scale by sqrt of eigenvalues
    X = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    return X

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _mds_coords, _mds_instance_id, _start_time, _DEADLINE

    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT

    if time.time() > _DEADLINE:
        return int(depot)

    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)

    if len(unvisited_arr) == 0:
        return int(depot)

    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]

    # Compute MDS coordinates if not already done for this instance
    # We use the distance_matrix as a proxy for instance identity (hashing it is expensive, 
    # but we can check if _mds_coords is None or if the depot/size matches a new instance).
    # Since the evaluator runs 8 instances sequentially, we need to reset state between instances.
    # How to detect a new instance? 
    # The distance matrix will change. We can check the first few elements or just assume 
    # the function is called for one instance at a time and reset if the matrix changes significantly.
    # A safer bet: The problem says "eight generated 50-customer instances". 
    # We can track the shape and a checksum of the first row.
    
    # Simple check: If we haven't computed coords, or if the depot index or matrix size changed?
    # Actually, the simplest robust way is to re-compute if we detect a "reset".
    # But MDS is fast. Let's just compute it every time if we are at the depot? No, that's wasteful.
    # Let's assume the distance matrix is static per instance. 
    # We can store a fingerprint of the distance matrix.
    
    # Fingerprint: sum of first 10 elements of flattened matrix
    fp = np.sum(distance_matrix.flat[:10])
    if _mds_instance_id != fp:
        _mds_instance_id = fp
        _mds_coords = _compute_mds_coords(distance_matrix)
    
    # The depot is node 0. In MDS, the centroid is at the origin.
    # We need to shift coordinates so the depot is at (0,0) for angle calculation.
    depot_coords = _mds_coords[depot]
    coords = _mds_coords - depot_coords
    
    # Calculate angles for all nodes
    # angle = arctan2(y, x)
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    # Normalize angles to [0, 2*pi)
    angles[angles < 0] += 2 * np.pi

    current_angle = angles[current_node]

    # Strategy: Angular Sweep
    # We want to pick the feasible node with the smallest positive angular difference from current_angle.
    # diff = (angle_j - current_angle) % (2*pi)
    # We prefer small diff.
    # However, if we are at the depot, we can start anywhere. 
    # Usually, we pick the one with the smallest angle to start the sweep, or the nearest.
    # Let's stick to the sweep logic:
    # If at depot, pick the feasible node with the smallest angle (start of sweep) OR nearest.
    # Nearest is often better for the first leg. But "Angular Sweep" implies ordering.
    # Let's use: If at depot, pick nearest feasible.
    # If not at depot, pick feasible node with min positive angular difference.
    
    if current_node == depot:
        # Pick nearest feasible node to start
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # For non-depot, compute angular differences
    diffs = (angles[feasible_nodes] - current_angle) % (2 * np.pi)
    
    # We want the smallest positive difference.
    # However, if the smallest difference is very large (e.g., > pi/2), it might be better to close the route?
    # Or pick the next one in the sweep.
    # A pure sweep might jump across the map if the next angular neighbor is far in distance.
    # To balance, we can use a score: score = diff + lambda * distance
    # But the directive says "minimize angular deviation".
    # Let's try pure angular first, with a capacity check.
    
    min_diff_idx = np.argmin(diffs)
    best_node = feasible_nodes[min_diff_idx]
    best_diff = diffs[min_diff_idx]
    
    # Heuristic: If the angular jump is too big (e.g., > 60 degrees = pi/3), 
    # and the distance is also large, maybe close the route?
    # Let's add a distance tie-breaker or penalty to prevent long jumps.
    # score = best_diff + 0.1 * distance_matrix[current_node, best_node] / max_distance?
    
    # Let's refine: Pick the node that minimizes (angular_diff + 0.5 * normalized_distance)
    # This enforces the sweep but allows nearby nodes to win if the angle is similar.
    
    dists = distance_matrix[current_node, feasible_nodes]
    max_dist = np.max(dists)
    if max_dist > 0:
        norm_dists = dists / max_dist
    else:
        norm_dists = np.zeros_like(dists)
        
    scores = diffs + 0.5 * norm_dists
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
