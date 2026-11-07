"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def _fast_aco_sim_v2(dm, demands, capacity, H, n_ants=20, n_iterations=10, seed=42, time_limit=5.0):
    """
    Faster, higher-fidelity simulation of the evaluator's ACO.
    Matches n_ants=20.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = row_tau * row_eta
                weights[~valid_mask] = 0.0

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost


def _compute_flow_potential(dm, demands, capacity, n):
    """
    Compute a flow-based potential heuristic.
    The idea: solve a minimum-cost flow relaxation where each customer needs to be served,
    and the depot has unlimited supply. Edge costs are distances.
    The dual variables (node potentials) give us a heuristic: edges with small potential
    difference + edge cost are more attractive.
    
    Since we can't easily solve LP here, we approximate with a simple potential:
    potential[i] = distance from depot to i (via shortest path in the complete graph).
    Then edge attractiveness ~ 1 / (dm[i,j] + |potential[i] - potential[j]| + epsilon)
    This encodes whether moving from i to j is "consistent" with the global flow from depot.
    """
    try:
        # Shortest path from depot (node 0) to all nodes
        # In a complete graph with non-negative weights, shortest path is just the direct edge
        # unless there's a shorter path through intermediate nodes.
        # For Euclidean-like distances, direct is usually shortest, but let's compute properly.
        
        # Simple Bellman-Ford / SPFA for small n
        INF = 1e18
        dist = np.full(n, INF)
        dist[0] = 0.0
        in_queue = np.ones(n, dtype=bool)
        queue = list(range(n))
        
        # Actually for n=51, just do repeated relaxation
        for _ in range(n - 1):
            updated = False
            for i in range(n):
                if dist[i] == INF:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    new_dist = dist[i] + dm[i, j]
                    if new_dist < dist[j] - 1e-12:
                        dist[j] = new_dist
                        updated = True
            if not updated:
                break
        
        # Potential: distance from depot
        potential = dist.copy()
        # Ensure finite
        potential = np.where(np.isfinite(potential), potential, 0.0)
        
        # Edge attractiveness: 1 / (dm[i,j] + |potential[i] - potential[j]| + epsilon)
        # This is high when the edge cost is small AND the potential difference is small
        # (i.e., the edge is "along the flow" rather than "against it")
        pot_diff = np.abs(potential[:, None] - potential[None, :])
        edge_cost_adj = dm + pot_diff + 1e-6
        H_flow = 1.0 / edge_cost_adj
        
        # Also create a variant that weights by demand
        d_demands = demands.reshape(-1, 1)
        H_flow_demand = H_flow / (1.0 + 0.1 * d_demands)
        
        # Another variant: use potential difference as a penalty only for "backward" edges
        # If potential[j] > potential[i], moving from i to j is "forward" (away from depot)
        # If potential[j] < potential[i], moving from i to j is "backward" (toward depot)
        # We want to encourage forward moves early, backward moves late.
        # For a static heuristic, let's just penalize large potential differences.
        pot_diff_signed = potential[None, :] - potential[:, None]  # pot[j] - pot[i]
        # Positive means j is farther from depot than i
        # We can boost edges that go forward (positive pot_diff) and penalize backward
        H_flow_signed = H_flow * (1.0 + 0.3 * np.tanh(0.5 * pot_diff_signed))
        
        return H_flow, H_flow_demand, H_flow_signed
    except Exception:
        return None, None, None


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Structural candidates
    H1 = 1.0 / dm_safe
    H2 = 1.0 / (dm_safe ** 2)
    
    d_demands = demands.reshape(-1, 1)
    H3 = 1.0 / (dm_safe * (1.0 + 0.1 * d_demands))
    H4 = 1.0 / ((dm_safe ** 2) * (1.0 + 0.1 * d_demands))
    
    dist_depot = dm[0, :]
    dist_depot_safe = np.where(dist_depot > 1e-10, dist_depot, 1e10)
    boost = 1.0 / (1.0 + dist_depot_safe).T
    H5 = H1 * boost
    H6 = H2 * boost
    
    candidates = []
    for h in [H1, H2, H3, H4, H5, H6]:
        h_safe = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        candidates.append(h_safe)

    # MST-based heuristic
    try:
        dm_sparse = csr_matrix(dm_safe)
        mst = minimum_spanning_tree(dm_sparse)
        mst_dense = mst.toarray()
        
        mst_mask = (mst_dense > 0).astype(float)
        
        H7 = H1 * (1.0 + 1.0 * mst_mask)
        H8 = H2 * (1.0 + 1.0 * mst_mask)
        
        h7_safe = np.where(np.isfinite(H7) & (H7 > 0), H7, 1e-9)
        h8_safe = np.where(np.isfinite(H8) & (H8 > 0), H8, 1e-9)
        
        candidates.append(h7_safe)
        candidates.append(h8_safe)
    except Exception:
        pass

    # KNN-based heuristic
    try:
        tree = cKDTree(coords)
        knn_dists, knn_idxs = tree.query(coords, k=6)
        
        H_nn = np.zeros_like(dm)
        for i in range(n):
            for rank in range(1, 6):
                j = knn_idxs[i, rank]
                d = knn_dists[i, rank]
                weight = (1.0 / (rank + 1)) * (1.0 / (1.0 + d))
                H_nn[i, j] = weight
                H_nn[j, i] = weight
                
        H_nn_safe = np.where(np.isfinite(H_nn) & (H_nn > 0), H_nn, 1e-9)
        candidates.append(H_nn_safe)
    except Exception:
        pass

    # Flow-based potential heuristic
    H_flow, H_flow_demand, H_flow_signed = _compute_flow_potential(dm, demands, capacity, n)
    if H_flow is not None:
        hf_safe = np.where(np.isfinite(H_flow) & (H_flow > 0), H_flow, 1e-9)
        candidates.append(hf_safe)
    
    if H_flow_demand is not None:
        hfd_safe = np.where(np.isfinite(H_flow_demand) & (H_flow_demand > 0), H_flow_demand, 1e-9)
        candidates.append(hfd_safe)
    
    if H_flow_signed is not None:
        hfs_safe = np.where(np.isfinite(H_flow_signed) & (H_flow_signed > 0), H_flow_signed, 1e-9)
        candidates.append(hfs_safe)

    best_H = candidates[0]
    best_score = np.inf
    
    # Evaluate candidates
    # We have ~11 candidates now.
    # 3 seeds each, 5s limit each -> 15s per candidate.
    # 11 * 15 = 165s. Plenty of time.
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
            
        costs = []
        for s in [0, 1, 2]:
            cost = _fast_aco_sim_v2(dm, demands, capacity, h, 
                                    n_ants=20, n_iterations=10, 
                                    seed=s, time_limit=5.0)
            if cost < np.inf:
                costs.append(cost)
                
        if costs:
            avg_cost = np.mean(costs)
            if avg_cost < best_score:
                best_score = avg_cost
                best_H = h

    # Refine with mixing
    if time.time() < deadline - 40.0:
        for alpha in [0.5, 0.7, 0.8, 0.9]:
            H_mix = alpha * best_H + (1 - alpha) * H1
            H_mix = np.where(np.isfinite(H_mix) & (H_mix > 0), H_mix, 1e-9)
            
            costs = []
            for s in [0, 1, 2]:
                cost = _fast_aco_sim_v2(dm, demands, capacity, H_mix, 
                                        n_ants=20, n_iterations=10, 
                                        seed=s, time_limit=5.0)
                if cost < np.inf:
                    costs.append(cost)
            
            if costs:
                avg_cost = np.mean(costs)
                if avg_cost < best_score:
                    best_score = avg_cost
                    best_H = H_mix

    return best_H

# EVOLVE-BLOCK-END
