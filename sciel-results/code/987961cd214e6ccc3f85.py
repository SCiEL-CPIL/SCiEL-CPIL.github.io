"""Combined CVRP heuristic: geometric sweep + savings + normalized lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Calculate Savings (Clarke-Wright)
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Geometric sweep term: project onto principal axis derived from distance matrix
    # Use the distance matrix to derive a 1D projection that captures spatial structure
    # Project each feasible node onto the axis defined by (depot, current_node) direction
    # This creates a "heading" vector and we prefer nodes that maintain compact sectors
    
    # Compute heading vector from depot to current_node
    # Since we only have distance matrix, use a proxy: 
    # The "angle" concept can be approximated by how well a node fits the current route direction
    # Use: geometric_score = -d_cur_feas + alpha * (d_dep_feas - d_cur_dep)
    # This favors nodes that are close to current and far from depot (extending the route)
    
    # Alternative: use the savings-based geometric interpretation
    # savings = d_cur_dep + d_dep_feas - d_cur_feas
    # High savings means the node is "aligned" with extending the route away from depot
    
    # Compute continuation cost: min distance from each candidate to (depot or other feasible nodes)
    if n_nodes > 1:
        # Distance matrix among feasible nodes
        d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
        # For each row i, find min over j != i
        np.fill_diagonal(d_feas_feas, np.inf)
        min_to_other = np.min(d_feas_feas, axis=1)
        
        # Continuation cost = min(d(cand, depot), min_to_other)
        cont_cost = np.minimum(d_dep_feas, min_to_other)
    else:
        cont_cost = d_dep_feas.copy()
    
    # Normalize terms to have comparable scales
    eps = 1e-10
    savings_norm = (savings - np.min(savings)) / (np.max(savings) - np.min(savings) + eps)
    dist_norm = (d_cur_feas - np.min(d_cur_feas)) / (np.max(d_cur_feas) - np.min(d_cur_feas) + eps)
    cont_norm = (cont_cost - np.min(cont_cost)) / (np.max(cont_cost) - np.min(cont_cost) + eps)
    
    # Weighted score combining savings, distance penalty, and continuation cost
    # W_sav: weight for savings (higher = prefer high-savings nodes)
    # W_dist: weight for distance penalty (higher = prefer shorter edges)
    # W_cont: weight for continuation cost (higher = prefer nodes with easy next steps)
    W_sav = 1.0
    W_dist = 0.6
    W_cont = 0.4
    
    scores = W_sav * savings_norm - W_dist * dist_norm - W_cont * cont_norm
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
