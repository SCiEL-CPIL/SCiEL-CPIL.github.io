import numpy as np
import time

_EVAL_START = time.time()
DEADLINE = _EVAL_START + 140

def _time_remaining():
    return DEADLINE - time.time()

def _project_onto_principal_axis(distance_matrix, node_indices):
    """
    Project nodes onto a principal axis derived from the distance matrix.
    This creates a 1D representation that helps identify spatial clusters.
    """
    if len(node_indices) < 2:
        return np.zeros(len(node_indices))
    
    # Extract submatrix for these nodes
    sub_matrix = distance_matrix[np.ix_(node_indices, node_indices)]
    
    # Use eigendecomposition to find principal axis
    # For computational efficiency with small n, use SVD
    try:
        _, _, vt = np.linalg.svd(sub_matrix, full_matrices=False)
        # Principal direction is the first right singular vector
        principal_axis = vt[0]
        
        # Project each node onto this axis
        projections = sub_matrix @ principal_axis
        
        # Normalize to [0, 1] range
        proj_min = projections.min()
        proj_max = projections.max()
        if proj_max > proj_min:
            normalized = (projections - proj_min) / (proj_max - proj_min)
        else:
            normalized = np.zeros(len(projections))
            
        return normalized
    except:
        # Fallback: use distance to depot as projection
        depot_idx = node_indices[0]  # Assume first is depot or use first node
        return distance_matrix[node_indices, depot_idx] / (distance_matrix[node_indices, depot_idx].max() + 1e-10)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
       if _time_remaining() < 1.0:
           if len(unvisited_nodes) == 0:
               return 0
           feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
           if len(feasible) == 0:
               return 0
           return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

       if len(unvisited_nodes) == 0:
           return 0

       unvisited_arr = np.array(unvisited_nodes)
       feasible_mask = demands[unvisited_arr] <= rest_capacity
       feasible = unvisited_arr[feasible_mask]

       if len(feasible) == 0:
           return 0

       if len(feasible) == 1:
           return int(feasible[0])

       # Distances
       d_cur = distance_matrix[current_node, feasible]
       d_dep = distance_matrix[feasible, depot]
       
       # Compute spatial projection for clustering
       # Include current_node and depot in the projection to maintain context
       all_nodes = np.concatenate([[current_node, depot], feasible])
       projections = _project_onto_principal_axis(distance_matrix, all_nodes)
       
       # Extract projections for feasible nodes (skip first two: current and depot)
       feasible_projections = projections[2:]
       current_projection = projections[0]
       
       # Spatial proximity score: prefer nodes close to current node in projected space
       spatial_score = np.abs(feasible_projections - current_projection)
       
       # Normalize spatial score
       spatial_max = spatial_score.max()
       if spatial_max > 0:
           spatial_score = spatial_score / spatial_max
       
       # Combined score: savings + spatial clustering
       # Savings: minimize d(current, j) - d(depot, j)
       savings_score = d_cur - d_dep
       
       # Normalize savings score
       savings_min = savings_score.min()
       savings_max = savings_score.max()
       if savings_max > savings_min:
           savings_score = (savings_score - savings_min) / (savings_max - savings_min)
       else:
           savings_score = np.zeros(len(savings_score))
       
       # Weighted combination: 70% savings, 30% spatial clustering
       # This maintains compact routes while leveraging savings
       combined_score = 0.7 * savings_score + 0.3 * spatial_score
       
       best_idx = np.argmin(combined_score)
       return int(feasible[best_idx])
