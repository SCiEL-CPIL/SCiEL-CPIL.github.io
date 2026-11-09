"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix using a node-potential gradient
    derived from the linear assignment problem (Hungarian algorithm).
    
    The heuristic H[i,j] = exp(-(c_ij - u_i - v_j) / tau) where c_ij is the
    distance, u and v are the dual variables from the assignment problem,
    and tau is a temperature parameter controlling the sharpness.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Solve the linear assignment problem to get dual variables
    # linear_sum_assignment returns row_ind, col_ind, and we can get costs
    row_ind, col_ind = linear_sum_assignment(d)
    
    # Compute the reduced costs: c_ij - u_i - v_j
    # We need to recover u and v from the assignment.
    # In the Hungarian algorithm, for assigned pairs (i, j): c_ij - u_i - v_j = 0
    # For unassigned: c_ij - u_i - v_j >= 0
    # We can compute u and v by solving a system, but a simpler approach:
    # The reduced cost matrix R has R[i,j] = d[i,j] - u_i - v_j >= 0
    # with R[i, col_ind[i]] = 0 for each i.
    
    # A practical approach: use the assignment to define a baseline.
    # Let's compute the "regret" of each edge relative to the optimal assignment.
    # For each node i, the best assigned edge is (i, col_ind[i]) with cost d[i, col_ind[i]].
    # For each node j, the best assigned edge is (row_ind[k], j) where col_ind[k]=j, with cost d[row_ind[k], j].
    
    # Simpler and more robust: compute reduced costs using a dual solution.
    # We can obtain u and v by noting that for the optimal assignment,
    # u_i + v_j = d[i, col_ind[i]] for all i, and u_i + v_j = d[row_ind[k], j] for all j.
    # This is a system of 2n equations in 2n unknowns (with one degree of freedom).
    # Let's fix u_0 = 0 and solve.
    
    u = np.zeros(n)
    v = np.zeros(n)
    
    # For each assigned pair (i, j=col_ind[i]): u_i + v_j = d[i, j]
    # We have n such equations. We need n more or use the structure.
    # Actually, the dual solution is not unique. Let's use a different approach.
    
    # Alternative: compute the reduced cost matrix directly using the fact that
    # in the optimal solution, the reduced costs of assigned edges are 0.
    # We can set u_i = 0 for all i and v_j = min_i d[i,j] - min_i d[i,j] ... no.
    
    # Let's use a simpler heuristic: the "assignment regret" for edge (i,j) is
    # how much worse it is compared to the best edge incident to i and the best
    # edge incident to j, normalized by the assignment cost.
    
    # Best edge from each node i (in the assignment):
    assigned_cost_from_i = d[np.arange(n), col_ind]
    # Best edge to each node j (in the assignment):
    # For node j, find i such that col_ind[i] == j
    inv_col = np.full(n, -1, dtype=int)
    for i in range(n):
        inv_col[col_ind[i]] = i
    assigned_cost_to_j = d[row_ind, np.arange(n)]
    
    # The total assignment cost
    total_assign_cost = np.sum(assigned_cost_from_i)
    
    # For edge (i, j), compute a "regret" measure:
    # regret[i,j] = d[i,j] - 0.5*(assigned_cost_from_i[i] + assigned_cost_to_j[j])
    # But this can be negative. Let's use:
    # regret[i,j] = max(0, d[i,j] - 0.5*(assigned_cost_from_i[i] + assigned_cost_to_j[j]))
    
    # Actually, let's use the proper reduced cost approach.
    # Compute u and v by solving the dual.
    # Set u_0 = 0. For each i, v_{col_ind[i]} = d[i, col_ind[i]] - u_i.
    # Then for consistency, check if this gives valid v.
    
    u[0] = 0.0
    for i in range(n):
        j = col_ind[i]
        v[j] = d[i, j] - u[i]
        # Now we can determine u for other nodes if they are assigned to already-known v
    # This doesn't fully determine u. Let's iterate.
    
    # Better: build the system. The dual constraints are u_i + v_j <= d[i,j] for all i,j,
    # with equality for assigned pairs. We want to find u, v that satisfy this.
    # A simple feasible solution: u_i = 0, v_j = -max_i d[i,j] ... no, v should make u_i+v_j <= d[i,j].
    # Let u_i = 0, then v_j <= min_i d[i,j]. Set v_j = min_i d[i,j].
    # Then reduced cost R[i,j] = d[i,j] - 0 - min_k d[k,j] >= 0.
    # This is a valid reduced cost matrix!
    
    v_min = np.min(d, axis=0)  # v_j = min_i d[i,j]
    u_zero = np.zeros(n)
    
    # Reduced cost: R[i,j] = d[i,j] - u_zero[i] - v_min[j] = d[i,j] - min_k d[k,j]
    reduced_costs = d - v_min[np.newaxis, :]
    
    # Ensure non-negative (should be by construction)
    reduced_costs = np.maximum(reduced_costs, 0.0)
    
    # Temperature parameter for the exponential
    # Scale by the mean reduced cost to make it dimensionless
    mean_rc = np.mean(reduced_costs)
    if mean_rc < 1e-12:
        mean_rc = 1.0
    tau = mean_rc * 0.5
    
    # Heuristic: H[i,j] = exp(-reduced_costs[i,j] / tau)
    # This gives higher attractiveness to edges with lower reduced cost
    heuristic = np.exp(-reduced_costs / tau)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
