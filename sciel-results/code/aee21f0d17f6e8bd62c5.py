# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using an MST-based heuristic.
    1. Computes MST of unvisited nodes.
    2. For each candidate, estimates the residual MST cost of the remaining nodes.
       - If the candidate is a leaf in the current MST, the residual MST is exact (MST - edge_to_parent).
       - If the candidate is internal, we approximate the residual MST by subtracting the sum of incident MST edges 
         and adding a penalty based on the minimum edge weight of the original MST to account for reconnection costs.
    3. Selects the candidate minimizing: dist(current, u) + dist(u, destination) + ResidualMST(u).
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Compute MST of the current unvisited set
    # We need to map local indices (0..n-1) to global node IDs
    # For efficiency, we work with the submatrix
    
    # Create a dense submatrix for the unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute MST
    try:
        mst_sparse = minimum_spanning_tree(sub_matrix)
        # Convert to dense for easier edge lookup if n is small, or keep sparse
        # n is at most 50, so dense is fine for local operations
        mst_dense = mst_sparse.toarray()
        
        # Calculate total MST weight
        total_mst_weight = np.sum(mst_dense)
        
        # Find the minimum edge weight in the MST (excluding 0s on diagonal)
        # We need to be careful with the diagonal
        np.fill_diagonal(mst_dense, np.inf)
        min_edge_weight = np.min(mst_dense)
        
        # Build adjacency list of MST to identify leaves and parents
        # mst_dense is symmetric
        mst_adj = {}
        for i in range(n_unvisited):
            mst_adj[i] = []
            for j in range(n_unvisited):
                if mst_dense[i, j] > 0:
                    mst_adj[i].append((j, mst_dense[i, j]))
                    
    except Exception:
        # Fallback to nearest neighbor if MST fails
        dists = distance_matrix[current_node, unvisited_nodes]
        min_idx = np.argmin(dists)
        return int(unvisited_nodes[min_idx])

    best_score = float('inf')
    best_node = -1

    # 2. Evaluate each candidate
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Cost to destination
        dest_dist = distance_matrix[candidate, destination_node]
        
        # Estimate Residual MST Cost for U \ {i}
        # Identify neighbors of i in the MST
        neighbors = mst_adj[i]
        
        if len(neighbors) == 0:
            # Isolated node in MST? Should not happen for connected graph, but handle it.
            # If i is isolated, removing it doesn't change MST of others?
            # Actually if n_unvisited > 1, MST is connected.
            residual_mst = total_mst_weight
        else:
            # Sum of weights of edges incident to i in the MST
            incident_weight = sum(w for _, w in neighbors)
            
            if len(neighbors) == 1:
                # i is a leaf. Removing i just removes that edge.
                # The remaining graph is still connected via the rest of the MST.
                residual_mst = total_mst_weight - incident_weight
            else:
                # i is internal. Removing i splits the tree into len(neighbors) components.
                # We need to reconnect them.
                # Approximation: 
                # The new MST will be roughly (Total MST - Incident Weight) + Reconnection Cost.
                # Reconnection Cost is at least the sum of the minimum edges connecting the new components to the rest.
                # A simple lower bound/estimate: 
                # We assume the reconnection cost is proportional to the number of new components 
                # and the minimum edge weight of the original graph (or MST).
                # Let's use: Reconnection Cost = (len(neighbors) - 1) * min_edge_weight * 2 
                # (Factor 2 is a heuristic buffer for detours).
                
                # A more robust estimate: 
                # The MST of the remaining nodes must connect the neighbors of i.
                # The cheapest way is to connect them via the rest of the graph.
                # We approximate this by adding the sum of the smallest (len(neighbors)-1) edges 
                # in the original submatrix that are NOT incident to i? 
                # That's expensive.
                
                # Simpler heuristic: 
                # residual_mst = (total_mst_weight - incident_weight) + (len(neighbors) - 1) * min_edge_weight
                
                residual_mst = (total_mst_weight - incident_weight) + (len(neighbors) - 1) * min_edge_weight
        
        # Total Score
        # We combine immediate, destination, and residual.
        # The residual represents the cost of the sub-tour for the remaining nodes.
        # The tour is current -> u -> (sub-tour) -> destination.
        # The sub-tour cost is approx residual_mst.
        # However, the sub-tour must start at u and end at destination? 
        # No, the sub-tour visits the remaining nodes. The path from u to the first remaining node 
        # and from the last remaining node to destination are not fully captured by MST(remaining).
        # MST(remaining) is the cost to connect the remaining nodes to each other.
        # We need to connect u to this cluster and this cluster to destination.
        # Let's add min_dist(u, remaining) and min_dist(destination, remaining) as in the previous version, 
        # but only if n_unvisited > 2.
        
        if n_unvisited > 2:
            # Find min distance from candidate to any other unvisited node
            # We can use the sub_matrix row i, excluding i
            other_indices = [j for j in range(n_unvisited) if j != i]
            min_dist_u_rem = np.min(sub_matrix[i, other_indices])
            
            # Find min distance from destination to any other unvisited node
            # Note: destination_node might be in unvisited_nodes or not.
            # If destination_node is in unvisited_nodes, it is one of the 'other' nodes.
            # We need min distance from destination to ANY node in U \ {u}.
            # If destination is in U \ {u}, min dist is 0? No, dist(dest, dest) = 0.
            # But the tour ends AT destination.
            # So if destination is in the remaining set, the "exit" cost is 0 (we are already there).
            # If destination is NOT in the remaining set, we need min dist from dest to U \ {u}.
            
            # Check if destination_node is in unvisited_nodes
            # unvisited_nodes is an array of node IDs.
            if destination_node in unvisited_nodes:
                # Destination is in the set.
                # If destination_node != candidate (which is true if n>1 and we are iterating),
                # then destination is in "other_indices".
                # The cost to "exit" to destination is effectively handled by the fact that 
                # the sub-tour must end at destination.
                # The MST connects them. The "exit" cost is min_dist from the cluster to dest.
                # Since dest is IN the cluster, min_dist is 0? 
                # Actually, the path goes through the cluster.
                # Let's just use min_dist from the *other* nodes to destination.
                # If destination is in other_indices, min_dist is 0.
                min_dist_dest_rem = np.min(sub_matrix[other_indices, np.where(unvisited_nodes == destination_node)[0][0]])
            else:
                # Destination is not in unvisited.
                min_dist_dest_rem = np.min(distance_matrix[destination_node, unvisited_nodes[other_indices]])
            
            residual_mst += min_dist_u_rem + min_dist_dest_rem
            
        total_score = immediate_dist + dest_dist + residual_mst
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate

    return int(best_node)
# EVOLVE-BLOCK-END
