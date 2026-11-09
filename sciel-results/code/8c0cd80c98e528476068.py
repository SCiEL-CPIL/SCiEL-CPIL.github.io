"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # If the set of unvisited nodes is small, we can afford a more expensive global check
    # or simply switch to a high-precision lookahead which is effectively optimal for small n.
    # For larger n, we use the MST-Leaf heuristic.
    
    # Strategy: MST Leaf Pruning Heuristic
    # 1. Identify the "center of mass" or the node in candidates closest to the destination.
    #    This acts as the anchor for the sub-tour.
    # 2. Compute a rough MST structure (or just the minimum outgoing edge for each node) 
    #    to identify which nodes are "leaves" (peripheral) and which are "hubs" (central).
    # 3. We prefer to visit leaves first to avoid getting trapped in a cluster, 
    #    but we also need to be close to the current node.
    # 4. Specifically, we calculate for each candidate:
    #    - dist_to_current: distance from current_node to candidate
    #    - min_dist_to_other: minimum distance from candidate to any other unvisited node or destination
    #    A node with a very small min_dist_to_other is likely a hub (connected to many).
    #    A node with a larger min_dist_to_other is likely a leaf.
    #    However, simple MST leaf detection is O(N^2) which is fine for N=50.
    
    # Let's implement a fast approximation of "MST Leaf-ness".
    # For each node in candidates, find its nearest neighbor in (candidates U {destination}).
    # If a node's nearest neighbor is the destination, it's a "tail" node.
    # If a node's nearest neighbor is another candidate, it's part of the chain.
    
    # Actually, a simpler and more robust structural heuristic for TSP construction 
    # that differs from pure NN is the "2-opt aware greedy" or "Angle-based".
    # But the directive asks for "MST leaf pruning".
    
    # Let's compute the MST of the subgraph consisting of candidates + destination.
    # We only need the degree of each node in the MST to identify leaves (degree 1).
    # Since N is small (<=50), we can run Prim's algorithm from the destination node 
    # to build an MST rooted at the destination.
    
    # Nodes involved in the MST calculation:
    # 0: destination_node
    # 1..n: candidates
    
    # We will map indices:
    # 0 -> destination_node
    # i -> candidates[i-1]
    
    num_nodes = n + 1
    # Distance matrix for the subgraph
    # dm_sub[i, j] is distance between subgraph node i and j
    
    # Optimization: We don't need to build the full dense matrix if we can compute on the fly,
    # but for N=50, building it is trivial.
    
    # Build index map
    # sub_idx 0 is dest
    # sub_idx 1..n are candidates
    
    # Initialize Prim's algorithm
    in_mst = np.zeros(num_nodes, dtype=bool)
    in_mst[0] = True  # Start with destination
    
    # Edge weights to MST for each node not in MST
    # We track the minimum edge connecting a node to the current MST
    min_dist_to_mst = np.full(num_nodes, np.inf)
    parent_in_mst = np.full(num_nodes, -1, dtype=int)
    
    # Distances from destination (node 0) to all candidates
    for i in range(1, num_nodes):
        cand_idx = candidates[i-1]
        min_dist_to_mst[i] = distance_matrix[destination_node, cand_idx]
        parent_in_mst[i] = 0
        
    # Prim's algorithm loop
    for _ in range(num_nodes - 1):
        # Find the node not in MST with the smallest min_dist_to_mst
        # We can use argmin on the vector, ignoring those already in MST
        # Set in_mst nodes to inf temporarily for argmin
        temp_min_dist = min_dist_to_mst.copy()
        temp_min_dist[in_mst] = np.inf
        next_node = np.argmin(temp_min_dist)
        
        if next_node == num_nodes: # Should not happen
            break
            
        in_mst[next_node] = True
        
        # Update distances for neighbors of next_node
        # The actual node index in the original graph
        if next_node == 0:
            orig_node = destination_node
        else:
            orig_node = candidates[next_node - 1]
            
        for j in range(num_nodes):
            if in_mst[j]:
                continue
            if j == 0:
                orig_j = destination_node
            else:
                orig_j = candidates[j - 1]
                
            d = distance_matrix[orig_node, orig_j]
            if d < min_dist_to_mst[j]:
                min_dist_to_mst[j] = d
                parent_in_mst[j] = next_node
                
    # Now we have the MST structure.
    # A node is a "leaf" if its degree in the MST is 1.
    # Note: The root (destination) might have degree > 1.
    # We want to pick a candidate that is a leaf in this MST, 
    # OR if no leaves are "good", pick the one that minimizes the path cost.
    
    # Calculate degree for each candidate in the MST
    # Degree = 1 (parent) + count of children
    degree = np.zeros(num_nodes, dtype=int)
    for i in range(1, num_nodes):
        degree[i] += 1 # Edge to parent
        p = parent_in_mst[i]
        if p != -1:
            degree[p] += 1
            
    # Identify leaves among the candidates (indices 1 to n)
    leaf_mask = np.zeros(n, dtype=bool)
    for i in range(1, num_nodes):
        if degree[i] == 1:
            leaf_mask[i-1] = True
            
    # If there are leaves, we prefer them.
    # Why? Because visiting a hub (degree > 1) early might force us to backtrack 
    # or cross the MST edges, which is bad for TSP.
    # Visiting leaves allows us to "peel" the tree, which is a known good strategy 
    # for TSP on tree-like structures.
    
    # However, we must also consider the distance from the CURRENT node.
    # If the nearest leaf is very far, and a hub is very close, we might still want the hub?
    # No, the heuristic is to follow the skeleton.
    # Let's score candidates:
    # Score = dist(current, candidate) + penalty_if_not_leaf
    
    # A better structural score:
    # For each candidate, calculate the cost to "enter" it from current 
    # and the cost of the edge it contributes to the MST.
    
    # Actually, a simpler and effective variant of "MST leaf pruning" in TSP heuristics 
    # is to select the node that minimizes:
    # dist(current, candidate) + alpha * (1 if candidate is a leaf in MST else 0)
    # Wait, we WANT leaves. So we should give a bonus (lower score) to leaves.
    
    # Let's refine:
    # 1. Compute dist_to_current for all candidates.
    # 2. Identify leaves.
    # 3. If there are leaves, restrict search to leaves? 
    #    If the current node is far from all leaves, this might be bad.
    #    Better: Score = dist_to_current - bonus * is_leaf
    
    # Let's try: Score = dist_to_current. 
    # But only among leaves?
    # If we restrict to leaves, we ensure we don't visit the "center" of a cluster 
    # until the end of the cluster is visited.
    
    # Let's check if there are any leaves.
    if np.any(leaf_mask):
        # Among the leaves, pick the one closest to current_node
        leaf_indices = np.where(leaf_mask)[0]
        leaf_candidates = candidates[leaf_indices]
        dist_to_leaf = distance_matrix[current_node, leaf_candidates]
        best_leaf_idx = leaf_indices[np.argmin(dist_to_leaf)]
        return int(candidates[best_leaf_idx])
    else:
        # No leaves? This can happen if the MST is a star (destination connected to all).
        # In that case, all nodes are leaves (degree 1) except destination.
        # Wait, if destination is connected to all, degree of candidates is 1.
        # So leaf_mask should be all True.
        # If leaf_mask is empty, something is wrong or N=1 (handled).
        # Fallback to nearest neighbor
        return int(candidates[np.argmin(distance_matrix[current_node, candidates])])

# EVOLVE-BLOCK-END
