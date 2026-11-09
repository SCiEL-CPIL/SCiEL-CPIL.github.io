import math
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
'''
Phylogenetic reconstruction using Neighbor Joining (NJ) followed by Parsimony NNI optimization.
'''

def _jc_distance_vec(A):
    """
    Compute Jukes-Cantor corrected distances between rows of A.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n distance matrix.
    """
    n, L = A.shape
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / L
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return d

def _nj_newick(D, labels):
    """
    Neighbor Joining returning Newick string.
    D: n x n distance matrix.
    labels: list of labels.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    active = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(n) if i != j}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        r = {}
        for i in active:
            r[i] = sum(dist[(i, j)] for j in active if j != i)
        
        min_q = float('inf')
        min_i, min_j = -1, -1
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        d_ij = dist[(min_i, min_j)]
        d_i = 0.5 * (d_ij + (r[min_i] - r[min_j]) / (m - 2))
        d_j = d_ij - d_i
        
        new_node = next_id
        next_id += 1
        node_struct[new_node] = (min_i, min_j, d_i, d_j)
        
        for k in active:
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist[(min_i, k)] + dist[(min_j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_node)
        
    a, b, c = active
    d_ab = dist[(a, b)]
    d_a = 0.5 * (d_ab + dist[(a, c)] - dist[(b, c)])
    d_b = d_ab - d_a
    
    new_node = next_id
    next_id += 1
    node_struct[new_node] = (a, b, d_a, d_b)
    d_c_branch = 0.5 * (dist[(a, c)] + dist[(b, c)] - d_ab)
    
    def to_newick(node_id):
        struct = node_struct[node_id]
        if isinstance(struct, str):
            return struct
        else:
            left_id, right_id, len_left, len_right = struct
            return f"({to_newick(left_id)}:{len_left:.4f},{to_newick(right_id)}:{len_right:.4f})"
    
    return f"({to_newick(a)}:{d_a:.4f},{to_newick(b)}:{d_b:.4f},{to_newick(c)}:{d_c_branch:.4f});"

def _build_tree_structure(newick, labels):
    """
    Parse Newick string into a tree structure suitable for NNI.
    Returns:
    - nodes: dict node_id -> list of neighbors
    - leaf_ids: dict label -> node_id
    """
    # Simple parser for unrooted binary trees
    # We will assign IDs to internal nodes.
    # This is a bit tricky. Let's use a recursive descent parser.
    
    # Actually, for NNI, we need the topology.
    # Let's represent the tree as a list of edges.
    # But we need to map leaves to node IDs.
    
    # Let's use a simpler representation:
    # A list of splits? No.
    # Let's use a dictionary of adjacency lists.
    
    # We will parse the Newick string.
    # Since we generated it with _nj_newick, we know the format.
    # But _nj_newick returns a string.
    # It's better to return the tree structure from _nj_newick directly.
    
    pass

def _nj_tree_structure(D, labels):
    """
    Returns the tree structure as:
    - nodes: list of node IDs (0 to n-1 are leaves, n to ... are internal)
    - adjacency: dict node_id -> set of neighbor node_ids
    - leaf_labels: dict node_id -> label
    """
    n = len(labels)
    if n == 1:
        return {0: set()}, {0: labels[0]}
    if n == 2:
        return {0: {1}, 1: {0}}, {0: labels[0], 1: labels[1]}
    
    active = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(n) if i != j}
    next_id = n
    
    # Adjacency list
    adj = {i: set() for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        r = {}
        for i in active:
            r[i] = sum(dist[(i, j)] for j in active if j != i)
        
        min_q = float('inf')
        min_i, min_j = -1, -1
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        d_ij = dist[(min_i, min_j)]
        d_i = 0.5 * (d_ij + (r[min_i] - r[min_j]) / (m - 2))
        d_j = d_ij - d_i
        
        new_node = next_id
        next_id += 1
        node_struct[new_node] = (min_i, min_j, d_i, d_j)
        
        # Update adjacency
        # Remove min_i - min_j edge? No, they are not directly connected yet in the final tree?
        # In NJ, we join min_i and min_j into new_node.
        # So min_i and min_j become children of new_node.
        # But in unrooted tree, new_node is connected to min_i and min_j.
        # And new_node takes the place of min_i and min_j in the active list.
        # The edges from min_i and min_j to other active nodes are transferred to new_node.
        
        # Actually, the adjacency in the final tree is built differently.
        # Let's just build the Newick and parse it?
        # No, let's build the adjacency directly.
        
        # The new node is connected to min_i and min_j.
        adj[new_node] = {min_i, min_j}
        adj[min_i].add(new_node)
        adj[min_j].add(new_node)
        
        # But min_i and min_j might have other neighbors?
        # In the "active" list, min_i and min_j are leaves of the current reduced tree.
        # Their other neighbors in the reduced tree are the other active nodes.
        # But we are building the final tree.
        # The edges to other active nodes are not directly added to new_node yet?
        # Yes, in NJ, the distance from new_node to k is calculated.
        # The topology is: new_node is the parent of min_i and min_j.
        # The "rest of the tree" is connected to new_node.
        
        # So, we need to update the adjacency of the other active nodes?
        # No, the other active nodes are not connected to min_i or min_j directly in the final tree.
        # They are connected to new_node.
        # So we don't add edges to adj[k] for k in active.
        # We just keep track of dist.
        
        for k in active:
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist[(min_i, k)] + dist[(min_j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_node)
        
    a, b, c = active
    d_ab = dist[(a, b)]
    d_a = 0.5 * (d_ab + dist[(a, c)] - dist[(b, c)])
    d_b = d_ab - d_a
    
    new_node = next_id
    next_id += 1
    node_struct[new_node] = (a, b, d_a, d_b)
    d_c_branch = 0.5 * (dist[(a, c)] + dist[(b, c)] - d_ab)
    
    # Final adjacency
    # new_node is connected to a and b.
    # c is connected to new_node?
    # In the final step, we have 3 nodes a, b, c.
    # We join a and b into new_node.
    # Then new_node and c are the remaining two?
    # No, we have 3 nodes.
    # The tree is (a, b, c) with a central node?
    # No, it's a binary tree.
    # So we join a and b. The new node is connected to c.
    
    adj[new_node] = {a, b, c}
    adj[a].add(new_node)
    adj[b].add(new_node)
    adj[c].add(new_node)
    
    # Wait, if a, b, c are internal nodes from previous steps, they already have neighbors.
    # We just add the edge to new_node.
    
    leaf_labels = {i: labels[i] for i in range(n)}
    return adj, leaf_labels

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170  # 10s margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
        
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = [sequences[l] for l in labels]
    A = np.array([[char_map[c] for c in seq] for seq in seqs], dtype=np.uint8)
    
    # Calculate distances
    D = _jc_distance_vec(A)
    
    # Get initial tree structure
    adj, leaf_labels = _nj_tree_structure(D, labels)
    
    # Convert sequences to bitmasks for Parsimony
    # A=1, C=2, G=4, T=8
    bitmask_map = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    L = len(seqs[0])
    # data: shape (n, L) of uint8
    data = np.array([[bitmask_map[c] for c in seq] for seq in seqs], dtype=np.uint8)
    
    # Initialize Fitch sets
    # fitch_sets: dict node_id -> np.array of shape (L,)
    # fitch_cost: dict node_id -> int
    fitch_sets = {}
    fitch_cost = {}
    
    # Initialize leaves
    for i in range(n):
        fitch_sets[i] = data[i]
        fitch_cost[i] = 0
    
    # Initialize internal nodes
    # We need to process internal nodes in a topological order (leaves to root).
    # Since it's a tree, we can do a BFS/DFS from leaves.
    # Or simply, since we built it with NJ, we know the parent-child relationships?
    # No, it's unrooted.
    # But we can root it arbitrarily for calculation.
    # Let's root at node 0.
    
    # Actually, we can just calculate Fitch sets for all internal nodes by iterating.
    # But we need the children's sets first.
    # Let's pick a root, say node 0.
    # Do a DFS to get post-order traversal.
    
    parent = {i: -1 for i in adj}
    visited = set()
    stack = [0]
    order = []
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in adj[node]:
            if neighbor != parent[node]:
                parent[neighbor] = node
                stack.append(neighbor)
                
    # Reverse order to get children before parents
    for node in reversed(order):
        if node < n:
            continue
        # Internal node
        children = [nei for nei in adj[node] if nei != parent[node]]
        # Combine Fitch sets of children
        # Start with first child's set
        if not children:
            continue
        s = fitch_sets[children[0]].copy()
        cost = 0
        for c in children[1:]:
            sc = fitch_sets[c]
            inter = s & sc
            # If inter is 0 for any site, cost +1
            # s becomes union
            # Vectorized:
            # inter is array.
            # cost += sum(inter == 0)
            # s = s | sc
            # But we need to do this per site.
            
            # inter_non_zero = inter != 0
            # cost += np.sum(~inter_non_zero)
            # s = s | sc
            
            # To be efficient, we can accumulate cost and update s.
            # But s changes per site? No, s is an array.
            # s = s | sc
            # cost += sum(inter == 0)
            
            # This is O(L) per child.
            # Since n is small, this is fine.
            
            inter = s & sc
            cost += int(np.sum(inter == 0))
            s = s | sc
            
        fitch_sets[node] = s
        fitch_cost[node] = cost
        
    total_cost = sum(fitch_cost.values())
    
    # NNI Optimization
    # Get list of internal edges
    # An edge (u, v) is internal if both u and v are internal?
    # Or if the edge is not a leaf edge?
    # NNI moves apply to internal edges.
    # In a binary tree, an edge is internal if it is not incident to a leaf.
    
    internal_edges = []
    for u in adj:
        if u >= n:
            for v in adj[u]:
                if v > u and v >= n:
                    internal_edges.append((u, v))
                    
    # Also, edges connecting internal node to leaf are not NNI-able?
    # NNI requires 4 subtrees.
    # So the edge must connect two internal nodes.
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for u, v in internal_edges:
            # Get neighbors of u (excluding v)
            nu = [nei for nei in adj[u] if nei != v]
            # Get neighbors of v (excluding u)
            nv = [nei for nei in adj[v] if nei != u]
            
            # In a binary tree, u has 3 neighbors. One is v. So 2 others.
            # v has 3 neighbors. One is u. So 2 others.
            if len(nu) != 2 or len(nv) != 2:
                continue
                
            a, b = nu
            c, d = nv
            
            # Get Fitch sets
            Sa = fitch_sets[a]
            Sb = fitch_sets[b]
            Sc = fitch_sets[c]
            Sd = fitch_sets[d]
            
            # Calculate costs for 3 topologies
            # Topology 1 (Original): u combines a,b; v combines c,d
            # Cost is fitch_cost[u] + fitch_cost[v]
            # But fitch_cost[u] is the cost incurred at u.
            # We need the incremental cost.
            # Actually, fitch_cost[u] is the sum of costs for all sites at node u.
            # So C1 = fitch_cost[u] + fitch_cost[v].
            
            C1 = fitch_cost[u] + fitch_cost[v]
            
            # Topology 2: a-c and b-d
            # New node N1 combines a, c.
            # New node N2 combines b, d.
            # Inter1 = Sa & Sc. Cost1 = sum(Inter1 == 0). S1 = Sa | Sc.
            # Inter2 = Sb & Sd. Cost2 = sum(Inter2 == 0). S2 = Sb | Sd.
            # C2 = Cost1 + Cost2.
            
            Inter1 = Sa & Sc
            Cost1 = int(np.sum(Inter1 == 0))
            S1 = Sa | Sc
            
            Inter2 = Sb & Sd
            Cost2 = int(np.sum(Inter2 == 0))
            S2 = Sb | Sd
            
            C2 = Cost1 + Cost2
            
            # Topology 3: a-d and b-c
            Inter3 = Sa & Sd
            Cost3 = int(np.sum(Inter3 == 0))
            S3 = Sa | Sd
            
            Inter4 = Sb & Sc
            Cost4 = int(np.sum(Inter4 == 0))
            S4 = Sb | Sc
            
            C3 = Cost3 + Cost4
            
            best_cost = min(C1, C2, C3)
            
            if best_cost < C1:
                # Update tree
                if best_cost == C2:
                    # Swap b and d? No, connect a-c and b-d.
                    # Remove edges u-a, u-b, v-c, v-d, u-v.
                    # Add edges N1-a, N1-c, N2-b, N2-d, N1-N2.
                    # But we don't have new node IDs.
                    # We can reuse u and v?
                    # Let u become N1 (connects a, c).
                    # Let v become N2 (connects b, d).
                    # Connect u and v.
                    
                    # Update adjacency
                    adj[u] = {a, c, v}
                    adj[v] = {b, d, u}
                    adj[a].remove(u)
                    adj[a].add(u) # already there?
                    # Wait, a was connected to u.
                    # Now a is connected to u.
                    # But c was connected to v.
                    # Now c is connected to u.
                    
                    # So:
                    # Remove u from adj[a]? No, u is still the node.
                    # But the set of neighbors changes.
                    
                    # Original:
                    # u connected to a, b, v.
                    # v connected to c, d, u.
                    
                    # New:
                    # u connected to a, c, v.
                    # v connected to b, d, u.
                    
                    # So:
                    # adj[u].remove(b)
                    # adj[u].add(c)
                    # adj[v].remove(c)
                    # adj[v].add(b)
                    
                    # And update adj of a, b, c, d?
                    # a is still connected to u.
                    # b is now connected to v (was u).
                    # c is now connected to u (was v).
                    # d is still connected to v.
                    
                    adj[u].remove(b)
                    adj[u].add(c)
                    adj[v].remove(c)
                    adj[v].add(b)
                    
                    adj[b].remove(u)
                    adj[b].add(v)
                    adj[c].remove(v)
                    adj[c].add(u)
                    
                    # Update Fitch sets
                    fitch_sets[u] = S1
                    fitch_cost[u] = Cost1
                    fitch_sets[v] = S2
                    fitch_cost[v] = Cost2
                    
                else: # C3
                    # Connect a-d and b-c
                    # u becomes N1 (a, d)
                    # v becomes N2 (b, c)
                    
                    adj[u].remove(b)
                    adj[u].add(d)
                    adj[v].remove(d)
                    adj[v].add(b)
                    
                    adj[b].remove(u)
                    adj[b].add(v)
                    adj[d].remove(v)
                    adj[d].add(u)
                    
                    fitch_sets[u] = S3
                    fitch_cost[u] = Cost3
                    fitch_sets[v] = S4
                    fitch_cost[v] = Cost4
                    
                total_cost = best_cost + (total_cost - C1)
                improved = True
                
    # Generate Newick from adj
    # Root at 0
    parent = {i: -1 for i in adj}
    visited = set()
    stack = [0]
    
    def to_newick(node, par):
        children = [nei for nei in adj[node] if nei != par]
        if not children:
            return leaf_labels[node]
        child_strs = [to_newick(c, node) for c in children]
        return "(" + ",".join(child_strs) + ")"
        
    newick = to_newick(0, -1) + ";"
    return newick

# EVOLVE-BLOCK-END
