"""Distance-proportional parsimony (DPP) inspired topology reconstruction."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.75:
        return 1.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _implied_branch_lengths(tree_edges, taxa, dist):
    """
    Given a tree topology (as a set of edges) and a taxon to add,
    compute the implied branch lengths for all edges.
    Returns total tree length or None if invalid.
    """
    # This is a simplified version - for DPP stepwise addition,
    # we need to solve for branch lengths given the topology
    
    pass


def _build_tree_dpp(sequences, labels):
    """
    Build tree using DPP stepwise addition.
    Start with 3 taxa, then add remaining taxa one by one,
    placing each on the existing tree to minimize total implied tree length.
    """
    n = len(labels)
    dist = _compute_distance_matrix(sequences, labels)
    
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]});\n"
    
    # Represent tree as a dictionary: node -> (children, is_leaf, label)
    # For unrooted tree, we'll use a different representation
    # Tree as list of edges: (node1, node2, branch_length)
    # Nodes: 0..n-1 are taxa, n.. are internal nodes
    
    # Start with 3 taxa: star tree ((0,1),2)
    # Internal node 3 connects to 0, 1, 2
    # Edges: (3,0), (3,1), (3,2)
    
    # For DPP, we need to compute branch lengths that minimize total tree length
    # given the topology and distance matrix
    
    # Let's use a different representation:
    # Tree as a list of splits (each split is a frozenset of taxa on one side)
    # But this is complex. Let me use a simpler approach.
    
    # I'll represent the tree as a nested tuple structure
    # For n=3: ((0,1),2)
    # When we add taxon k, we try inserting it on each edge
    
    # Tree representation: 
    # - Leaf: (taxon_index,)
    # - Internal: (left_subtree, right_subtree)
    # But for unrooted trees, this is tricky. Let me use a different approach.
    
    # Use a list of taxa in the tree and a list of internal nodes
    # Each internal node has 3 neighbors (for unrooted binary tree)
    
    # Simpler: use Newick string and parse/modify it
    # Or use a flat representation
    
    # Let me use a proper tree structure:
    # nodes: dict node_id -> set of neighbor node_ids
    # leaves: node_ids 0..n-1 correspond to taxa 0..n-1
    # internal nodes: n, n+1, ...
    
    # Initialize with 3 taxa
    next_node_id = n  # next available internal node id
    nodes = {
        0: {1, 2},  # taxon 0 connected to internal nodes
        1: {0, 2},  # taxon 1 connected to internal nodes  
        2: {0, 1},  # taxon 2 connected to internal nodes
    }
    # Wait, for 3 taxa, we need one internal node
    # Let me redo: for 3 taxa, tree is a star with one central node
    # Central node = n (which is 3)
    nodes = {
        0: {3},
        1: {3},
        2: {3},
        3: {0, 1, 2},
    }
    
    def _get_all_edges():
        """Get all edges in the tree."""
        edges = set()
        for u, neighbors in nodes.items():
            for v in neighbors:
                if u < v:
                    edges.add((u, v))
        return edges
    
    def _compute_branch_lengths_for_topology():
        """
        Given the current tree topology, compute branch lengths
        that minimize the sum of squared differences between
        implied distances and observed distances.
        Returns total tree length (sum of branch lengths).
        """
        # For a given topology, we can solve for branch lengths
        # by minimizing sum over all pairs of (d_implied - d_observed)^2
        # This is a linear system that can be solved
        
        # Get all edges
        edges = list(_get_all_edges())
        m = len(edges)
        
        if m == 0:
            return 0.0
        
        # Build the system: for each pair of taxa, the implied distance
        # is the sum of branch lengths along the path between them
        # We want to minimize sum (implied_dist - observed_dist)^2
        
        # This is a least squares problem: A * x = b
        # where A is (n_taxa_pairs x m) matrix, x is branch lengths, b is observed distances
        
        # Get all pairs of taxa
        taxa_indices = list(range(n))  # only the original taxa, not internal nodes
        pairs = []
        for i in range(len(taxa_indices)):
            for j in range(i + 1, len(taxa_indices)):
                pairs.append((taxa_indices[i], taxa_indices[j]))
        
        p = len(pairs)
        
        # Build A matrix: A[k, e] = 1 if edge e is on path between pair k
        A = np.zeros((p, m))
        b = np.zeros(p)
        
        for k, (taxon_i, taxon_j) in enumerate(pairs):
            b[k] = dist[taxon_i, taxon_j]
            # Find path from taxon_i to taxon_j
            # BFS
            from collections import deque
            parent = {taxon_i: None}
            queue = deque([taxon_i])
            while queue:
                node = queue.popleft()
                if node == taxon_j:
                    break
                for neighbor in nodes[node]:
                    if neighbor not in parent:
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            # Reconstruct path
            path = []
            current = taxon_j
            while current != taxon_i:
                path.append(current)
                current = parent[current]
            path.append(taxon_i)
            
            # Edges on path
            for idx in range(len(path) - 1):
                u, v = path[idx], path[idx + 1]
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    e_idx = edges.index(edge)
                    A[k, e_idx] = 1
        
        # Solve least squares: minimize ||A*x - b||^2
        # x = A^T * A * x = A^T * b
        # Use numpy's lstsq
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return float('inf')
        
        # Check if solution is valid (non-negative branch lengths)
        if np.any(x < -1e-6):
            # Negative branch lengths - not ideal but we can still use it
            # For DPP, we might want to handle this differently
            pass
        
        total_length = np.sum(np.abs(x))
        return total_length
    
    def _try_insert_taxon(taxon_idx):
        """
        Try inserting taxon_idx on every edge of the current tree.
        Returns the best edge and the resulting tree.
        """
        edges = list(_get_all_edges())
        best_total_length = float('inf')
        best_edge = None
        best_new_nodes = None
        
        for edge in edges:
            u, v = edge
            
            # Create a copy of the tree
            new_nodes = {key: set(val) for key, val in nodes.items()}
            
            # Remove edge (u, v)
            new_nodes[u].remove(v)
            new_nodes[v].remove(u)
            
            # Add new internal node w
            w = next_node_id
            new_nodes[u].add(w)
            new_nodes[v].add(w)
            new_nodes[w] = {u, v, taxon_idx}
            new_nodes[taxon_idx] = {w}
            
            # Compute total tree length for this topology
            # We need to temporarily update the global nodes
            global_nodes_backup = {key: set(val) for key, val in nodes.items()}
            nodes.clear()
            nodes.update(new_nodes)
            
            total_length = _compute_branch_lengths_for_topology()
            
            # Restore
            nodes.clear()
            nodes.update(global_nodes_backup)
            
            if total_length < best_total_length:
                best_total_length = total_length
                best_edge = edge
                best_new_nodes = new_nodes
        
        return best_edge, best_new_nodes, best_total_length
    
    # Stepwise addition
    for taxon_idx in range(3, n):
        best_edge, best_new_nodes, best_length = _try_insert_taxon(taxon_idx)
        
        if best_new_nodes is not None:
            # Update the tree
            nodes = best_new_nodes
            next_node_id += 1
    
    # Convert tree to Newick string
    def _tree_to_newick():
        """Convert the tree structure to a Newick string."""
        # Find a leaf to start from
        leaf = 0
        
        # Build Newick by traversing the tree
        visited = set()
        
        def _build_subtree(node, parent):
            """Recursively build Newick string for subtree rooted at node, coming from parent."""
            neighbors = [nb for nb in nodes[node] if nb != parent]
            
            if len(neighbors) == 0:
                # Leaf
                return labels[node]
            
            if len(neighbors) == 1:
                # Only one neighbor besides parent - this is a leaf or internal node
                # If it's a taxon, it's a leaf
                if node < n:
                    return labels[node]
                else:
                    # Internal node with only one neighbor (shouldn't happen in binary tree)
                    child_str = _build_subtree(neighbors[0], node)
                    return child_str
            
            if len(neighbors) == 2:
                # Binary node
                child1_str = _build_subtree(neighbors[0], node)
                child2_str = _build_subtree(neighbors[1], node)
                return f"({child1_str},{child2_str})"
            
            # More than 2 neighbors - polytomy
            child_strs = []
            for nb in neighbors:
                child_strs.append(_build_subtree(nb, node))
            return "(" + ",".join(child_strs) + ")"
        
        # For unrooted tree, pick an arbitrary edge as the "root"
        # Start from a leaf, go to its neighbor
        start_leaf = 0
        start_neighbor = list(nodes[start_leaf])[0]
        
        result = _build_subtree(start_neighbor, start_leaf)
        return result + ";\n"
    
    return _tree_to_newick()


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};\n"
    if n == 2:
        return f"({labels[0]},{labels[1]});\n"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]});\n"
    
    # Use DPP stepwise addition
    return _build_tree_dpp(sequences, labels)


# EVOLVE-BLOCK-END

