import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def compute_fiedler_partition(adj_matrix, rng):
    """
    Compute Fiedler vector of the graph and use it to partition into two sets.
    Returns a boolean mask for one side of the partition.
    """
    N = adj_matrix.shape[0]
    # Build Laplacian: L = D - A
    degrees = adj_matrix.sum(axis=1)
    L = np.diag(degrees) - adj_matrix.astype(float)
    
    # Find eigenvalues and eigenvectors
    # Use sparse eigenvalue solver for efficiency
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigs
        L_sparse = csr_matrix(L)
        # Find smallest non-trivial eigenvalue (Fiedler)
        # eigs finds largest by default, so use shift-invert
        vals, vecs = eigs(L_sparse, k=2, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(vals.real)
        fiedler = vecs[:, idx[1]].real  # Second smallest eigenvalue's eigenvector
    except Exception:
        # Fallback: use numpy for small graphs
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
    
    # Partition: nodes with positive fiedler value go to set A, negative to set B
    # Add slight random perturbation to break ties
    fiedler += rng.normal(0, 0.01, size=N)
    partition = fiedler > 0
    
    return partition

def kempe_chain_improve(selected_mask, adj_matrix, deadline, rng):
    """
    Improve the independent set using Kempe-chain flips.
    Given a 2-coloring (selected/not-selected), try to flip connected components
    in the subgraph to increase the size of the selected set.
    """
    N = adj_matrix.shape[0]
    current_selected = selected_mask.copy()
    improved = True
    max_iterations = 500
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Get the current coloring: 1 = selected, 0 = not selected
        # Find "bad" edges: edges where both endpoints are selected (shouldn't happen for valid IS)
        # or edges where both are not selected (potential for improvement)
        
        # Actually, for Kempe chains in independent set context:
        # We have a partition (S, V\S) where S is an independent set.
        # A Kempe chain is a connected component in the subgraph induced by 
        # edges between S and V\S. Flipping the chain means swapping membership.
        
        # Build the bipartite subgraph: edges between selected and unselected
        # For each unselected node, check if it has any selected neighbors
        # If an unselected node has NO selected neighbors, we can add it.
        
        # Simpler approach: find unselected nodes with no selected neighbors and add them
        # Then find selected nodes that now conflict and remove them
        # This is essentially a 1-swap improvement
        
        # Find unselected nodes that can be added (no conflicts with selected)
        unselected = np.where(~current_selected)[0]
        if len(unselected) == 0:
            break
        
        # For each unselected node, count conflicts with selected
        # conflicts_with_selected[i] = number of selected neighbors of i
        selected_indices = np.where(current_selected)[0]
        if len(selected_indices) == 0:
            break
        
        # Check which unselected nodes have no selected neighbors
        # adj_matrix[unselected][:, selected_indices] gives the adjacency
        sub_adj = adj_matrix[np.ix_(unselected, selected_indices)]
        has_conflict = sub_adj.any(axis=1)
        
        # Nodes that can be added
        addable = unselected[~has_conflict]
        
        if len(addable) > 0:
            # Pick one at random
            node = addable[rng.integers(len(addable))]
            current_selected[node] = True
            improved = True
            continue
        
        # Try Kempe chain: find a connected component in the "boundary" graph
        # that we can flip to gain size
        # Build graph where edges exist between selected and unselected nodes
        # A Kempe chain flip: take a connected component of unselected nodes
        # that are all adjacent to selected nodes, and flip them all to selected,
        # but only if the net gain is positive
        
        # Find unselected nodes adjacent to selected
        boundary_unselected = unselected[has_conflict]
        if len(boundary_unselected) == 0:
            break
        
        # For each connected component of boundary_unselected in the full graph,
        # check if flipping it (making them selected and removing their selected neighbors)
        # increases the set size
        visited = np.zeros(N, dtype=bool)
        
        for start_node in boundary_unselected:
            if time.time() > deadline:
                break
            if visited[start_node]:
                continue
            
            # BFS to find connected component in unselected subgraph
            component = []
            queue = [start_node]
            visited[start_node] = True
            
            while queue:
                u = queue.pop()
                component.append(u)
                # Find unselected neighbors of u
                neighbors = np.where(adj_matrix[u])[0]
                for nb in neighbors:
                    if not visited[nb] and not current_selected[nb]:
                        visited[nb] = True
                        queue.append(nb)
            
            if len(component) == 0:
                continue
            
            # Find all selected neighbors of this component
            comp_array = np.array(component)
            comp_neighbors = set()
            for c in component:
                nbs = np.where(adj_matrix[c])[0]
                for nb in nbs:
                    if current_selected[nb]:
                        comp_neighbors.add(nb)
            
            comp_neighbors = list(comp_neighbors)
            
            # Net change: add len(component), remove len(comp_neighbors)
            net_change = len(component) - len(comp_neighbors)
            
            if net_change > 0:
                # Perform the flip
                for c in component:
                    current_selected[c] = True
                for nb in comp_neighbors:
                    current_selected[nb] = False
                improved = True
                break  # Restart the outer loop
    
    return current_selected

def greedy_mis_fast(adj_matrix, deadline, rng):
    """
    Fast greedy MIS with random ordering.
    """
    N = adj_matrix.shape[0]
    order = rng.permutation(N)
    selected = np.zeros(N, dtype=bool)
    
    for i in order:
        if time.time() > deadline:
            break
        if selected[i]:
            continue
        # Check if i conflicts with any already selected
        if selected[adj_matrix[i]].any():
            continue
        selected[i] = True
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Generate weight-4 vectors
    w4_vectors = generate_weight_4_vectors()
    N = len(w4_vectors)
    
    # Generate weight-1 vectors (norm^2 = 4, compatible with weight-4)
    w1_vectors = generate_weight_1_vectors()
    
    if N == 0:
        return w1_vectors
    
    # Compute dot products for weight-4 vectors
    dots = w4_vectors @ w4_vectors.T
    # Conflict if dot product > 2 (need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    adj_matrix = (dots > 2).astype(np.int8)
    np.fill_diagonal(adj_matrix, 0)
    
    rng = np.random.default_rng(42)
    best_selected = np.zeros(N, dtype=bool)
    best_size = 0
    
    # Phase 1: Spectral partition + Kempe improvement
    # Try multiple spectral partitions with different random seeds
    spectral_deadline = deadline - 60  # Leave 60s for greedy fallbacks
    
    for attempt in range(20):
        if time.time() > spectral_deadline:
            break
        
        # Compute Fiedler partition
        partition = compute_fiedler_partition(adj_matrix, rng)
        
        # The partition gives two sets. Pick the larger one as initial IS.
        # But we need to verify it's actually an independent set.
        # The Fiedler partition may not be a perfect independent set, so we need to clean it up.
        
        # Start with the larger side
        if partition.sum() >= (N - partition.sum()):
            current = partition.copy()
        else:
            current = ~partition.copy()
        
        # Clean up: remove any node that conflicts with another selected node
        # Iterate until stable
        changed = True
        while changed:
            if time.time() > spectral_deadline:
                break
            changed = False
            selected_indices = np.where(current)[0]
            for i in selected_indices:
                if time.time() > spectral_deadline:
                    break
                # Check if i has any selected neighbor
                if current[adj_matrix[i]].any():
                    # Remove the one with higher degree to break conflict
                    conflict_nbs = np.where(current[adj_matrix[i]])[0]
                    if len(conflict_nbs) > 0:
                        # Remove i (simpler)
                        current[i] = False
                        changed = True
                        break
        
        # Now current should be an independent set. Apply Kempe improvement.
        improved = kempe_chain_improve(current, adj_matrix, min(spectral_deadline, time.time() + 5), rng)
        
        size = improved.sum()
        if size > best_size:
            best_size = size
            best_selected = improved.copy()
    
    # Phase 2: Random greedy restarts to try to beat the spectral result
    greedy_deadline = deadline - 5
    
    while time.time() < greedy_deadline:
        selected = greedy_mis_fast(adj_matrix, greedy_deadline, rng)
        size = selected.sum()
        if size > best_size:
            best_size = size
            best_selected = selected.copy()
        
        # Also try Kempe improvement on the greedy result
        if time.time() < greedy_deadline - 2:
            improved = kempe_chain_improve(selected, adj_matrix, greedy_deadline - 1, rng)
            size = improved.sum()
            if size > best_size:
                best_size = size
                best_selected = improved.copy()
    
    if best_size == 0:
        return w1_vectors
    
    result_w4 = w4_vectors[best_selected].astype(np.int64)
    
    # Combine with weight-1 vectors
    final_vectors = np.vstack([result_w4, w1_vectors])
    
    return final_vectors
