import numpy as np
import math
import time

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 165.0  # Safety margin for 20 instances

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)

    if n_taxa <= 2:
        if n_taxa == 1:
            return f"({labels[0]});"
        else:
            return f"({labels[0]},{labels[1]});"

    # Map sequences to integer arrays for faster comparison
    # A:0, C:1, G:2, T:3
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    # Vectorized conversion
    seq_array = np.empty((n_taxa, L), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        # Using list comprehension + numpy array is faster than loop for string to int array
        seq_array[i] = [seq_map[c] for c in s]

    # Compute pairwise Hamming distances
    # diff[i, j, k] = seq_array[i, k] != seq_array[j, k]
    # This is n^2 * L memory. For n=32, L=2000, it's 2M bytes. Safe.
    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    hamming = diff.sum(axis=2)

    # Apply Jukes-Cantor correction
    p = hamming / L
    
    d_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    
    # Handle p=0 and p=1
    # JC(d) = -0.75 * ln(1 - 4/3 * p)
    valid_mask = (p > 0) & (p < 0.75)
    
    if np.any(valid_mask):
        d_valid = -0.75 * np.log(1 - (4.0 / 3.0) * p[valid_mask])
        d_matrix[valid_mask] = d_valid
        large_val = np.max(d_valid) * 1.5
    else:
        large_val = 1.0
        
    invalid_mask = p >= 0.75
    d_matrix[invalid_mask] = large_val
    
    # Ensure diagonal is 0
    np.fill_diagonal(d_matrix, 0.0)

    # Neighbor Joining
    active = list(range(n_taxa))
    trees = {i: labels[i] for i in range(n_taxa)}
    
    D = d_matrix.copy()
    next_id = n_taxa
    
    while len(active) > 3:
        if time.time() > deadline:
            # Fallback: return a star tree or the current partial tree?
            # Better to return a valid tree.
            # If we break here, we have a partial tree.
            # We should finish the NJ quickly or return a star tree.
            # For safety, return a star tree connecting all remaining active nodes.
            # But we need to map back to original labels.
            # This is complex. Let's just ensure the loop finishes.
            # With N=32, the loop runs 29 times. It should be very fast.
            # If we hit the deadline, we likely have a bug or extremely slow hardware.
            # Let's just break and try to assemble what we have.
            break
            
        n = len(active)
        
        # Row sums
        r = D.sum(axis=1)
        
        # Q matrix
        # Q[i, j] = (n - 2) * D[i, j] - r[i] - r[j]
        Q = (n - 2) * D - r[:, None] - r[None, :]
        
        # Find min Q
        # We want the pair (i, j) with min Q[i, j] where i != j
        # Mask diagonal
        Q_masked = Q.copy()
        np.fill_diagonal(Q_masked, np.inf)
        
        min_val = np.min(Q_masked)
        min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
        i, j = min_idx
        
        # Get the original node ids
        node_i = active[i]
        node_j = active[j]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Update trees
        trees[new_id] = f"({trees[node_i]},{trees[node_j]})"
        
        # Remove i and j from active, add new_id
        # We need to update the distance matrix D.
        # The new row/col for new_id should be calculated for all other active nodes.
        
        # Indices of remaining nodes
        remaining_indices = [k for k in range(n) if k not in (i, j)]
        
        # Calculate new distances
        # D[new, k] = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        
        # We will construct a new D matrix of size (n-1, n-1)
        # The new active list will be: remaining nodes + new_id
        
        # Let's build the new D
        new_n = n - 1
        new_D = np.zeros((new_n, new_n), dtype=np.float64)
        
        # Map old indices to new indices
        # The new active list order: remaining_indices (in original order) + [new_id]
        # But we need to map the distances correctly.
        
        # Let's create a list of new active nodes
        new_active = [active[k] for k in remaining_indices] + [new_id]
        
        # Fill new_D for the remaining-remaining part
        # For each pair in remaining_indices, copy from D
        for a_idx, a in enumerate(remaining_indices):
            for b_idx, b in enumerate(remaining_indices):
                new_D[a_idx, b_idx] = D[a, b]
        
        # Fill new_D for the new node (last row/col)
        # The new node is at index new_n - 1
        for a_idx, a in enumerate(remaining_indices):
            dist_new_a = 0.5 * (D[i, a] + D[j, a] - D[i, j])
            new_D[a_idx, new_n - 1] = dist_new_a
            new_D[new_n - 1, a_idx] = dist_new_a
            
        new_D[new_n - 1, new_n - 1] = 0.0
        
        # Update state
        active = new_active
        D = new_D

    # Final step: 3 nodes remain
    # They form a star tree (unrooted)
    # Connect them in a triangle? No, unrooted tree with 3 leaves is a star.
    # The topology is ((a,b),c) or similar.
    # In Newick, we just join them.
    # The order doesn't matter for topology score if it's a star.
    # But we need to return a valid Newick string.
    
    # The remaining 3 nodes are in `active`
    # We just need to combine their trees
    # e.g. (tree1, tree2, tree3) is not valid Newick.
    # It must be (tree1, (tree2, tree3)) or similar.
    
    # Let's just pick the first two, join them, then join with the third.
    # (a, b, c) -> ((a, b), c)
    
    # However, the branch lengths are not strictly required, but the topology is.
    # For 3 taxa, the only unrooted topology is the star.
    
    # Construct the final Newick
    # We have trees for each node in active.
    
    # Pick 0, 1, 2
    if len(active) == 3:
        t0 = trees[active[0]]
        t1 = trees[active[1]]
        t2 = trees[active[2]]
        
        # Join 0 and 1
        join_01 = f"({t0},{t1})"
        
        # Join with 2
        final_tree = f"({join_01},{t2})"
        
        return final_tree + ";"
    else:
        # Fallback if we broke out of the loop early
        # This shouldn't happen for n<=32 within 165s
        # Just return a star tree of all remaining active nodes?
        # Or just the first 3?
        # Let's try to finish it quickly with a simple heuristic if we ran out of time
        # But the loop is O(N^2) per step, so 29 steps * 32^2 is ~30k ops. Very fast.
        # If we get here, it's likely an error.
        # Return a valid tree for the first 3, and ignore others? No, must include all.
        # Let's just build a star tree from the remaining active nodes.
        # But they might be internal nodes with subtrees.
        # We need to join them all.
        
        # Simple fallback: join them all in a chain
        if not active:
             return "();"
        
        current = trees[active[0]]
        for node in active[1:]:
            current = f"({current},{trees[node]})"
        return current + ";"
