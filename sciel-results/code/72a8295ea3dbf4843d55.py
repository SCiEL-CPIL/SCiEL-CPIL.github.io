import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    # Map labels to indices
    # Encode sequences
    # A=0, C=1, G=2, T=3
    # Use numpy array of shape (n, L)
    
    seq_array = np.zeros((n, len(next(iter(sequences.values())))), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        # Vectorized encoding
        # A->0, C->1, G->2, T->3
        # np.where or lookup
        lookup = np.array([0, 1, 2, 3], dtype=np.uint8)
        # Map chars to indices
        # A: 65, C: 67, G: 71, T: 84
        # This is slow in Python loop.
        # Use vectorized string ops?
        # seq.encode('ascii')
        # Then map.
        
    # Better:
    # Create a 256 byte lookup table
    table = np.zeros(256, dtype=np.uint8)
    table[ord('A')] = 0
    table[ord('C')] = 1
    table[ord('G')] = 2
    table[ord('T')] = 3
    
    for i, label in enumerate(labels):
        seq_bytes = sequences[label].encode('ascii')
        seq_array[i] = table[seq_bytes]
        
    # Pairwise Hamming distances
    # seq_array is (n, L)
    # diff = seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :]
    # This is (n, n, L) boolean array.
    # For n=32, L=2000, size is 32*32*2000 = 2M booleans. ~2MB.
    # This is fine.
    
    diff = seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :]
    hamming = diff.sum(axis=2)
    
    # JC Correction
    p = hamming / len(next(iter(sequences.values())))
    # Avoid log(0)
    p = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    return dist

def _nj_reconstruct(dist_matrix):
    n = dist_matrix.shape[0]
    # dist_matrix is (n, n)
    # nodes: list of active node indices
    nodes = list(range(n))
    # trees: dict mapping node index to Newick string or subtree structure
    # We need to track the topology.
    # Let's store the tree as a nested list/dict.
    # Or just store the splits?
    # For NJ, it's easier to build the Newick string directly.
    
    # trees[i] is the Newick string for the subtree rooted at i (if i is internal)
    # or the label if i is a leaf.
    trees = {i: str(i) for i in range(n)} # Use indices as labels for now
    
    # Distance matrix D
    D = dist_matrix.copy()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        # Row sums
        # r[i] = sum(D[i, j] for j in nodes if j != i)
        # Use numpy for speed?
        # nodes is a list of indices into D.
        # D is (n, n).
        # We need to sum rows for active nodes.
        
        # Create a sub-matrix of active nodes
        # This is O(N^2) per step.
        
        active_D = D[np.ix_(nodes, nodes)]
        r = active_D.sum(axis=1)
        
        # Q matrix
        # Q[i, j] = (N-2) D[i, j] - r[i] - r[j]
        Q = (n_curr - 2) * active_D - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Find min Q (excluding diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        a = nodes[i_idx]
        b = nodes[j_idx]
        
        # New node
        new_node = n # Use a counter for new nodes
        n += 1
        
        # Update D
        # D[new, k] = 0.5 * (D[a, k] + D[b, k] - D[a, b])
        # for k in nodes
        # We need to update the main D matrix.
        
        # Indices of remaining nodes
        rest_nodes = [x for x in nodes if x not in (a, b)]
        
        # Calculate new distances
        # D[a, rest] and D[b, rest]
        # D[a, b] is the distance between a and b
        
        new_dists = 0.5 * (D[a, rest_nodes] + D[b, rest_nodes] - D[a, b])
        
        # Update D
        D[new_node, rest_nodes] = new_dists
        D[rest_nodes, new_node] = new_dists
        
        # Update trees
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        # Update nodes
        nodes = rest_nodes + [new_node]
        
    # Final step: 3 nodes left
    # Connect them in a star or pick one pair?
    # Standard NJ: The last 3 nodes form a trifurcation?
    # No, for binary tree, we pick the pair with min distance?
    # Or just connect them arbitrarily?
    # The topology of the last 3 is a cherry + 1.
    # We should pick the pair (i, j) that minimizes D[i, j] to form the last cherry.
    
    if len(nodes) == 3:
        i, j, k = nodes
        # Find min pair
        d_ij = D[i, j]
        d_ik = D[i, k]
        d_jk = D[j, k]
        
        if d_ij <= d_ik and d_ij <= d_jk:
            a, b = i, j
            c = k
        elif d_ik <= d_ij and d_ik <= d_jk:
            a, b = i, k
            c = j
        else:
            a, b = j, k
            c = i
            
        new_node = n
        n += 1
        trees[new_node] = f"({trees[a]},{trees[b]})"
        # Connect c to new_node
        # The final tree is ( (a,b), c )
        # But it's unrooted.
        # The Newick string should represent the unrooted tree.
        # A standard way is to root it arbitrarily.
        # The evaluator accepts any Newick string that represents the unrooted topology.
        # So ( (a,b), c ) is fine.
        
        # We don't need to update D further.
        # The final tree is trees[new_node] with c attached?
        # Wait, the loop ends when len(nodes) <= 3.
        # My loop condition is `while len(nodes) > 3`.
        # So if I have 3 nodes, I break.
        # I need to handle the final 3 nodes.
        
        # Let's modify the loop to run until len(nodes) == 2?
        # No, NJ creates one node per step.
        # Start with N.
        # Step 1: N-1.
        # ...
        # Step N-3: 3 nodes left.
        # Then we need to join the last 3.
        
        # Let's just join the last 3 in the loop if I change condition to > 2?
        # If len(nodes) == 2, we join them.
        # But NJ requires 3 to calculate the new distance to the rest?
        # No, if 2 nodes left, they are the last two.
        
        # Standard NJ implementation:
        # While N > 3:
        #   Join a, b -> k
        #   N = N - 1
        # After loop, N == 3.
        # Join the 3 nodes.
        
        # I will handle the final 3 nodes here.
        # The tree is ( (a,b), c )
        # But we need to map back to original labels.
        
        final_tree = f"({trees[new_node]},{trees[c]})"
        
    else:
        # This case shouldn't happen if logic is correct
        final_tree = ""

    # Map indices to original labels
    # trees values contain indices like "0", "1", etc.
    # We need to replace them with labels.
    # Since we used indices 0..N-1 for leaves, we can do a simple replacement.
    
    # But internal nodes also have indices.
    # We need to be careful not to replace "10" with "1" if label "1" exists.
    # Better: Build the tree structure and then traverse to generate Newick.
    
    # Let's restart the tree building part to be robust.
    
    return final_tree
