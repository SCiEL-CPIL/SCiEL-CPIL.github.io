<<<<<<< SEARCH
    def improve_2swap(labels):
        """
        Local improvement via 2-swap moves.
        A 2-swap move exchanges the labels of two vertices.
        This is useful when single flips are stuck but swapping two vertices helps.
        
        For vertices i and j with labels li and lj:
        If li == lj, swapping doesn't change anything.
        If li != lj, swapping is equivalent to flipping both.
        
        Actually, a more useful 2-move is: flip two vertices on the SAME side.
        Let's implement: try flipping pairs of vertices that are on the same side.
        """
        improved = True
        while improved and (time.time() < deadline - 2.0):
            improved = False
            # For each pair of vertices on the same side, check if flipping both improves
            # To keep it efficient, we only try a subset or use a heuristic
            # For n=100, O(n^2) per iteration is 10000 pairs, which is fine
            
            for i in range(n):
                if time.time() >= deadline - 2.0:
                    break
                li = labels[i]
                for j in range(i + 1, n):
                    lj = labels[j]
                    if li != lj:
                        continue  # Only consider pairs on the same side
                    
                    # Compute gain from flipping both i and j
                    # Before: both have label li
                    # After: both have label 1-li
                    # 
                    # For vertex i:
                    #   Before: same_i = # neighbors with label li, opp_i = # neighbors with label 1-li
                    #   After:  same_i' = # neighbors with label 1-li = opp_i (but j is now on same side)
                    #   Actually, let's compute the change in cut size directly.
                    #
                    # The cut size change from flipping vertex i alone is:
                    #   delta_i = opp_i - same_i
                    # Flipping vertex j alone:
                    #   delta_j = opp_j - same_j
                    # But when flipping both, there's an interaction term for the edge (i,j).
                    #
                    # Let's compute more carefully.
                    # Let S0 = set of vertices with label 0, S1 = set with label 1.
                    # Suppose li = lj = 0. So i, j in S0.
                    # After flip, i, j in S1.
                    #
                    # Edges affected:
                    # 1. Edge (i,j): was internal to S0, now internal to S1. No change in cut.
                    # 2. Edges from i to S0 (excluding j): were internal, now crossing. Gain: +1 each.
                    # 3. Edges from i to S1: were crossing, now internal. Loss: -1 each.
                    # 4. Edges from j to S0 (excluding i): were internal, now crossing. Gain: +1 each.
                    # 5. Edges from j to S1: were crossing, now internal. Loss: -1 each.
                    #
                    # Let:
                    #   a = # neighbors of i in S0 (excluding j)
                    #   b = # neighbors of i in S1
                    #   c = # neighbors of j in S0 (excluding i)
                    #   d = # neighbors of j in S1
                    #
                    # Gain = a + c - b - d
                    #
                    # Note: same_i (before) = a + 1 (if j is neighbor) + ... wait
                    # same_i = # neighbors of i with label 0 = a + (1 if adj[i,j] else 0)
                    # opp_i = # neighbors of i with label 1 = b
                    # Similarly for j.
                    
                    same_i = np.sum(adj[i] & (labels == li))
                    opp_i = np.sum(adj[i] & (labels != li))
                    same_j = np.sum(adj[j] & (labels == li))
                    opp_j = np.sum(adj[j] & (labels != li))
                    
                    # same_i includes j if adj[i,j] is 1
                    # Let's compute a, b, c, d
                    if adj[i, j]:
                        a = same_i - 1  # neighbors of i in S0 excluding j
                        c = same_j - 1  # neighbors of j in S0 excluding i
                    else:
                        a = same_i
                        c = same_j
                    b = opp_i
                    d = opp_j
                    
                    gain = a + c - b - d
                    if gain > 0:
                        labels[i] = 1 - li
                        labels[j] = 1 - li
                        improved = True
                        break  # Restart the outer loop since labels changed
        return labels
=======
    def improve_2swap(labels):
        """
        Efficient 2-swap improvement.
        We limit the number of checks to avoid timeout.
        """
        improved = True
        checks = 0
        max_checks = 5000 # Limit total pair checks across all iterations to be safe
        while improved and (time.time() < deadline - 2.0) and checks < max_checks:
            improved = False
            # Sample a few random pairs or check all?
            # For n=100, all pairs is 4950. This is fine if done once.
            # But doing it in a loop is risky.
            # Let's just do one pass of all pairs if we have time.
            
            for i in range(n):
                if time.time() >= deadline - 2.0 or checks >= max_checks:
                    return labels
                li = labels[i]
                for j in range(i + 1, n):
                    checks += 1
                    if checks >= max_checks:
                        return labels
                    lj = labels[j]
                    if li != lj:
                        continue
                    
                    # Calculate gain for flipping i and j
                    # Gain = (c0_i - c1_i) + (c0_j - c1_j) - 2 * adj[i,j] * (1 if li==0 else -1)? 
                    # Let's use the previous logic but vectorized for the pair?
                    # No, simple calculation is fine for n=100.
                    
                    # Re-calculate counts for i and j quickly
                    # c0_i = sum(adj[i] * (labels==0))
                    # c1_i = sum(adj[i] * (labels==1))
                    
                    c0_i = np.sum(adj[i] * (labels == 0))
                    c1_i = np.sum(adj[i] * (labels == 1))
                    c0_j = np.sum(adj[j] * (labels == 0))
                    c1_j = np.sum(adj[j] * (labels == 1))
                    
                    # Gain for flipping i alone: (1-2*li)*(c0_i - c1_i)
                    # Gain for flipping j alone: (1-2*lj)*(c0_j - c1_j)
                    # Interaction term:
                    # If we flip both, the edge (i,j) status changes?
                    # If li == lj, edge (i,j) was internal, remains internal. No change in cut for this edge.
                    # However, the "counts" c0/c1 used in the single flip gain include each other?
                    # If li == lj == 0:
                    # c0_i includes j if adj[i,j]=1.
                    # c0_j includes i if adj[i,j]=1.
                    # Single flip gain for i: c0_i - c1_i. (Assuming we flip i to 1).
                    # But if we flip j too, the "c0_i" for the state where j is flipped is different.
                    
                    # Let's use the explicit formula derived earlier:
                    # a = c0_i - (1 if adj[i,j] and li==0 else 0)
                    # b = c1_i
                    # c = c0_j - (1 if adj[i,j] and li==0 else 0)
                    # d = c1_j
                    # Gain = a + c - b - d
                    
                    if li == 0:
                        a = c0_i - (1 if adj[i,j] else 0)
                        c = c0_j - (1 if adj[i,j] else 0)
                    else:
                        a = c1_i - (1 if adj[i,j] else 0)
                        c = c1_j - (1 if adj[i,j] else 0)
                        
                    b = c1_i if li == 0 else c0_i
                    d = c1_j if li == 0 else c0_j
                    
                    gain = a + c - b - d
                    if gain > 0:
                        labels[i] = 1 - li
                        labels[j] = 1 - li
                        improved = True
                        return labels # Restart search
        return labels
>>>>>>> REPLACE

