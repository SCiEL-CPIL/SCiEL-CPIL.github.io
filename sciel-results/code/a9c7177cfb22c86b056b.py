"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using multi-start local search with simulated annealing
    for escape from local optima, bounded by a time deadline.
    
    Strategy:
    1. Convert adjacency to a numpy boolean/integer matrix for fast operations.
    2. Run multiple restarts. Each restart:
       a. Initialize with random or alternating labels.
       b. Perform hill-climbing (flip vertices that increase cut size).
       c. If stuck, use simulated annealing to potentially escape local optima.
    3. Keep the best solution found.
    4. Return the best labels before the deadline.
    
    Cut size for a labeling x is: sum over edges (i,j) where x[i] != x[j].
    Equivalently, for each vertex v, the number of neighbors in the opposite set.
    Total cut = (1/2) * sum_v (number of neighbors of v in opposite set).
    
    We can compute the gain of flipping vertex v as:
    gain = 2 * (num_opposite_neighbors - num_same_neighbors)
    where num_opposite_neighbors + num_same_neighbors = degree(v).
    So gain = 2 * (num_opposite - num_same) = 2 * (2 * num_opposite - degree(v)).
    If gain > 0, flipping v increases the cut.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin

    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.int8)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Precompute degrees
    degrees = A.sum(axis=1)
    
    # Function to compute cut size
    def cut_size(labels):
        # For each vertex, count neighbors in opposite set
        # labels: 0/1 array
        # For vertex v with label l, opposite neighbors are those with label 1-l
        # We can compute: for each v, sum A[v, :] * (labels != labels[v])
        # But faster: cut = sum over v of (A[v, :] @ (1 - labels)) where labels[v] == 1
        # Actually, let's use: cut = (1/2) * sum_v sum_u A[v,u] * (labels[v] != labels[u])
        # = (1/2) * sum_v sum_u A[v,u] * (1 - 2*labels[v]*labels[u] + labels[v] + labels[u]) ... this is complex
        # Simpler: cut = sum over edges where labels differ
        # Using numpy: create a matrix M where M[i,j] = 1 if labels[i] != labels[j]
        # cut = (A * M).sum() / 2  (since each edge counted twice)
        # But creating M is O(n^2) memory. For n=100, it's fine.
        
        # Alternative: for each vertex v, the number of neighbors in opposite set:
        # If labels[v] == 0: opposite neighbors are those with label 1
        # If labels[v] == 1: opposite neighbors are those with label 0
        # So: opposite_count[v] = A[v, :] @ (1 - labels) if labels[v] == 1, else A[v, :] @ labels
        # Wait: if labels[v] == 0, we want neighbors with label 1. So A[v,:] @ labels gives that.
        # If labels[v] == 1, we want neighbors with label 0. So A[v,:] @ (1-labels) gives that.
        
        # Let's compute it vectorized:
        # For all v, opposite_count[v] = A[v,:] @ (1 - labels) if labels[v]==1 else A[v,:] @ labels
        # We can do: opposite_count = np.where(labels == 1, A @ (1 - labels), A @ labels)
        # But A @ labels gives for each v: sum_u A[v,u] * labels[u]
        # A @ (1-labels) gives for each v: sum_u A[v,u] * (1 - labels[u])
        
        labels_int = labels.astype(np.int8)
        A_labels = A @ labels_int  # for each v, sum of labels of neighbors
        A_one_minus_labels = A @ (1 - labels_int)  # for each v, sum of (1-labels) of neighbors
        
        # For v with label 0: opposite neighbors have label 1, so count = A_labels[v]
        # For v with label 1: opposite neighbors have label 0, so count = A_one_minus_labels[v]
        opposite_counts = np.where(labels_int == 0, A_labels, A_one_minus_labels)
        
        # Total cut = sum of opposite_counts (each edge counted once? No, each edge (u,v) is counted 
        # in opposite_counts[u] if labels differ, and in opposite_counts[v] if labels differ.
        # So sum(opposite_counts) = 2 * cut_size
        return int(opposite_counts.sum() // 2)
    
    # Function to compute gain of flipping vertex v
    def flip_gain(v, labels):
        """
        Return the change in cut size if we flip vertex v.
        gain = 2 * (num_opposite - num_same)
        num_same = degrees[v] - num_opposite
        gain = 2 * (2 * num_opposite - degrees[v])
        """
        labels_int = labels.astype(np.int8)
        # num_opposite for v:
        if labels_int[v] == 0:
            num_opposite = A[v, :] @ labels_int
        else:
            num_opposite = A[v, :] @ (1 - labels_int)
        
        num_same = degrees[v] - num_opposite
        return 2 * (num_opposite - num_same)
    
    # Hill climbing from a given labeling
    def hill_climb(labels):
        """
        Perform hill climbing. Returns improved labels and final cut size.
        """
        labels = labels.copy()
        labels_int = labels.astype(np.int8)
        
        # Precompute current opposite counts for all vertices
        A_labels = A @ labels_int
        A_one_minus_labels = A @ (1 - labels_int)
        opposite_counts = np.where(labels_int == 0, A_labels, A_one_minus_labels)
        
        improved = True
        while improved:
            improved = False
            # Find the vertex with the maximum positive gain
            # gain[v] = 2 * (2 * opposite_counts[v] - degrees[v])
            gains = 2 * (2 * opposite_counts - degrees)
            
            # Find max gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain <= 0:
                break
            
            # Flip vertex max_gain_idx
            v = max_gain_idx
            old_label = labels_int[v]
            new_label = 1 - old_label
            
            # Update labels
            labels_int[v] = new_label
            labels[v] = new_label
            
            # Update opposite_counts for all vertices
            # When we flip v, for each neighbor u of v:
            #   if u and v were in same set, now they're in opposite sets
            #   if u and v were in opposite sets, now they're in same set
            # So for each neighbor u of v:
            #   if old_label == labels_int[u]: u's opposite_count decreases by 1
            #   if old_label != labels_int[u]: u's opposite_count increases by 1
            # For v itself:
            #   v's opposite_count becomes degrees[v] - old_opposite_count
            
            neighbors = np.where(A[v, :] == 1)[0]
            
            for u in neighbors:
                if labels_int[u] == old_label:
                    # u and v were same, now opposite: u gains one opposite neighbor
                    opposite_counts[u] += 1
                else:
                    # u and v were opposite, now same: u loses one opposite neighbor
                    opposite_counts[u] -= 1
            
            # Update v's opposite count
            # v's new opposite count = number of neighbors with label != new_label
            # = number of neighbors with label == old_label
            # = old opposite count of v? No.
            # Before flip: v had label old_label. opposite_counts[v] = number of neighbors with label != old_label.
            # After flip: v has label new_label = 1 - old_label. 
            # New opposite count for v = number of neighbors with label != new_label = number of neighbors with label == old_label.
            # = degrees[v] - (number of neighbors with label != old_label) = degrees[v] - old_opposite_counts[v]
            # But we already updated neighbors. Let's just recompute for v.
            # Actually, the neighbors' opposite_counts have been updated. For v itself:
            # The number of neighbors with label == old_label (which is now the "same" label for v's new perspective... wait)
            # Let me think again. 
            # Before: v has label L. Neighbors with label L: same. Neighbors with label 1-L: opposite.
            # opposite_counts[v] = #neighbors with label 1-L.
            # After: v has label 1-L. 
            # New opposite for v: neighbors with label L.
            # #neighbors with label L = degrees[v] - #neighbors with label 1-L = degrees[v] - old_opposite_counts[v].
            # So new_opposite_counts[v] = degrees[v] - old_opposite_counts[v].
            
            old_v_opposite = opposite_counts[v]
            opposite_counts[v] = degrees[v] - old_v_opposite
            
            improved = True
        
        # Compute final cut size
        final_cut = int(opposite_counts.sum() // 2)
        return labels, final_cut
    
    # Simulated annealing step to escape local optima
    def simulated_annealing(labels, current_cut, max_steps=500):
        """
        Perform simulated annealing to try to escape local optima.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        labels_int = labels.astype(np.int8)
        
        # Precompute current opposite counts
        A_labels = A @ labels_int
        A_one_minus_labels = A @ (1 - labels_int)
        opposite_counts = np.where(labels_int == 0, A_labels, A_one_minus_labels)
        
        T = 5.0  # initial temperature
        cooling = 0.99
        
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            
            old_label = labels_int[v]
            new_label = 1 - old_label
            
            # Compute gain of flipping v
            if old_label == 0:
                num_opposite = A[v, :] @ labels_int
            else:
                num_opposite = A[v, :] @ (1 - labels_int)
            
            num_same = degrees[v] - num_opposite
            gain = 2 * (num_opposite - num_same)
            
            # Accept if gain > 0 or with probability exp(gain / T) if gain < 0
            if gain > 0:
                accept = True
            else:
                # gain is negative or zero
                if gain == 0:
                    accept = np.random.random() < 0.5
                else:
                    # Avoid overflow/underflow
                    if gain / T < -700:
                        accept = False
                    else:
                        accept = np.random.random() < np.exp(gain / T)
            
            if accept:
                # Flip v
                labels_int[v] = new_label
                labels[v] = new_label
                
                # Update opposite_counts
                neighbors = np.where(A[v, :] == 1)[0]
                for u in neighbors:
                    if labels_int[u] == old_label:
                        opposite_counts[u] += 1
                    else:
                        opposite_counts[u] -= 1
                
                old_v_opposite = opposite_counts[v]
                opposite_counts[v] = degrees[v] - old_v_opposite
                
                current_cut += gain // 2  # gain is change in 2*cut, so cut changes by gain/2
        
        final_cut = int(opposite_counts.sum() // 2)
        return labels, final_cut
    
    # Multi-start search
    best_labels = None
    best_cut = -1
    
    # First, try alternating start
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = hill_climb(labels_alt)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()
    
    # Try random restarts
    num_restarts = 0
    while time.time() - start_time < DEADLINE:
        # Random initialization
        labels_rand = np.random.randint(0, 2, n)
        
        # Hill climb
        labels_improved, cut_improved = hill_climb(labels_rand)
        
        if cut_improved > best_cut:
            best_cut = cut_improved
            best_labels = labels_improved.copy()
        
        # Try simulated annealing from the best solution so far
        if time.time() - start_time < DEADLINE - 5:  # leave some time
            labels_sa, cut_sa = simulated_annealing(best_labels, best_cut, max_steps=300)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
        
        num_restarts += 1
    
    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
