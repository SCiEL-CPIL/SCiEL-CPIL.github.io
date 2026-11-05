# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using multiple random restarts with incremental local search.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    # Precompute neighbors list for fast iteration
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    
    best_labels = None
    best_cut = -1
    
    def calc_cut(labels):
        x = np.where(labels == 0, 1, -1)
        val = x @ (adj @ x)
        total_edges = np.sum(adj) // 2
        return total_edges - 0.5 * val

    def local_search(labels):
        """
        Local search with incremental updates.
        """
        labels = labels.copy()
        # Initialize n0, n1
        mask0 = (labels == 0)
        n0 = adj.dot(mask0)
        n1 = adj.dot(1 - mask0) # Or adj.dot(labels == 1)
        
        improved = True
        # Limit passes to avoid infinite loops (shouldn't happen, but safe)
        max_passes = 100
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # Check time
            if time.time() - start_time > DEADLINE:
                return labels

            for v in range(n):
                # Calculate gain for flipping v
                if labels[v] == 0:
                    # Currently in 0. Gain = n0[v] - n1[v]
                    gain = n0[v] - n1[v]
                else:
                    # Currently in 1. Gain = n1[v] - n0[v]
                    gain = n1[v] - n0[v]
                
                if gain > 0:
                    # Flip v
                    labels[v] = 1 - labels[v]
                    improved = True
                    
                    # Update n0, n1 for v and its neighbors
                    if labels[v] == 0:
                        # Flipped from 1 to 0
                        # v's neighbors:
                        # For u in neighbors[v]:
                        # u's neighbor v moved from 1 to 0.
                        # So u's n1 decreases by 1, n0 increases by 1.
                        for u in neighbors[v]:
                            n0[u] += 1
                            n1[u] -= 1
                        
                        # Update v's own counts
                        # v was 1, now 0.
                        # v's neighbors in 0 (old n1) are now in same set? No.
                        # Let's just recompute v's n0, n1 from scratch? 
                        # Or use the swap logic.
                        # Old state: v=1.
                        # n0[v] was count of neighbors in 0.
                        # n1[v] was count of neighbors in 1.
                        # New state: v=0.
                        # New n0[v] (neighbors in 0) = Old n1[v] (neighbors in 1, since v is now 0, those are "opposite"? No.)
                        
                        # Let's trace carefully.
                        # n0[x] = number of neighbors of x with label 0.
                        # If v changes 1 -> 0:
                        # For any neighbor u:
                        #   If u was 0: u's n1 (neighbors in 1) loses v. u's n0 (neighbors in 0) gains v.
                        #   If u was 1: u's n1 (neighbors in 1) loses v? No, v was 1, so v was in u's n1. Now v is 0, so v moves to u's n0.
                        #   So for ALL neighbors u:
                        #     n0[u] += 1
                        #     n1[u] -= 1
                        
                        # For v itself:
                        #   v's neighbors are the same set.
                        #   Old n0[v] = count of neighbors with label 0.
                        #   Old n1[v] = count of neighbors with label 1.
                        #   New n0[v] = count of neighbors with label 0.
                        #   Wait, the labels of neighbors haven't changed. Only v changed.
                        #   So n0[v] (count of neighbors in 0) is UNCHANGED?
                        #   Yes! n0[v] depends on neighbors' labels, not v's label.
                        #   So n0[v] and n1[v] do NOT need to be updated for v itself?
                        #   Let's check definition.
                        #   n0[v] = sum_{u in N(v)} [label[u] == 0].
                        #   This does not depend on label[v].
                        #   So n0[v] and n1[v] remain the same values.
                        #   
                        #   BUT, the GAIN calculation uses n0[v] and n1[v].
                        #   If v is 0, gain = n0[v] - n1[v].
                        #   If v is 1, gain = n1[v] - n0[v].
                        #   So we just need to update the neighbors' n0/n1.
                        
                        pass # No update needed for v's own n0/n1 values
                    else:
                        # Flipped from 0 to 1
                        for u in neighbors[v]:
                            n0[u] -= 1
                            n1[u] += 1
                        # No update needed for v's own n0/n1 values
        
        return labels

    # Multiple random restarts
    rng = np.random.default_rng()
    
    # Start with alternating pattern
    labels = np.arange(n) % 2
    labels = local_search(labels)
    c = calc_cut(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Try random restarts
    while time.time() - start_time < DEADLINE:
        # Random initial labeling
        labels = rng.integers(0, 2, size=n)
        labels = local_search(labels)
        c = calc_cut(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
