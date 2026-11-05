"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solves Maximum Cut using Simulated Annealing.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Ensure adjacency is a numpy array of bools for fast operations
    adj = np.asarray(adjacency, dtype=bool)
    
    # Precompute degrees to quickly calculate cut size changes
    # cut_size = sum(adj[v, labels != labels[v]])
    # If we flip v:
    # new_cut = old_cut - (edges from v to same side) + (edges from v to other side)
    # edges_same = sum(adj[v, labels == labels[v]])
    # edges_other = sum(adj[v, labels != labels[v]])
    # delta = edges_other - edges_same
    
    # Initialize with random labels
    labels = np.random.randint(0, 2, n, dtype=np.int8)
    
    # Calculate initial cut size
    # Cut edges are those where labels differ.
    # Since adj is symmetric, we can just sum adj[labels != labels.T] but that's O(n^2) memory.
    # Better: sum over all v of edges to other side.
    # Or simply: cut = 0.5 * sum( (1 - labels[i]*labels[j]) * adj[i][j] ) for i!=j?
    # Let's use the property: cut = sum(adj * (labels[:, None] != labels[None, :])) / 2? No, adj is symmetric.
    # Actually, just summing adj[labels[:, None] != labels[None, :]] counts each edge twice if we are not careful?
    # If adj[i][j]=1 and labels[i]!=labels[j], it contributes 1 to adj[i][j] and 1 to adj[j][i].
    # So total sum is 2 * cut_size.
    
    # Efficient initial cut calculation:
    # mask = labels[:, None] != labels[None, :]
    # cut = np.sum(adj & mask) // 2
    
    # Even faster for init:
    # cut = 0
    # for v in range(n):
    #     cut += np.sum(adj[v, labels != labels[v]])
    # This is O(n^2) but with numpy sums it's fast.
    
    # Let's use the matrix multiplication trick or boolean mask.
    # For n=100, boolean mask is 100x100 = 10k elements. Very fast.
    mask = labels[:, None] != labels[None, :]
    current_cut = np.sum(adj & mask) // 2
    
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Simulated Annealing Parameters
    # Time limit: 150s wall clock. We should leave some buffer.
    # Let's aim for 140s to be safe, or check time frequently.
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer
    
    # Temperature schedule
    # Start high enough to accept bad moves, cool down.
    # Typical SA for MaxCut: T0 ~ E / ln(N) or similar.
    # Let's estimate max possible cut. Max cut <= m.
    # Let's just use a heuristic T0.
    # A common approach: T0 = 100.0, alpha = 0.999?
    # Or geometric cooling: T = T0 * alpha^k
    # We need to decide how many steps.
    # Each step is O(n) if we do it smartly, or O(1) if we update incrementally?
    # Incremental update is better.
    
    # Let's implement incremental update.
    # When flipping vertex v:
    # delta = (edges from v to other side) - (edges from v to same side)
    # We can precompute these sums for each vertex?
    # same_counts[v] = sum(adj[v, labels == labels[v]])
    # other_counts[v] = sum(adj[v, labels != labels[v])
    # If we flip v:
    #   new_same_counts[v] = other_counts[v]
    #   new_other_counts[v] = same_counts[v]
    #   For each neighbor u of v:
    #       If u was same as v, now u is other -> same_counts[u]--, other_counts[u]++
    #       If u was other as v, now u is same -> same_counts[u]++, other_counts[u]--
    
    # This incremental update is O(degree(v)) per move.
    # Average degree in random graph ~ n/2 = 50.
    # So each step is ~50 ops.
    # 145 seconds / 50 ops per step = ~1.4M steps? Maybe more with numpy overhead.
    # Let's try to be efficient.
    
    # Precompute neighbor lists?
    # adj is bool matrix.
    # neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    # Initialize counts
    # same_counts[i] = number of neighbors of i with same label
    # other_counts[i] = number of neighbors of i with different label
    
    # Efficient init of counts:
    # same_counts = np.array([np.sum(adj[i, labels == labels[i]]) for i in range(n)])
    # other_counts = np.array([np.sum(adj[i, labels != labels[i]]) for i in range(n)])
    
    # Vectorized init:
    # mask = labels[:, None] == labels[None, :]
    # same_counts = np.sum(adj & mask, axis=1)
    # other_counts = np.sum(adj & ~mask, axis=1)
    
    mask_same = labels[:, None] == labels[None, :]
    same_counts = np.sum(adj & mask_same, axis=1)
    other_counts = np.sum(adj & ~mask_same, axis=1)
    
    # Neighbors list for fast iteration
    # Using lists of indices
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    # SA Loop
    T = 10.0  # Initial temperature
    # Cooling rate: we want to reach near 0 by deadline.
    # Let's estimate steps per second.
    # Just run until deadline.
    
    # To make cooling adaptive, we can just use a fixed number of iterations per temperature step?
    # Or simply decrease T slowly.
    # Let's use a simple geometric cooling with a small alpha to keep T high for longer,
    # or linear cooling?
    # Given the time limit, let's just use a fixed small step count per T update?
    # No, let's just update T every, say, 1000 steps?
    # Or better: T = T0 * exp(-lambda * t)?
    
    # Let's try: T starts at 5.0.
    # Each move has delta E in [-n, n].
    # If delta < 0, always accept.
    # If delta > 0, accept with prob exp(-delta / T).
    
    # We need to pick a good T0.
    # Max delta is n. So T0 should be comparable to n/10 or so?
    # Let's start T = n / 10.
    T = n / 10.0
    alpha = 0.9999  # Very slow cooling?
    # If we do 100k steps, T becomes 5 * 0.9999^100k = 5 * e^-10 = 0.002. Good.
    # How many steps can we do?
    # Let's measure.
    
    # Optimization:
    # Instead of random vertex every time, we can pick random vertex.
    # Update counts for that vertex and its neighbors.
    
    # Let's add a "shake" if no improvement for a while?
    # Or just rely on SA.
    
    # Time guard inside loop
    # To avoid checking time.time() every step (expensive), check every 1000 steps.
    
    step = 0
    while time.time() < deadline:
        # Pick random vertex
        v = random.randint(0, n - 1)
        
        # Calculate delta
        # delta = other_counts[v] - same_counts[v]
        delta = other_counts[v] - same_counts[v]
        
        # Decide to flip
        if delta < 0:
            flip = True
        else:
            # Accept with probability exp(-delta / T)
            # If T is very small, this is 0.
            if T > 1e-9:
                # Use log to avoid underflow?
                # random.random() < exp(-delta/T)
                # delta/T might be large.
                try:
                    prob = np.exp(-delta / T)
                except OverflowError:
                    prob = 0.0
                flip = random.random() < prob
            else:
                flip = False
        
        if flip:
            # Flip v
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            # Update current_cut
            current_cut += delta
            
            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            
            # Update counts
            # v's counts swap
            same_counts[v], other_counts[v] = other_counts[v], same_counts[v]
            
            # Update neighbors
            # For each neighbor u of v:
            # u's relationship with v changed.
            # If u had same label as old_v, it now has different label.
            # So for u: same_count decreases by 1, other_count increases by 1.
            # If u had different label from old_v, it now has same label.
            # So for u: same_count increases by 1, other_count decreases by 1.
            
            # We can vectorize this?
            # neighbors_v = neighbors[v]
            # same_as_old = labels[neighbors_v] == old_label
            # For those, same_counts[u] -= 1, other_counts[u] += 1
            # For others, same_counts[u] += 1, other_counts[u] -= 1
            
            # But careful: labels[v] has already changed.
            # We need to compare neighbors' labels to old_label.
            
            nbrs = neighbors[v]
            if len(nbrs) > 0:
                # labels of neighbors
                nbr_labels = labels[nbrs]
                # Note: labels[v] is new_label.
                # We want to know which neighbors were same as old_label.
                # old_label = 1 - new_label
                # So same_as_old = (nbr_labels == old_label)
                
                was_same = (nbr_labels == old_label)
                
                # Update counts for neighbors
                # For was_same: same--, other++
                # For not was_same: same++, other--
                
                # We can do this with fancy indexing or loops.
                # Since n is small, loop might be okay, but vectorized is better.
                
                # same_counts[nbrs[was_same]] -= 1
                # other_counts[nbrs[was_same]] += 1
                # same_counts[nbrs[~was_same]] += 1
                # other_counts[nbrs[~was_same]] -= 1
                
                # Using np.add.at to handle duplicates? No, nbrs are unique indices.
                # But fancy indexing with boolean mask works if we assign directly?
                # same_counts[nbrs[was_same]] -= 1 works if no duplicates in nbrs[was_same].
                # nbrs are unique.
                
                idx_same = nbrs[was_same]
                idx_other = nbrs[~was_same]
                
                if len(idx_same) > 0:
                    same_counts[idx_same] -= 1
                    other_counts[idx_same] += 1
                if len(idx_other) > 0:
                    same_counts[idx_other] += 1
                    other_counts[idx_other] -= 1
            
            # Cool down
            T *= alpha
            
            # Check time every 1000 steps to avoid overhead
            step += 1
            if step % 1000 == 0:
                if time.time() > deadline:
                    break
    
    return best_labels
# EVOLVE-BLOCK-END
