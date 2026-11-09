"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _walksat(adjacency, labels, deadline, alpha=1.0):
    """
    WalkSAT-style stochastic local search for Max-2-SAT formulation of Max-Cut.
    
    Max-Cut is equivalent to Max-2-SAT where each edge (u,v) is a clause (x_u != x_v).
    Unsatisfied clauses correspond to uncut edges (edges within the same partition).
    
    WalkSAT flips variables (nodes) with a probability biased by the number of 
    unsatisfied clauses they appear in, allowing random walks to escape local optima.
    """
    n = len(labels)
    current_labels = labels.copy()
    
    # Precompute adjacency as boolean matrix for fast operations
    adj_bool = adjacency.astype(bool)
    
    # Track unsatisfied edges (edges within same partition)
    # An edge (i,j) is unsatisfied if labels[i] == labels[j] and adj[i,j] is True
    # We'll maintain this incrementally.
    
    # For each vertex, track:
    # - same_count[v]: number of neighbors with same label (unsatisfied edges incident to v)
    # - diff_count[v]: number of neighbors with different label (satisfied edges incident to v)
    
    # Initialize counts
    diff_mat = current_labels[None, :] != current_labels[:, None]
    same_counts = (adj_bool & ~diff_mat).sum(axis=1)
    diff_counts = (adj_bool & diff_mat).sum(axis=1)
    
    # Total unsatisfied edges = sum(same_counts) // 2
    current_unsat = int(same_counts.sum() // 2)
    
    best_labels = current_labels.copy()
    best_unsat = current_unsat
    
    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj_bool[v])[0] for v in range(n)]
    
    start_time = time.time()
    rng = np.random.RandomState()
    
    while time.time() < deadline:
        # If no unsatisfied edges, we found a perfect cut (bipartite graph)
        if current_unsat == 0:
            break
        
        # Weighted selection: pick a vertex with probability proportional to same_count
        # Vertices with more unsatisfied edges are more likely to be flipped
        weights = same_counts.astype(float)
        if weights.sum() == 0:
            # All satisfied, but shouldn't happen if current_unsat > 0
            break
        weights /= weights.sum()
        
        # Pick vertex v with probability proportional to its unsatisfied edge count
        v = rng.choice(n, p=weights)
        
        # With probability alpha, flip v (even if it increases unsat)
        # With probability 1-alpha, only flip if it decreases unsat
        if rng.random() < alpha:
            # Always flip v
            flip_v = True
        else:
            # Only flip if it reduces unsatisfied edges
            # Flipping v changes same_count[v] and diff_count[v]
            # Delta in unsat = new_same - old_same for v, but also affects neighbors
            # Actually, let's just compute the delta properly
            
            # If we flip v:
            # - v's same_count becomes diff_count[v], and vice versa
            # - For each neighbor u of v:
            #   If u had same label as v (before), after flip u has different label
            #     so u's same_count decreases by 1, diff_count increases by 1
            #   If u had different label as v (before), after flip u has same label
            #     so u's same_count increases by 1, diff_count decreases by 1
            
            # Total change in unsatisfied edges:
            # For v: same_count[v] changes to diff_count[v], so delta_v = diff_count[v] - same_count[v]
            # For each neighbor u:
            #   If labels[u] == labels[v] (before): u's same_count -= 1, so delta_u = -1
            #   If labels[u] != labels[v] (before): u's same_count += 1, so delta_u = +1
            # But wait, each edge is counted twice in the sum of same_counts.
            # Let's think in terms of total_unsat = sum(same_counts) // 2
            
            # Change in sum(same_counts):
            # v contributes: diff_count[v] - same_count[v]
            # Each neighbor u contributes:
            #   If u same as v: -1 (u's same_count decreases)
            #   If u diff from v: +1 (u's same_count increases)
            
            same_neighbors = np.sum(current_labels[neighbors[v]] == current_labels[v])
            diff_neighbors = len(neighbors[v]) - same_neighbors
            
            delta_sum = (diff_count[v] - same_count[v]) + (-same_neighbors) + (diff_neighbors)
            # Since diff_neighbors = deg(v) - same_neighbors
            # delta_sum = diff_count[v] - same_count[v] - same_neighbors + deg(v) - same_neighbors
            # = diff_count[v] - same_count[v] + deg(v) - 2*same_neighbors
            
            # But we want to know if total_unsat decreases.
            # total_unsat = sum(same_counts) // 2
            # So if delta_sum < 0, total_unsat decreases.
            
            flip_v = delta_sum < 0
        
        if flip_v:
            # Perform the flip
            old_label = current_labels[v]
            current_labels[v] = 1 - old_label
            
            # Update counts
            # v's counts swap
            same_counts[v], diff_counts[v] = diff_counts[v], same_counts[v]
            
            # Update neighbors
            for u in neighbors[v]:
                if current_labels[u] == 1 - old_label:
                    # u's new label is same as v's NEW label
                    # Before flip: u's label vs old_label
                    # If u's new label == v's new label, then u's old label == v's old label? No.
                    # Let's be careful.
                    # Before flip: labels[u] vs old_label
                    # After flip: labels[u] vs (1 - old_label)
                    # If labels[u] == 1 - old_label, then labels[u] != old_label
                    # So before flip, u and v were different (u in diff for v)
                    # After flip, u and v are same (u in same for v)
                    # So for u: u's same_count increases by 1, diff_count decreases by 1
                    same_counts[u] += 1
                    diff_counts[u] -= 1
                else:
                    # labels[u] == old_label
                    # Before flip: u and v were same
                    # After flip: u and v are different
                    # So for u: u's same_count decreases by 1, diff_count increases by 1
                    same_counts[u] -= 1
                    diff_counts[u] += 1
            
            # Update current_unsat
            current_unsat = int(same_counts.sum() // 2)
            
            # Track best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_labels = current_labels.copy()
    
    return best_labels

def partition(adjacency):
    """
    Main partition function using WalkSAT-style stochastic local search.
    """
    n = len(adjacency)
    deadline = time.time() + 140.0  # Safety margin
    
    # Multiple random restarts with different initializations
    best_labels = None
    best_unsat = float('inf')
    
    # Try several random initializations
    rng = np.random.RandomState()
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        # Random initialization
        labels = rng.randint(0, 2, size=n)
        
        # Run WalkSAT with a portion of the time
        restart_deadline = time.time() + time_per_restart * 0.9  # Leave some margin
        
        result = _walksat(adjacency, labels, restart_deadline, alpha=0.5)
        
        # Evaluate result
        unsat = int((adjacency.astype(bool) & (result[None, :] == result[:, None])).sum() // 2)
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_labels = result
        
        # If we found a perfect cut, stop early
        if best_unsat == 0:
            break
    
    return best_labels
# EVOLVE-BLOCK-END
