"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # An edge (i, j) crosses if labels[i] != labels[j]
    # For an undirected graph, each edge is counted once.
    # Using vectorization: for each row, count neighbors with different label.
    # Since the matrix is symmetric, sum of (adjacency @ (labels != labels[None, :])) 
    # would count each crossing edge twice. So divide by 2.
    # Actually, let's just sum over i < j or use the formula.
    # Efficient way:
    # cut = 0.5 * sum_i sum_j A_ij * (1 - x_i x_j) / 2 ... 
    # Let's use: cut = 0.5 * (sum(A) - sum(A * (labels[:, None] == labels[None, :])))
    # Wait, A * (labels match) counts internal edges. 
    # Total edges = sum(A) / 2.
    # Internal edges = sum(A * (labels match)) / 2.
    # Cut = Total - Internal = (sum(A) - sum(A * (labels match))) / 2.
    
    # Let's stick to a robust vectorized calculation.
    # labels is a 1D array of 0/1.
    # diff = (labels[:, None] != labels[None, :])
    # cut = np.sum(adjacency * diff) / 2.0
    
    # Even faster:
    # Let x be labels in {-1, 1}.
    # Cut = 0.25 * (sum(A) - sum(A * x x^T))
    # But labels are 0/1. Let's convert or just use the boolean mask.
    
    # For n=100, this is very fast.
    diff = (labels[:, None] != labels[None, :])
    return 0.5 * np.sum(adjacency * diff)


def partition(adjacency):
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_labels = None
    best_score = -1.0
    
    # Multiple random restarts with local search
    # We will try a mix of random starts and greedy starts.
    
    num_restarts = 50  # Max restarts, limited by time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialization
        if r == 0:
            # Greedy-like initialization or alternating
            labels = np.random.randint(0, 2, n)
        else:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        
        # Local Search: Hill Climbing
        # We iterate until no improvement or time limit.
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
                
            improved = False
            # Randomize order of vertices to break symmetry
            order = np.random.permutation(n)
            
            for v in order:
                if time.time() > deadline - 2.0:
                    break
                    
                # Calculate gain of flipping v
                # Current contribution of v to cut:
                # Edges from v to same side are "wasted".
                # Edges from v to other side are "cut".
                
                # Let's compute the change in cut size if we flip v.
                # Let L be the set of vertices with label 1, and 0 otherwise.
                # If v is in L (label 1):
                #   Current cut edges incident to v: sum(A[v, ~L])
                #   After flip (v in 0):
                #     New cut edges incident to v: sum(A[v, L])
                #     Change = sum(A[v, L]) - sum(A[v, ~L])
                #     Note: sum(A[v, L]) includes A[v,v] which is 0.
                #     sum(A[v, ~L]) is neighbors not in L.
                
                # More simply:
                # Let gain = 2 * (sum(A[v, same_side]) - sum(A[v, diff_side])) ? No.
                
                # Let's use the formula:
                # Cut = 0.5 * sum(A * (x_i != x_j))
                # If we flip x_v, the terms involving v change.
                # Terms A[v, u] for u != v.
                # If x_u == x_v, term was 0, becomes 1.
                # If x_u != x_v, term was 1, becomes 0.
                # So change in sum(A * diff) for row v is:
                #   +1 for each u where x_u == x_v
                #   -1 for each u where x_u != x_v
                # Total change in sum = (# same neighbors) - (# diff neighbors)
                # Change in Cut = 0.5 * Change in sum.
                
                same_mask = (labels == labels[v])
                diff_mask = (labels != labels[v])
                
                same_count = np.sum(adjacency[v, same_mask])
                diff_count = np.sum(adjacency[v, diff_mask])
                
                delta = 0.5 * (same_count - diff_count)
                
                if delta > 1e-9:
                    labels[v] = 1 - labels[v]
                    improved = True
                    
            # After a full pass, check for early stopping if no improvement
            # The 'improved' flag is reset at the start of the while loop.
        
        # Evaluate final score
        score = _cut_size(adjacency, labels)
        
        if score > best_score:
            best_score = score
            best_labels = labels.copy()
            
        # If we hit a perfect cut (all edges crossing? impossible for non-bipartite, 
        # but max possible is total edges), we can stop.
        # Max possible cut size is sum(A)/2.
        max_possible = np.sum(adjacency) / 2.0
        if best_score >= max_possible - 1e-9:
            break

    if best_labels is None:
        # Fallback
        best_labels = np.random.randint(0, 2, n)
        
    return best_labels
# EVOLVE-BLOCK-END
