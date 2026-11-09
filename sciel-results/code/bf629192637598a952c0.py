"""
Strategy: Goemans-Williamson Spectral Initialization + Vectorized Single-Node Simulated Annealing.

1. Initialization: 
   - Compute the top k eigenvectors of the graph Laplacian (or related matrix).
   - Use a random rotation matrix to project these eigenvectors into a high-dimensional space.
   - Assign labels based on the sign of the projection onto a random vector.
   - This is the Goemans-Williamson (GW) approximation for Max Cut, which is theoretically strong.

2. Optimization:
   - Use a vectorized Simulated Annealing (SA) algorithm.
   - In each step, calculate the gain of flipping every single node simultaneously.
   - Gain for flipping node i: 
     Let x be the current label vector (0/1).
     Let A be the adjacency matrix.
     The cut size is 0.5 * sum_{i,j} A_{ij} (x_i != x_j).
     If we flip node i, the change in cut size is:
     Delta_i = sum_{j} A_{ij} * (1 if x_i == x_j else -1)
             = sum_{j} A_{ij} * (1 - 2*x_i*(1-x_j) - 2*(1-x_i)*x_j) ... this is getting messy.
     
     Simpler formulation using -1/1 labels:
     Let y_i = 1 if x_i=1 else -1.
     Cut = 0.5 * (||y||^2 * E_total - sum_{i,j} A_{ij} y_i y_j)
     Actually, Cut = 0.5 * sum_{i,j} A_{ij} (1 - y_i y_j) / 2? No.
     Cut = number of edges (i,j) where y_i != y_j.
     Cut = 0.5 * sum_{i,j} A_{ij} (1 - y_i y_j) is for weighted? 
     Let's stick to 0/1 for delta calculation.
     
     Delta for flipping i:
     Current contribution of edges incident to i:
     For each neighbor j:
       If x_i == x_j, edge is NOT cut. Flipping i makes it cut. Gain +1.
       If x_i != x_j, edge IS cut. Flipping i makes it not cut. Loss -1.
     So Delta_i = (num neighbors j with x_j == x_i) - (num neighbors j with x_j != x_i)
     Let S_i = sum_{j} A_{ij} x_j.
     Let d_i = sum_{j} A_{ij} (degree).
     If x_i = 0:
       Neighbors with x_j=0 contribute +1. Neighbors with x_j=1 contribute -1.
       Delta_i = S_i - (d_i - S_i) = 2*S_i - d_i.
     If x_i = 1:
       Neighbors with x_j=1 contribute +1 (become 0, so cut increases? No.
       If x_i=1, x_j=1: edge not cut. Flip i to 0. x_i!=x_j. Edge becomes cut. Gain +1.
       If x_i=1, x_j=0: edge cut. Flip i to 0. x_i==x_j. Edge not cut. Loss -1.
       Delta_i = (num neighbors with x_j=1) - (num neighbors with x_j=0)
       S_i = sum A_{ij} x_j = num neighbors with x_j=1.
       d_i = num neighbors total.
       Delta_i = S_i - (d_i - S_i) = 2*S_i - d_i.
       
     Wait, let's re-verify.
     If x_i=0, x_j=0: Not cut. Flip i->1. x_i!=x_j. Cut. Change: +1.
     If x_i=0, x_j=1: Cut. Flip i->1. x_i==x_j. Not cut. Change: -1.
     Delta = (# x_j=0) - (# x_j=1) = (d_i - S_i) - S_i = d_i - 2*S_i.
     
     If x_i=1, x_j=1: Not cut. Flip i->0. x_i!=x_j. Cut. Change: +1.
     If x_i=1, x_j=0: Cut. Flip i->0. x_i==x_j. Not cut. Change: -1.
     Delta = (# x_j=1) - (# x_j=0) = S_i - (d_i - S_i) = 2*S_i - d_i.
     
     So:
     If x_i == 0: Delta_i = d_i - 2*S_i
     If x_i == 1: Delta_i = 2*S_i - d_i
     
     We can compute S = A @ x (vector of sums of neighbor labels).
     Then compute Delta for all nodes at once.
     
     This allows O(N^2) per step (for the matrix-vector multiply) but with very low constant factor.
     We can then pick the best positive delta, or a random node if no positive delta (with prob exp(-delta/T)).
"""

import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels is 0/1
    # diff is 1 if labels differ
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _gw_init(adjacency):
    """
    Goemans-Williamson initialization.
    1. Compute Laplacian L.
    2. Get eigenvectors corresponding to the smallest non-zero eigenvalues.
    3. Randomly rotate and project.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        # Get eigenvalues and vectors
        vals, vecs = np.linalg.eigh(L)
        
        # We want the eigenvectors for the smallest non-zero eigenvalues.
        # Usually, the first eigenvector is the all-ones vector (eigenvalue 0).
        # We take the next 'k' eigenvectors. For N=100, taking k=10 is usually good.
        k = min(10, n-1)
        eigvecs = vecs[:, 1:1+k]
        
        # Generate a random rotation matrix R (k x k)
        R = np.random.normal(0, 1, (k, k))
        # Orthonormalize R using SVD or QR to ensure it's a rotation
        Q, _ = np.linalg.qr(R)
        
        # Project eigenvectors onto the rotated space
        # M = eigvecs @ Q.T  (n x k)
        M = eigvecs @ Q.T
        
        # Generate a random vector v in R^k
        v = np.random.normal(0, 1, k)
        
        # Project M onto v
        # z = M @ v (n,)
        z = M @ v
        
        # Assign labels based on sign of z
        labels = (z >= 0).astype(np.int8)
        
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n, dtype=np.int8)

def _vectorized_sa(adjacency, labels, start_temp, end_temp, max_iters, deadline):
    """
    Vectorized Simulated Annealing.
    Computes delta for all nodes simultaneously, picks best or random move.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute
    adj_float = adjacency.astype(float)
    degrees = adj_float.sum(axis=1)
    
    start_time = time.time()
    if max_iters <= 0:
        return best_labels, best_cut
        
    alpha = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        T = start_temp * (alpha ** i)
        
        # Calculate S = A @ labels
        # labels is 0/1 int8, adj_float is float64
        S = adj_float @ labels.astype(float)
        
        # Calculate Delta for all nodes
        # If x_i == 0: Delta = d_i - 2*S_i
        # If x_i == 1: Delta = 2*S_i - d_i
        # We can compute this as:
        # Delta_i = (1 - 2*x_i) * (2*S_i - d_i)
        # Let's verify:
        # x_i=0: (1) * (2S - d) = 2S - d. Wait, my derivation said d - 2S for x=0.
        # Let's re-check derivation.
        # x=0: Delta = (# x_j=0) - (# x_j=1) = (d-S) - S = d - 2S.
        # Formula (1-2x)(2S-d) for x=0 gives 2S-d. This is - (d-2S).
        # So if x=0, the formula gives negative of the correct delta?
        # Let's re-derive carefully.
        
        # Objective: Maximize Cut.
        # Cut = sum_{i<j} A_{ij} [x_i != x_j]
        # Change in cut when flipping i:
        # Edges (i, j) for all j != i.
        # If x_i == x_j, edge is NOT cut. Flipping i makes x_i != x_j. Edge becomes cut. Change +1.
        # If x_i != x_j, edge IS cut. Flipping i makes x_i == x_j. Edge becomes not cut. Change -1.
        
        # Case x_i = 0:
        # Neighbors j with x_j = 0: x_i == x_j. Change +1.
        # Neighbors j with x_j = 1: x_i != x_j. Change -1.
        # Delta = count(x_j=0) - count(x_j=1)
        # count(x_j=1) = S_i
        # count(x_j=0) = d_i - S_i
        # Delta = (d_i - S_i) - S_i = d_i - 2*S_i.
        
        # Case x_i = 1:
        # Neighbors j with x_j = 1: x_i == x_j. Change +1.
        # Neighbors j with x_j = 0: x_i != x_j. Change -1.
        # Delta = count(x_j=1) - count(x_j=0)
        # count(x_j=1) = S_i
        # count(x_j=0) = d_i - S_i
        # Delta = S_i - (d_i - S_i) = 2*S_i - d_i.
        
        # So:
        # Delta_i = d_i - 2*S_i if x_i == 0
        # Delta_i = 2*S_i - d_i if x_i == 1
        
        # We can write this as:
        # Delta_i = (1 - 2*x_i) * (d_i - 2*S_i)
        # Check x=0: 1 * (d - 2S) = d - 2S. Correct.
        # Check x=1: -1 * (d - 2S) = 2S - d. Correct.
        
        deltas = (1 - 2 * labels.astype(float)) * (degrees - 2 * S)
        
        # Find the best move
        max_delta = np.max(deltas)
        
        if max_delta > 0:
            # Always take the best improving move
            best_idx = np.argmax(deltas)
            # Flip the node
            labels[best_idx] = 1 - labels[best_idx]
            current_cut += max_delta
        else:
            # No improving move. Try a random move with probability exp(-delta/T)
            # To avoid iterating over all nodes, pick a random node and check it.
            # Or, more efficiently, we can accept a random worsening move if T is high enough.
            # For simplicity and speed, let's just pick a random node and calculate its delta.
            # Since we already have deltas, we can just pick a random index.
            idx = np.random.randint(n)
            delta = deltas[idx]
            
            if T > 0:
                # If delta is negative, prob = exp(-delta/T)
                # If delta is 0, prob = 1
                if delta < 0:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        labels[idx] = 1 - labels[idx]
                        current_cut += delta
                else:
                    # delta == 0 (shouldn't happen often if no max > 0, but possible)
                    labels[idx] = 1 - labels[idx]
                    current_cut += delta
            else:
                # T is very small, only accept improving moves. Since none exist, do nothing.
                pass
                
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function.
    1. Goemans-Williamson Initialization.
    2. Vectorized Single-Node Simulated Annealing.
    3. Restarts if time permits.
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: GW Init + SA
    labels = _gw_init(adjacency)
    
    # Temperature scaling
    # Max delta is bounded by degree. Max degree <= n.
    # So max delta ~ n.
    start_temp = 5.0 * n
    end_temp = 0.1
    
    # Run SA until close to deadline
    # We leave 5s for safety and potential restarts
    labels, cut = _vectorized_sa(adjacency, labels, start_temp, end_temp, 1000000, deadline - 5.0)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Phase 2: Restarts
    # If we have time left, do a few more restarts with different GW inits
    remaining = deadline - time.time() - 2.0
    if remaining > 1.0:
        # Estimate time per restart
        # Each SA run takes ~ time. 
        # Let's just try 3-5 more restarts if we have > 10s.
        num_restarts = min(5, int(remaining / 3.0))
        
        for r in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            labels = _gw_init(adjacency)
            # Shorter SA for restarts
            sub_deadline = time.time() + (remaining / num_restarts) - 0.5
            labels, cut = _vectorized_sa(adjacency, labels, start_temp, end_temp, 1000000, sub_deadline)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
                
    if best_labels is None:
        best_labels = _gw_init(adjacency)
        
    return best_labels
