"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels is a 1D array of 0/1
    # An edge (i,j) is cut if labels[i] != labels[j]
    # Sum over all i<j where adjacency[i,j]=1 and labels[i]!=labels[j]
    # Efficient: use matrix multiplication trick
    # Let x = labels as float (0 or 1)
    # cut = 0.5 * (sum(A) - sum(A * (1 - 2*x_i + 2*x_i*x_j))) ... this is getting complicated
    # Simpler: 
    # diff = (labels[:, None] != labels[None, :])
    # return int((adjacency & diff).sum() // 2)
    # But for speed, let's use:
    x = labels.astype(np.float64)
    # The number of edges in the cut is:
    # sum_{i<j} A[i,j] * (x[i] != x[j])
    # = sum_{i<j} A[i,j] * (1 - (1-2*x[i])*(1-2*x[j])/2 ... no
    # Let's use: cut = 0.5 * (sum(A) - sum(A * ((1-2*x_i)*(1-2*x_j) + 1)/2)) ... 
    # Actually, the standard formula:
    # Let s_i = 1 if label 0, -1 if label 1 (or vice versa)
    # cut = (M - s^T A s) / 4 where M = total edges * 2 ... 
    # Let's just do the direct computation which is fine for n=100
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Get an initial partition using spectral method (Fiedler vector)."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    # Degree matrix
    D = A.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(D) - A
    # Find eigenvector corresponding to second smallest eigenvalue (Fiedler vector)
    # Use numpy's eigvalsh for symmetric matrices
    # For n=100, this is fast
    eigvals, eigvecs = np.linalg.eigh(L)
    # Second smallest eigenvalue is index 1
    fiedler = eigvecs[:, 1]
    # Partition based on sign
    labels = (fiedler < 0).astype(np.int64)
    return labels


def _local_search(adjacency, labels, max_passes=10):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # Compute gain if we flip vertex v
            # Current state: labels[v] is current label
            # Edges from v to same-side: contribute 0 to cut
            # Edges from v to other-side: contribute 1 to cut
            # If we flip v:
            #   Edges to same-side become crossing: gain = count_same
            #   Edges to other-side become non-crossing: loss = count_other
            # Net gain = count_same - count_other
            same_side = (labels == labels[v])
            other_side = (labels != labels[v])
            count_same = A[v, same_side].sum() - A[v, v]  # exclude self
            count_other = A[v, other_side].sum()
            
            if count_same > count_other:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    
    return labels


def _walksat(adjacency, labels, iterations=5000, p=0.5):
    """WalkSAT-style stochastic local search for Max-Cut as Max-2-SAT."""
    n = len(adjacency)
    A = adjacency.astype(np.int64)
    labels = labels.copy()
    
    # Precompute neighbor lists
    neighbors = []
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        neighbors.append(nbrs)
    
    best_labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    for it in range(iterations):
        # Find all unsatisfied clauses (edges not in the cut)
        # An edge (i,j) is unsatisfied if labels[i] == labels[j]
        unsatisfied = []
        for i in range(n):
            for j in neighbors[i]:
                if j > i and labels[i] == labels[j]:
                    unsatisfied.append((i, j))
        
        if not unsatisfied:
            # All edges are cut - perfect solution
            break
        
        # Choose a random unsatisfied clause
        i, j = random.choice(unsatisfied)
        
        # With probability p, flip the variable that appears in the most satisfied clauses
        # With probability 1-p, flip a random variable from the clause
        if random.random() < p:
            # Compute for each variable in the clause, how many satisfied clauses it would break
            # For variable i: count satisfied clauses involving i (excluding the current unsatisfied one)
            # A clause (a,b) is satisfied if labels[a] != labels[b]
            gain_i = 0
            gain_j = 0
            for a in neighbors[i]:
                if a == j:
                    continue
                if labels[i] != labels[a]:
                    # This clause is currently satisfied. Flipping i would break it.
                    gain_i -= 1
                else:
                    # This clause is currently unsatisfied. Flipping i would satisfy it.
                    gain_i += 1
            for b in neighbors[j]:
                if b == i:
                    continue
                if labels[j] != labels[b]:
                    gain_j -= 1
                else:
                    gain_j += 1
            
            # Flip the variable with higher gain (less negative)
            if gain_i >= gain_j:
                labels[i] = 1 - labels[i]
            else:
                labels[j] = 1 - labels[j]
        else:
            # Random flip
            if random.random() < 0.5:
                labels[i] = 1 - labels[i]
            else:
                labels[j] = 1 - labels[j]
        
        # Track best
        current_cut = _cut_size(adjacency, labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the number of cut edges.
    Uses spectral initialization, local search, and WalkSAT for further improvement.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n = len(adjacency)
    A = np.array(adjacency, dtype=bool)
    
    # Start with spectral partition
    labels = _spectral_partition(A)
    
    # Local search to improve
    labels = _local_search(A, labels, max_passes=5)
    
    best_labels = labels.copy()
    best_cut = _cut_size(A, labels)
    
    # Try multiple WalkSAT restarts with different random seeds
    random.seed(42)
    
    num_restarts = 10
    walksat_iters = 3000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Start WalkSAT from current best or a perturbed version
        if restart == 0:
            ws_labels = best_labels.copy()
        else:
            # Perturb the best solution randomly
            ws_labels = best_labels.copy()
            # Flip a random subset of vertices
            num_flips = max(1, n // 10)
            flip_indices = random.sample(range(n), num_flips)
            for idx in flip_indices:
                ws_labels[idx] = 1 - ws_labels[idx]
        
        # Run WalkSAT
        ws_labels = _walksat(A, ws_labels, iterations=walksat_iters, p=0.5)
        
        # Local search polish
        ws_labels = _local_search(A, ws_labels, max_passes=3)
        
        current_cut = _cut_size(A, ws_labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = ws_labels.copy()
    
    # Final local search polish
    final_labels = _local_search(A, best_labels, max_passes=10)
    final_cut = _cut_size(A, final_labels)
    if final_cut > best_cut:
        best_labels = final_labels
        best_cut = final_cut
    
    return best_labels
# EVOLVE-BLOCK-END
