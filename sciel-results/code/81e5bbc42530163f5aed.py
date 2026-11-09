"""Rewritten program for Maximum Cut with stronger algorithm."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # For each vertex, count neighbors in the opposite partition
    # Sum over all vertices, divide by 2 (each edge counted twice)
    n = len(labels)
    # Use matrix multiplication for speed
    # labels as 0/1, opposite = 1 - labels
    # cut_size = sum over i,j of A[i,j] * (labels[i] != labels[j])
    # = sum over i,j of A[i,j] * (labels[i] * (1-labels[j]) + (1-labels[i]) * labels[j])
    # Simpler: use the fact that for binary labels,
    # cut = sum_i labels[i] * sum_j A[i,j]*(1-labels[j]) + ...
    # Actually, let's just do it with numpy efficiently
    # Create a matrix where M[i,j] = 1 if labels[i] != labels[j] else 0
    # cut = sum(A * M)
    # But we can compute more efficiently:
    # Let L = labels (0/1 vector)
    # For each i, the number of neighbors of i in the opposite set is:
    #   if labels[i] == 0: sum_j A[i,j] * labels[j]
    #   if labels[i] == 1: sum_j A[i,j] * (1 - labels[j])
    # Total cut = sum_i [if labels[i]==0: (A @ labels)[i] else: (A @ (1-labels))[i]]
    # = sum_i labels[i] * (A @ (1-labels))[i] + (1-labels[i]) * (A @ labels)[i]
    # = sum_i labels[i] * (sum_j A[i,j] - (A @ labels)[i]) + (1-labels[i]) * (A @ labels)[i]
    # = sum_i labels[i] * (deg[i] - (A @ labels)[i]) + (1-labels[i]) * (A @ labels)[i]
    # = sum_i labels[i]*deg[i] - sum_i labels[i]*(A @ labels)[i] + sum_i (A @ labels)[i] - sum_i labels[i]*(A @ labels)[i]
    # = sum_i labels[i]*deg[i] + sum_i (A @ labels)[i] - 2 * sum_i labels[i]*(A @ labels)[i]
    
    # Even simpler: 
    # cut = 0.5 * (sum of degrees - sum of internal edges * 2)
    # Or: cut = (total_edges - internal_edges)
    # internal_edges = 0.5 * (labels^T A labels + (1-labels)^T A (1-labels))
    # Let's just use the direct approach with numpy for clarity and speed
    
    L = labels.astype(np.int32)
    A = adjacency.astype(np.int32)
    # Number of edges from vertices in group 1 to group 0, plus from group 0 to group 1
    # But since A is symmetric, we just need:
    # cut = sum_i,j A[i,j] * L[i] * (1-L[j])
    # = L^T A (1-L)
    # Since A is symmetric, this equals (1-L)^T A L as well, and the total cut is this value
    # (because each crossing edge (i in 1, j in 0) is counted once in L^T A (1-L))
    cut = int(L @ A @ (1 - L))
    return cut


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses simulated annealing with multiple restarts and a deadline guard.
    """
    n = len(adjacency)
    A = adjacency.astype(np.int32)
    deg = A.sum(axis=1)
    
    # Precompute for fast cut size updates
    # When we flip vertex v, the change in cut size is:
    # If v is in group 0 (label 0): 
    #   Currently contributes: sum_j A[v,j] * L[j] (edges from v to group 1)
    #   After flip (to group 1): contributes sum_j A[v,j] * (1-L[j]) = deg[v] - sum_j A[v,j]*L[j]
    #   Delta = deg[v] - 2 * sum_j A[v,j]*L[j]
    # If v is in group 1 (label 1):
    #   Currently contributes: sum_j A[v,j] * (1-L[j]) = deg[v] - sum_j A[v,j]*L[j]
    #   After flip (to group 0): contributes sum_j A[v,j] * L[j]
    #   Delta = 2 * sum_j A[v,j]*L[j] - deg[v]
    # In general: delta = (1 - 2*L[v]) * (2 * (A @ L)[v] - deg[v])
    
    def flip_delta(v, L, A_L):
        """Compute change in cut size if we flip vertex v."""
        # A_L[v] = sum_j A[v,j] * L[j]
        return (1 - 2 * L[v]) * (2 * A_L[v] - deg[v])
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 50
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            # Start with alternating pattern
            L = np.arange(n) % 2
        elif restart == 1:
            # Start with random
            L = rng.getrandbits(1) and np.ones(n, dtype=np.int32) or np.zeros(n, dtype=np.int32)
            L = np.array([rng.getrandbits(1) for _ in range(n)], dtype=np.int32)
        else:
            # Random initialization
            L = np.array([rng.getrandbits(1) for _ in range(n)], dtype=np.int32)
        
        # Compute initial cut
        A_L = A @ L
        current_cut = int(L @ A @ (1 - L))
        
        # Simulated annealing
        # Temperature schedule: geometric cooling
        T_start = 10.0
        T_end = 0.01
        # Number of iterations per restart
        # We'll iterate until deadline or a fixed number of steps
        max_iters = 20000
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        T = T_start
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = rng.randint(0, n - 1)
            
            # Compute delta
            delta = (1 - 2 * L[v]) * (2 * A_L[v] - deg[v])
            
            # Accept if delta > 0 or with probability exp(delta/T)
            if delta > 0 or rng.random() < np.exp(delta / T):
                # Flip vertex v
                L[v] = 1 - L[v]
                # Update A_L: A_L changes for all neighbors of v
                # A_L[u] += A[u,v] * (2*L[v] - 1) for all u... 
                # Actually, A_L[v] = sum_j A[v,j]*L[j]. When we flip L[v], 
                # A_L[u] for u != v changes by A[u,v] * (new_L[v] - old_L[v]) = A[u,v] * (2*L[v]_new - 1)
                # Wait, let's be more careful.
                # Old L[v] = l, new L[v] = 1-l
                # A_L[u] = sum_j A[u,j] * L[j]
                # For u != v: A_L[u] changes by A[u,v] * (1-l - l) = A[u,v] * (1 - 2l) = A[u,v] * (2*(1-l) - 1)
                # For u = v: A_L[v] = sum_j A[v,j]*L[j], this doesn't change because L[v] is not in the sum for A_L[v] (no self-loops assumed, but even if there are, A[v,v]*L[v] changes)
                # Assuming no self-loops (typical for undirected graphs), A[v,v] = 0, so A_L[v] doesn't change.
                # But to be safe, let's update properly.
                
                # The change in L is: L[v] goes from l to 1-l, so delta_L = 1 - 2*l = 2*(1-l) - 1
                # A_L[u] += A[u,v] * delta_L for all u
                # Since A is symmetric, A[u,v] = A[v,u]
                delta_L = 2 * L[v] - 1  # This is the change: new - old = (1-l) - l = 1-2l, but L[v] is now 1-l, so 2*L[v]-1 = 2(1-l)-1 = 1-2l. Correct.
                A_L += delta_L * A[:, v]
                
                current_cut += delta
            
            # Update temperature
            T *= cooling_rate
        
        # Check if this is the best so far
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = L.copy()
        
        # If we're already at a good solution, maybe do some local improvement
        if time.time() > deadline:
            break
    
    if best_labels is None:
        # Fallback: return alternating pattern
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
