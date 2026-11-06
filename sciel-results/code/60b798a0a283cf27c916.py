import numpy as np
import time
import random

N = 256

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(42)
    
    # 1. Precompute APs and Incidence Matrix
    ap_indices = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices.append(indices)
                
    M = len(ap_indices)
    
    # Build A (M x N) as int8
    A = np.zeros((M, N), dtype=np.int8)
    for i, indices in enumerate(ap_indices):
        for idx in indices:
            A[i, idx - 1] = 1
            
    # Precompute B = A.T @ A (N x N) and C = diag(B)
    # B[j, k] is the number of APs containing both position j and k
    B = A.T.astype(np.int32) @ A.astype(np.int32)
    C = np.diag(B).copy()
    
    # 2. Initialization
    best_max_abs = float('inf')
    best_seq = None
    
    # We will run multiple restarts if time permits, but the L2 optimizer is strong enough that one long run might suffice.
    # Let's do a few restarts to be safe.
    
    num_restarts = 0
    
    while time.time() < deadline - 5.0:
        num_restarts += 1
        # Random start
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        # Initial S and Q
        # S = A @ s
        S = A.astype(np.int32) @ s.astype(np.int32)
        # Q = A.T @ S
        Q = A.T.astype(np.int32) @ S
        
        current_max_abs = int(np.max(np.abs(S)))
        if current_max_abs < best_max_abs:
            best_max_abs = current_max_abs
            best_seq = s.copy()
            
        # Annealing parameters
        T = 10.0
        cooling_rate = 0.99
        min_T = 0.01
        
        # Main Loop
        while T > min_T and time.time() < deadline - 2.0:
            # Compute delta E for all positions
            # delta_E[j] = 4 * (C[j] - s[j] * Q[j])
            # s[j] is +/- 1
            # Note: Q is int32, s is int8
            delta_E = 4 * (C - s * Q)
            
            # Find the best move (min delta_E)
            # We can just pick the index with minimum delta_E
            j = np.argmin(delta_E)
            dE = delta_E[j]
            
            # Metropolis acceptance
            accept = False
            if dE <= 0:
                accept = True
            else:
                # Avoid overflow in exp
                if dE / T < 50: 
                    p = np.exp(-dE / T)
                    if rng.random() < p:
                        accept = True
            
            if accept:
                # Flip s[j]
                old_s_j = s[j]
                s[j] = -old_s_j
                
                # Update S
                # S_new = S - 2 * old_s_j * A[:, j]
                # A[:, j] is a column of 0s and 1s
                # We can update S in place
                S -= 2 * old_s_j * A[:, j]
                
                # Update Q
                # Q_new = Q - 2 * old_s_j * B[:, j]
                Q -= 2 * old_s_j * B[:, j]
                
                # Update max abs
                # We could track this incrementally, but recomputing is O(M)
                # To save time, we only recompute max every K steps or if we accept a move that might reduce it?
                # Actually, just checking max is O(M). M=80k. 80k ops is ~0.1ms.
                # We can do it every step.
                current_max_abs = int(np.max(np.abs(S)))
                
                if current_max_abs < best_max_abs:
                    best_max_abs = current_max_abs
                    best_seq = s.copy()
            
            T *= cooling_rate
            
            # If we are stuck at a very low max, we can speed up cooling
            if current_max_abs <= 3:
                T *= 0.95

    # Final Polish: Greedy descent on max-abs directly on the best found
    if best_seq is not None and time.time() < deadline - 1.0:
        s = best_seq.copy()
        S = A.astype(np.int32) @ s.astype(np.int32)
        Q = A.T.astype(np.int32) @ S
        current_max_abs = int(np.max(np.abs(S)))
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for j in range(N):
                # Try flipping j
                old_s_j = s[j]
                # Calculate new max without actually flipping?
                # S_new = S - 2 * old_s_j * A[:, j]
                # This is O(M). Doing it for all j is O(N*M) = 20M ops.
                # We can do this in a vectorized way?
                # S_new_matrix = S[None, :] - 2 * s[:, None] * A.T
                # This creates a 256 x 80000 matrix. 20M ints. 80MB.
                # Might be too slow/memory heavy.
                
                # Let's stick to simple loop for polish, it's short.
                S_temp = S - 2 * old_s_j * A[:, j]
                new_max = int(np.max(np.abs(S_temp)))
                if new_max < current_max_abs:
                    # Accept
                    s[j] = -old_s_j
                    S = S_temp
                    Q -= 2 * old_s_j * B[:, j]
                    current_max_abs = new_max
                    improved = True
                    if current_max_abs < best_max_abs:
                        best_max_abs = current_max_abs
                        best_seq = s.copy()
                    if time.time() > deadline - 0.5:
                        break

    return [int(x) for x in best_seq]
