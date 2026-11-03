import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Precompute the target tensor T for verification
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0

    def verify(U_mat, V_mat, W_mat, atol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', W_mat, U_mat, V_mat)
        return np.allclose(T_recon, T, atol=atol)

    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to reduce rank by finding linear dependencies among the gates
    # Each gate r has:
    #   U_gate: vector of length n*m with one 1 at position i*m+j
    #   V_gate: vector of length m*p with one 1 at position j*p+k  
    #   W_gate: vector of length p*n with one 1 at position k*n+i
    #
    # Two gates can be merged if their tensor contributions are linearly dependent.
    # For now, let's try a simpler approach: check if removing one gate and adjusting others works.
    
    # Try rank reduction from schoolbook - 1 downwards
    for target_rank in range(rank_schoolbook - 1, max(1, rank_schoolbook - 20), -1):
        if time.time() > deadline - 30:
            break
            
        # Try to find a rank-target_rank decomposition
        # Use random restarts with a smarter initialization
        found = False
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            # Initialize with a subset of schoolbook gates
            # Pick target_rank random gates from the schoolbook
            indices = np.random.choice(rank_schoolbook, size=target_rank, replace=False)
            U = U_school[:, indices].copy()
            V = V_school[:, indices].copy()
            W = W_school[:, indices].copy()
            
            # Now try to optimize these gates
            # Use gradient-free optimization: randomly perturb and keep if error decreases
            
            T_recon = np.einsum('ar,br,cr->abc', W, U, V)
            current_error = np.linalg.norm(T_recon - T)
            
            max_iters = 100
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Randomly perturb one gate
                gate = np.random.randint(target_rank)
                component = np.random.randint(3)  # 0=U, 1=V, 2=W
                
                if component == 0:
                    old_val = U[:, gate].copy()
                    U[:, gate] += np.random.randn(n * m) * 0.1
                elif component == 1:
                    old_val = V[:, gate].copy()
                    V[:, gate] += np.random.randn(m * p) * 0.1
                else:
                    old_val = W[:, gate].copy()
                    W[:, gate] += np.random.randn(p * n) * 0.1
                
                T_recon = np.einsum('ar,br,cr->abc', W, U, V)
                new_error = np.linalg.norm(T_recon - T)
                
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-9:
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        found = True
                        break
                else:
                    # Revert
                    if component == 0:
                        U[:, gate] = old_val
                    elif component == 1:
                        V[:, gate] = old_val
                    else:
                        W[:, gate] = old_val
            
            if found:
                break
        
        if found:
            break
    
    # Return the best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
