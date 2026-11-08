"""Program for matrix multiplication tensor decomposition 3x3x5."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Search for rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Uses randomized search with exact verification.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,j}
    # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] which is 1 if this is a valid multiplication term, 0 otherwise
    # More precisely, the tensor has entries T[(i,j), j, (j,k)] = 1 for each (i,j,k)
    # In our indexing:
    # U index: i*m + j (row of A)
    # V index: j*p + k (column of B)  
    # W index: k*n + i (entry of C)
    # The tensor T has T[u,v,w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some valid (i,j,k)
    # But actually for the decomposition, we need:
    # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    
    # The target tensor is: T[u,v,w] = 1 if there exists (i,j,k) such that
    # u = i*m+j, v = j*p+k, w = k*n+i, AND the j in u equals the j in v
    # Wait, let me reconsider. The matrix multiplication tensor is:
    # For C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[(i,j), (j,k), (i,k)] = 1
    # In flat indexing:
    # U row: i*m + j
    # V row: j*p + k  
    # W row: k*n + i
    # T[u,v,w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    # Note: the j in u must equal the j in v for a valid term
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that U @ diag(V) @ W^T reconstructs the target tensor."""
        # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # Efficiently: for each r, add U[:,r] outer V[:,r] outer W[:,r]
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(U_mat.shape[1]):
            T_recon += np.einsum('u,v,w->uvw', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return np.allclose(T_recon, target, atol=tol)
    
    def verify_exact(U_list, V_list, W_list):
        """Exact verification using fractions."""
        rank = len(U_list[0])
        T_recon = {}
        for r in range(rank):
            for u in range(len(U_list)):
                if U_list[u][r] != 0:
                    for v in range(len(V_list)):
                        if V_list[v][r] != 0:
                            for w in range(len(W_list)):
                                if W_list[w][r] != 0:
                                    key = (u, v, w)
                                    T_recon[key] = T_recon.get(key, 0) + U_list[u][r] * V_list[v][r] * W_list[w][r]
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    expected = 1
                    actual = T_recon.get((u, v, w), 0)
                    if actual != expected:
                        return False
        
        # Check no extra non-zero entries
        for key, val in T_recon.items():
            if val != 0:
                u, v, w = key
                # Check if this is a valid target entry
                i = u // m
                j = u % m
                j2 = v // p
                k = v % p
                k2 = w // n
                i2 = w % n
                if not (i == i2 and j == j2 and k == k2):
                    return False
        
        return True
    
    # Start with schoolbook
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank), dtype=np.int64)
    V_sb = np.zeros((m*p, schoolbook_rank), dtype=np.int64)
    W_sb = np.zeros((p*n, schoolbook_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    # Try to find lower rank decompositions using randomized search
    # Strategy: try ranks from schoolbook-1 down, using random initialization and local search
    
    np.random.seed(42)
    
    for trial_rank in range(schoolbook_rank - 1, max(15, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U_rand = np.zeros((n*m, trial_rank), dtype=np.float64)
            V_rand = np.zeros((m*p, trial_rank), dtype=np.float64)
            W_rand = np.zeros((p*n, trial_rank), dtype=np.float64)
            
            # Initialize with random small values
            for r in range(trial_rank):
                U_rand[:, r] = np.random.choice([-1, 0, 1], size=n*m)
                V_rand[:, r] = np.random.choice([-1, 0, 1], size=m*p)
                W_rand[:, r] = np.random.choice([-1, 0, 1], size=p*n)
            
            # Local search: try to adjust entries to match target
            success = False
            for iter in range(200):
                if time.time() > deadline:
                    break
                    
                # Compute current reconstruction
                T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for r in range(trial_rank):
                    T_recon += np.einsum('u,v,w->uvw', U_rand[:,r], V_rand[:,r], W_rand[:,r])
                
                # Check if we've matched
                if np.allclose(T_recon, target, atol=1e-6):
                    # Try to round to integers
                    U_round = np.round(U_rand).astype(np.int64)
                    V_round = np.round(V_rand).astype(np.int64)
                    W_round = np.round(W_rand).astype(np.int64)
                    if verify(U_round, V_round, W_round, tol=1e-9):
                        success = True
                        break
                
                # Find the largest error
                error = T_recon - target
                # Find position with max absolute error
                max_err_idx = np.unravel_index(np.argmax(np.abs(error)), error.shape)
                u_idx, v_idx, w_idx = max_err_idx
                err_val = error[u_idx, v_idx, w_idx]
                
                if abs(err_val) < 1e-6:
                    break
                
                # Try to fix by adjusting one entry
                # The entry U[u_idx, r] affects all (u_idx, v, w) for all v, w
                # Similarly for V and W
                # Try adjusting U[u_idx, r] for random r
                r = np.random.randint(trial_rank)
                
                # Compute gradient-like update
                # d(T[u,v,w])/d(U[u,r]) = V[v,r] * W[w,r]
                # We want to reduce error, so update U[u,r] -= err_val / (sum of V[v',r]*W[w',r] for all v',w' where error exists)
                
                # Simpler: just try flipping or adjusting
                if err_val > 0:
                    # Need to decrease T[u_idx, v_idx, w_idx]
                    # Try decreasing U[u_idx, r] or V[v_idx, r] or W[w_idx, r]
                    choice = np.random.randint(3)
                    if choice == 0:
                        U_rand[u_idx, r] -= 1.0
                    elif choice == 1:
                        V_rand[v_idx, r] -= 1.0
                    else:
                        W_rand[w_idx, r] -= 1.0
                else:
                    # Need to increase T[u_idx, v_idx, w_idx]
                    choice = np.random.randint(3)
                    if choice == 0:
                        U_rand[u_idx, r] += 1.0
                    elif choice == 1:
                        V_rand[v_idx, r] += 1.0
                    else:
                        W_rand[w_idx, r] += 1.0
            
            if success:
                best_rank = trial_rank
                best_U = U_round.tolist()
                best_V = V_round.tolist()
                best_W = W_round.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
