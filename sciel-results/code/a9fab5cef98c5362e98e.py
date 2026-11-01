import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Target tensor T where T[i*m+j, j*p+k, k*n+i] = 1
    # Dimensions: (n*m, m*p, p*n) = (4, 6, 6)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_solution(U, V, W):
        rank = U.shape[1]
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_calc, T_target, atol=1e-9)

    # Schoolbook decomposition as fallback (Rank 12)
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Strategy: Construct Rank 10 solution
    # 1. Use 7 Strassen terms for the first 2 columns of B (k=0,1)
    # 2. Solve for 3 additional terms for the residual (k=2)
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((6, 7))
    W_strassen = np.zeros((6, 7))
    
    # Standard Strassen for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices (first 2 cols): 00, 01, 10, 11 -> 0, 1, 3, 4
    # C indices (first 2 cols): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # T1: (A00+A11)(B00+B11)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[4, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # T2: (A10+A11)B00
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = 1
    
    # T3: A00(B11-B10)
    U_strassen[0, 2] = 1
    V_strassen[4, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[1, 2] = 1; W_strassen[3, 2] = -1
    
    # T4: A11(B00-B01)
    U_strassen[3, 3] = 1
    V_strassen[0, 3] = 1; V_strassen[1, 3] = -1
    W_strassen[2, 3] = 1; W_strassen[0, 3] = -1
    
    # T5: (A00+A01)B11
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[4, 4] = 1
    W_strassen[1, 4] = 1; W_strassen[3, 4] = 1
    
    # T6: (A10-A00)(B00+B01)
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[2, 5] = 1; W_strassen[0, 5] = 1
    
    # T7: (A01-A11)(B10+B11)
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[3, 6] = 1; V_strassen[4, 6] = 1
    W_strassen[1, 6] = 1; W_strassen[3, 6] = 1

    # Calculate Residual R = T_target - T_strassen
    T_strassen = np.zeros((n*m, m*p, p*n))
    for r in range(7):
        T_strassen += np.einsum('i,j,k->ijk', U_strassen[:, r], V_strassen[:, r], W_strassen[:, r])
    
    R = T_target - T_strassen
    
    # R should have non-zeros only at:
    # (0, 2, 4), (1, 5, 4), (2, 2, 5), (3, 5, 5)
    # We want to decompose R into 3 rank-1 tensors.
    
    found_rank_10 = False
    U_10 = None
    V_10 = None
    W_10 = None
    
    # Randomized search for 3 vectors u, v, w
    # We will try 1000 random initializations
    # To speed up, we fix the supports to the non-zeros of R?
    # Actually, the vectors can have non-zeros elsewhere, but they must cancel out.
    # However, for a sparse target, it's likely the vectors are sparse too.
    # Let's try sparse vectors first.
    
    # Support of R:
    # i in {0, 1, 2, 3}
    # j in {2, 5}
    # k in {4, 5}
    
    # Let's try to solve using alternating minimization on the 3 vectors.
    # Initialize u, v, w randomly.
    
    max_iters = 100
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize u, v, w
        # Try to initialize with values that might work.
        # Since R is small, we can try to set specific entries.
        
        u = np.zeros((4, 3))
        v = np.zeros((6, 3))
        w = np.zeros((6, 3))
        
        # Random initialization on the support of R
        # u[i] for i in 0..3
        # v[j] for j in 2,5
        # w[k] for k in 4,5
        
        for r in range(3):
            u[0, r] = random.uniform(-2, 2)
            u[1, r] = random.uniform(-2, 2)
            u[2, r] = random.uniform(-2, 2)
            u[3, r] = random.uniform(-2, 2)
            
            v[2, r] = random.uniform(-2, 2)
            v[5, r] = random.uniform(-2, 2)
            
            w[4, r] = random.uniform(-2, 2)
            w[5, r] = random.uniform(-2, 2)
            
        # Alternating minimization
        for it in range(max_iters):
            # Solve for u given v, w
            # R[i,j,k] = sum_r u[i,r] v[j,r] w[k,r]
            # For fixed i, this is a linear system in u[i,:]? No.
            # R[i, :, :] = sum_r u[i,r] (v[:,r] \otimes w[:,r])
            # This is not linear in u[i] directly because of the product v*w.
            # However, we can flatten.
            # Let M_r = v[:,r] \otimes w[:,r].
            # R[i] = sum_r u[i,r] M_r.
            # This is a linear system for u[i] if we treat M_r as basis vectors.
            # But M_r are fixed for a given step.
            
            # Actually, alternating minimization for tensor decomposition is:
            # Fix v, w. Solve for u.
            # R[i,j,k] = sum_r u[i,r] v[j,r] w[k,r]
            # For a fixed i, let x_r = u[i,r].
            # R[i,j,k] = sum_r x_r (v[j,r] w[k,r])
            # This is a linear system in x? No, it's a sum of rank-1 matrices.
            # We can solve for x by least squares.
            
            # Let's do it for all i.
            # Construct matrix A (size 36x3) where columns are v_r \otimes w_r.
            # Then R_flat (size 36) = A @ u_flat? No.
            # R[i, j, k] = sum_r u[i,r] v[j,r] w[k,r]
            # Let's fix i.
            # R[i, :, :] is a 6x6 matrix.
            # It is the sum of 3 rank-1 matrices: sum_r u[i,r] (v[:,r] w[:,r]^T)
            # Let V_w = [v[:,0] w[:,0]^T, v[:,1] w[:,1]^T, v[:,2] w[:,2]^T] (6x6 x 3)
            # We want to find u[i, :] such that sum_r u[i,r] V_w[:, :, r] approx R[i, :, :]
            # This is a linear least squares problem.
            
            # Build the basis matrix B (size 36x3)
            B = np.zeros((36, 3))
            for r in range(3):
                B[:, r] = np.kron(v[:, r], w[:, r])
            
            # Solve for u
            u_new = np.zeros((4, 3))
            for i in range(4):
                R_i = R[i, :, :].flatten()
                # Solve B x = R_i
                try:
                    u_new[i, :] = np.linalg.lstsq(B, R_i, rcond=None)[0]
                except:
                    break
                    
            # Check convergence or update
            u = u_new
            
            # Similarly for v and w
            # Fix u, w. Solve for v.
            # R[i, j, k] = sum_r u[i,r] v[j,r] w[k,r]
            # For fixed j, R[:, j, :] is 4x6.
            # sum_r v[j,r] (u[:,r] w[:,r]^T)
            B_v = np.zeros((24, 3))
            for r in range(3):
                B_v[:, r] = np.kron(u[:, r], w[:, r])
                
            v_new = np.zeros((6, 3))
            for j in range(6):
                R_j = R[:, j, :].flatten()
                try:
                    v_new[j, :] = np.linalg.lstsq(B_v, R_j, rcond=None)[0]
                except:
                    break
            v = v_new
            
            # Fix u, v. Solve for w.
            # R[i, j, k] = sum_r u[i,r] v[j,r] w[k,r]
            # For fixed k, R[:, :, k] is 4x6.
            # sum_r w[k,r] (u[:,r] v[:,r]^T)
            B_w = np.zeros((24, 3))
            for r in range(3):
                B_w[:, r] = np.kron(u[:, r], v[:, r])
                
            w_new = np.zeros((6, 3))
            for k in range(6):
                R_k = R[:, :, k].flatten()
                try:
                    w_new[k, :] = np.linalg.lstsq(B_w, R_k, rcond=None)[0]
                except:
                    break
            w = w_new
            
            # Check if converged
            T_curr = np.zeros((4, 6, 6))
            for r in range(3):
                T_curr += np.einsum('i,j,k->ijk', u[:, r], v[:, r], w[:, r])
            if np.allclose(T_curr, R, atol=1e-8):
                # Found a solution!
                U_10 = np.hstack([U_strassen, u])
                V_10 = np.hstack([V_strassen, v])
                W_10 = np.hstack([W_strassen, w])
                found_rank_10 = True
                break
                
        if found_rank_10:
            break

    if found_rank_10 and check_solution(U_10, V_10, W_10):
        best_U, best_V, best_W = U_10, V_10, W_10
        best_rank = 10

    if time.time() > deadline - 10:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
