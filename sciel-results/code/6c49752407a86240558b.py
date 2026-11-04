"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    Uses structured decomposition with known low-rank building blocks.
    """
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    
    # Start with schoolbook as fallback
    rank_sb = n * m * p  # 42
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def snap_to_half(arr):
        """Snap values to nearest half-integer."""
        snapped = np.round(arr * 2) / 2
        return snapped
    
    def verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
        """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_list(arr):
        """Convert numpy array to list with integer/half-integer entries."""
        result = []
        for row in arr:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                elif abs(2 * v - round(2 * v)) < 1e-9:
                    r.append(int(round(2 * v)) / 2.0)
                else:
                    r.append(float(v))
            result.append(r)
        return result
    
    # Strategy: Use block decomposition
    # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    # Use rank-5 for 2x3x3 blocks, rank-6 for 2x3x1 block
    # Total rank: 5 + 5 + 6 = 16
    
    # Rank-5 decomposition for 2x3x3 multiplication
    # A is 2x3, B is 3x3, C = A*B is 2x3
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Known rank-5 algorithm for 2x3x3:
    # P1 = A[0,0] * (B[0,0] + B[0,1] + B[0,2])
    # P2 = A[0,1] * (B[1,0] + B[1,1] + B[1,2])
    # P3 = A[0,2] * (B[2,0] + B[2,1] + B[2,2])
    # P4 = (A[0,0] + A[1,0]) * (B[0,0] + B[0,1] + B[0,2])
    # P5 = A[1,0] * (B[0,0] + B[0,1] + B[0,2])
    # This doesn't work cleanly. Let me use a different known decomposition.
    
    # Actually, let me try a more systematic approach using ALS with better initialization.
    
    # Try to find a low-rank decomposition using numerical optimization
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ranks from 12 up to schoolbook
    for rank in range(12, rank_sb):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + rank * 100 + attempt)
            U_r = np.random.randn(nm, rank)
            V_r = np.random.randn(mp, rank)
            W_r = np.random.randn(pn, rank)
            
            # ALS optimization
            max_iter = 500
            U_work = U_r.copy()
            V_work = V_r.copy()
            W_work = W_r.copy()
            
            for it in range(max_iter):
                if time.time() > deadline - 25:
                    break
                
                # Update W: fix U and V
                # For each (k,i): sum_c W[k,i,c] * sum_j U[i,j,c] * V[j,k,c] = 1
                # M[k,i,c] = sum_j U[i,j,c] * V[j,k,c]
                # W[k,i,:] = M[k,i,:] / (M[k,i,:] · M[k,i,:])
                
                U3 = U_work.reshape(n, m, rank)
                V3 = V_work.reshape(m, p, rank)
                
                # M[k,i,c] = sum_j U3[i,j,c] * V3[j,k,c]
                M = np.einsum('ijc,jkc->kic', U3, V3)  # shape (p, n, rank)
                
                # For each (k,i), solve: M[k,i,:] · W[k,i,:] = 1
                # Optimal: W[k,i,:] = M[k,i,:] / ||M[k,i,:]||^2
                M_norm_sq = np.sum(M ** 2, axis=2)  # shape (p, n)
                M_norm_sq[M_norm_sq < 1e-12] = 1e-12
                W_work = M / M_norm_sq[:, :, np.newaxis]
                
                # Update U: fix V and W
                # For each (i,j): sum_c U[i,j,c] * sum_k V[j,k,c] * W[k,i,c] = 1
                # N[i,j,c] = sum_k V[j,k,c] * W[k,i,c]
                # U[i,j,:] = N[i,j,:] / (N[i,j,:] · N[i,j,:])
                
                V3 = V_work.reshape(m, p, rank)
                W3 = W_work.reshape(p, n, rank)
                
                # N[i,j,c] = sum_k V3[j,k,c] * W3[k,i,c]
                N = np.einsum('jkc,kic->ijc', V3, W3)  # shape (n, m, rank)
                
                N_norm_sq = np.sum(N ** 2, axis=2)  # shape (n, m)
                N_norm_sq[N_norm_sq < 1e-12] = 1e-12
                U_work = N / N_norm_sq[:, :, np.newaxis]
                
                # Update V: fix U and W
                # For each (j,k): sum_c V[j,k,c] * sum_i U[i,j,c] * W[k,i,c] = 1
                # P[j,k,c] = sum_i U[i,j,c] * W[k,i,c]
                # V[j,k,:] = P[j,k,:] / (P[j,k,:] · P[j,k,:])
                
                U3 = U_work.reshape(n, m, rank)
                W3 = W_work.reshape(p, n, rank)
                
                # P[j,k,c] = sum_i U3[i,j,c] * W3[k,i,c]
                P = np.einsum('ijc,kic->jkc', U3, W3)  # shape (m, p, rank)
                
                P_norm_sq = np.sum(P ** 2, axis=2)  # shape (m, p)
                P_norm_sq[P_norm_sq < 1e-12] = 1e-12
                V_work = P / P_norm_sq[:, :, np.newaxis]
            
            # Check if we found a valid decomposition
            U_s = snap_to_half(U_work)
            V_s = snap_to_half(V_work)
            W_s = snap_to_half(W_work)
            
            if verify_decomposition(U_s, V_s, W_s, n, m, p, rank, tol=1e-4):
                best_rank = rank
                best_U, best_V, best_W = U_s, V_s, W_s
                break
        
        if best_rank == rank:
            break
    
    # Return the best decomposition found
    if best_rank < rank_sb:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        # Fall back to schoolbook
        return {
            "U": to_list(U_sb),
            "V": to_list(V_sb),
            "W": to_list(W_sb)
        }
# EVOLVE-BLOCK-END
