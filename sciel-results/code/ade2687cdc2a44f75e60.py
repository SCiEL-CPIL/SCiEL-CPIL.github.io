import numpy as np
from itertools import product
import time

def construct():
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    total = n * m * p  # 50
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Schoolbook solution (rank 50)
    U_school = np.zeros((nm, total), dtype=float)
    V_school = np.zeros((mp, total), dtype=float)
    W_school = np.zeros((pn, total), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        """Check if the decomposition is exact (within tolerance)"""
        # Vectorized verification
        # T_ijk = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We need T_ijk == 1 for all i,j,k
        
        # Reshape U to (n, m, rank)
        # Reshape V to (m, p, rank)
        # Reshape W to (p, n, rank)
        
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Compute sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a contraction.
        # First, compute M[i, j, k] = sum_c U[i,j,c] * V[j,k,c]
        # M = np.einsum('ijc,jkc->ijk', U_3d, V_3d)
        # Then T = sum_k M[i, j, k] * W[k, i, c]? No, W is indexed by k, i.
        # T[i, j] = sum_k (sum_c U[i,j,c] V[j,k,c]) W[k,i,c]
        # Note: W[k*n+i, c] is W_3d[k, i, c].
        
        # Let's do it step by step to be safe and fast enough
        # M[i, j, k] = sum_c U[i, j, c] * V[j, k, c]
        M = np.einsum('ijc,jkc->ijk', U_3d, V_3d)
        # T[i, j] = sum_k M[i, j, k] * W_3d[k, i, c] ??? No, c is summed in M already.
        # T[i, j] = sum_k M[i, j, k] * W_3d[k, i, :] ... wait, W depends on c?
        # The formula is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # So we must sum over c FIRST, or do it in one go.
        # T[i, j] = sum_k ( sum_c U[i,j,c] V[j,k,c] W[k,i,c] )
        
        # Using einsum: 'ijc,jkc,kic->ij'
        # U: i,j,c
        # V: j,k,c
        # W: k,i,c
        # Result: i,j
        
        T = np.einsum('ijc,jkc,kic->ij', U_3d, V_3d, W_3d)
        
        # Target is 1.0 for all i,j
        # Note: The matrix multiplication (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor decomposition computes (AB)_{ik} = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]?
        # Wait, the standard tensor decomposition for matrix mult is:
        # (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # The bilinear form is sum_c (sum_i A_{ij} U_{ij,c}) (sum_j B_{jk} V_{jk,c}) ... no.
        
        # Let's re-read the contract carefully.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # "decompose the matrix multiplication tensor"
        # Standard tensor $T_{ijk}$ for $A_{ij} B_{jk} \to C_{ik}$ is usually $T_{ijk} = \delta_{...}$?
        # Actually, the tensor is defined by $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # In tensor decomposition, we write $C_{ik} = \sum_c U_{i, \cdot, c} \cdot A \cdot V_{\cdot, k, c}$?
        # No, the task defines U, V, W shapes.
        # U: (n*m, rank) -> U[i*m+j, c]
        # V: (m*p, rank) -> V[j*p+k, c]
        # W: (p*n, rank) -> W[k*n+i, c]
        
        # The condition is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        # This implies that the computed product is:
        # sum_j sum_k ( sum_c U[i,j,c] V[j,k,c] W[k,i,c] ) A_{ij} B_{jk}
        # If the sum is 1 for all i,j,k, then the result is sum_j A_{ij} B_{jk} = C_{ik}.
        # So yes, the target tensor is all 1s.
        
        error = np.sum((T - 1.0)**2)
        return error < 1e-8

    # Try to find lower-rank solutions using gradient descent
    best_rank = 50
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try various ranks
    # We'll try ranks from 40 down to 20
    for rank in range(40, 14, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        found = False
        for attempt in range(20): # Increased restarts
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            np.random.seed(rank * 100 + attempt)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Reshape for vectorized ops
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            lr = 0.05
            for step in range(5000):
                if time.time() > deadline - 2:
                    break
                
                # Compute T = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # T_ijk is not needed, we need gradients.
                # Let S_c[i,j,k] = U[i,j,c] V[j,k,c] W[k,i,c]
                # T[i,j] = sum_k sum_c S_c[i,j,k]
                # This is hard to vectorize for gradients directly without large intermediate arrays if rank is high.
                # However, rank is small (<=40).
                
                # Gradient for U[i,j,c]:
                # dL/du = 2 * sum_{k} (T[i,j] - 1) * V[j,k,c] * W[k,i,c]
                # Note: T[i,j] depends on all c.
                
                # Let's compute T first.
                # T[i,j] = sum_k sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                
                # We can compute M_c[i,j,k] = U[i,j,c] V[j,k,c]
                # Then T_c[i,j] = sum_k M_c[i,j,k] W[k,i,c]
                # This is expensive.
                
                # Alternative: Use the fact that we are minimizing sum_{i,j} (T_{ij} - 1)^2.
                # Let's just use a simpler, slightly slower but robust update if vectorization is tricky.
                # But 5000 steps of slow loop is bad.
                
                # Let's try a vectorized gradient step.
                # G_U[i,j,c] = 2 * (T[i,j] - 1) * sum_k V[j,k,c] W[k,i,c]
                # G_V[j,k,c] = 2 * (T[i,j] - 1) * sum_i U[i,j,c] W[k,i,c]  <-- Wait, T is indexed by i,j.
                # The error term is (T[i,j] - 1).
                # dL/dU[i,j,c] = 2 * sum_k (T[i,j] - 1) V[j,k,c] W[k,i,c]
                # dL/dV[j,k,c] = 2 * sum_i (T[i,j] - 1) U[i,j,c] W[k,i,c]
                # dL/dW[k,i,c] = 2 * sum_j (T[i,j] - 1) U[i,j,c] V[j,k,c]
                
                # Step 1: Compute T[i,j]
                # T[i,j] = sum_k sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                
                # We can compute this using einsum
                # U: (n, m, r)
                # V: (m, p, r)
                # W: (p, n, r)
                
                # T[i,j] = sum_{k,c} U[i,j,c] V[j,k,c] W[k,i,c]
                # Let's contract V and W first?
                # HW[j,k,i,c] = V[j,k,c] W[k,i,c] -> (m, p, n, r)
                # Then T[i,j] = sum_{k,c} U[i,j,c] HW[j,k,i,c]
                
                # This creates a (m, p, n, r) array. 5*5*2*40 = 2000 elements. Small.
                HW = np.einsum('jkc,kic->jkic', V_3d, W_3d)
                
                # T[i,j] = sum_{k,c} U[i,j,c] HW[j,k,i,c]
                # U: (n, m, r) -> (i, j, c)
                # HW: (j, k, i, c)
                # We need to match indices:
                # T[i,j] = sum_{k,c} U[i,j,c] HW[j,k,i,c]
                # This is a bit of a permutation.
                # Let's permute HW to (j, i, k, c) ? No, i is the last index of HW?
                # HW[j,k,i,c]. We want to sum over k and c.
                # U[i,j,c] needs to be aligned with HW[j,k,i,c].
                
                # Let's use a different contraction.
                # T[i,j] = sum_c sum_k U[i,j,c] V[j,k,c] W[k,i,c]
                
                # Let's compute M_c[i,j,k] = U[i,j,c] V[j,k,c]
                # M = np.einsum('ijc,jkc->ijkc', U_3d, V_3d)
                # Then T[i,j] = sum_{k,c} M[i,j,k,c] W[k,i,c]
                # M: (n, m, p, r)
                # W: (p, n, r) -> (k, i, c)
                
                M = np.einsum('ijc,jkc->ijkc', U_3d, V_3d)
                # T[i,j] = sum_{k,c} M[i,j,k,c] W[k,i,c]
                # W is (p, n, r).
                # We need to sum M[i,j,k,c] * W[k,i,c] over k,c.
                # This is equivalent to:
                # T = np.einsum('ijkc,kic->ij', M, W_3d)
                
                T = np.einsum('ijkc,kic->ij', M, W_3d)
                
                error = np.sum((T - 1.0)**2)
                
                if error < 1e-10:
                    break
                
                # Gradients
                # G_U[i,j,c] = 2 * (T[i,j] - 1) * sum_k V[j,k,c] W[k,i,c]
                # Let S_U[j,i,c] = sum_k V[j,k,c] W[k,i,c]
                # S_U = np.einsum('jkc,kic->jic', V_3d, W_3d)
                # G_U[i,j,c] = 2 * (T[i,j] - 1) * S_U[j,i,c]
                # Note: T is (i,j). S_U is (j,i,c).
                # We need to align T[i,j] with S_U[j,i,c].
                # T[i,j] is at index i,j. S_U is at index j,i,c.
                # So G_U[i,j,c] = 2 * (T[i,j] - 1) * S_U[j,i,c]
                
                S_U = np.einsum('jkc,kic->jic', V_3d, W_3d)
                # T is (n, m). We need to broadcast (T[i,j]-1) to (n, m, 1) and multiply with S_U permuted?
                # S_U[j,i,c] has indices j, i, c.
                # We want G_U[i,j,c].
                # So we need to take S_U, swap first two axes?
                # S_U_perm[i,j,c] = S_U[j,i,c]
                S_U_perm = S_U.transpose(1, 0, 2) # (n, m, r) -> i, j, c
                
                # err_U[i,j] = 2 * (T[i,j] - 1)
                err_U = 2.0 * (T - 1.0)
                G_U = err_U[:, :, np.newaxis] * S_U_perm
                
                # G_V[j,k,c] = 2 * sum_i (T[i,j] - 1) U[i,j,c] W[k,i,c]
                # Let S_V[i,j,k,c] = U[i,j,c] W[k,i,c]
                # S_V = np.einsum('ijc,kic->ijkc', U_3d, W_3d)
                # G_V[j,k,c] = 2 * sum_i (T[i,j] - 1) S_V[i,j,k,c]
                # S_V is (n, m, p, r) -> i, j, k, c
                # T is (n, m) -> i, j
                # We sum over i.
                # G_V[j,k,c] = 2 * sum_i (T[i,j] - 1) U[i,j,c] W[k,i,c]
                
                S_V = np.einsum('ijc,kic->ijkc', U_3d, W_3d)
                # err_V[i,j] = 2 * (T[i,j] - 1)
                # G_V[j,k,c] = sum_i err_V[i,j] S_V[i,j,k,c]
                # This is a contraction over i.
                # G_V = np.einsum('ij,ijkc->jkc', err_V, S_V)
                
                G_V = np.einsum('ij,ijkc->jkc', err_V, S_V)
                
                # G_W[k,i,c] = 2 * sum_j (T[i,j] - 1) U[i,j,c] V[j,k,c]
                # Let S_W[i,j,k,c] = U[i,j,c] V[j,k,c]
                # S_W = np.einsum('ijc,jkc->ijkc', U_3d, V_3d)
                # G_W[k,i,c] = 2 * sum_j (T[i,j] - 1) S_W[i,j,k,c]
                
                S_W = np.einsum('ijc,jkc->ijkc', U_3d, V_3d)
                # G_W[k,i,c] = sum_j err_W[i,j] S_W[i,j,k,c]
                # err_W[i,j] = 2 * (T[i,j] - 1)
                # G_W = np.einsum('ij,ijkc->kic', err_W, S_W)
                
                G_W = np.einsum('ij,ijkc->kic', err_W, S_W)
                
                # Update
                U -= lr * G_U.reshape(nm, rank)
                V -= lr * G_V.reshape(mp, rank)
                W -= lr * G_W.reshape(pn, rank)
                
                # Update 3d views
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                if error < 1e-10:
                    break
            
            # Check if we found a solution
            if error < 1e-8:
                # Try to snap to integer/half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify(U_snap, V_snap, W_snap):
                    best_rank = rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    found = True
                    break
                else:
                    # Use the floating point solution if it's exact enough
                    if verify(U, V, W):
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        found = True
                        break
        
        if found:
            # Try to go even lower
            continue
    
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
