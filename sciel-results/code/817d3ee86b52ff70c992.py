import time
import numpy as np
from itertools import combinations

# Constants for 2x3 * 3x3 matrix multiplication
_N, _M, _P = 2, 3, 3
_NM = _N * _M  # 6
_MP = _M * _P  # 9
_PN = _P * _N  # 6

# Index helpers (row-major)
def _u_idx(i, j):  return i * _M + j   # A[i,j]
def _v_idx(j, k):  return j * _P + k   # B[j,k]
def _w_idx(k, i):  return k * _N + i   # C[k,i]  (trace convention W[k*n+i, r])

# Known rank-17 fallback decomposition
_FALLBACK_RANK = 17

def _fallback_decomposition():
    """Return the known rank-17 decomposition as (U, V, W) numpy arrays."""
    r = _FALLBACK_RANK
    U = np.zeros((_NM, r))
    V = np.zeros((_MP, r))
    W = np.zeros((_PN, r))
    
    # Part 1: Strassen for the 2x2 x 2x2 block (A[:, 0:2] * B[0:2, :])
    # We'll use 0-indexed for the 2x2 block: A[0:2, 0:2], B[0:2, 0:2]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[_u_idx(0,0), 0] = 1; U[_u_idx(1,1), 0] = 1
    V[_v_idx(0,0), 0] = 1; V[_v_idx(1,1), 0] = 1
    W[_w_idx(0,0), 0] = 1; W[_w_idx(1,1), 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[_u_idx(1,0), 1] = 1; U[_u_idx(1,1), 1] = 1
    V[_v_idx(0,0), 1] = 1
    W[_w_idx(0,1), 1] = 1; W[_w_idx(1,1), 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[_u_idx(0,0), 2] = 1
    V[_v_idx(0,1), 2] = 1; V[_v_idx(1,1), 2] = -1
    W[_w_idx(1,0), 2] = 1; W[_w_idx(1,1), 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[_u_idx(1,1), 3] = 1
    V[_v_idx(1,0), 3] = 1; V[_v_idx(0,0), 3] = -1
    W[_w_idx(0,0), 3] = 1; W[_w_idx(0,1), 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[_u_idx(0,0), 4] = 1; U[_u_idx(0,1), 4] = 1
    V[_v_idx(1,1), 4] = 1
    W[_w_idx(0,0), 4] = -1; W[_w_idx(1,0), 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[_u_idx(1,0), 5] = 1; U[_u_idx(0,0), 5] = -1
    V[_v_idx(0,0), 5] = 1; V[_v_idx(0,1), 5] = 1
    W[_w_idx(1,1), 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[_u_idx(0,1), 6] = 1; U[_u_idx(1,1), 6] = -1
    V[_v_idx(1,0), 6] = 1; V[_v_idx(1,1), 6] = 1
    W[_w_idx(0,0), 6] = 1
    
    # Part 2: Schoolbook for cross terms A[:,2] * B[2,:]
    U[_u_idx(0,2), 7] = 1; V[_v_idx(2,0), 7] = 1; W[_w_idx(0,0), 7] = 1
    U[_u_idx(0,2), 8] = 1; V[_v_idx(2,1), 8] = 1; W[_w_idx(1,0), 8] = 1
    U[_u_idx(1,2), 9] = 1; V[_v_idx(2,0), 9] = 1; W[_w_idx(0,1), 9] = 1
    U[_u_idx(1,2), 10] = 1; V[_v_idx(2,1), 10] = 1; W[_w_idx(1,1), 10] = 1
    
    # Part 3: Schoolbook for B[:,2] * A rows -> C[:,2]
    U[_u_idx(0,0), 11] = 1; V[_v_idx(0,2), 11] = 1; W[_w_idx(2,0), 11] = 1
    U[_u_idx(0,1), 12] = 1; V[_v_idx(1,2), 12] = 1; W[_w_idx(2,0), 12] = 1
    U[_u_idx(0,2), 13] = 1; V[_v_idx(2,2), 13] = 1; W[_w_idx(2,0), 13] = 1
    U[_u_idx(1,0), 14] = 1; V[_v_idx(0,2), 14] = 1; W[_w_idx(2,1), 14] = 1
    U[_u_idx(1,1), 15] = 1; V[_v_idx(1,2), 15] = 1; W[_w_idx(2,1), 15] = 1
    U[_u_idx(1,2), 16] = 1; V[_v_idx(2,2), 16] = 1; W[_w_idx(2,1), 16] = 1
    
    return U, V, W


def _validate(U, V, W):
    """
    Check if (U, V, W) exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    
    # Tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Expected: T[_u_idx(i,j), _v_idx(j,k), _w_idx(k,i)] = 1 for all i,j,k
    #           0 otherwise
    
    # Build the decomposition tensor
    # U is (NM, r), V is (MP, r), W is (PN, r)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # We can compute this as a sum of rank-1 tensors
    # For efficiency, check the 18 required non-zero entries and a subset of zeros
    
    # First, check all 18 required entries
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                a = _u_idx(i, j)
                b = _v_idx(j, k)
                c = _w_idx(k, i)
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that entries where j != middle index are 0
    # The "middle" index in (a,b) correspondence is: a = i*M+j, b = j'*P+k
    # The decomposition is correct if and only if for all (a,b,c), 
    # T[a,b,c] = 1 if a's j == b's j and c = k*N+i where a=i*M+j, b=j*P+k, c=k*N+i
    # T[a,b,c] = 0 otherwise
    
    # Let's check all zero entries
    for a in range(_NM):
        i_a = a // _M
        j_a = a % _M
        for b in range(_MP):
            j_b = b // _P
            k_b = b % _P
            # If j_a != j_b, all c should be 0
            if j_a != j_b:
                for c in range(_PN):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
            else:
                # j_a == j_b == j
                i_val = i_a
                k_val = k_b
                c_expected = _w_idx(k_val, i_val)
                for c in range(_PN):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if c == c_expected:
                        if abs(val - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(val) > 1e-9:
                            return False
    return True

def _try_rank_16_search(deadline):
    """
    Attempt to find a rank-16 decomposition using random restarts and local optimization.
    Returns (U, V, W) if found, else None.
    """
    T = np.zeros((_NM, _MP, _PN))
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                T[_u_idx(i, j), _v_idx(j, k), _w_idx(k, i)] = 1.0
    
    # Pre-compute flattened indices for faster access if needed, but np.einsum is fast enough for small sizes
    # Let's use a simple alternating least squares approach
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W with random values
        # Using small random values to avoid blowup
        U = np.random.randn(_NM, 16) * 0.5
        V = np.random.randn(_MP, 16) * 0.5
        W = np.random.randn(_PN, 16) * 0.5
        
        # Run ALS for a few iterations
        for it in range(100):
            if time.time() > deadline:
                return None
            
            # Update U
            # T = sum U_r V_r W_r
            # For each column r of U, we want to minimize || T - sum U_r V_r W_r ||_F
            # This is equivalent to solving a linear system for U[:, r]
            # However, since it's multilinear, we fix V and W and solve for U.
            # T_reshaped (NM, MP*PN) = U @ (V ⊗ W)^T ? No.
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Fix V, W. For a fixed i, T[i, :, :] = sum_r U[i, r] (V[:, r] ⊗ W[:, r])
            # This is a linear system for U[i, :]
            
            # Let's vectorize:
            # T_flat (NM, MP*PN)
            # A_r (MP*PN, 1) = V[:, r] ⊗ W[:, r]
            # T_flat = U @ A.T where A is (16, MP*PN)
            
            # Construct A matrix (16, MP*PN)
            A = np.zeros((16, _MP * _PN))
            for r in range(16):
                A[r, :] = np.kron(V[:, r], W[:, r]).flatten()
            
            T_flat = T.reshape(_NM, _MP * _PN)
            
            # Solve U = T_flat @ A.T @ inv(A @ A.T)
            # Least squares: U = argmin || T_flat - U A ||_F
            # U = T_flat @ A.T @ (A @ A.T)^{-1}
            try:
                ATA = A @ A.T
                # Add small regularization for stability
                ATA += 1e-6 * np.eye(16)
                U_new = T_flat @ A.T @ np.linalg.inv(ATA)
                U = U_new
            except np.linalg.LinAlgError:
                pass
            
            # Update V
            # T_flat (MP, NM*PN)
            B = np.zeros((16, _NM * _PN))
            for r in range(16):
                B[r, :] = np.kron(U[:, r], W[:, r]).flatten()
            
            T_flat_V = T.transpose(1, 0, 2).reshape(_MP, _NM * _PN)
            
            try:
                BTA = B @ B.T
                BTA += 1e-6 * np.eye(16)
                V_new = T_flat_V @ B.T @ np.linalg.inv(BTA)
                V = V_new
            except np.linalg.LinAlgError:
                pass
            
            # Update W
            # T_flat (PN, NM*MP)
            C_mat = np.zeros((16, _NM * _MP))
            for r in range(16):
                C_mat[r, :] = np.kron(U[:, r], V[:, r]).flatten()
            
            T_flat_W = T.transpose(2, 0, 1).reshape(_PN, _NM * _MP)
            
            try:
                CTA = C_mat @ C_mat.T
                CTA += 1e-6 * np.eye(16)
                W_new = T_flat_W @ C_mat.T @ np.linalg.inv(CTA)
                W = W_new
            except np.linalg.LinAlgError:
                pass
            
            # Check convergence
            # Compute residual
            # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            # res = np.linalg.norm(T - T_approx)
            # If res < 1e-6, we might have found it.
            # But we need exact integers/half-integers.
            # We'll check validation at the end of each restart or periodically.
            
            if it % 10 == 0:
                # Round to nearest half-integer to check if we are close to an exact solution
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _validate(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded

    return None

def construct():
    deadline = time.time() + 115.0  # Leave 5s margin
    
    # Start with the known rank-17 solution
    U_best, V_best, W_best = _fallback_decomposition()
    best_rank = _FALLBACK_RANK
    
    if not _validate(U_best, V_best, W_best):
        raise RuntimeError("Fallback decomposition failed validation")
    
    # Try to find a rank-16 decomposition
    result = _try_rank_16_search(deadline)
    if result is not None:
        U_best, V_best, W_best = result
        best_rank = 16
    
    # Final validation
    if not _validate(U_best, V_best, W_best):
        # Fall back to the known solution
        U_best, V_best, W_best = _fallback_decomposition()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
