"""Matrix multiplication 2x4x7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_rank50():
    """Build a verified rank-50 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 2x4x4: 4 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    # 2x4x2: 2 Strassen blocks
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    # 2x4x1: 8 direct terms
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1

    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify: for all i,k,j,j': sum_c W[k*n+i,c]*U[i*m+j,c]*V[j*p+kp,c] == delta_{j,j'}."""
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    val = 0.0
                    for c in range(r):
                        val += W[k*n+i, c] * U[i*m+j, c] * V[j*p+kp, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using einsum."""
    r = U.shape[1]
    # Build tensor T[i,j,j',k] = sum_c W[k*n+i,c] U[i*m+j,c] V[j*p+kp,c]
    # We need T[i,j,j',k] == delta_{j,j'} for all i,k
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # For each (i,k): sum_c W3[k,i,c] U3[i, :, c] V3[:, k, c] should be I_m
    # T[i,k,j,j'] = sum_c W3[k,i,c] U3[i,j,c] V3[j',k,c]
    # = einsum('kic,ijc,j'kc->ijkj', ...) - but that's 4D. Let's do per (i,k).
    for i in range(n):
        for k in range(p):
            # M[j,j'] = sum_c W3[k,i,c] U3[i,j,c] V3[j',k,c]
            # = (W3[k,i,:] * U3[i,:,:].T) @ V3[:,k,:].T  -> shape (m, m)
            M = (W3[k, i, :] * U3[i, :, :]).reshape(1, -1) @ V3[:, k, :].T
            expected = np.eye(m)
            if not np.allclose(M, expected, atol=tol):
                return False
    return True


def _newton_reduce(U, V, W, n, m, p, target_rank, max_iters=50, tol=1e-9):
    """
    Try to reduce rank to target_rank by Newton iterations.
    Strategy: truncate to target_rank, then alternate fixing two factors and solving
    for the third via linear least squares.
    
    The equation is: for all i,k: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+kp,c] = delta_{j,j'}
    
    If we fix U and V, the equation is linear in W.
    If we fix U and W, the equation is linear in V.
    If we fix V and W, the equation is linear in U.
    """
    r = target_rank
    U0 = U[:, :r].copy()
    V0 = V[:, :r].copy()
    W0 = W[:, :r].copy()
    
    # Build the target tensor
    # T[i,k,j,j'] = delta_{j,j'}
    # We need: sum_c W[k*n+i,c] U[i*m+j,c] V[j*p+kp,c] = T[i,k,j,j']
    
    # Reshape for convenience
    def reshape_U(Um):
        return Um.reshape(n, m, r)
    def reshape_V(Vm):
        return Vm.reshape(m, p, r)
    def reshape_W(Wm):
        return Wm.reshape(p, n, r)
    
    def unreshape_U(U3):
        return U3.reshape(n*m, r)
    def unreshape_V(V3):
        return V3.reshape(m*p, r)
    def unreshape_W(W3):
        return W3.reshape(p*n, r)
    
    U3 = reshape_U(U0)
    V3 = reshape_V(V0)
    W3 = reshape_W(W0)
    
    best_err = np.inf
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    for it in range(max_iters):
        # Solve for W given U, V:
        # For each (i,k): sum_c W[k*n+i,c] * (U[i,m,j,c] * V[j',p,k,c]) = delta_{j,j'}
        # Let A[i,k,j,j',c] = U[i,j,c] * V[j',k,c]  (shape n,p,m,m,r)
        # Then for each (i,k): sum_c W[k*n+i,c] * A[i,k,j,j',c] = delta_{j,j'}
        # This is a linear system in W for each (i,k).
        
        # Build A: A[i,k,j,j',c] = U3[i,j,c] * V3[j',k,c]
        # Shape: (n, p, m, m, r)
        A = U3[:, :, None, :, :] * V3[None, :, :, None, :]  # (n, m, 1, m, r) * (1, p, m, 1, r) -> need to be careful
        # Actually: U3[i,j,c] has shape (n,m,r), V3[j',k,c] has shape (m,p,r)
        # A[i,k,j,j',c] = U3[i,j,c] * V3[j',k,c]
        # = U3[:, :, None, :, :] * V3[None, None, :, :, :] won't work directly due to axes
        # Let's do it explicitly
        A = np.zeros((n, p, m, m, r))
        for c in range(r):
            A[:, :, :, :, c] = U3[:, :, c][:, None, :, None] * V3[:, :, c][None, :, None, :]
            # U3[:, :, c] is (n, m), V3[:, :, c] is (m, p)
            # A[i,k,j,j',c] = U3[i,j,c] * V3[j',k,c]
            # = U3[:, :, c][:, None, :, None] * V3[:, :, c][None, :, None, :]
            # Shape: (n, 1, m, 1) * (1, p, 1, m) -> (n, p, m, m)
        
        # For each (i,k): A[i,k,j,j',c] * W[k*n+i,c] = delta_{j,j'}
        # This is: sum_c A[i,k,j,j',c] * W[k*n+i,c] = I_m[j,j']
        # So for each (i,k), we have a system: A_ik @ W_ik = I_m
        # where A_ik has shape (m*m, r) and W_ik has shape (r,)
        # If r >= m*m, we can solve least squares.
        
        # Actually, m*m = 16, and r could be as low as ~16.
        # For each (i,k), solve: A_ik @ w = I_m.flatten()
        # A_ik is (m*m, r), w is (r,), I_m.flatten() is (m*m,)
        
        # Let's solve for W given U, V
        W_new3 = W3.copy()
        for i in range(n):
            for k in range(p):
                A_ik = A[i, k, :, :, :].reshape(m*m, r)
                b = np.eye(m).flatten()
                # Solve A_ik @ w = b in least squares
                if r >= m*m:
                    w, _, _, _ = np.linalg.lstsq(A_ik, b, rcond=None)
                else:
                    # Underdetermined: use minimum norm solution
                    w = A_ik.T @ np.linalg.pinv(A_ik) @ b
                W_new3[k, i, :] = w
        
        W3 = W_new3
        W0 = unreshape_W(W3)
        
        # Solve for U given V, W:
        # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+kp,c] = delta_{j,j'}
        # For fixed i: sum_c U[i*m+j,c] * (W[k*n+i,c] * V[j*p+kp,c]) = delta_{j,j'} for all k,j,j'
        # Let B[i,j,j',k,c] = W[k*n+i,c] * V[j*p+kp,c]
        # Then for each i: sum_c U[i*m+j,c] * B[i,j,j',k,c] = delta_{j,j'} for all k
        # This is a linear system in U for each i.
        
        # Actually, let's fix V and W and solve for U.
        # For each i: we need sum_c U[i,j,c] * (sum over k of W[k,i,c]*V[j',k,c]) ... no
        # Let me re-derive.
        # C[i,k] = sum_j sum_j' U[i,j,c] V[j',k,c] W[i,k,c] summed over c
        # We need this to equal sum_j A[i,j] B[j,k] for all A,B
        # Which means: for all i,k,j,j': sum_c W[i,k,c] U[i,j,c] V[j',k,c] = delta_{j,j'}
        
        # Fix V and W. For each i, the equation involves U[i,j,c] for j=0..m-1, c=0..r-1.
        # For a fixed i, the equations are: for all k, j, j':
        #   sum_c W[k*n+i,c] U[i*m+j,c] V[j*p+kp,c] = delta_{j,j'}
        # Let's group by (j, c): unknowns are U[i*m+j, c]
        # For each (j, j', k): sum_c U[i*m+j,c] * (W[k*n+i,c] * V[j*p+kp,c]) = delta_{j,j'}
        # So for fixed i, j, j': sum_c U[i*m+j,c] * D[i,j,j',k,c] = delta_{j,j'} where D[i,j,j',k,c] = W[k*n+i,c]*V[j*p+kp,c]
        # Wait, j is fixed in the unknown, but appears in D too.
        
        # Let me just do it numerically. For fixed i, build a linear system.
        # Unknowns: U[i, j, c] for j in 0..m-1, c in 0..r-1. Total m*r unknowns.
        # Equations: for each k in 0..p-1, j in 0..m-1, j' in 0..m-1:
        #   sum_c W[k*n+i,c] * U[i,j,c] * V[j*p+kp,c] = delta_{j,j'}
        # Number of equations: p * m * m
        # Number of unknowns: m * r
        
        U_new3 = U3.copy()
        for i in range(n):
            # Build system matrix
            # Unknowns indexed by (j, c), total m*r
            # Equation indexed by (k, j, j'), total p*m*m
            # Coefficient of U[i,j,c] in equation (k,j,j') is W[k*n+i,c]*V[j*p+kp,c]
            # Note: the j in the equation is the same as the j in the unknown? 
            # Yes! The equation for (k, j, j') only involves U[i, j, c] (not other j values).
            # So the system is block diagonal in j!
            # For each j: unknowns are U[i,j,c] for c=0..r-1 (r unknowns)
            # Equations for this j: for each k, j': sum_c W[k*n+i,c]*U[i,j,c]*V[j*p+kp,c] = delta_{j,j'}
            # So for fixed j: r unknowns, p*m equations
            for j in range(m):
                # Build A_mat: (p*m, r)
                # A_mat[k*m + j', c] = W[k*n+i,c] * V[j*p+kp,c]
                A_mat = np.zeros((p*m, r))
                for k in range(p):
                    for jp in range(m):
                        A_mat[k*m+jp, :] = W3[k, i, :] * V3[j, k, :]
                b = np.zeros(p*m)
                for jp in range(m):
                    if j == jp:
                        b[:] += 1.0  # delta_{j,j'} = 1 for all k when j==j'
                        # Wait, delta_{j,j'} is 1 if j==j' else 0, independent of k
                        # So b[k*m+jp] = 1 if j==jp else 0
                b = np.zeros(p*m)
                for k in range(p):
                    for jp in range(m):
                        b[k*m+jp] = 1.0 if j == jp else 0.0
                
                if r >= p*m:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                else:
                    sol = A_mat.T @ np.linalg.pinv(A_mat) @ b
                U_new3[i, j, :] = sol
        
        U3 = U_new3
        U0 = unreshape_U(U3)
        
        # Solve for V given U, W:
        # For each k: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+kp,c] = delta_{j,j'} for all i,j,j'
        # Fix U and W. For each k, unknowns are V[j, k, c] for j=0..m-1, c=0..r-1. Total m*r.
        # Equations: for all i, j, j': sum_c W[k*n+i,c] U[i*m+j,c] V[j*p+kp,c] = delta_{j,j'}
        # Note: V[j*p+kp,c] appears with j as the first index of V.
        # For fixed k, the unknown is V[j, k, c] for j in 0..m-1.
        # Equation (i, j, j'): sum_c W[k*n+i,c] * U[i,j,c] * V[j,k,c] = delta_{j,j'}
        # Again, j in the unknown matches j in the equation. Block diagonal in j!
        # For fixed k, j: unknowns V[j,k,c] for c=0..r-1 (r unknowns)
        # Equations: for all i, j': sum_c W[k*n+i,c] * U[i,j,c] * V[j,k,c] = delta_{j,j'}
        # So for fixed (k,j): r unknowns, n*m equations
        
        V_new3 = V3.copy()
        for k in range(p):
            for j in range(m):
                A_mat = np.zeros((n*m, r))
                for i in range(n):
                    for jp in range(m):
                        A_mat[i*m+jp, :] = W3[k, i, :] * U3[i, j, :]
                b = np.zeros(n*m)
                for i in range(n):
                    for jp in range(m):
                        b[i*m+jp] = 1.0 if j == jp else 0.0
                
                if r >= n*m:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                else:
                    sol = A_mat.T @ np.linalg.pinv(A_mat) @ b
                V_new3[j, k, :] = sol
        
        V3 = V_new3
        V0 = unreshape_V(V3)
        
        # Check error
        if it % 5 == 0 or it == max_iters - 1:
            if _verify_fast(U0, V0, W0, n, m, p, tol=1e-9):
                return U0, V0, W0
            # Compute error
            err = 0.0
            for i in range(n):
                for k in range(p):
                    M = (W3[k, i, :] * U3[i, :, :]).reshape(1, -1) @ V3[:, k, :].T
                    err += np.max(np.abs(M - np.eye(m)))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
            if best_err < 1e-12:
                break
    
    # Check best
    if _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-9):
        return best_U, best_V, best_W
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 56
    
    # Build rank-50 Strassen decomposition
    U50, V50, W50, rank50 = _build_rank50()
    
    # Verify rank-50
    if not _verify_fast(U50, V50, W50, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    
    # Try to reduce rank using Newton iterations
    for target_rank in [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations for the truncated factors
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Truncate with some random perturbation
            if attempt == 0:
                U_t, V_t, W_t = U50[:, :target_rank].copy(), V50[:, :target_rank].copy(), W50[:, :target_rank].copy()
            else:
                np.random.seed(attempt * 1000)
                U_t = U50[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.01
                V_t = V50[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.01
                W_t = W50[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.01
            
            result = _newton_reduce(U_t, V_t, W_t, n, m, p, target_rank, max_iters=100)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we couldn't find target_rank, try a few more and stop
            if target_rank < 47:
                break
    
    # Final verification
    if not _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fall back to rank-50
        best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
        best_rank = rank50
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
