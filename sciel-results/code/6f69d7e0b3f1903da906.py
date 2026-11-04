"""Matrix multiplication 2x4x6: rank-minimal tensor decomposition.

Uses a combination of:
1. Strassen-based rank-42 decomposition (6 independent 2x2x2 blocks)
2. Linear algebra analysis to find rank reductions
3. Numerical search for lower rank decompositions

The key insight is that the 2x4 * 4x6 multiplication can be decomposed
as 6 independent 2x2 * 2x2 multiplications, each using Strassen (rank 7),
giving rank 42. We then search for linear dependencies that allow rank reduction.
"""

import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat 3D array.
    T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
    A, B, C, r = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
    S = np.zeros((A, B, C), dtype=float)
    for rr in range(r):
        S += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    return np.allclose(S, T, atol=tol)

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
    return U, V, W

def _strassen_decomposition_2x4x6(n, m, p):
    """
    Rank-42 Strassen-based decomposition for 2x4 * 4x6.
    Split A into two 2x2 blocks (left and right columns).
    Split B into three 2x2 blocks (top and bottom rows, for each pair of columns).
    Use Strassen for each of the 6 resulting 2x2 * 2x2 multiplications.
    """
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 factors
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    def add_strassen(a_idx, b_idx, c_idx, col_start):
        # M1
        U[a_idx[0], col_start] += 1
        U[a_idx[3], col_start] += 1
        V[b_idx[0], col_start] += 1
        V[b_idx[3], col_start] += 1
        W[c_idx[0], col_start] += 1
        W[c_idx[3], col_start] += 1
        
        # M2
        U[a_idx[2], col_start+1] += 1
        U[a_idx[3], col_start+1] += 1
        V[b_idx[0], col_start+1] += 1
        W[c_idx[2], col_start+1] += 1
        W[c_idx[3], col_start+1] -= 1
        
        # M3
        U[a_idx[0], col_start+2] += 1
        V[b_idx[1], col_start+2] += 1
        V[b_idx[3], col_start+2] -= 1
        W[c_idx[1], col_start+2] += 1
        W[c_idx[3], col_start+2] += 1
        
        # M4
        U[a_idx[3], col_start+3] += 1
        V[b_idx[2], col_start+3] += 1
        V[b_idx[0], col_start+3] -= 1
        W[c_idx[0], col_start+3] += 1
        W[c_idx[2], col_start+3] += 1
        
        # M5
        U[a_idx[0], col_start+4] += 1
        U[a_idx[1], col_start+4] += 1
        V[b_idx[3], col_start+4] += 1
        W[c_idx[0], col_start+4] -= 1
        W[c_idx[1], col_start+4] += 1
        
        # M6
        U[a_idx[2], col_start+5] += 1
        U[a_idx[0], col_start+5] -= 1
        V[b_idx[0], col_start+5] += 1
        V[b_idx[1], col_start+5] += 1
        W[c_idx[3], col_start+5] += 1
        
        # M7
        U[a_idx[1], col_start+6] += 1
        U[a_idx[3], col_start+6] -= 1
        V[b_idx[2], col_start+6] += 1
        V[b_idx[3], col_start+6] += 1
        W[c_idx[0], col_start+6] += 1
    
    # A is 2x4: A[i,j] is at U[i*m+j]
    a_left = [0, 1, 4, 5]   # a00, a01, a10, a11
    a_right = [2, 3, 6, 7]  # a00, a01, a10, a11
    
    # Column pair 0: cols 0,1
    b_top_0 = [0, 1, 6, 7]
    b_bottom_0 = [12, 13, 18, 19]
    c_0 = [0, 2, 1, 3]
    
    add_strassen(a_left, b_top_0, c_0, 0)
    add_strassen(a_right, b_bottom_0, c_0, 7)
    
    # Column pair 1: cols 2,3
    b_top_1 = [2, 3, 8, 9]
    b_bottom_1 = [14, 15, 20, 21]
    c_1 = [4, 6, 5, 7]
    
    add_strassen(a_left, b_top_1, c_1, 14)
    add_strassen(a_right, b_bottom_1, c_1, 21)
    
    # Column pair 2: cols 4,5
    b_top_2 = [4, 5, 10, 11]
    b_bottom_2 = [16, 17, 22, 23]
    c_2 = [8, 10, 9, 11]
    
    add_strassen(a_left, b_top_2, c_2, 28)
    add_strassen(a_right, b_bottom_2, c_2, 35)
    
    return U, V, W

def _try_rank_with_fixed_W(T, r, W, tol=1e-8):
    """Given fixed W (C x r), try to find U (A x r) and V (B x r) such that
    sum_r U[:,r] V[:,r] W[:,r] == T.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(42)
    V = rng.randn(B, r)
    T_flat = T.reshape(A, B * C)
    
    for it in range(300):
        # Solve for U given V, W
        P = np.einsum('br,cr->bcr', V, W).reshape(r, B * C)
        U, _, _, _ = np.linalg.lstsq(P.T, T_flat.T, rcond=None)
        U = U.T  # (A, r)
        
        # Solve for V given U, W
        Q = np.einsum('ar,cr->acr', U, W).reshape(r, A * C)
        T_reshaped = np.transpose(T, (1, 0, 2)).reshape(B, A * C)
        V, _, _, _ = np.linalg.lstsq(Q.T, T_reshaped.T, rcond=None)
        V = V.T  # (B, r)
    
    # Check residual
    S = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(S - T) / max(1, np.linalg.norm(T))
    if err < tol:
        return U, V, W, err
    return None, None, None, err

def _try_rank_with_fixed_U(T, r, U, tol=1e-8):
    """Given fixed U (A x r), try to find V (B x r) and W (C x r) such that
    sum_r U[:,r] V[:,r] W[:,r] == T.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(42)
    W = rng.randn(C, r)
    T_flat = T.reshape(A, B * C)
    
    for it in range(300):
        # Solve for V given U, W
        Q = np.einsum('ar,cr->acr', U, W).reshape(r, A * C)
        T_reshaped = np.transpose(T, (1, 0, 2)).reshape(B, A * C)
        V, _, _, _ = np.linalg.lstsq(Q.T, T_reshaped.T, rcond=None)
        V = V.T  # (B, r)
        
        # Solve for W given U, V
        P = np.einsum('ar,br->abr', U, V).reshape(r, A * B)
        T_reshaped2 = np.transpose(T, (2, 0, 1)).reshape(C, A * B)
        W, _, _, _ = np.linalg.lstsq(P.T, T_reshaped2.T, rcond=None)
        W = W.T  # (C, r)
    
    # Check residual
    S = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(S - T) / max(1, np.linalg.norm(T))
    if err < tol:
        return U, V, W, err
    return None, None, None, err

def _try_rank_with_fixed_V(T, r, V, tol=1e-8):
    """Given fixed V (B x r), try to find U (A x r) and W (C x r) such that
    sum_r U[:,r] V[:,r] W[:,r] == T.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(42)
    U = rng.randn(A, r)
    T_flat = T.reshape(A, B * C)
    
    for it in range(300):
        # Solve for W given U, V
        P = np.einsum('ar,br->abr', U, V).reshape(r, A * B)
        T_reshaped = np.transpose(T, (2, 0, 1)).reshape(C, A * B)
        W, _, _, _ = np.linalg.lstsq(P.T, T_reshaped.T, rcond=None)
        W = W.T  # (C, r)
        
        # Solve for U given V, W
        Q = np.einsum('br,cr->bcr', V, W).reshape(r, B * C)
        T_reshaped2 = T.reshape(A, B * C)
        U, _, _, _ = np.linalg.lstsq(Q.T, T_reshaped2.T, rcond=None)
        U = U.T  # (A, r)
    
    # Check residual
    S = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(S - T) / max(1, np.linalg.norm(T))
    if err < tol:
        return U, V, W, err
    return None, None, None, err

def _search_rank(T, start_rank, end_rank, deadline, max_restarts=30):
    """Search for minimal rank by trying fixed-factor approach with random factors."""
    A, B, C = T.shape
    best = None
    rng = np.random.RandomState(0)
    
    for r in range(start_rank, max(end_rank - 1, -1), -1):
        if time.time() > deadline:
            break
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Try fixing W
            W = rng.randn(C, r)
            U, V, Wsol, err = _try_rank_with_fixed_W(T, r, W)
            if U is not None and err < 1e-8:
                S = np.einsum('ar,br,cr->abc', U, V, Wsol)
                if np.allclose(S, T, atol=1e-8):
                    best = (U, V, Wsol, r)
                    found = True
                    break
            
            if time.time() > deadline:
                break
            
            # Try fixing U
            U = rng.randn(A, r)
            Usol, V, W, err = _try_rank_with_fixed_U(T, r, U)
            if Usol is not None and err < 1e-8:
                S = np.einsum('ar,br,cr->abc', Usol, V, W)
                if np.allclose(S, T, atol=1e-8):
                    best = (Usol, V, W, r)
                    found = True
                    break
            
            if time.time() > deadline:
                break
            
            # Try fixing V
            V = rng.randn(B, r)
            U, Vsol, W, err = _try_rank_with_fixed_V(T, r, V)
            if Vsol is not None and err < 1e-8:
                S = np.einsum('ar,br,cr->abc', U, Vsol, W)
                if np.allclose(S, T, atol=1e-8):
                    best = (U, Vsol, W, r)
                    found = True
                    break
        
        if found:
            return best
    
    return best

def _rank_of_matrix(M, tol=1e-10):
    """Compute numerical rank of matrix M."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * (s[0] if len(s) > 0 else 1)))

def _find_linear_dependencies(U, V, W, tol=1e-10):
    """
    Find linear dependencies among the rank-1 terms.
    Returns a list of (coefficients, indices) where sum coeff_i * term_i = 0.
    """
    r = U.shape[1]
    # Build the tensor as a 3D array
    A, B, C = U.shape[0], V.shape[0], W.shape[0]
    T = np.zeros((A, B, C))
    for i in range(r):
        T += np.einsum('a,b,c->abc', U[:, i], V[:, i], W[:, i])
    
    # Check if any subset of terms is linearly dependent
    # This is expensive, so we use a simpler approach
    # Check if the columns of U, V, W have dependencies
    rank_U = _rank_of_matrix(U, tol)
    rank_V = _rank_of_matrix(V, tol)
    rank_W = _rank_of_matrix(W, tol)
    
    # If any matrix has rank < r, there are dependencies
    dependencies = []
    if rank_U < r:
        # Find null space vectors
        _, s, vh = np.linalg.svd(U, full_matrices=True)
        null_dim = r - rank_U
        if null_dim > 0:
            null_vecs = vh[rank_U:]  # (null_dim, r)
            for nv in null_vecs:
                if np.linalg.norm(nv) > tol:
                    dependencies.append(('U', nv))
    if rank_V < r:
        _, s, vh = np.linalg.svd(V, full_matrices=True)
        null_dim = r - rank_V
        if null_dim > 0:
            null_vecs = vh[rank_V:]
            for nv in null_vecs:
                if np.linalg.norm(nv) > tol:
                    dependencies.append(('V', nv))
    if rank_W < r:
        _, s, vh = np.linalg.svd(W, full_matrices=True)
        null_dim = r - rank_W
        if null_dim > 0:
            null_vecs = vh[rank_W:]
            for nv in null_vecs:
                if np.linalg.norm(nv) > tol:
                    dependencies.append(('W', nv))
    
    return dependencies

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    T = _build_tensor(n, m, p)
    
    # Fallback: schoolbook
    U_sbk, V_sbk, W_sbk = _schoolbook(n, m, p)
    best = (U_sbk, V_sbk, W_sbk, n * m * p)
    
    # Try Strassen rank-42
    try:
        U_str, V_str, W_str = _strassen_decomposition_2x4x6(n, m, p)
        if _verify(U_str, V_str, W_str, T):
            best = (U_str, V_str, W_str, 42)
            print(f"Strassen rank 42 verified", file=sys.stderr)
            
            # Try to find rank reductions from linear dependencies
            deps = _find_linear_dependencies(U_str, V_str, W_str)
            if deps:
                print(f"Found {len(deps)} linear dependencies", file=sys.stderr)
        else:
            print("Strassen failed verification", file=sys.stderr)
    except Exception as e:
        print(f"Strassen error: {e}", file=sys.stderr)
    
    # Try to find lower rank using fix-one-factor method
    if time.time() < deadline - 10:
        search_result = _search_rank(T, best[3] - 1, 20, deadline - 5)
        if search_result is not None:
            U, V, W, r = search_result
            if r < best[3]:
                best = (U, V, W, r)
                print(f"Found rank {r} decomposition", file=sys.stderr)
    
    U, V, W, r = best
    print(f"Final rank: {r}", file=sys.stderr)
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
