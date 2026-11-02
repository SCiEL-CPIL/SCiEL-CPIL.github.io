"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 with search for rank 27."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def build_strassen_28():
        """Build the rank-28 Strassen decomposition for 2x4 x 4x4."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
        V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
        W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

        # Strassen 7 terms for 2x2 matrix multiplication
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Local 2x2 index mapping: (r,c) -> r*2+c
        # U local: A[i, j] where i in {0,1}, j in {0,1} -> flat: i*2+j
        # V local: B[i, j] where i in {0,1}, j in {0,1} -> flat: i*2+j
        # W local: C[i, j] where i in {0,1}, j in {0,1} -> flat: i*2+j
        
        # Strassen terms as (u_indices, v_indices, w_indices) with coefficients
        strassen_terms = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        col = 0
        # 4 block products: A_L*B_TL, A_R*B_BL -> C_L; A_L*B_TR, A_R*B_BR -> C_R
        for b_col_block in range(2):  # 0: left cols of C, 1: right cols of C
            for a_col_block in range(2):  # 0: left cols of A, 1: right cols of A
                # A block: columns 2*a_col_block + {0,1}
                # B block: rows 2*a_col_block + {0,1}, columns 2*b_col_block + {0,1}
                # C block: rows {0,1}, columns 2*b_col_block + {0,1}
                
                a_base = 2 * a_col_block  # column offset in A
                b_row_base = 2 * a_col_block  # row offset in B
                b_col_base = 2 * b_col_block  # column offset in B
                c_col_base = 2 * b_col_block  # column offset in C
                
                for u_t, v_t, w_t in strassen_terms:
                    # U: A[i, a_base + j] -> flat index i*4 + (a_base + j)
                    for loc, coeff in u_t:
                        r = loc // 2
                        c = loc % 2
                        U[r * 4 + a_base + c, col] += coeff
                    
                    # V: B[b_row_base + r, b_col_base + c] -> flat index (b_row_base+r)*4 + (b_col_base+c)
                    for loc, coeff in v_t:
                        r = loc // 2
                        c = loc % 2
                        V[(b_row_base + r) * 4 + b_col_base + c, col] += coeff
                    
                    # W: C[r, c_col_base + c] -> flat index (c_col_base+c)*2 + r
                    for loc, coeff in w_t:
                        r = loc // 2
                        c = loc % 2
                        W[(c_col_base + c) * 2 + r, col] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    def validate_numpy(U, V, W):
        """Validate decomposition using numpy. Returns (valid, max_error)."""
        rank = U.shape[1]
        max_err = 0.0
        
        # Check all entries of the tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Should be 1
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    max_err = max(max_err, abs(s - 1.0))
        
        # Check off-diagonal entries should be 0
        for a in range(n * m):
            i1 = a // m
            j1 = a % m
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                for c in range(p * n):
                    i2 = c // n
                    k2 = c % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    u_row = U[a]
                    v_row = V[b]
                    w_row = W[c]
                    s = np.dot(u_row, v_row * w_row)
                    max_err = max(max_err, abs(s))
        
        return max_err < 1e-9, max_err

    def tensor_from_decomp(U, V, W):
        """Compute the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        return T

    def compute_rank_lower_bound(T):
        """
        Compute a lower bound on the rank of the 3-tensor T.
        Uses the slice rank method: for each slice direction, compute the rank
        of the matrix obtained by fixing one index.
        
        The slice rank of T is min over all ways to write T as sum of slices.
        A simple lower bound: the rank of any matrix flattening.
        
        For a 3-tensor T[i,j,k], the rank is at least max(rank(T_{i,:,:}), rank(T_{:,j,:}), rank(T_{:,:,k}))
        where these are the matrix unfoldings.
        """
        # Matrix unfolding along i: shape (n, m*p)
        T_i = np.zeros((n, m * p))
        T_j = np.zeros((m, n * p))
        T_k = np.zeros((p, n * m))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_i[i, j * p + k] = T[i, j, k]
                    T_j[j, i * p + k] = T[i, j, k]
                    T_k[k, i * m + j] = T[i, j, k]
        
        rank_i = np.linalg.matrix_rank(T_i, tol=1e-10)
        rank_j = np.linalg.matrix_rank(T_j, tol=1e-10)
        rank_k = np.linalg.matrix_rank(T_k, tol=1e-10)
        
        return max(rank_i, rank_j, rank_k)

    def check_rank27_feasibility(U28, V28, W28):
        """
        Check if rank 27 is feasible by looking for linear dependencies among the 28 tensor slices.
        Each column r defines a rank-1 tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        We need to check if any one of these 28 tensors is in the span of the other 27.
        
        Returns True if rank 27 might be achievable (i.e., the 28 tensors are linearly dependent).
        """
        rank = 28
        # Build the matrix of tensor slices
        # Each slice is a vector in R^(n*m*p) = R^32
        slices = np.zeros((rank, n * m * p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        slices[r, idx] = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
        
        # Check if the 28 vectors are linearly dependent
        # If rank of the 28x32 matrix is < 28, then they are dependent
        mat_rank = np.linalg.matrix_rank(slices, tol=1e-10)
        return mat_rank < rank, mat_rank

    # Build baseline rank-28
    base = build_strassen_28()
    best_result = {
        "U": base["U"].tolist(),
        "V": base["V"].tolist(),
        "W": base["W"].tolist()
    }
    best_rank = 28
    best_err = 0.0
    
    # Verify baseline
    valid, err = validate_numpy(base["U"], base["V"], base["W"])
    if not valid:
        return best_result
    
    # Compute tensor and check rank lower bound
    T = tensor_from_decomp(base["U"], base["V"], base["W"])
    lb = compute_rank_lower_bound(T)
    
    # Check if the 28 Strassen tensors are linearly independent
    U28 = base["U"].copy()
    V28 = base["V"].copy()
    W28 = base["W"].copy()
    
    dependent, mat_rank = check_rank27_feasibility(U28, V28, W28)
    
    # If the tensors are linearly independent, rank 27 is impossible for this decomposition
    # But a different rank-27 decomposition might exist. Still, this gives us information.
    
    # Try to find rank 27
    target_rank = 27
    found = False
    
    if dependent:
        # The Strassen tensors are linearly dependent, so we might be able to reduce rank
        # Try to find which tensor is in the span of the others
        for remove_col in range(28):
            if time.time() > deadline - 20:
                break
            
            keep_cols = [c for c in range(28) if c != remove_col]
            
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                seed = 12345 + remove_col * 1000 + attempt * 7919
                np.random.seed(seed)
                
                # Initialize from Strassen with removed column + perturbation
                scale = 0.3 + 0.05 * attempt
                U = U28[:, keep_cols].copy() + np.random.randn(n*m, target_rank) * scale
                V = V28[:, keep_cols].copy() + np.random.randn(m*p, target_rank) * scale
                W = W28[:, keep_cols].copy() + np.random.randn(p*n, target_rank) * scale
                
                # Alternating least squares with adaptive step
                for step in range(40):
                    if time.time() > deadline - 5:
                        break
                    
                    # Update U rows
                    for a in range(n*m):
                        i1 = a // m
                        j1 = a % m
                        M = np.zeros((m*p*p*n, target_rank))
                        rhs = np.zeros(m*p*p*n)
                        for b in range(m*p):
                            j2 = b // p
                            k1 = b % p
                            for c in range(p*n):
                                i2 = c // n
                                k2 = c % n
                                M[b*p*n + c, :] = V[b, :] * W[c, :]
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    rhs[b*p*n + c] = 1.0
                        try:
                            U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        except:
                            pass
                    
                    # Update V rows
                    for b in range(m*p):
                        j2 = b // p
                        k1 = b % p
                        M = np.zeros((n*m*p*n, target_rank))
                        rhs = np.zeros(n*m*p*n)
                        for a in range(n*m):
                            i1 = a // m
                            j1 = a % m
                            for c in range(p*n):
                                i2 = c // n
                                k2 = c % n
                                M[a*p*n + c, :] = U[a, :] * W[c, :]
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    rhs[a*p*n + c] = 1.0
                        try:
                            V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        except:
                            pass
                    
                    # Update W rows
                    for c in range(p*n):
                        i2 = c // n
                        k2 = c % n
                        M = np.zeros((n*m*m*p, target_rank))
                        rhs = np.zeros(n*m*m*p)
                        for a in range(n*m):
                            i1 = a // m
                            j1 = a % m
                            for b in range(m*p):
                                j2 = b // p
                                k1 = b % p
                                M[a*m*p + b, :] = U[a, :] * V[b, :]
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    rhs[a*m*p + b] = 1.0
                        try:
                            W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        except:
                            pass
                    
                    # Check convergence
                    valid, err = validate_numpy(U, V, W)
                    if valid:
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
                if found:
                    break
            
            if found:
                break
    
    # If not found via ALS, try a more targeted approach:
    # Look for a linear combination of 2 columns that can replace them with 1
    if not found and time.time() < deadline - 10:
        for c1 in range(28):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, 28):
                if time.time() > deadline - 5:
                    break
                
                t1 = np.zeros(n*m*p*n)
                t2 = np.zeros(n*m*p*n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = (i*m+j)*p*n + (j*p+k)*n + (k*n+i)
                            t1[idx] = U28[i*m+j, c1] * V28[j*p+k, c1] * W28[k*n+i, c1]
                            t2[idx] = U28[i*m+j, c2] * V28[j*p+k, c2] * W28[k*n+i, c2]
                
                mask = np.abs(t1) > 1e-10
                if np.sum(mask) > 0:
                    alpha_estimates = t2[mask] / t1[mask]
                    if np.std(alpha_estimates) < 1e-10:
                        alpha = alpha_estimates[0]
                        keep = [c for c in range(28) if c != c2]
                        U_new = U28[:, keep].copy()
                        V_new = V28[:, keep].copy()
                        W_new = W28[:, keep].copy()
                        U_new[:, c1] += alpha * U28[:, c2]
                        V_new[:, c1] += alpha * V28[:, c2]
                        W_new[:, c1] += alpha * W28[:, c2]
                        
                        valid, err = validate_numpy(U_new, V_new, W_new)
                        if valid:
                            best_rank = 27
                            best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                            found = True
                            break
            if found:
                break
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = U.shape[1]
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", rank)
