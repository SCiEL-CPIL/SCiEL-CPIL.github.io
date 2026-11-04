import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin

def _build_structured_rank77():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 x 2x2 (cols 0,1)
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Schoolbook for 2x2 x 2x1 (col 2)
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-6):
    """Check if U,V,W exactly decompose the 4x4*4x6 matrix multiplication tensor (vectorized)."""
    n, m, p = 4, 4, 6
    R = U.shape[1]
    
    # Build the multiplication tensor T[a, b, c] where:
    # a = i*m + j (index into A), b = j'*p + k (index into B), c = k'*n + i' (index into C)
    # T[a, b, c] = 1 if j == j' AND k == k' AND i == i', else 0
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor entry T[(i,j), (j',k), (k',i')] = 1 if j==j' and k==k' and i==i'
    # But we only care about the "diagonal" where all indices match: T[(i,j),(j,k),(k,i)] = 1
    
    # The decomposition condition: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all (i,j,j',k) where j != j': sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = 0
    
    # Vectorized approach:
    # For each r, the outer product U[:,r] @ V[:,r].T @ W[:,r] (broadcast) contributes to the tensor.
    # More efficiently: compute the tensor as sum_r outer(outer(U[:,r], V[:,r]), W[:,r])
    # But that's 16*24*24 = 9216 entries, which is small.
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute actual tensor
    T_actual = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        # outer product of U[:,r], V[:,r], W[:,r]
        T_actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if np.max(np.abs(T_actual - T_expected)) > tol:
        return False
    return True

def _rank_reduce(U, V, W, max_time=130.0):
    """Try to reduce rank by identifying and removing redundant terms."""
    R = U.shape[1]
    start = time.time()
    
    # First, let's check if the current decomposition is valid
    if not _verify_decomposition(U, V, W):
        return U, V, W  # Can't reduce if not valid
    
    # Identify columns with all-zero U, V, or W (definitely redundant)
    keep = list(range(R))
    for r in range(R):
        if time.time() - start > max_time - 10:
            break
        if np.allclose(U[:, r], 0, atol=1e-9) or np.allclose(V[:, r], 0, atol=1e-9) or np.allclose(W[:, r], 0, atol=1e-9):
            keep.remove(r)
    
    if len(keep) < R:
        U_reduced = U[:, keep]
        V_reduced = V[:, keep]
        W_reduced = W[:, keep]
        if _verify_decomposition(U_reduced, V_reduced, W_reduced):
            return U_reduced, V_reduced, W_reduced
    
    # Try removing pairs of columns
    for r in range(R):
        if time.time() - start > max_time - 10:
            break
        for s in range(r + 1, R):
            if time.time() - start > max_time - 10:
                break
            keep = [x for x in range(R) if x != r and x != s]
            U_reduced = U[:, keep]
            V_reduced = V[:, keep]
            W_reduced = W[:, keep]
            if _verify_decomposition(U_reduced, V_reduced, W_reduced):
                return U_reduced, V_reduced, W_reduced
    
    # Try linear-combination elimination: for each pair (r,s), find alpha,beta such that
    # new_col_r = col_r - alpha * col_s is zero in the tensor sense, or
    # new_col_s = col_s - beta * col_r is zero.
    # This means the two columns are "proportional" in the rank-1 tensor sense.
    # Specifically, if U[:,r] = a*U[:,s], V[:,r] = b*V[:,s], W[:,r] = c*W[:,s] with a*b*c=1,
    # then we can eliminate one of them.
    
    for r in range(R):
        if time.time() - start > max_time - 5:
            break
        for s in range(r + 1, R):
            if time.time() - start > max_time - 5:
                break
            
            # Check if col r is a scalar multiple of col s in the tensor sense
            # U[:,r] = a * U[:,s], V[:,r] = b * V[:,s], W[:,r] = c * W[:,s], a*b*c = 1
            # Find a from U, b from V, check if c = 1/(a*b) matches W
            
            # Find a: ratio of non-zero entries in U
            mask_u = np.abs(U[:, s]) > 1e-9
            if np.sum(mask_u) > 0:
                a = U[mask_u, r] / U[mask_u, s]
                # Check consistency
                if np.max(np.abs(a - a[0])) < 1e-9:
                    a = a[0]
                    # Find b from V
                    mask_v = np.abs(V[:, s]) > 1e-9
                    if np.sum(mask_v) > 0:
                        b = V[mask_v, r] / V[mask_v, s]
                        if np.max(np.abs(b - b[0])) < 1e-9:
                            b = b[0]
                            # Check W
                            mask_w = np.abs(W[:, s]) > 1e-9
                            if np.sum(mask_w) > 0:
                                c_expected = W[mask_w, r] / W[mask_w, s]
                                if np.max(np.abs(c_expected - c_expected[0])) < 1e-9:
                                    c = c_expected[0]
                                    # Check a*b*c ≈ 1
                                    if abs(a * b * c - 1.0) < 1e-9:
                                        # Column r is redundant given column s
                                        # Remove column r
                                        keep = [x for x in range(R) if x != r]
                                        U_reduced = U[:, keep]
                                        V_reduced = V[:, keep]
                                        W_reduced = W[:, keep]
                                        if _verify_decomposition(U_reduced, V_reduced, W_reduced):
                                            return U_reduced, V_reduced, W_reduced
    
    return U, V, W

def to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

def construct():
    # Build the structured rank-77 decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify
    if not _verify_decomposition(U, V, W):
        # Fallback: return as-is (shouldn't happen)
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try to reduce rank
    U, V, W = _rank_reduce(U, V, W)
    
    # Final polish: round to nearest integer/half-integer if close
    for mat in [U, V, W]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < 1e-9:
                    mat[i, j] = nearest_half
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
