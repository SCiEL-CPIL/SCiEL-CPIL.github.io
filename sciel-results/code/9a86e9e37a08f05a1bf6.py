"""Matrix multiplication 3,4,4 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, keep safety margin
    
    n, m, p = DIMENSIONS
    # Tensor indices: U[i*m + j], V[j*p + k], W[k*n + i]
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the tensor as a 3D array for validation
    # T[a, b, c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0), dtype=np.float64)
    V0 = np.zeros((m*p, rank0), dtype=np.float64)
    W0 = np.zeros((p*n, rank0), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    def validate(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        # Compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.array_equal(T_recon, T)
    
    def try_round(U, V, W):
        """Try to round to integers or half-integers and validate."""
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if validate(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try rounding to half-integers (multiply by 2, round, divide by 2)
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if validate(U_h, V_h, W_h):
            return U_h, V_h, W_h
        
        return None
    
    # Convert to list of columns for manipulation
    def to_columns(U, V, W):
        r = U.shape[1]
        cols = []
        for c in range(r):
            cols.append((U[:, c].copy(), V[:, c].copy(), W[:, c].copy()))
        return cols
    
    def from_columns(cols):
        r = len(cols)
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        for c, (uc, vc, wc) in enumerate(cols):
            U[:, c] = uc
            V[:, c] = vc
            W[:, c] = wc
        return U, V, W
    
    # Local search: try to remove or merge columns
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    # Try removing columns one at a time
    cols = to_columns(U0, V0, W0)
    random.seed(42)
    random.shuffle(cols)
    
    improved = True
    while improved:
        improved = False
        for idx in range(len(cols)):
            if time.time() > deadline - 10:
                break
            # Try removing column idx
            new_cols = cols[:idx] + cols[idx+1:]
            U_t, V_t, W_t = from_columns(new_cols)
            if validate(U_t, V_t, W_t):
                cols = new_cols
                improved = True
                break
        if time.time() > deadline - 10:
            break
    
    U, V, W = from_columns(cols)
    if validate(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = len(cols)
    
    # Try to find a better decomposition using random restarts with lower rank
    # Use a heuristic: start from random initialization and try to fit
    
    # For now, return the best we found
    # Convert to lists with integer or half-integer values
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                val2 = round(val * 2) / 2
                if abs(val2 - round(val2)) < 1e-9:
                    row_list.append(int(round(val2)))
                else:
                    row_list.append(val2)
            result.append(row_list)
        return result
    
    if time.time() > deadline - 5:
        # Return schoolbook if we're running out of time
        return {"U": to_list(U0), "V": to_list(V0), "W": to_list(W0)}
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
