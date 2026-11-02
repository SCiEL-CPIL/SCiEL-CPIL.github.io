"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Fallback: schoolbook rank 50 (always valid)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Target rank 25: fix U and V sparsely, solve for W.
    # U[i,j,c] = 1 if c == i*m + j  (each U cell used in exactly one column)
    # V[j,k,c] = 1 if c == j*p + k  (each V cell used in exactly one column)
    # Then for each (i,k): sum_c W[k,i,c] * (terms) = sum_j A[i,j] B[j,k]
    rank = m * p  # 25

    # Build the linear system: for each (i,k), equation over c.
    # LHS of (i,k) equation = sum_c W[k,i,c] * T[i,k,c] where
    # T[i,k,c] = sum over (j) of U[i,j,c]*V[j,k,c]  -- but with our structure,
    # U[i,j,c]=1 only if c==i*m+j, V[j,k,c]=1 only if c==j*p+k.
    # So for a given c, U[i,j,c]=1 iff j == c - i*m (and 0<=j<m),
    #   and V[j,k,c]=1 iff c == j*p+k  i.e. j == (c-k)//p if c% p == k.
    # T[i,k,c] = 1 if there exists j with (c == i*m + j) AND (c == j*p + k).

    # Equation (i,k): sum_c W[k,i,c] * T[i,k,c] = sum_j A[i,j]*B[j,k]
    # But the RHS is a bilinear symbol; we match coefficients.
    # Actually the tensor identity: for the multiplication tensor,
    #   (A B)[i,k] = sum_j A[i,j] B[j,k]
    # must equal sum_c U[i,:,c] . (A)  *  (B .) V[:,k,c] ... 
    # Let me set it up as: sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k])
    #   = sum_c U[i,:,c].a_i * V[:,k,c].b_k
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    #   sum_c U[i,j,c] V[j',k,c] = delta_{j,j'}  for all i,k,j,j'
    # That's the condition: sum_c U[i,j,c] V[j',k,c] = delta_{j,j'} * (something independent)
    # More precisely the tensor: T[i,k,j] (multiplication) must factor as
    #   T[i,k,j] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]  ... 
    # Hmm, the standard factorization: 
    #   (A B)[i,k] = sum_c  (sum_j U[i,j,c] A[i,j]) * (sum_l V[l,k,c] B[l,k])
    # We need: sum_c U[i,j,c] V[l,k,c] = delta_{j,l}  (independent of i,k)
    # and W[k,i,c] = 1 for all. Actually then (AB)[i,k] = sum_c (sum_j U[i,j,c]A[ij])(sum_l V[lk,c]Blk)
    # = sum_{j,l} (sum_c U[i,j,c] V[l,k,c]) A[i,j] B[l,k] = sum_{j,l} delta_{j,l} A[ij] B[lk] = sum_j A[ij] B[jk]. 
    # So W is all 1s and we need sum_c U[i,j,c] V[l,k,c] = delta_{j,l}.

    # With U[i,j,c]=1 iff c=i*m+j, V[l,k,c]=1 iff c=l*p+k:
    # sum_c U[i,j,c] V[l,k,c] = 1 iff (i*m+j) == (l*p+k) i.e. j + i*m = k + l*p.
    # For this to equal delta_{j,l} for ALL i,k, we'd need j=i*m+j only equals l*p+k when j=l.
    # That requires: the map c=i*m+j and c=l*p+k intersect only at j=l. 
    # c from U: i*m+j in [0, m*p) since i in [0,n), j in [0,m) -> i*m+j in [0, n*m).
    # c from V: l*p+k in [0, m*p) since l in [0,m), k in [0,p) -> l*p+k in [0, m*p).
    # We need for all i,k: {j: i*m+j in range} ∩ {l: l*p+k in range} ... 
    # This is getting complex. Let me just do the direct least-squares on W with the
    # constraint that the factored tensor matches the multiplication tensor.

    # Direct approach: unknowns W[k,i,c] for k in [0,p), i in [0,n), c in [0,rank).
    # The factorized tensor: F[i,k,j] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We need F[i,k,j] = delta_{j, j} ... no. Multiplication tensor M[i,k,j] = 1 (since (AB)[ik]=sum_j A[ij]B[jk], coefficient of A[ij]B[jk] is 1).
    # So we need: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,k,j.
    # That's n*p*m = 50 equations in n*p*rank = 2*5*25 = 250 unknowns. Underdetermined -> solvable.

    # Build U, V as 0/1 matrices with the structure above.
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            c = i * m + j
            if c < rank:
                U[i * m + j, c] = 1.0
    for l in range(m):
        for k in range(p):
            c = l * p + k
            if c < rank:
                V[l * p + k, c] = 1.0

    # Equations: for each (i,k,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Unknown W indexed by (k,i,c) -> flat index w = (k*n + i)*rank + c
    num_eq = n * p * m
    num_unk = p * n * rank
    A_mat = np.zeros((num_eq, num_unk), dtype=float)
    b_vec = np.zeros(num_eq, dtype=float)
    eq = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for c in range(rank):
                    val = U[i * m + j, c] * V[j * p + k, c]
                    if val != 0.0:
                        w = (k * n + i) * rank + c
                        A_mat[eq, w] = val
                b_vec[eq] = 1.0
                eq += 1

    # Solve least squares
    try:
        sol, residuals, rank_lev, svals = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except Exception:
        if time.time() > deadline - 2.0:
            return schoolbook()
        return schoolbook()

    # Verify: reconstruct and check residual
    W_flat = sol
    W = np.zeros((p * n, rank), dtype=float)
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i, c] = W_flat[(k * n + i) * rank + c]

    # Check exactness: F[i,k,j] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] should be 1
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err

    if max_err < 1e-9:
        # Round to nearest half-integer for cleanliness
        def to_half(x):
            # round to nearest 0.5
            return round(x * 2.0) / 2.0
        U_out = [[to_half(U[i, c]) for c in range(rank)] for i in range(n * m)]
        V_out = [[to_half(V[i, c]) for c in range(rank)] for i in range(m * p)]
        W_out = [[to_half(W[i, c]) for c in range(rank)] for i in range(p * n)]
        # Re-verify after rounding
        max_err2 = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += U_out[i * m + j][c] * V_out[j * p + k][c] * W_out[k * n + i][c]
                    err = abs(s - 1.0)
                    if err > max_err2:
                        max_err2 = err
        if max_err2 < 1e-9:
            return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
