"""
Tensor decomposition for matrix multiplication M_{2,3,10}.
Dimensions: n=2, m=3, p=10
Standard rank: 60

Strategy: Use the known algebraic structure. For n=2, we can exploit the fact
that C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k].

The key insight for improving beyond rank 60: for each column k of B, we need
to compute two dot products (one for each row of A). We can share the B[j,k]
factors across the two rows.

For a rank-r decomposition, we need r rank-1 tensors. Each rank-1 tensor is
u ⊗ v ⊗ w where:
- u has length 6 (indices of A)
- v has length 30 (indices of B)
- w has length 20 (indices of C)

The trivial decomposition uses 60 rank-1 tensors, one for each (i,j,k) triple.

A better approach: For each (j,k), we can create a pair of rank-1 tensors that
share the v component. Specifically:
- For fixed j,k, we need A[0,j]*B[j,k] in C[0,k] and A[1,j]*B[j,k] in C[1,k].
- We can write this as:
  (A[0,j], 0) ⊗ (B[j,k], 0) ⊗ (0, C[0,k]) + (0, A[1,j]) ⊗ (B[j,k], 0) ⊗ (0, C[1,k])
  
  But this is still 2 tensors per (j,k) pair, giving 2*3*10 = 60.

Let me think differently. The tensor is:
T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i

For n=2, m=3, p=10:
- a ranges over 0..5 (i=0,1; j=0,1,2)
- b ranges over 0..29 (j=0,1,2; k=0..9)
- c ranges over 0..19 (k=0..9; i=0,1)

T[i*3+j, j*10+k, k*2+i] = 1

One approach: use the structure to find a rank < 60 decomposition.

Known result: For M_{2,3,p}, the rank is 3p for p >= some threshold, but for small p it might be different. Actually, the rank of M_{2,3,10} is known to be 30 in some references? No, let me think more carefully.

The rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30.

For M_{2,3,10}, I'll try to find a rank-30 or slightly higher decomposition using a structured approach.

Idea: For each j (0,1,2) and k (0..9), the contribution to C is:
C[0,k] gets A[0,j]*B[j,k]
C[1,k] gets A[1,j]*B[j,k]

We can group by k. For fixed k, we need:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]

This is like multiplying a 2x3 matrix by a 3x1 vector for each k.

The rank of M_{2,3,1} is known to be 7 (Strassen-like). So for M_{2,3,10}, we might be able to use 7*10 = 70? That's worse.

Actually, let me just try to find a good decomposition numerically using a smarter approach.

Let me use the fact that we can write the tensor as a sum of rank-1 terms and use a greedy or ALS approach with good initialization.
"""
import time
import sys
import numpy as np

n, m, p = 2, 3, 10
DEADLINE = 145

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _check(U, V, W, T, tol=1e-8):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_recon - T)) < tol

def _als(n, m, p, rank, max_iters=500, seed=42, tol=1e-10):
    """Alternating Least Squares for tensor decomposition."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    T_u = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
    T_v = T.transpose(1, 0, 2).reshape(m*p, -1)  # (m*p, n*m*p*n)
    T_w = T.transpose(2, 0, 1).reshape(p*n, -1)  # (p*n, n*m*m*p)
    
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U: minimize ||T_u - U * (V ⊙ W)||_F
        # T_u[a, (b,c)] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed U, this is linear. For updating U:
        # U[a,:] = argmin sum_{b,c} (T_u[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        # Gradient: -2 sum_{b,c} T_u[a,b,c] V[b,r] W[c,r] + 2 sum_{b,c} (sum_s U[a,s] V[b,s] W[c,s]) V[b,r] W[c,r]
        # = -2 * einsum over b,c of T_u[a,b,c] * V[b,r] * W[c,r] + 2 * U[a,:] @ (sum over b,c of V[b,:]*W[c,:])
        
        # Compute the "cross term" matrix: M[r,s] = sum_{b,c} V[b,r] W[c,r] V[b,s] W[c,s]
        # Actually: sum_{b,c} V[b,r] V[b,s] W[c,r] W[c,s] = (V^T V)[r,s] * (W^T W)[r,s]? No.
        # sum_{b,c} V[b,r] V[b,s] W[c,r] W[c,s] = sum_b V[b,r]V[b,s] * sum_c W[c,r]W[c,s] = (V^T V)[r,s] * (W^T W)[r,s]
        
        VtV = V.T @ V  # (rank, rank)
        WtW = W.T @ W  # (rank, rank)
        M = VtW = VtV * WtW  # element-wise product, (rank, rank)
        
        # RHS for U: rhs[a,r] = sum_{b,c} T_u[a,b,c] V[b,r] W[c,r]
        # = einsum('abc,br,cr->ar', T, V, W) where T is the full tensor
        rhs_U = np.einsum('abc,br,cr->ar', T, V, W)
        
        try:
            U = np.linalg.solve(M + 1e-12 * np.eye(rank), rhs_U)
        except:
            pass
        
        # Update V similarly
        UtU = U.T @ U
        WtW = W.T @ W
        Mv = UtU * WtW
        rhs_V = np.einsum('abc,ar,cr->br', T, U, W)
        try:
            V = np.linalg.solve(Mv + 1e-12 * np.eye(rank), rhs_V)
        except:
            pass
        
        # Update W
        UtU = U.T @ U
        VtV = V.T @ V
        Mw = UtU * VtV
        rhs_W = np.einsum('abc,ar,br->cr', T, U, V)
        try:
            W = np.linalg.solve(Mw + 1e-12 * np.eye(rank), rhs_W)
        except:
            pass
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            return U, V, W, err
    
    return best_U, best_V, best_W, best_err

def construct():
    start = time.time()
    T = _build_tensor(n, m, p)
    
    # Try ranks from 30 upward
    for rank in range(30, 61):
        if time.time() - start > DEADLINE - 10:
            break
        for seed in range(5):
            if time.time() - start > DEADLINE - 10:
                break
            U, V, W, err = _als(n, m, p, rank, max_iters=200, seed=seed)
            if err < 1e-8:
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to trivial
    U, V, W = _trivial(n, m, p)
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
