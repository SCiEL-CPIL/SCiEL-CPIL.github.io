"""
Rank-minimal tensor decomposition for 2x3 * 3x8 matrix multiplication.

We use a structured search that exploits the known multiplication tensor.
The key insight: for small matrix multiplication, the rank is often bounded
by known results. For 2x3 x 3x8, we can try ranks below the schoolbook 48.

Strategy:
1. Use the known structure of matrix multiplication to set up a linear system.
2. Try ranks from a lower bound up to schoolbook rank.
3. Use randomized initialization + ALS with multiple restarts.
4. Verify exact integer recovery.
"""

import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 8]

# Deadline: 150s wall-clock, return best result with safety margin
_DEADLINE = 150.0
_START_TIME = None

def _time_remaining():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _build_multiplication_tensor():
    """Build the 2x3 * 3x8 multiplication tensor T[i,j,k] = 1 if i,j,k are the (row, col-of-A, col-of-B) indices."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    # T[a, b, c] where a in [0,n), b in [0,m), c in [0,p)
    # The multiplication: C[a,c] = sum_b A[a,b] * B[b,c]
    # Tensor: T[a, b, c] = 1 (each term contributes)
    # Actually the tensor has T[a, b, c] = 1 for all valid a,b,c
    # Wait, no. The matrix multiplication tensor T_{ijk} where i is row of C, j is column of A (row of B), k is column of C
    # T_{ijk} = 1 if the product A_{ij} * B_{jk} contributes to C_{ik}
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # That's just the all-ones tensor of shape (n, m, p)? No wait...
    
    # Let me reconsider. The matrix multiplication tensor for n x m times m x p:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has entries T[i,j,k] where:
    #   i indexes the row of A / row of C (0..n-1)
    #   j indexes the column of A / row of B (0..m-1)  
    #   k indexes the column of B / column of C (0..p-1)
    # T[i,j,k] = 1 (the coefficient of A_{i,j} * B_{j,k} in C_{i,k})
    # So yes, T is the all-ones tensor of shape (n, m, p).
    
    # But wait, that can't be right for rank purposes. The all-ones tensor of shape (n,m,p) has rank... 
    # Actually the rank of the all-ones tensor (n,m,p) is min(n,m,p) if we allow real entries? No.
    # The all-ones tensor 1_{n,m,p} can be written as sum_{r=1}^{min(n,m,p)} u_r v_r w_r where
    # u_r[i] = 1, v_r[j] = 1, w_r[k] = 1... but that only gives rank 1 if n=m=p=1.
    
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k is rank 1: 
    # T[i,j,k] = (1)(1)(1) = u_i * v_j * w_k where u, v, w are all-ones vectors.
    # So the all-ones tensor has rank 1!
    
    # But that's not the matrix multiplication tensor. Let me re-examine.
    
    # The matrix multiplication tensor is NOT the all-ones tensor. 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a bilinear computation. The tensor T satisfies:
    # T[i, j, k] = delta_{?} ... 
    
    # Actually, the way to think about it: we want to compute C = A @ B where A is n x m and B is m x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # In tensor form, the multiplication tensor T has:
    # T[i, j, k] = 1 for all i, j, k? No...
    #
    # The standard way: the tensor T is of shape (n, m, p) and T[i,j,k] represents the coefficient
    # such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]. Since T[i,j,k] = 1 for all valid indices,
    # yes it IS the all-ones tensor.
    #
    # But then the rank would be 1, which is clearly wrong for matrix multiplication.
    
    # I think I'm confusing two things. Let me use the correct formulation:
    # The matrix multiplication tensor T_{ijk} for multiplying an n x m matrix by an m x p matrix
    # is defined as T_{ijk} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # Since every (i,j,k) triple is valid, T_{ijk} = 1 for all i,j,k.
    # 
    # BUT the rank of this tensor is NOT 1. The issue is that in the tensor decomposition,
    # we need: T_{ijk} = sum_{r} U_{ir} V_{jr} W_{kr}
    # If T_{ijk} = 1 for all i,j,k, then we need U V^T W-like decomposition...
    # Actually 1_{n,m,p} = u v^T w^T? No, that's a rank-1 tensor.
    # u v^T w^T has entries u_i * v_j * w_k. For this to equal 1 for all i,j,k, we need
    # u_i * v_j * w_k = 1 for all i,j,k, which means u_i = c, v_j = 1/c, w_k = 1 for any c != 0.
    # So yes, the all-ones tensor IS rank 1.
    
    # This means my understanding of the tensor is wrong. Let me look at it differently.
    
    # The correct tensor for matrix multiplication: we have m separate "channels" j.
    # The tensor should be: T[i, j, k] where j is summed over.
    # But in the standard tensor decomposition framework, we don't sum - we just decompose.
    
    # I think the issue is that the matrix multiplication tensor is NOT the all-ones tensor.
    # Let me reconsider: C = A B, C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]
    # This can be seen as: for each j, we have a term A[i,j] * B[j,k] contributing to C[i,k].
    # So the "tensor" has one component for each j: T_j[i,k] = A[i,j] B[j,k]
    # 
    # In the tensor decomposition framework, we write:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k])
    # Wait, that's not standard either.
    
    # Let me use the standard formulation from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for each (i, j, k) in the multiplication,
    # the contribution to C[i,k] from A[i,j]*B[j,k] is captured.
    
    # The multiplication tensor T has dimensions (n, m, p) and T[i,j,k] = 1
    # (since each triple (i,j,k) corresponds to exactly one product term).
    
    # The tensor decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    # This is a system of n*m*p = 48 equations. The unknowns are the entries of U, V, W.
    # For rank r, we have r*(n*m + m*p + p*n) = r*(6 + 24 + 16) = 46r unknowns.
    # For r=1: 46 unknowns, 48 equations - might be possible but unlikely.
    # For r=2: 92 unknowns, 48 equations - definitely possible.
    
    # But the RANK is the minimum r such that this system has a solution.
    # The all-ones tensor T[i,j,k]=1 is NOT rank 1 in this parameterization because
    # the indices are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    
    # So it's NOT simply the all-ones tensor in the usual sense. The coupling of indices
    # means the structure is more complex.
    
    # Let me just build it correctly and search for the minimal rank.
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    return T

def _verify(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _als_solve(T_flat, n, m, p, rank, max_iter=500, tol=1e-10, seed=None):
    """
    ALS for tensor decomposition.
    T_flat is the flattened tensor of shape (n*m*p,) with T_flat[i*m*p + j*p + k] = T[i,j,k]
    
    We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Unfolding:
    - Mode 0 (index i*m+j): reshape T to (n*m, p*n) where T0[im, kn] = T[i, j, k]
      Wait, need to be careful. T[i,j,k] with i in [0,n), j in [0,m), k in [0,p).
      Index in U: a = i*m + j, range [0, n*m)
      Index in V: b = j*p + k, range [0, m*p)
      Index in W: c = k*n + i, range [0, p*n)
      
      T[i,j,k] = sum_r U[a,r] V[b,r] W[c,r]
      
      Mode-0 unfolding: T0[a, (b,c)] = T[i,j,k]
      T0[a, (b,c)] = sum_r U[a,r] * (V[b,r] * W[c,r])
      So T0 = U @ (V ⊙ W)^T where ⊙ is Khatri-Rao product.
      V ⊙ W has shape (m*p * p*n, rank)? No, Khatri-Rao of V (m*p x rank) and W (p*n x rank)
      is (m*p * p*n) x rank? That doesn't seem right.
      
      Actually, Khatri-Rao product of two matrices A (d1 x r) and B (d2 x r) gives (d1*d2 x r)
      where (A ⊙ B)[i*d2 + j, r] = A[i,r] * B[j,r].
      
      So for mode 0: T0[a, b*p*n + c] = sum_r U[a,r] * V[b,r] * W[c,r]
      T0 = U @ (V ⊙ W)^T where V ⊙ W has shape (m*p * p*n, rank)
      Wait, that's (24 * 16, rank) = (384, rank). And T0 is (6, 384).
      T0[a, bc] = sum_r U[a,r] * (V ⊙ W)[bc, r]
      So T0 = U @ (V ⊙ W)^T. ✓
      
    Similarly:
    Mode 1: T1[b, a*p*n + c] = sum_r V[b,r] * U[a,r] * W[c,r]
    T1 = V @ (U ⊙ W)^T where U ⊙ W has shape (n*m * p*n, rank) = (96, rank)
    T1 is (24, 96).
    
    Mode 2: T2[c, a*m*p + b] = sum_r W[c,r] * U[a,r] * V[b,r]
    T2 = W @ (U ⊙ V)^T where U ⊙ V has shape (n*m * m*p, rank) = (144, rank)
    T2 is (16, 144).
    """
    np.random.seed(seed)
    
    n, m, p = n, m, p
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Build unfolded matrices
    # T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    # Mode 0: T0[a, b*p*n + c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T0 has shape (n*m, m*p * p*n)
    T0 = np.zeros((dim_U, dim_V * dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T0[a, b * dim_W + c] = T[i, j, k]
    
    # Mode 1: T1[b, a*p*n + c]
    # Wait, need consistent ordering. Let me use:
    # T1[b, a * dim_W + c] = T[i,j,k]
    T1 = np.zeros((dim_V, dim_U * dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T1[b, a * dim_W + c] = T[i, j, k]
    
    # Mode 2: T2[c, a * dim_V + b]
    T2 = np.zeros((dim_W, dim_U * dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T2[c, a * dim_V + b] = T[i, j, k]
    
    # Initialize
    U = np.random.randn(dim_U, rank)
    V = np.random.randn(dim_V, rank)
    W = np.random.randn(dim_W, rank)
    
    for iteration in range(max_iter):
        if time.time() - _START_TIME > _DEADLINE - 30:
            break
            
        U_old = U.copy()
        
        # Update U: T0 = U @ (V ⊙ W)^T
        # V ⊙ W: (dim_V * dim_W, rank)
        VW = np.zeros((dim_V * dim_W, rank))
        for r in range(rank):
            VW[:, r] = np.kron(V[:, r], W[:, r])
        # Actually Khatri-Rao: (V ⊙ W)[b*dim_W + c, r] = V[b,r] * W[c,r]
        VW = np.zeros((dim_V * dim_W, rank))
        for r in range(rank):
            for b in range(dim_V):
                for c in range(dim_W):
                    VW[b * dim_W + c, r] = V[b, r] * W[c, r]
        
        # U = lstsq((VW)^T, T0^T)^T
        try:
            U_new, _, _, _ = np.linalg.lstsq(VW.T, T0.T, rcond=None)
            U = U_new.T
        except:
            break
        
        # Update V: T1 = V @ (U ⊙ W)^T
        UW = np.zeros((dim_U * dim_W, rank))
        for r in range(rank):
            for a in range(dim_U):
                for c in range(dim_W):
                    UW[a * dim_W + c, r] = U[a, r] * W[c, r]
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW.T, T1.T, rcond=None)
            V = V_new.T
        except:
            break
        
        # Update W: T2 = W @ (U ⊙ V)^T
        UV = np.zeros((dim_U * dim_V, rank))
        for r in range(rank):
            for a in range(dim_U):
                for b in range(dim_V):
                    UV[a * dim_V + b, r] = U[a, r] * V[b, r]
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV.T, T2.T, rcond=None)
            W = W_new.T
        except:
            break
        
        # Check convergence
        change = np.linalg.norm(U - U_old)
        if change < tol:
            break
    
    # Convert to list of lists with rational approximation
    def to_list(mat):
        rows = []
        for row in mat:
            r_list = []
            for val in row:
                # Try to find a close half-integer
                r2 = np.round(val * 2)
                if abs(val - r2 / 2.0) < 1e-4:
                    r_list.append(r2 / 2.0 if r2 % 2 != 0 else float(r2 // 2))
                    if r2 % 2 == 0:
                        r_list[-1] = int(r2 // 2)
                else:
                    r_list.append(val)
            rows.append(r_list)
        return rows
    
    return to_list(U), to_list(V), to_list(W)

def _verify_solution(U, V, W, n, m, p, rank):
    """Verify the solution with rational entries."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-6

def _search_rank(n, m, p, min_rank, max_rank, time_budget=120):
    """Search for minimal rank using ALS with multiple restarts."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build unfolded matrices once
    T = np.ones((n, m, p))
    
    T0 = np.zeros((dim_U, dim_V * dim_W))
    T1 = np.zeros((dim_V, dim_U * dim_W))
    T2 = np.zeros((dim_W, dim_U * dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T0[a, b * dim_W + c] = 1.0
                T1[b, a * dim_W + c] = 1.0
                T2[c, a * dim_V + b] = 1.0
    
    for rank in range(min_rank, max_rank + 1):
        if time.time() - _START_TIME > _DEADLINE - 30:
            break
        
        num_restarts = 20 if rank <= 10 else 10
        for restart in range(num_restarts):
            if time.time() - _START_TIME > _DEADLINE - 30:
                break
            
            np.random.seed(42 + restart * 1000 + rank)
            U = np.random.randn(dim_U, rank)
            V = np.random.randn(dim_V, rank)
            W = np.random.randn(dim_W, rank)
            
            max_iter = 300
            tol = 1e-10
            for iteration in range(max_iter):
                if time.time() - _START_TIME > _DEADLINE - 30:
                    break
                    
                U_old = U.copy()
                
                # Update U
                VW = np.zeros((dim_V * dim_W, rank))
                for r in range(rank):
                    VW[:, r] = np.kron(V[:, r], W[:, r])
                try:
                    U = np.linalg.lstsq(VW.T, T0.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                UW = np.zeros((dim_U * dim_W, rank))
                for r in range(rank):
                    UW[:, r] = np.kron(U[:, r], W[:, r])
                try:
                    V = np.linalg.lstsq(UW.T, T1.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                UV = np.zeros((dim_U * dim_V, rank))
                for r in range(rank):
                    UV[:, r] = np.kron(U[:, r], V[:, r])
                try:
                    W = np.linalg.lstsq(UV.T, T2.T, rcond=None)[0].T
                except:
                    break
                
                if np.linalg.norm(U - U_old) < tol:
                    break
            
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for r in range(rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Found a valid solution at this rank
                def to_list(mat):
                    rows = []
                    for row in mat:
                        r_list = []
                        for val in row:
                            r2 = np.round(val * 2)
                            if abs(val - r2 / 2.0) < 1e-4:
                                if r2 % 2 == 0:
                                    r_list.append(int(r2 // 2))
                                else:
                                    r_list.append(r2 / 2.0)
                            else:
                                r_list.append(val)
                        rows.append(r_list)
                    return rows
                
                return rank, to_list(U), to_list(V), to_list(W)
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p  # 48
    U_school = [[0] * rank_schoolbook for _ in range(dim_U)]
    V_school = [[0] * rank_schoolbook for _ in range(dim_V)]
    W_school = [[0] * rank_schoolbook for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a better rank
    # Lower bound: max(n, m, p) = 8? Actually for matrix multiplication, 
    # the rank is at least max(n, m, p) but can be higher.
    # Known results: 2x3 x 3x3 has rank 7 (vs schoolbook 9).
    # For 2x3 x 3x8, let's search from rank 8 upward.
    
    # Search for minimal rank
    result = _search_rank(n, m, p, min_rank=8, max_rank=rank_schoolbook, time_budget=120)
    
    if result is not None:
        rank, U, V, W = result
        # Final verification
        if _verify_solution(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
