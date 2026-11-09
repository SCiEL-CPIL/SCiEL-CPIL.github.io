"""Matrix multiplication 4x4 x 4x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # The multiplication tensor: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the (i,k) entry of (A*B) depends on A[i,j]*B[j,k]
    # So T is 1 at positions (i,j,k) for all i,j,k - it's the full tensor of 1s
    # Wait, no. The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Actually the tensor is: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1
    # So T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    
    # A rank-1 decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is vector in R^n, V_c in R^m, W_c in R^p
    # T[i,j,k] = sum_c U_c[i] * V_c[j] * W_c[k]
    
    # We need: sum_c U_c[i] * V_c[j] * W_c[k] = 1 for all i,j,k
    
    # This is equivalent to: for each (i,j,k), the sum equals 1.
    # The tensor is all-ones of shape (n,m,p).
    
    # A trivial decomposition: use n*m*p rank-1 terms each with a single 1.
    # But we can do better using the structure.
    
    # Key insight: The all-ones tensor of shape (n,m,p) can be decomposed as:
    # sum_c U_c[i] * V_c[j] * W_c[k] = 1
    # 
    # If we let U_c[i] = a_{c,i}, V_c[j] = b_{c,j}, W_c[k] = c_{c,k}
    # We need sum_c a_{c,i} b_{c,j} c_{c,k} = 1 for all i,j,k
    #
    # This is a 3D matrix equation. One approach: use the fact that 
    # 1 = sum over some basis.
    
    # Let's try: if we set W_c[k] = 1 for all k for some terms, and vary U and V
    # Then we need sum_c U_c[i] V_c[j] = 1 for all i,j (if W is constant)
    # That's rank-1 decomposition of the all-ones n×m matrix, which needs rank 1!
    # U[i]=1, V[j]=1 gives sum = 1. So one term with W[k]=1 for all k works?
    # No wait: sum_c U_c[i] V_c[j] W_c[k] = U[i]*V[j]*W[k] for a single term.
    # For this to equal 1 for all i,j,k, we need U[i]*V[j]*W[k] = 1 for all i,j,k.
    # That means U[i] = 1/u, V[j] = 1/v, W[k] = 1/w where u*v*w=1.
    # So U[i] = a, V[j] = b, W[k] = c with a*b*c = 1.
    # E.g., U[i]=1, V[j]=1, W[k]=1 gives 1*1*1=1. YES!
    
    # Wait, that means the rank is 1?! That can't be right for matrix multiplication.
    
    # Let me re-examine. The matrix multiplication tensor is NOT the all-ones tensor.
    # 
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 if A[i,j] and B[j,k] contribute to C[i,k]
    # But the issue is that the same C[i,k] receives contributions from different j values.
    # 
    # The tensor decomposition represents: C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_k V_c[j,k] * B[j,k])
    # No, that's not right either.
    
    # Let me use the standard formulation:
    # The multiplication tensor M has M[i,j,k] = 1 for all i,j,k (in the range)
    # because the bilinear form is: C[i,k] = sum_{j} A[i,j] B[j,k]
    # 
    # A rank-r decomposition gives r trilinear forms:
    # C[i,k] = sum_c sum_j U_c[i,j] * A[i,j] * V_c[j,k] * B[j,k]
    # 
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U_c[i,j] V_c[j,k] = 1 for all j, for each (i,k)
    # 
    # Wait, that's not quite right. Let me be more careful.
    
    # Standard: The tensor T ∈ R^{n×m × m×p × n×p} where
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', j'=j, k=k'
    # i.e., T[(i,j), (j,k), (i,k)] = 1 and 0 otherwise.
    
    # So T is indexed by (U_idx, V_idx, W_idx) where:
    # U_idx = (i,j) with i in [0,n), j in [0,m)  -> n*m entries
    # V_idx = (j,k) with j in [0,m), k in [0,p)  -> m*p entries  
    # W_idx = (i,k) with i in [0,n), k in [0,p)  -> n*p entries
    # 
    # T[U_idx, V_idx, W_idx] = 1 if U_idx=(i,j), V_idx=(j,k), W_idx=(i,k) for some i,j,k
    # Otherwise 0.
    
    # A rank-1 term: U_c ⊗ V_c ⊗ W_c where U_c ∈ R^{n*m}, V_c ∈ R^{m*p}, W_c ∈ R^{n*p}
    # T = sum_c U_c ⊗ V_c ⊗ W_c means:
    # T[a,b,c] = sum_r U_r[a] V_r[b] W_r[c]
    
    # For our tensor: T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', j'=j, k=k'
    # So T[(i,j),(j,k),(i,k)] = 1, else 0.
    
    # We need: sum_r U_r[(i,j)] * V_r[(j,k)] * W_r[(i,k)] = 1 if indices align, else 0.
    
    # The schoolbook uses rank n*m*p with U_r[(i,j)]=1 for one (i,j), V_r[(j,k)]=1 for one (j,k), W_r[(i,k)]=1 for one (i,k).
    
    # To reduce rank, we need to find fewer terms. This is a hard problem.
    
    # Let me try a constructive approach using known techniques.
    
    # For 4×4 × 4×5, one approach: use the fact that 4 = 2+2 and apply Strassen recursively.
    
    # Actually, let me try to use a numerical optimization approach to find a low-rank decomposition.
    
    # Represent the tensor as a 3D array and try to find a CP decomposition.
    
    # Build the tensor T of shape (n*m, m*p, n*p) = (16, 20, 20)
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a low-rank CP decomposition using an iterative approach.
    # We'll use a variant of the ALS (Alternating Least Squares) for CP decomposition,
    # but with the constraint that we want an EXACT decomposition with small entries.
    
    # Since the tensor has integer entries, we hope for a rational/integer solution.
    
    # Let's try ranks from low to high and see if we can find an exact decomposition.
    
    def try_rank(r, max_iter=500, tol=1e-10):
        """Try to find a rank-r CP decomposition of T."""
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(n * p, r)
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Update U: for each column c of U, solve least squares
            for c in range(r):
                # Fix V[:,c] and W[:,c], solve for U[:,c]
                # T ≈ sum_c U_c ⊗ V_c ⊗ W_c
                # For fixed c, the contribution to T from term c is U_c ⊗ V_c ⊗ W_c
                # Residual: T - sum_{c' != c} U_{c'} ⊗ V_{c'} ⊗ W_{c'}
                # We want U_c to minimize ||residual - U_c ⊗ V_c ⊗ W_c||
                
                # Build the residual
                residual = T.copy()
                for c2 in range(r):
                    if c2 != c:
                        residual -= np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                
                # Now solve: minimize ||residual - U_c ⊗ V_c ⊗ W_c||_F
                # This is: minimize over U_c of ||residual - outer(outer(U_c, V_c), W_c)||
                # Flatten: residual_flat = residual.reshape(-1)
                # The term U_c ⊗ V_c ⊗ W_c flattened is a Kronecker product
                # Let's use the property: vec(U_c ⊗ V_c ⊗ W_c) = (W_c ⊗ V_c) ⊗ U_c ... 
                # Actually, it's easier to use the unfolding.
                
                # Alternatively, note that:
                # residual[i,j,k] ≈ U_c[i] * V_c[j] * W_c[k]
                # For fixed V_c and W_c, this is linear in U_c:
                # residual[i,j,k] = U_c[i] * (V_c[j] * W_c[k])
                # So for each i: sum_{j,k} residual[i,j,k] * (V_c[j]*W_c[k]) = U_c[i] * sum_{j,k} (V_c[j]*W_c[k])^2
                # Wait, that's not quite right. Let me think again.
                
                # We want to minimize: sum_{i,j,k} (residual[i,j,k] - U_c[i] V_c[j] W_c[k])^2
                # Taking derivative w.r.t. U_c[i]:
                # -2 sum_{j,k} (residual[i,j,k] - U_c[i] V_c[j] W_c[k]) V_c[j] W_c[k] = 0
                # sum_{j,k} residual[i,j,k] V_c[j] W_c[k] = U_c[i] sum_{j,k} V_c[j]^2 W_c[k]^2
                # U_c[i] = sum_{j,k} residual[i,j,k] V_c[j] W_c[k] / sum_{j,k} V_c[j]^2 W_c[k]^2
                
                Vc_Wc = np.outer(V[:, c], W[:, c])  # shape (m*p, n*p)
                denom = np.sum(Vc_Wc ** 2)
                if abs(denom) < 1e-15:
                    continue
                # numerator[i] = sum_{j,k} residual[i,j,k] Vc_Wc[j,k]
                numerator = np.einsum('ijk,jk->i', residual, Vc_Wc)
                U[:, c] = numerator / denom
            
            # Update V similarly
            for c in range(r):
                residual = T.copy()
                for c2 in range(r):
                    if c2 != c:
                        residual -= np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                
                Uc_Wc = np.outer(U[:, c], W[:, c])  # shape (n*m, n*p)
                denom = np.sum(Uc_Wc ** 2)
                if abs(denom) < 1e-15:
                    continue
                numerator = np.einsum('ijk,ik->j', residual, Uc_Wc)
                V[:, c] = numerator / denom
            
            # Update W similarly
            for c in range(r):
                residual = T.copy()
                for c2 in range(r):
                    if c2 != c:
                        residual -= np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                
                Uc_Vc = np.outer(U[:, c], V[:, c])  # shape (n*m, m*p)
                denom = np.sum(Uc_Vc ** 2)
                if abs(denom) < 1e-15:
                    continue
                numerator = np.einsum('ijk,ij->k', residual, Uc_Vc)
                W[:, c] = numerator / denom
            
            # Compute error
            approx = np.zeros_like(T)
            for c in range(r):
                approx += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            error = np.linalg.norm(T - approx)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < tol:
                break
        
        return best_error, best_U, best_V, best_W
    
    # Try various ranks
    best_rank = None
    best_decomp = None
    
    # Start from a reasonable lower bound. The tensor has "rank" at least max(n*m, m*p, n*p) = 20
    # But for matrix multiplication, the rank is at least the border rank, which for n×n × n×n is at least n^2.
    # For 4×4 × 4×5, let's try ranks from 20 upward.
    
    for r in range(20, 81):
        if time.time() - start_time > DEADLINE:
            break
        
        error, U, V, W = try_rank(r, max_iter=200, tol=1e-8)
        
        if error < 1e-6:
            best_rank = r
            best_decomp = (U, V, W)
            break
    
    if best_decomp is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((n * p, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_decomp
    
    # Try to round to small rationals/ints if close
    def try_round(mat):
        # Try to find a matrix with small integer/half-integer entries close to mat
        # Scale to make entries small
        max_val = np.max(np.abs(mat))
        if max_val < 1e-10:
            return mat
        # Try scaling by powers of 2
        for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            scaled = mat * scale
            rounded = np.round(scaled)
            # Check if rounding is close
            if np.linalg.norm(scaled - rounded) / max(np.linalg.norm(scaled), 1e-15) < 1e-6:
                return rounded / scale
        return mat
    
    U_r = try_round(U)
    V_r = try_round(V)
    W_r = try_round(W)
    
    # Verify the decomposition with rounded values
    approx = np.zeros_like(T)
    for c in range(best_rank):
        approx += np.einsum('i,j,k->ijk', U_r[:, c], V_r[:, c], W_r[:, c])
    if np.linalg.norm(T - approx) > 1e-6:
        # Rounding broke it, use unrounded
        U_r, V_r, W_r = U, V, W
    
    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
