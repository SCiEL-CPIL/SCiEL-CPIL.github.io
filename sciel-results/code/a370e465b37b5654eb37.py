"""Rewritten program: search for a low-rank exact tensor decomposition of 2x3 by 3x7 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# Known lower bound for 2x3 * 3x7 is 21 (from n*m*p / max(n,m,p) heuristic? Actually lower bound is max(n*m, m*p, p*n) = max(6,21,14)=21)
# Schoolbook rank is 42. We aim to find rank 21 or lower.

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = 1 if (i,j) in A, (j,k) in B, output (i,k).
    The tensor is over indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    For matmul C = A*B, C[i,k] = sum_j A[i,j] * B[j,k].
    The bilinear tensor T has T[i,j,k] = 1 for all i,j,k (since every product term contributes).
    Actually, the tensor decomposition is: C[i,k] = sum_r U[i, r] * V[k, r] ... no.
    
    Let me reconsider. The standard tensor for matrix multiplication of (n x m) and (m x p) is:
    T[i, j, k] where i indexes rows of A and C, j indexes columns of A / rows of B, k indexes columns of B and C.
    T[i, j, k] = 1 for all i, j, k.
    
    A rank-r decomposition gives: T[i,j,k] = sum_{r=1}^r U[i, r] * V[j, r] * W[k, r]
    where U is n x r, V is m x r, W is p x r.
    
    But the task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    And the output factor follows W[k*n+i, r].
    
    So the decomposition is:
    C[i, k] = sum_{j} A[i, j] * B[j, k] = sum_r (sum_j U[i*m + j, r] * A[i,j]) * (sum_k ... )
    
    Actually, the standard formulation:
    Let A be n x m, B be m x p.
    C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    A rank-r bilinear algorithm:
    C[i, k] = sum_{r=0}^{r-1} (sum_j U[i, j, r] * A[i, j]) * (sum_k V[j, k, r] * B[j, k])
    
    But that's not quite right either. Let me think about the standard tensor decomposition.
    
    The matrix multiplication tensor T ∈ R^{n x m x p} is defined by T[i, j, k] = 1 for all i, j, k.
    A rank-r decomposition is: T[i, j, k] = sum_{r=0}^{r-1} U[i, r] * V[j, r] * W[k, r]
    
    But the task specifies U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    This suggests a different grouping.
    
    Looking at the seed:
    U[i*m + j, column] = 1
    V[j*p + k, column] = 1
    W[k*n + i, column] = 1
    
    So the decomposition is:
    For each (i, j, k), we have a rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    The bilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * (something that picks out the right output)
    
    Actually, I think the convention is:
    The product C = A*B has entries C[i, k] = sum_j A[i, j] * B[j, k].
    
    The tensor T[i, j, k] = 1 for all i, j, k represents the fact that each A[i,j] and B[j,k] pair contributes to C[i,k].
    
    A rank-r decomposition: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that doesn't make sense dimensionally for a bilinear form. Let me re-read.
    
    "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    So U has rows indexed by (i, j), V by (j, k), W by (k, i).
    
    The decomposition condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    This is a 3-way tensor decomposition where the "tensor" T has entries T[(i,j), (j,k), (k,i)] = 1.
    
    So T is a 3-tensor with indices (i*m+j), (j*p+k), (k*n+i) and T = 1 for all valid (i,j,k).
    
    We need: sum_r U[a, r] * V[b, r] * W[c, r] = 1 where a = i*m+j, b = j*p+k, c = k*n+i.
    
    This is equivalent to the standard matrix multiplication tensor rank problem.
    """
    n, m, p = n, m, p
    # The tensor is implicitly 1 for all (i, j, k)
    # We don't need to build it explicitly, just verify the decomposition
    return n, m, p

def _verify(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _round_to_half_integer(M, tol=1e-8):
    """Round entries to nearest half-integer."""
    M2 = 2.0 * M
    M_rounded = np.round(M2)
    if np.max(np.abs(M2 - M_rounded)) > tol:
        return None
    return M_rounded / 2.0

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook rank 42
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # 42
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # So rank >= 21
    
    # Strategy: Use random initialization and tensor train / CP decomposition search
    # Or use known results: for 2x3 * 3x7, the rank is known to be 21? Let's try to construct one.
    
    # Actually, let's try a structured approach.
    # The tensor T[i,j,k] = 1 can be decomposed if we can find rank-r factors.
    
    # Try rank 21 first (the lower bound)
    for target_rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for each rank
        max_restarts = 50 if target_rank <= 25 else 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            random.seed(restart * 1000 + target_rank)
            np.random.seed(restart * 1000 + target_rank)
            
            # Initialize randomly
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize to help convergence
            for r in range(target_rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    # Scale so that product of norms is 1
                    scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Use ALS (Alternating Least Squares) to minimize ||T - U*V*W||_F
            # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
            
            converged = False
            for it in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each (i,j), minimize sum_k W[k*n+i, :] dot (V[j*p+k, :] * ...)
                # Actually, the constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed V, W: U[i*m+j, :] = (sum_k V[j*p+k, :] * W[k*n+i, :])^T * 1 / ||...||^2
                # More precisely, we want: sum_r U[a, r] * (sum_k V[b, r] * W[c, r]) = 1
                # where a = i*m+j, b = j*p+k, c = k*n+i
                # This is: U[a, :] dot (sum_k V[b, :] * W[c, :]) = 1
                # So U[a, :] = (sum_k V[b, :] * W[c, :]) / ||sum_k V[b, :] * W[c, :]||^2
                # But this only works if we're fitting to 1.
                
                # Let's define: for each (i,j,k), the "effective" vector is g[i,j,k, r] = V[j*p+k, r] * W[k*n+i, r]
                # We want sum_r U[i*m+j, r] * g[i,j,k, r] = 1
                # This is a linear system: U[i*m+j, :] * G[i,j,:] = 1 where G[i,j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                
                # For each (i,j), we have multiple k values, so we have an overdetermined system:
                # sum_r U[i*m+j, r] * (sum_{k} V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
                # Wait, no. For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is p equations for each (i,j), with target_rank unknowns.
                
                # Let's build the system for each (i,j):
                # For k in 0..p-1: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let h_k[r] = V[j*p+k, r] * W[k*n+i, r]
                # Then: U[i*m+j, :] dot h_k = 1 for all k
                # This is a linear system: H * U[i*m+j, :]^T = 1 where H[k, r] = h_k[r]
                
                new_U = U.copy()
                for i in range(n):
                    for j in range(m):
                        # Build H matrix of shape (p, target_rank)
                        H = np.zeros((p, target_rank))
                        for k in range(p):
                            H[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        # Solve H * u = 1 (least squares)
                        try:
                            u, residuals, rank_h, sv = np.linalg.lstsq(H, np.ones(p), rcond=None)
                            new_U[i*m + j, :] = u
                        except:
                            pass
                
                U = new_U
                
                # Similarly update V and W
                new_V = V.copy()
                for j in range(m):
                    for k in range(p):
                        # For each (j,k), we need: sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
                        # Let h_i[r] = U[i*m+j, r] * W[k*n+i, r]
                        # H[i, r] = h_i[r], solve H * v = 1
                        H = np.zeros((n, target_rank))
                        for i in range(n):
                            H[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        try:
                            v, residuals, rank_h, sv = np.linalg.lstsq(H, np.ones(n), rcond=None)
                            new_V[j*p + k, :] = v
                        except:
                            pass
                V = new_V
                
                new_W = W.copy()
                for k in range(p):
                    for i in range(n):
                        # For each (k,i), we need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                        H = np.zeros((m, target_rank))
                        for j in range(m):
                            H[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        try:
                            w, residuals, rank_h, sv = np.linalg.lstsq(H, np.ones(m), rcond=None)
                            new_W[k*n + i, :] = w
                        except:
                            pass
                W = new_W
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0) ** 2
                if err < 1e-12:
                    converged = True
                    break
                
                if it % 20 == 0 and it > 0:
                    # Check if error is decreasing
                    pass
            
            if converged or _verify(U, V, W, n, m, p):
                # Try to round to half-integers
                U_rounded = _round_to_half_integer(U)
                V_rounded = _round_to_half_integer(V)
                W_rounded = _round_to_half_integer(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        found = True
                        break
                else:
                    # Use the unrounded solution if it's exact
                    if _verify(U, V, W, n, m, p):
                        # Check if entries are close to half-integers
                        if (np.max(np.abs(2*U - np.round(2*U))) < 1e-6 and
                            np.max(np.abs(2*V - np.round(2*V))) < 1e-6 and
                            np.max(np.abs(2*W - np.round(2*W))) < 1e-6):
                            best_U = np.round(2*U) / 2.0
                            best_V = np.round(2*V) / 2.0
                            best_W = np.round(2*W) / 2.0
                            best_rank = target_rank
                            found = True
                            break
        
        if found:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < n * m * p and best_U is not None:
        return {
            "U": [[float(x) for x in best_U[i, :]] for i in range(n * m)],
            "V": [[float(x) for x in best_V[i, :]] for i in range(m * p)],
            "W": [[float(x) for x in best_W[i, :]] for i in range(p * n)]
        }
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
