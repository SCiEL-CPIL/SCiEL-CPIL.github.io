import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # Build the matrix multiplication tensor
    # T[c, a, b] where c = k*n+i (output), a = i*m+j (A index), b = j*p+k (B index)
    # T[k*n+i, i*m+j, j*p+k] = 1
    
    # Shape: (p*n, n*m, m*p) = (8, 6, 12)
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    
    # Try to find minimal rank decomposition
    # The tensor T can be written as sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Try ranks from 18 upward
    for rank in range(18, 25):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Initialize with trivial decomposition (rank = n*m*p = 24)
        # Then try to reduce
        
        # Use greedy algorithm:
        # Start with residual = T
        # For each step, find the best rank-1 tensor to subtract
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        residual = T.copy()
        
        for r in range(rank):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Find the best rank-1 approximation to the residual
            # A rank-1 tensor is u ⊗ v ⊗ w
            # We want to minimize ||residual - u⊗v⊗w||_F^2
            
            # Use iterative approach: fix two, optimize the third
            # Start with random initialization
            
            best_u, best_v, best_w = None, None, None
            best_norm = np.inf
            
            # Try several random initializations
            ntries = 50
            for trial in range(ntries):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                # Random initialization
                u = np.random.randn(n * m)
                v = np.random.randn(m * p)
                w = np.random.randn(p * n)
                
                # Iterate: fix two, solve for the third
                for it in range(50):
                    # Solve for u: residual - v⊗w contracted over v,w
                    # (v⊗w)[a,b,c] = v[b]*w[c]
                    # We want u such that sum_b,c u[a]*v[b]*w[c] approximates residual[a,b,c]
                    # This is: u = argmin_u ||residual - u ⊗ v ⊗ w||
                    # = argmin_u sum_{a,b,c} (residual[a,b,c] - u[a]*v[b]*w[c])^2
                    # Derivative: sum_{b,c} v[b]*w[c]*residual[a,b,c] - u[a]*sum_{b,c} v[b]^2*w[c]^2 = 0
                    # u[a] = sum_{b,c} v[b]*w[c]*residual[a,b,c] / sum_{b,c} v[b]^2*w[c]^2
                    
                    vw = np.outer(v, w)  # (m*p, p*n)
                    # Contract residual with vw over b,c
                    # u[a] = sum_{b,c} residual[a,b,c] * vw[b,c]
                    u_new = np.tensordot(residual, vw, axes=([1, 2], [0, 1]))
                    denom = np.sum(vw ** 2)
                    if denom > 1e-12:
                        u = u_new / denom
                    
                    # Solve for v
                    uw = np.outer(u, w)  # (n*m, p*n)
                    v_new = np.tensordot(residual, uw, axes=([0, 2], [0, 1]))
                    denom = np.sum(uw ** 2)
                    if denom > 1e-12:
                        v = v_new / denom
                    
                    # Solve for w
                    uv = np.outer(u, v)  # (n*m, m*p)
                    w_new = np.tensordot(residual, uv, axes=([0, 1], [0, 1]))
                    denom = np.sum(uv ** 2)
                    if denom > 1e-12:
                        w = w_new / denom
                
                # Compute the norm of the approximation
                approx = np.einsum('a,b,c->abc', u, v, w)
                norm = np.linalg.norm(residual - approx)
                if norm < best_norm:
                    best_norm = norm
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_u is None:
                break
            
            # Subtract the best rank-1 tensor
            approx = np.einsum('a,b,c->abc', best_u, best_v, best_w)
            residual = residual - approx
            U[:, r] = best_u
            V[:, r] = best_v
            W[:, r] = best_w
        
        # Check if the decomposition is valid
        # Reconstruct T from U, V, W
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Check if T_recon matches T
        if np.allclose(T_recon, T, atol=1e-9):
            # Valid decomposition found!
            # Convert to integers or half-integers if possible
            U_int = U.astype(int)
            V_int = V.astype(int)
            W_int = W.astype(int)
            
            # Check if they're already integers
            if np.allclose(U, U_int) and np.allclose(V, V_int) and np.allclose(W, W_int):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
            # Try multiplying by 2 to get half-integers
            U_half = (U * 2).astype(int)
            V_half = (V * 2).astype(int)
            W_half = (W * 2).astype(int)
            
            if np.allclose(U * 2, U_half) and np.allclose(V * 2, V_half) and np.allclose(W * 2, W_half):
                # Return as fractions or keep as floats
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
            
            # Otherwise, return as is (shouldn't happen for exact decomposition)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: use the known rank-22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        raise TimeoutError("Exceeded time limit")
    
    return {"U": U, "V": V, "W": W}

