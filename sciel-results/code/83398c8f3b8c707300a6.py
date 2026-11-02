import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Target tensor T is sparse: T[i*m+j, j*p+k, k*n+i] = 1
    # We need to find U (N x R), V (M x R), W (P x R) such that
    # sum_r U[:,r] V[:,r] W[:,r] = T
    
    # 1. Schoolbook Decomposition (Rank 45) - Always valid
    rank_schoolbook = n * m * p
    U_sb = np.zeros((N, rank_schoolbook))
    V_sb = np.zeros((M, rank_schoolbook))
    W_sb = np.zeros((P, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def residual(U, V, W):
        # Compute sum U_r V_r W_r
        return np.einsum('ar,br,cr->abc', U, V, W)

    def check_solution(U, V, W):
        # Check if U, V, W exactly decompose T
        # We only need to check the non-zero entries of T and a sample of zero entries?
        # No, "exactly decompose" means the tensor identity must hold.
        # However, checking the full tensor is O(N*M*P*R).
        # N*M*P = 9*15*15 = 2025. R is ~30-45. 2025*45 is small.
        
        # Construct T for comparison? Or just check the specific non-zero spots?
        # The tensor T is zero everywhere except at (i*m+j, j*p+k, k*n+i).
        # If the residual matches T at those points AND is zero elsewhere, it's valid.
        # But if we assume the structure of the problem (multiplication tensor), 
        # usually finding the non-zero parts is the constraint. 
        # Wait, the tensor is the *multiplication* tensor. 
        # The decomposition defines a bilinear map. 
        # If the map is correct, the tensor is correct.
        
        # Let's verify the bilinear map directly. It's more robust.
        # Generate random A, B
        # C = A B
        # C_decomp = sum_r (U_r^T A_vec) (V_r^T B_vec) W_r ?
        # No, the indices are flattened.
        # A is n x m. A_vec is length N. A_vec[i*m+j] = A[i,j].
        # B is m x p. B_vec is length M. B_vec[j*p+k] = B[j,k].
        # C is n x p. C_vec is length P? No, C is n x p. 
        # W is (p*n, R). Index k*n+i.
        # So the output vector C_vec has length P (p*n)?
        # The task says W shape is (p*n, rank).
        # The output of multiplication is n x p. 
        # The trace convention W[k*n+i, r] suggests the output index is k*n+i.
        # This corresponds to C[k, i]? No, C is n x p. 
        # Usually C[i, k]. 
        # If W index is k*n+i, then the output vector is indexed by (k, i).
        # This implies the output matrix is p x n? Or the indices are transposed?
        # "Output factor follows the official trace convention W[k*n+i,r]"
        # If A is n x m, B is m x p, C is n x p.
        # Term A[i,j] B[j,k] contributes to C[i,k].
        # The tensor index for C is usually (i, k).
        # Here W index is k*n + i. This is row k, col i? 
        # If the output matrix is C' of size p x n, where C'[k, i] = C[i, k] = C.T?
        # Or maybe the indices are just labeled that way.
        # Let's assume the standard tensor definition:
        # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        # This defines the tensor. We just need to match this tensor.
        
        # Construct T
        T = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        Res = residual(U, V, W)
        
        if not np.allclose(Res, T, atol=1e-9):
            return False
            
        # Check entries are integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    def try_rank(R):
        # Random restarts with simple gradient descent? 
        # Or just random search? 
        # Given time limit, random search in high dimensions is hopeless.
        # We need a smarter approach or a hardcoded known decomposition if possible.
        # Since we don't have one, we will try a heuristic local search.
        
        # Initialize with random values
        for restart in range(20):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # Simple gradient descent on ||Res - T||^2
            # Res = sum U_r V_r W_r
            # dL/dU = 2 * sum (Res - T) * V_r W_r
            # This is expensive to compute full tensor every step.
            # But N*M*P is small (2025).
            
            T = np.zeros((N, M, P))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m + j, j*p + k, k*n + i] = 1
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    return None
                
                Res = residual(U, V, W)
                Err = Res - T
                loss = np.sum(Err**2)
                
                # Check convergence
                if loss < 1e-8:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2.0
                    V_snap = np.round(V * 2) / 2.0
                    W_snap = np.round(W * 2) / 2.0
                    if check_solution(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    else:
                        # Try to fix snapping issues with a few more steps
                        pass
                
                # Gradients
                # dL/dU[:, r] = 2 * sum_{b,c} Err[a,b,c] * V[b,r] * W[c,r]
                # This is einsum('abc,br,cr->ar', Err, V, W)
                grad_U = 2 * np.einsum('abc,br,cr->ar', Err, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', Err, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', Err, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent blowup?
                # Not strictly necessary for small steps.
            
            # Final check after loop
            Res = residual(U, V, W)
            if np.allclose(Res, T, atol=1e-6):
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if check_solution(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
        
        return None

    # Try ranks lower than schoolbook
    # Theoretical lower bound is max(N, M, P) = 15.
    # Schoolbook is 45.
    # We try a few ranks. If we fail, we return schoolbook.
    
    # Note: Gradient descent on tensor rank is non-convex and very hard.
    # It's unlikely to find a solution for R < 45 in 100s with random init.
    # However, we must try to satisfy the "stronger algorithm" requirement,
    # even if it likely falls back to schoolbook.
    
    # To increase chance, we can try a "structured" initialization.
    # But without knowing the structure, random is all we have.
    
    # Let's try R = 44, 43, 42... down to 30.
    # We allocate time per rank.
    ranks_to_try = range(44, 29, -1)
    
    for R in ranks_to_try:
        if time.time() > deadline - 10: # Leave 10s for final check
            break
            
        result = try_rank(R)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            # If we found a lower rank, we could stop, but let's see if we can go lower?
            # Usually, if we find one, it's a big deal. But we might find a suboptimal one.
            # For now, we keep the best (lowest) found.
            # Since we iterate downwards, the first one found is the lowest in this loop.
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
