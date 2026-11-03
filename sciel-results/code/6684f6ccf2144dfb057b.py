import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T is defined by T[i, j, k] = 1 for all i, j, k
    # The decomposition condition is sum_r U[i, j, r] V[j, k, r] W[i, k, r] = 1
    # Note: The problem statement says W[k*n+i, r] in the trace convention description,
    # but the standard tensor decomposition for matrix multiplication C = A * B is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor form T_{i, j, k} = 1.
    # Decomposition: T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r}
    # Wait, the standard decomposition is usually:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # This can be written as sum_r (A_{i, j} U_{j, r}) (B_{j, k} V_{j, r}) ... no.
    
    # Let's stick to the contract:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This corresponds to:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # If we set:
    # U_{i, j, r} = A_{i, j} * u_{j, r} ? No, U is a factor.
    # Usually: C_{i, k} = sum_r (sum_j A_{i, j} U_{j, r}) (sum_l B_{l, k} V_{r, l}) ...
    
    # Let's look at the verify function in the seed:
    # val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
    # This is exactly sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} (with W indexed by k, i)
    # This must equal 1 for all i, j, k.
    
    # This is the tensor rank of the matrix multiplication tensor.
    # For 3x3x4, the known minimal rank is 32 (using Laderman for 3x3x3 part? No, 3x3x3 is 23).
    # Actually, the rank of 3x3x3 matrix multiplication is 23.
    # The rank of 3x3x4 matrix multiplication?
    # It is known that R(3,3,4) = 32.
    # Reference: "The rank of 3x3x4 matrix multiplication" or similar.
    # Actually, I recall that R(3,3,3)=23, R(4,4,4)=49.
    # R(3,3,4) is likely 32.
    
    # Strategy:
    # 1. Try to find a solution of rank 32 using random search + refinement.
    # 2. If that fails, use a known construction or fall back to schoolbook (rank 36).
    # 3. To improve the search, we can use the structure.
    #    The tensor is T_{i,j,k} = 1.
    #    We can decompose this into 4 tensors of size 3x3x1? No.
    #    We can decompose 3x3x4 as 4 * (3x3x1) ? No, the rank is not additive in that simple way.
    
    # Let's try a randomized search for rank 32.
    # We will use a local search approach.
    # Initialize U, V, W with random small integers.
    # Then, iteratively update one factor at a time to minimize the error.
    # Since the target is integer/half-integer, we can try to snap to these values.
    
    max_rank = n * m * p  # 36
    target_rank = 32
    
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(arr, tol=1e-6):
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    # Try rank 32
    for r in [32, 33, 34, 35, 36]:
        for attempt in range(50):
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(n * m, r)).astype(float)
            V = np.random.randint(-2, 3, size=(m * p, r)).astype(float)
            W = np.random.randint(-2, 3, size=(p * n, r)).astype(float)
            
            # Refinement loop
            improved = True
            max_iter = 100
            iter_count = 0
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # We want sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1 for all k
                        # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # We want A @ u = ones(p)
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        b = np.ones(p)
                        try:
                            u, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                            # Check if this reduces error
                            # Calculate new error
                            new_val = A @ u
                            old_val = A @ U[i * m + j, :]
                            if np.sum((new_val - 1)**2) < np.sum((old_val - 1)**2) - 1e-10:
                                U[i * m + j, :] = u
                                improved = True
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # We want sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = 1 for all i
                        # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # We want B @ v = ones(n)
                        B = np.zeros((n, r))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        b = np.ones(n)
                        try:
                            v, residuals, rank, s = np.linalg.lstsq(B, b, rcond=None)
                            new_val = B @ v
                            old_val = B @ V[j * p + k, :]
                            if np.sum((new_val - 1)**2) < np.sum((old_val - 1)**2) - 1e-10:
                                V[j * p + k, :] = v
                                improved = True
                        except:
                            pass
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        # We want sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1 for all j
                        # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # We want C @ w = ones(m)
                        C = np.zeros((m, r))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        b = np.ones(m)
                        try:
                            w, residuals, rank, s = np.linalg.lstsq(C, b, rcond=None)
                            new_val = C @ w
                            old_val = C @ W[k * n + i, :]
                            if np.sum((new_val - 1)**2) < np.sum((old_val - 1)**2) - 1e-10:
                                W[k * n + i, :] = w
                                improved = True
                        except:
                            pass
            
            # Check if we have a solution
            if verify(U, V, W, r):
                U_rounded = is_half_integer(U)
                V_rounded = is_half_integer(V)
                W_rounded = is_half_integer(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if verify(U_rounded, V_rounded, W_rounded, r):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        # If we didn't find a solution for this rank, try next
        # But we want minimal rank. If r=32 fails, we should try r=33 etc.
        # However, the loop above tries 32, then 33...
        # If r=32 fails, it moves to 33.
        
    # Fallback: Schoolbook algorithm (Rank 36)
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

# EVOLVE-BLOCK-END
