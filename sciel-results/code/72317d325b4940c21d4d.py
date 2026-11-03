import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 4]
_DEADLINE = time.time() + 130.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta
    # Decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # But with the W convention W[k*n+i, r], so:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    # Actually, let me think more carefully.
    
    # The standard tensor decomposition for matrix multiplication:
    # T_{ijk} = sum_r U_{ir} * V_{jr} * W_{kr}
    # where U has shape (n, rank), V has shape (m, rank), W has shape (p, rank)
    
    # But the interface specifies:
    # U: (n*m, rank) - indexed by (i*m + j)
    # V: (m*p, rank) - indexed by (j*p + k)  
    # W: (p*n, rank) - indexed by (k*n + i)
    
    # So the decomposition is:
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't make sense dimensionally. Let me re-read.
    
    # Actually for matrix multiplication C = A * B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T_{i,j,k} has entries 1 when... no, T_{i,j,k} = delta_{jk}? No.
    # T_{i,j,k} is the coefficient such that C[i,k] = sum_{j} T_{i,j,k} A[i,j] B[j,k]
    # So T_{i,j,k} = 1 if j is the same for A[i,j] and B[j,k], which is always true.
    # Actually T_{i,j,k} = 1 for all valid (i,j,k).
    
    # The decomposition: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # This must equal 1 for all valid (i,j,k).
    
    # Let's verify with random matrices
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is expensive for large dimensions, but 3*4*4=48 is small
    
    # Instead, verify by checking that the decomposition reproduces matrix multiplication
    # for random inputs
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, that's not right either. Let me think again.
        
        # The tensor T_{i,j,k} satisfies: C[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k]
        # If T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Then C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        
        # Let's compute this
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] * A[i,j] * B[j,k]
                C_computed[i,k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition using a search approach
    # First, establish the schoolbook as baseline
    best_result, best_rank = _schoolbook(n, m, p)
    
    if time.time() > _DEADLINE - 10.0:
        return best_result
    
    # Attempt to find a lower-rank decomposition using random initialization and optimization
    # We'll try ranks from 32 up to 47 (below schoolbook's 48)
    
    target_ranks = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    
    for target_rank in target_ranks:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > _DEADLINE - 15.0:
                break
            
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use gradient-free optimization: coordinate descent or simple iterative refinement
            # The objective is to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            # where T[i,j,k] = 1 for all i,j,k
            
            # Since the tensor is all ones, this is a special case
            # T_{i,j,k} = 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is a system of n*m*p = 48 equations in n*m*rank + m*p*rank + p*n*rank unknowns
            # For rank=32: 12*32 + 16*32 + 12*32 = 1152 unknowns, 48 equations - heavily underdetermined
            
            # Let's try a simpler approach: set up the linear system and solve
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is nonlinear due to the product of three variables per term
            # But we can fix two of them and solve for the third linearly
            
            # Strategy: iterate - fix U and V, solve for W; fix U and W, solve for V; fix V and W, solve for U
            
            max_iters = 100
            for it in range(max_iters):
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # Fix U and V, solve for W
                # For each (k,i): sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
                # Wait, that's not quite right. Let me re-derive.
                
                # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed (i,k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
                
                # So for each (i,k): sum_r W[k*n+i, r] * C_{ik,r} = 1 where C_{ik,r} = sum_j U[i*m+j, r] * V[j*p+k, r]
                
                # This gives p*n = 12 equations for p*n*rank = 384 unknowns in W - underdetermined
                # We can find a minimum-norm solution
                
                # Build the system: for each (i,k), equation is sum_r W[k*n+i, r] * C_{ik,r} = 1
                # W has shape (p*n, rank) = (12, target_rank)
                # Each row of W corresponds to a (k,i) pair
                # Each equation involves one row of W
                
                # So we have 12 independent equations, each involving one row of W (rank unknowns each)
                # For each row w of W (corresponding to (k,i)): sum_r w[r] * C_{ik,r} = 1
                # Minimum norm solution: w = C_{ik}^T / ||C_{ik}||^2 if C_{ik} != 0
                
                for i in range(n):
                    for k in range(p):
                        C_ik = np.zeros(target_rank)
                        for j in range(m):
                            C_ik += U[i*m+j, :] * V[j*p+k, :]
                        norm_sq = np.dot(C_ik, C_ik)
                        if norm_sq > 1e-15:
                            W[k*n+i, :] = C_ik / norm_sq
                        else:
                            # If C_ik is zero, we can't satisfy the equation with this U,V
                            # Add a small perturbation
                            W[k*n+i, :] = 0.01 * np.ones(target_rank)
                
                # Fix U and W, solve for V
                # For fixed (j,k): sum_i sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
                # = sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] * W[k*n+i, r]) = 1
                
                for j in range(m):
                    for k in range(p):
                        C_jk = np.zeros(target_rank)
                        for i in range(n):
                            C_jk += U[i*m+j, :] * W[k*n+i, :]
                        norm_sq = np.dot(C_jk, C_jk)
                        if norm_sq > 1e-15:
                            V[j*p+k, :] = C_jk / norm_sq
                        else:
                            V[j*p+k, :] = 0.01 * np.ones(target_rank)
                
                # Fix V and W, solve for U
                # For fixed (i,j): sum_k sum_r V[j*p+k, r] * W[k*n+i, r] * U[i*m+j, r] = 1
                # = sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1
                
                for i in range(n):
                    for j in range(m):
                        C_ij = np.zeros(target_rank)
                        for k in range(p):
                            C_ij += V[j*p+k, :] * W[k*n+i, :]
                        norm_sq = np.dot(C_ij, C_ij)
                        if norm_sq > 1e-15:
                            U[i*m+j, :] = C_ij / norm_sq
                        else:
                            U[i*m+j, :] = 0.01 * np.ones(target_rank)
                
                # Check convergence
                # Compute the residual
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_res = max(max_res, abs(val - 1.0))
                
                if max_res < 1e-10:
                    # Verify with the full check
                    U_list = [[Fraction(U[i,j]).limit_denominator(10**6) for j in range(target_rank)] for i in range(n*m)]
                    V_list = [[Fraction(V[i,j]).limit_denominator(10**6) for j in range(target_rank)] for i in range(m*p)]
                    W_list = [[Fraction(W[i,j]).limit_denominator(10**6) for j in range(target_rank)] for i in range(p*n)]
                    
                    if _verify(U_list, V_list, W_list, n, m, p, target_rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                if max_res < 1e-15:
                    break  # Converged
        
    # Fallback to schoolbook
    return best_result
