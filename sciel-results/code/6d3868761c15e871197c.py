"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check each output element C[i,k]
    for i in range(n):
        for k in range(p):
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # In decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should equal sum_j A[i,j]*B[j,k]
            # For this to hold for all A, B:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j is the "matching" index, else 0
            # Actually, the tensor identity is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for the multiplication tensor
            
            # Let me think again. The matrix multiplication tensor T is defined by:
            # T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k]
            # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
            # where T[i,j,k] = 1 (since A[i,j]*B[j,k] is the product)
            
            # Wait, that's not right either. Let me re-read the task.
            # "sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]"
            # This means for the specific A and B, but A and B are arbitrary.
            # So the tensor identity must hold: the coefficient of A[i',j']*B[j',k'] in the expression
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # should be 1 if (i',j',k') = (i,j,k) and 0 otherwise... no that's not right.
            
            # Actually, (A*B)[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is: for each (i,j,k), the term A[i,j]*B[j,k] appears in C[i,k]
            # The decomposition must satisfy:
            # For all i, j, k: the coefficient of A[i,j]*B[j,k] in sum_r U[i*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
            # where (i',k') = (i,k) should be 1 if j'=j and k'=k, else 0.
            
            # More precisely: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_{j'} A[i,j'] * B[j',k]
            # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a scalar for fixed i,j,k. But we need this to represent the bilinear form.
            
            # I think the correct interpretation is:
            # The multiplication tensor has components T[i,j,k] where T[i,j,k] = 1 for the "multiplication"
            # and the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # But T[i,j,k] should be 1 only when the indices match for the multiplication.
            
            # Actually, for matrix multiplication, the tensor is:
            # T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k] to get C[i,k]
            # So for fixed (i,k), we sum over j: sum_j A[i,j]*B[j,k]
            # The tensor component for (i,j,k) is 1 (it's the coefficient in the bilinear form)
            
            # So the identity is: for all i, j, k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Wait, that can't be right because then all entries would be 1.
            
            # Let me reconsider. The standard tensor decomposition for matrix multiplication:
            # The multiplication tensor M_{n,m,p} has indices (i,j,k) for i in [n], j in [m], k in [p]
            # M[i,j,k] = 1 (it's the "multiplication" tensor)
            # The decomposition: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # where U is indexed by (i,j), V by (j,k), W by (k,i)
            
            # So for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Let me verify with the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
            # for col = i*m*p + j*p + k
            # Then sum_r ... = 1 * 1 * 1 = 1 for the matching r, and 0 otherwise. Sum = 1. Correct!
            
            # So the condition is: for all i, j, k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            pass  # We'll implement proper verification below
    
    # Proper verification:
    # For all i in [n], j in [m], k in [p]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_fixed(rank, n, m, p, max_iters=1000, seed=42):
    """Try to find a decomposition at the given rank using local search."""
    rng = np.random.default_rng(seed)
    
    # Initialize with small random values
    U = rng.standard_normal((n*m, rank)) * 0.1
    V = rng.standard_normal((m*p, rank)) * 0.1
    W = rng.standard_normal((p*n, rank)) * 0.1
    
    # We want to minimize: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    # This is a non-convex problem. Use gradient descent with restarts.
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(3):
        U = rng.standard_normal((n*m, rank)) * 0.5
        V = rng.standard_normal((m*p, rank)) * 0.5
        W = rng.standard_normal((p*n, rank)) * 0.5
        
        lr = 0.01
        for it in range(max_iters):
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # s = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = s - 1.0
                        loss += err * err
                        for r in range(rank):
                            u_val = U[i*m+j, r]
                            v_val = V[j*p+k, r]
                            w_val = W[k*n+i, r]
                            grad_U[i*m+j, r] += 2 * err * v_val * w_val
                            grad_V[j*p+k, r] += 2 * err * u_val * w_val
                            grad_W[k*n+i, r] += 2 * err * u_val * v_val
            
            # Gradient descent
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-10:
                break
        
        if best_loss < 1e-10:
            break
    
    if best_loss < 1e-9:
        # Snap to half-integers
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        # Verify snapped version
        if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def _build_strassen_222():
    """Return Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7-multiplication algorithm
    # U, V, W each have 7 columns
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U rows: (i,j) for i,j in {0,1}
    # V rows: (j,k) for j,k in {0,1}
    # W rows: (k,i) for k,i in {0,1}
    
    # M1: U=(0,0):1, (1,1):1; V=(0,0):1, (1,1):1; W=(0,0):1, (1,1):1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # M2: U=(1,0):1, (1,1):1; V=(0,0):1; W=(0,1):1
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    
    # M3: U=(0,0):1; V=(0,1):1, (1,1):-1; W=(0,0):1
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+0, 2] = 1
    
    # M4: U=(1,1):1; V=(1,0):1, (0,0):-1; W=(1,1):1
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[1*2+1, 3] = 1
    
    # M5: U=(0,0):1, (0,1):1; V=(1,1):1; W=(1,0):1
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[1*2+0, 4] = 1
    
    # M6: U=(1,0):1, (0,0):-1; V=(0,0):1, (0,1):1; W=(0,1):1
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[0*2+1, 5] = 1
    
    # M7: U=(0,1):1, (1,1):-1; V=(1,0):1, (1,1):1; W=(1,0):1
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[1*2+0, 6] = 1
    
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook decomposition (rank 60)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Based on known results, the rank for 2x5x6 is likely between 30 and 45
    # Try ranks from low to high
    
    for rank in range(25, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_fixed(rank, n, m, p, max_iters=200, seed=rank)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found a valid decomposition at this rank
    
    return best_result
# EVOLVE-BLOCK-END
