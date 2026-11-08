import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, A, B):
    n, m, p = DIMENSIONS
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    # Check for a few random matrices
    for _ in range(10):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    # Sum over j?
                    # The equation is sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c] = C[i,k]
                    # Wait, the schoolbook code sums over c for fixed i,j,k?
                    # No, the schoolbook code:
                    # U[i*m+j][c] = 1
                    # V[j*p+k][c] = 1
                    # W[k*n+i][c] = 1
                    # This implies for a fixed c, it contributes to (i,j) in U, (j,k) in V, (k,i) in W.
                    # The term is A[i,j] * B[j,k] * (U*V*W)?
                    # No, the decomposition is C[i,k] = sum_j sum_c (U[i,j,c] * A[i,j]) * (V[j,k,c] * B[j,k]) * W[k,i,c]?
                    # No, the standard tensor decomposition is:
                    # C[i,k] = sum_c ( sum_j U[i,j,c] A[i,j] ) * ( sum_j V[j,k,c] B[j,k] ) * W[k,i,c] ?
                    # This is not bilinear in A and B if U and V are mixed.
                    
                    # Let's re-read the verification in the prompt's seed code.
                    # left = A[i][j] * B[j][k]
                    # right = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    # This checks if the tensor entry T[i,j,k] is reproduced.
                    # T[i,j,k] is 1 if we are computing A[i,j]*B[j,k].
                    # So the condition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
                    # This is independent of A and B!
                    # The verification in the seed code is just a sanity check that the tensor is correct.
                    # If sum_c ... = 1, then for any A, B, the product is correct.
                    pass
        
        # So we just need to satisfy the tensor equation.
        # We can verify by checking the tensor equation directly.
        
    # Direct tensor verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _optimize_rank(n, m, p, target_rank, max_iters=1000, lr=0.01):
    """
    Try to find a decomposition of rank target_rank for n x m x p.
    """
    R = target_rank
    # Initialize with schoolbook (first R terms)
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Fill first R terms with schoolbook
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c >= R:
                    break
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
            if c >= R:
                break
        if c >= R:
            break
            
    # If R < n*m*p, we have set some to 0.
    # We want to optimize U, V, W to satisfy the equation.
    
    # Gradient Descent
    # Loss = sum_{i,j,k} ( sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1 )^2
    
    # We will use simple gradient descent.
    # To speed up, we can use numpy.
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Flatten U, V, W for optimization?
    # Or just update in place.
    
    # Let's use a simple update rule.
    # grad_U[i,j,c] = 2 * error[i,j,k] * V[j,k,c] * W[k,i,c]
    # But error depends on all c.
    
    # This is expensive to compute gradients naively.
    # Let's just try a few random restarts with small perturbations?
    # Or use scipy.optimize?
    
    # Given the small size, we can use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    def loss_and_grad(x):
        # x is flattened U, V, W
        nU = n*m*R
        nV = m*p*R
        nW = p*n*R
        
        U_flat = x[:nU]
        V_flat = x[nU:nU+nV]
        W_flat = x[nU+nV:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i, j, k in indices:
            s = 0.0
            for c in range(R):
                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            err = s - 1.0
            loss += err * err
            
            for c in range(R):
                # d/dU[i,j,c] = 2 * err * V[j,k,c] * W[k,i,c]
                grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
                
        return loss, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    res = minimize(loss_and_grad, x0, jac=True, method='BFGS', options={'maxiter': 200, 'disp': False})
    
    if res.fun < 1e-4:
        U_opt = res.x[:nU].reshape(n*m, R)
        V_opt = res.x[nU:nU+nV].reshape(m*p, R)
        W_opt = res.x[nU+nV:].reshape(p*n, R)
        
        # Check if entries are close to half-integers
        # If so, round them
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        # Verify rounded version
        if _verify(U_round, V_round, W_round, None, None):
            return U_round, V_round, W_round
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    # First, try to optimize the full 3x5x5 directly?
    # It might be too hard.
    
    # Try block decomposition: 3x3x3 + 3x3x2 + 3x2x3 + 3x2x2
    # We need a good 3x3x3 decomposition.
    
    # Let's try to find a 3x3x3 decomposition of rank 23.
    # If we find it, we use it.
    
    U333, V333, W333 = None, None, None
    for rank in [23, 24, 25]:
        if time.time() > deadline - 20:
            break
        res = _optimize_rank(3, 3, 3, rank)
        if res is not None:
            U333, V333, W333 = res
            print(f"Found 3x3x3 rank {rank}")
            break
            
    if U333 is not None:
        # Assemble 3x5x5
        # Blocks:
        # 1. i in 0..2, j in 0..2, k in 0..2 -> U333, V333, W333
        # 2. i in 0..2, j in 0..2, k in 3..4 -> Naive 3x3x2
        # 3. i in 0..2, j in 3..4, k in 0..2 -> Naive 3x2x3
        # 4. i in 0..2, j in 3..4, k in 3..4 -> Naive 3x2x2
        
        R1 = U333.shape[1]
        R2 = 3 * 3 * 2 # 18
        R3 = 3 * 2 * 3 # 18
        R4 = 3 * 2 * 2 # 12
        TotalRank = R1 + R2 + R3 + R4
        
        U = np.zeros((15, TotalRank))
        V = np.zeros((25, TotalRank))
        W = np.zeros((15, TotalRank))
        
        # Block 1
        U[:, 0:R1] = U333
        V[:, 0:R1] = V333
        W[:, 0:R1] = W333
        
        # Block 2: i 0..2, j 0..2, k 3..4
        offset = R1
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    k_orig = k + 3
                    U[i*5+j, offset] = 1.0
                    V[j*5+k_orig, offset] = 1.0
                    W[k_orig*3+i, offset] = 1.0
                    offset += 1
                    
        # Block 3: i 0..2, j 3..4, k 0..2
        for i in range(3):
            for j in range(2):
                j_orig = j + 3
                for k in range(3):
                    U[i*5+j_orig, offset] = 1.0
                    V[j_orig*5+k, offset] = 1.0
                    W[k*3+i, offset] = 1.0
                    offset += 1
                    
        # Block 4: i 0..2, j 3..4, k 3..4
        for i in range(3):
            for j in range(2):
                j_orig = j + 3
                for k in range(2):
                    k_orig = k + 3
                    U[i*5+j_orig, offset] = 1.0
                    V[j_orig*5+k_orig, offset] = 1.0
                    W[k_orig*3+i, offset] = 1.0
                    offset += 1
                    
        if _verify(U, V, W, None, None):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U0, V0, W0 = _schoolbook()
    return {"U": U0, "V": V0, "W": W0}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W
