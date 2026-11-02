import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0 - 5.0 # 5s safety margin
    
    # Target rank for 3x3x2 block
    # We assume rank 12 is possible for 3x3x2
    block_rank = 12
    
    # Try to find a decomposition for 3x3x2
    U_block, V_block, W_block = find_3x3x2_decomposition(block_rank, deadline)
    
    if U_block is not None:
        # Assemble 3x3x6 decomposition
        # The k indices are 0,1 for block 0; 2,3 for block 1; 4,5 for block 2
        rank = block_rank * 3
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for b in range(3):
            offset = b * block_rank
            # U is the same for all blocks (shared A)
            # But wait, if U is shared, the rank is not simply additive for U.
            # The decomposition is sum of rank-1 terms.
            # For block b, we have terms r=0..11.
            # U_{i,j, offset+r} = U_block_{i,j,r}
            # V_{j,k, offset+r} = V_block_{j, k-2b, r} if k in 2b..2b+1 else 0
            # W_{k,i, offset+r} = W_block_{k-2b, i, r} if k in 2b..2b+1 else 0
            
            for i in range(n):
                for j in range(m):
                    for r in range(block_rank):
                        U[i * m + j][offset + r] = U_block[i * m + j][r]
                        
            for j in range(m):
                for kb in range(2): # k index within block
                    k = 2 * b + kb
                    for r in range(block_rank):
                        V[j * p + k][offset + r] = V_block[j * 2 + kb][r]
                        
            for kb in range(2):
                k = 2 * b + kb
                for i in range(n):
                    for r in range(block_rank):
                        W[k * n + i][offset + r] = W_block[kb * n + i][r]
                        
        # Verify
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
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

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def find_3x3x2_decomposition(rank, deadline):
    n, m, p = 3, 3, 2
    target = np.ones((n, m, p))
    
    # Random restarts
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # ALS
        for it in range(100):
            if time.time() > deadline:
                return None, None, None
            
            # Update U
            # For each (i,j), solve sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
            # This is linear in U[i,j,:].
            # Let A[i,j] be the vector of length rank.
            # We want A[i,j] . M[i,j] = 1, where M[i,j,r] = sum_k V[j,k,r] W[k,i,r]
            # Actually, the equation is sum_k sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # sum_r U[i,j,r] (sum_k V[j,k,r] W[k,i,r]) = 1
            # Let C[i,j,r] = sum_k V[j,k,r] W[k,i,r]
            # Then sum_r U[i,j,r] C[i,j,r] = 1
            # This is a single linear equation for each (i,j).
            # We have 1 equation and `rank` unknowns for each (i,j).
            # We can set U[i,j,:] to be the minimum norm solution:
            # U[i,j,:] = C[i,j,:] / (C[i,j,:] . C[i,j,:])
            # But this might not lead to a valid global solution if the system is inconsistent or if we need to balance V and W.
            # Standard ALS for tensor decomposition minimizes the Frobenius norm of the difference.
            # Here we have equality constraints.
            # Let's use a least squares approach for the whole tensor.
            
            # Residual R[i,j,k] = target[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We want to minimize sum R^2.
            
            # Update U:
            # For fixed V, W, the problem for U is linear.
            # Flatten U to vector u.
            # The residual is linear in u.
            # R = target - M u, where M is a matrix of size (n*m*p) x (n*m*rank).
            # This is too large.
            # Instead, update one (i,j) at a time? No, they are coupled via V, W.
            # But for fixed V, W, the equations for different (i,j) are independent?
            # Equation for (i,j): sum_k sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # This equation only involves U[i,j,:].
            # So yes, we can solve for each U[i,j,:] independently!
            
            for i in range(n):
                for j in range(m):
                    C = np.zeros(rank)
                    for k in range(p):
                        C += V[j * p + k, :] * W[k * n + i, :]
                    # We want sum_r U[i,j,r] C[r] = 1
                    # If C is zero, we have a problem.
                    norm_sq = np.dot(C, C)
                    if norm_sq < 1e-12:
                        # Fallback: random or previous value
                        pass
                    else:
                        U[i * m + j, :] = C / norm_sq
                        
            # Update V:
            # Equation for (j,k): sum_i sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # sum_r V[j,k,r] (sum_i U[i,j,r] W[k,i,r]) = 1
            for j in range(m):
                for k in range(p):
                    C = np.zeros(rank)
                    for i in range(n):
                        C += U[i * m + j, :] * W[k * n + i, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq < 1e-12:
                        pass
                    else:
                        V[j * p + k, :] = C / norm_sq
                        
            # Update W:
            # Equation for (k,i): sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # sum_r W[k,i,r] (sum_j U[i,j,r] V[j,k,r]) = 1
            for k in range(p):
                for i in range(n):
                    C = np.zeros(rank)
                    for j in range(m):
                        C += U[i * m + j, :] * V[j * p + k, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq < 1e-12:
                        pass
                    else:
                        W[k * n + i, :] = C / norm_sq
                        
        # Check convergence
        residual = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual += (s - 1) ** 2
                    
        if residual < 1e-6:
            # Snap to rationals
            U_int = snap_to_rational(U, 8)
            V_int = snap_to_rational(V, 8)
            W_int = snap_to_rational(W, 8)
            
            # Verify exact
            if verify_numpy(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
                
    return None, None, None

def snap_to_rational(mat, denom):
    # Multiply by denom, round, divide
    scaled = mat * denom
    rounded = np.round(scaled)
    snapped = rounded / denom
    return snapped

def verify_numpy(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
