"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Build target tensor: T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Reshape to 3D tensors
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # For each (i,j,k), compute the contraction
    # sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a 4th order tensor contraction. Let's do it efficiently.
    # Create a 4D tensor: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can iterate over c and accumulate
    
    result = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        # For each i,j,k: uc[i,j] * vc[j,k] * wc[k,i]
        # This is: for each i,j: uc[i,j] * (vc[j,:] * wc[:,i])
        # vc[j,:] is (p,), wc[:,i] is (p,), element-wise product is (p,)
        # So for each i,j: uc[i,j] * sum_k vc[j,k]*wc[k,i]
        # Actually we need the full (i,j,k) tensor
        # Let's do: temp[j,k] = vc[j,k] * wc[k,i] for each i
        # Better: use einsum or explicit loops
        
        # For efficiency with small dims, just do nested loops
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        if vc[j, k] != 0 and wc[k, i] != 0:
                            result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    # Check if all entries are exactly 1 (within tolerance)
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound for 4x5x6 matrix mult is at least max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    # But actual rank could be higher. Let's try from lower bounds upward.
    
    # Lower bound: the rank must be at least the maximum of the three "slice" dimensions
    # For matrix multiplication, rank >= max(n*p, n*m, m*p) is not quite right
    # A better lower bound: rank >= max(n, m, p) * max(n, m, p) ... no
    # Actually, for matmul(n,m,p), rank >= max(n*p, m*p, n*m) is a known lower bound? 
    # No, the standard lower bound is rank >= max(n, m, p)
    # But we can use: rank >= n*p (since fixing j gives a bilinear map from R^n x R^p to R^{n*p}... no)
    
    # Let's just try ranks from a reasonable lower bound
    # The schoolbook is 120. Known results for 4x5x6... let's try to reduce.
    # Lower bound: at least max(n*m, m*p, n*p) = max(20, 30, 24) = 30? 
    # Actually no. The tensor rank of matmul(n,m,p) is at least max(n*m, m*p, n*p) / something...
    # Let's just try a range of ranks.
    
    # Try ranks from 30 to best_rank-1
    # Use randomized search: generate random U, V, W with integer/half-integer entries
    # and check if they decompose the tensor.
    
    # For small ranks, we can try to solve the system of equations
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 120 equations in rank^2 * (n*m + m*p + p*n) variables
    # which is nonlinear. Hard to solve directly.
    
    # Alternative: use the fact that we can fix U and V and solve for W, or vice versa.
    # If we fix U and V, then for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W! For each k, we have n*m equations in p*n unknowns for W[k*n:..., :]
    # Wait, W is indexed by [k*n+i, c]. For fixed k, the unknowns are W[k*n+i, c] for i in 0..n-1, c in 0..rank-1.
    # The equations for fixed k are: for each i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is n*m equations in n*rank unknowns (W[k*n+i, c] for i,c).
    # If n*rank >= n*m, i.e., rank >= m, we might have a solution.
    
    # Strategy: 
    # 1. Randomly generate U (n*m x rank) and V (m*p x rank) with small integer/half-integer entries
    # 2. For each k, solve the linear system for W[k*n:..., :]
    # 3. Check if the solution is exact and has nice entries (integer/half-integer)
    
    # Let's try this approach
    for target_rank in range(max(n, m, p), best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U and V with entries from {-1, 0, 1} or {-2, -1, 0, 1, 2}
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            
            # For each k, solve for W[k*n:..., :]
            W = np.zeros((p*n, target_rank))
            valid = True
            
            for k in range(p):
                # Build the linear system: for each (i,j), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a linear system in W[k*n+i, c]
                # Let's form the matrix A where A[(i,j), (i',c)] = U[i*m+j,c] * V[j*p+k,c] if i==i' else 0
                # Actually, the equation for (i,j) is: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
                # So for fixed i, the equation involves W[k*n+i, c] for all c
                # The coefficient of W[k*n+i,c] in equation (i,j) is U[i*m+j,c] * V[j*p+k,c]
                
                # So the system is: for each i, j: sum_c M[i,j,c] * W[k*n+i,c] = 1
                # where M[i,j,c] = U[i*m+j,c] * V[j*p+k,c]
                
                # This can be written as: for each i, the row vector M[i,:,c] dot W[k*n+i,:] = 1 for all j
                # So for each i, we have m equations in rank unknowns: W[k*n+i,:]
                
                for i in range(n):
                    # M[i, j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # We want: for all j, sum_c M[i,j,c] * W[k*n+i,c] = 1
                    # This is: M_i @ W[k*n+i,:] = ones(m)
                    # where M_i is (m x rank) with M_i[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    
                    M_i = np.zeros((m, target_rank))
                    for j in range(m):
                        M_i[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve M_i @ x = ones(m) for x = W[k*n+i, :]
                    # This is an underdetermined system (m equations, rank unknowns)
                    # We want a solution with small integer/half-integer entries
                    # Use least squares or find a particular solution
                    
                    try:
                        # Use np.linalg.lstsq to get a solution
                        x, residuals, rank_s, sv = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                        # Check if this is an exact solution
                        if not np.allclose(M_i @ x, np.ones(m), atol=1e-9):
                            valid = False
                            break
                        W[k*n+i, :] = x
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Check if W has nice entries (integer or half-integer)
                # Round to nearest half-integer and check
                W_rounded = np.round(W * 2) / 2
                if np.allclose(W, W_rounded, atol=1e-6):
                    # Verify the full decomposition
                    U_list = [[float(x) for x in row] for row in U.tolist()]
                    V_list = [[float(x) for x in row] for row in V.tolist()]
                    W_list = [[float(x) for x in row] for row in W_rounded.tolist()]
                    
                    if _verify(U_list, V_list, W_list):
                        found = True
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
