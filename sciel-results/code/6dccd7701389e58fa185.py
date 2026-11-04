import numpy as np
import random
import time

DIMENSIONS = [2, 3, 9]

def construct():
    # We try to find a low rank decomposition for mu_2,3,3 and replicate it.
    # If we fail, we fall back to schoolbook.
    
    n, m, p = 2, 3, 3
    # Target rank for mu_2,3,3
    # Schoolbook is 18.
    # Try 16, 17.
    
    # Define the tensor T for mu_2,3,3
    # T[i, j, k] = 1 if consistent
    # We check consistency by indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def is_consistent(i, j, k):
        # i from 0..n-1, j from 0..m-1, k from 0..p-1
        # The tensor component is 1 for all i,j,k?
        # No, the tensor is defined by the multiplication.
        # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # It is 1 for all i,j,k.
        # Wait, the decomposition is sum_r u_r v_r w_r.
        # The product is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # This sum must equal 1 for all i,j,k?
        # No.
        # The equation is:
        # For each i, k:
        # sum_j A_{i,j} B_{j,k} = C_{i,k}
        # This is equivalent to:
        # sum_{i,j,k} A_{i,j} B_{j,k} C'_{i,k} = sum_{i,k} C_{i,k} C'_{i,k}
        # The tensor T has T_{i,j,k} = 1?
        # Let's check the schoolbook.
        # For each (i,j,k), we have a term.
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
        # Sum_r ... = 1 if (i,j,k) matches the r-th term.
        # So T_{i,j,k} = 1 for all i,j,k.
        # Yes, the matrix multiplication tensor has all entries 1 in the standard basis?
        # No, the standard basis is $e_{i,j} \otimes e_{j,k} \otimes e_{i,k}$.
        # The tensor is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
        # So yes, T_{i,j,k} = 1 for all valid i,j,k.
        
        return True

    # We want to find U, V, W such that
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of equations.
    # Number of equations: n*m*p = 18.
    # Number of variables: (n*m + m*p + p*n) * R = (6 + 9 + 6) * R = 21 R.
    
    # We search for integer solutions.
    
    best_rank = 18
    best_U = None
    best_V = None
    best_W = None
    
    # Try rank 16
    for R in [16, 17]:
        # Random restarts
        for _ in range(100): # Limit restarts
            # Initialize with random integers
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
            W = np.random.randint(-2, 3, (p*n, R))
            
            # Local search
            for step in range(1000):
                # Check error
                # Compute tensor product
                # We can vectorize this
                # T_comp[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                
                # Flatten indices
                # i in 0..1, j in 0..2, k in 0..2
                # We can loop over i,j,k
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            if val != 1:
                                error += 1
                                
                if error == 0:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    break
                    
                # Perturb
                # Pick a random matrix and entry
                mat = random.choice([U, V, W])
                row = random.randint(0, mat.shape[0]-1)
                col = random.randint(0, R-1)
                mat[row, col] += random.choice([-1, 1])
                
            if best_rank < 18:
                break
                
    if best_rank < 18:
        # Construct the final decomposition for 2x3x9
        # Replicate the 2x3x3 decomposition 3 times
        # B is 3x9, C is 2x9
        # We split B into B1, B2, B3 (3x3 each)
        # C into C1, C2, C3 (2x3 each)
        # The U factors are the same.
        # V and W factors are concatenated.
        
        final_rank = best_rank * 3
        final_U = np.zeros((2*3, final_rank), dtype=int)
        final_V = np.zeros((3*9, final_rank), dtype=int)
        final_W = np.zeros((9*2, final_rank), dtype=int)
        
        for block in range(3):
            # U is the same
            final_U[:, block*best_rank : (block+1)*best_rank] = best_U
            
            # V: indices j*p + k
            # For block b, the columns of B are 3b..3b+2
            # So k_global = 3b + k_local
            # V index = j*9 + k_global
            for j in range(3):
                for k in range(3):
                    v_idx = j*9 + (3*block + k)
                    final_V[v_idx, block*best_rank : (block+1)*best_rank] = best_V[j*3 + k, :]
                    
            # W: indices k*n + i
            # For block b, the columns of C are 3b..3b+2
            # So k_global = 3b + k_local
            # W index = k_global*2 + i
            for k in range(3):
                for i in range(2):
                    w_idx = (3*block + k)*2 + i
                    final_W[w_idx, block*best_rank : (block+1)*best_rank] = best_W[k*2 + i, :]
                    
        return {
            "U": final_U.tolist(),
            "V": final_V.tolist(),
            "W": final_W.tolist()
        }
    else:
        # Fallback to schoolbook
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
