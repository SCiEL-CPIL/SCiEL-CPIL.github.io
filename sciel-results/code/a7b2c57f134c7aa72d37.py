"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 120
    
    # Build the target tensor T[i,j,k] = delta_{i,k} for matrix mult C[i,k] = sum_j A[i,j]*B[j,k]
    # In the trace convention: T[k*n+i, j] for W factor... 
    # Actually the tensor is: for each (i,j,k) triple, we need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c should give 1 if this is a valid decomposition
    
    # The tensor T is defined by: T[(i,j), (j,k), (k,i)] = 1 for all i,j,k
    # More precisely, the matrix multiplication tensor has entries:
    # T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want rank r such that T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is vector of size n*m, v_c of size m*p, w_c of size p*n
    
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Build target tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find low-rank decomposition
    # Strategy: random initialization + iterative refinement (like tensor train / CP decomposition)
    # But we need EXACT decomposition, so we'll use a different approach.
    
    # Approach: Try ranks from known bounds upward
    # Known: rank(4x5x5) <= 4*5*5 = 100 (schoolbook)
    # Better bounds exist. Let's try random CP decomposition with exact verification.
    
    def verify(U, V, W):
        """Check if U ⊗ V ⊗ W exactly reconstructs T."""
        # Compute sum over c of u_c ⊗ v_c ⊗ w_c
        # U is (nm, r), V is (mp, r), W is (pn, r)
        r = U.shape[1]
        # Reconstruct tensor
        recon = np.zeros((nm, mp, pn), dtype=float)
        for c in range(r):
            recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        return np.allclose(recon, T, atol=1e-10, rtol=1e-10)
    
    def try_rank(r, num_attempts=50, time_limit=10):
        """Try to find exact rank-r decomposition."""
        start_time = time.time()
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Iterative refinement: alternate fixing two factors and solving for the third
            # This is similar to ALS for CP decomposition
            for iteration in range(100):
                if time.time() > deadline - 5:
                    return None
                
                # Fix V, W -> solve for U
                # For each column c, we want U[:,c] such that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
                # This is a linear system. Flatten T and set up system.
                # T[a,b,c] = sum_c' U[a,c']*V[b,c']*W[c,c']
                # For fixed V, W: T = sum_c' (U[:,c'] ⊗ (V[:,c'] ⊙ W[:,c']))
                # Where ⊙ is element-wise product... no, that's not right.
                
                # Better: use the fact that for fixed V and W,
                # T = sum_c' U[:,c'] ⊗ (V[:,c'] ⊗ W[:,c'])
                # Flatten: T_flat = U @ M where M is constructed from V, W
                # T[a,b,c] = sum_c' U[a,c'] * V[b,c'] * W[c,c']
                # So T as matrix (nm x mp*pn): T_mat[a, b*pn+c] = sum_c' U[a,c'] * (V[b,c']*W[c,c'])
                # T_mat = U @ (V_Kron_W)^T where V_Kron_W[b*pn+c, c'] = V[b,c']*W[c,c']
                
                # Build the coefficient matrix
                # coeff[b*pn+c, c'] = V[b, c'] * W[c, c']
                coeff = np.zeros((mp * pn, r))
                for b in range(mp):
                    for c in range(pn):
                        for cprime in range(r):
                            coeff[b * pn + c, cprime] = V[b, cprime] * W[c, cprime]
                
                # T_mat[a, b*pn+c] = sum_c' U[a, c'] * coeff[b*pn+c, c']
                # So T_mat = U @ coeff^T
                T_mat = T.reshape(nm, mp * pn)
                
                # Solve U @ coeff^T = T_mat for U
                # U = T_mat @ (coeff^T)^+ (pseudo-inverse)
                try:
                    U_new = T_mat @ np.linalg.pinv(coeff.T)
                except:
                    continue
                
                # Fix U, W -> solve for V
                # T[a, b, c] = sum_c' U[a, c'] * V[b, c'] * W[c, c']
                # T_mat2[b, a*pn+c] = sum_c' V[b, c'] * (U[a, c'] * W[c, c'])
                coeff2 = np.zeros((nm * pn, r))
                for a in range(nm):
                    for c in range(pn):
                        for cprime in range(r):
                            coeff2[a * pn + c, cprime] = U[a, cprime] * W[c, cprime]
                
                T_mat2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new = T_mat2 @ np.linalg.pinv(coeff2.T)
                except:
                    continue
                
                # Fix U, V -> solve for W
                # T[a, b, c] = sum_c' U[a, c'] * V[b, c'] * W[c, c']
                # T_mat3[c, a*mp+b] = sum_c' W[c, c'] * (U[a, c'] * V[b, c'])
                coeff3 = np.zeros((nm * mp, r))
                for a in range(nm):
                    for b in range(mp):
                        for cprime in range(r):
                            coeff3[a * mp + b, cprime] = U[a, cprime] * V[b, cprime]
                
                T_mat3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = T_mat3 @ np.linalg.pinv(coeff3.T)
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                recon = np.zeros((nm, mp, pn), dtype=float)
                for c in range(r):
                    recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
                if np.allclose(recon, T, atol=1e-10, rtol=1e-10):
                    return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound for 4x5x5 is at least 4*5 + 5*5 - 1 = 39 (simple bound)
    # Actually, better lower bounds exist. Let's start from around 40-50.
    
    # First, let's try the schoolbook as fallback
    rank_school = nm * mp // p  # not right
    rank_school = n * m * p  # 100
    
    # Try a range of ranks
    best_result = None
    best_rank = rank_school
    
    # Try ranks in increasing order
    for r in range(40, 101):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_attempts=30, time_limit=min(8, max(1, (deadline - time.time()) - 15)))
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer entries if possible
            # Check if entries are close to integers or half-integers
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Round to nearest 0.5 if close
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[0])):
                        val = mat[i][j]
                        rounded = round(val * 2) / 2
                        if abs(val - rounded) < 1e-8:
                            mat[i][j] = rounded
                        else:
                            # Keep as is (should be integer for exact decompositions)
                            mat[i][j] = val
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
