"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the target tensor T[i, j, k] where:
    # i indexes A (n*m = 4), j indexes B (m*p = 10), k indexes output (p*n = 10)
    # T[i, j, k] = 1 if A[i] * B[j] contributes to output[k], else 0
    # A is 2x2, B is 2x5, output is 2x5
    # A[i] = A[i//m, i%m] = A[i//2, i%2]
    # B[j] = B[j//p, j%p] = B[j//5, j%5]
    # output[k] = out[k//n, k%n] = out[k//2, k%2]
    # out[r, c] = sum_s A[r,s] * B[s,c]
    # So T[i, j, k] = 1 if (i//2 == k//2) and (i%2 == j//5) and (j%5 == k%2)
    
    nm = n * m  # 4
    mp = m * p  # 10
    pn = p * n  # 10
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(nm):
        ar, as_ = i // m, i % m
        for j in range(mp):
            bs, bc = j // p, j % p
            if ar == bs:
                for k in range(pn):
                    or_, oc = k // n, k % n
                    if or_ == ar and oc == bc:
                        T[i, j, k] = 1.0
    
    # Strategy: use tensor flattening + SVD to find a low-rank decomposition
    # Flatten modes (i,j) into one axis: matrix of shape (nm*mp, pn) = (40, 10)
    T_flat_ij = T.reshape(nm * mp, pn)
    
    # The matrix rank of this flattening gives a lower bound
    # But we want the tensor rank. Let's try to find a decomposition.
    
    # Approach: Assume the decomposition has the form where each rank-1 term
    # has structure. For matrix multiplication, rank-1 terms correspond to
    # products of linear forms in A, linear forms in B, and linear forms in output.
    
    # Let's try rank r from 14 to 18 and see if we can find an exact solution.
    
    def try_find_decomposition(r):
        """Try to find a rank-r exact decomposition."""
        # We'll use the fact that for matrix multiplication, the U, V, W matrices
        # have entries that are typically -1, 0, 1 (or half-integers for Strassen).
        
        # Use ALS (Alternating Least Squares) with exact rational solutions
        # Initialize using SVD of a random flattening
        
        # Try multiple initializations
        for init_trial in range(20):
            if time.time() > deadline:
                return None
            
            # Random initialization with structure
            np.random.seed(init_trial * 1000 + r)
            
            # For matrix multiplication tensors, a good initialization uses
            # the structure: each rank-1 term corresponds to a "product" pattern
            # Let's initialize U, V, W with small random integers
            
            U = np.random.randint(-1, 2, (nm, r)).astype(float)
            V = np.random.randint(-1, 2, (mp, r)).astype(float)
            W = np.random.randint(-1, 2, (pn, r)).astype(float)
            
            # ALS iterations
            for iteration in range(50):
                if time.time() > deadline:
                    return None
                
                # Update U: for each i, U[i,:] = (V^T * W^T * T[i,:,:]) / (V^T * W^T)
                # T[i,:,:] is (mp, pn), V is (mp, r), W is (pn, r)
                # We need: sum_s U[i,s] * V[j,s] * W[k,s] = T[i,j,k]
                # For fixed i: sum_j,k T[i,j,k] V[j,s] W[k,s] = sum_s U[i,s] (sum_j,k V[j,s'] W[k,s'] ... )
                # Actually: sum_s U[i,s] * (sum_j,k V[j,s] W[k,s] * delta) 
                # Let's do it properly:
                # T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
                # For fixed i, we have: for all j,k: T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
                # This is: vec(T[i,:,:]) = sum_s U[i,s] * (V[:,s] * W[:,s])
                # So: vec(T[i,:,:]) = M_i * U[i,:] where M_i[j*n+k, s] = V[j,s]*W[k,s]
                
                # Build M for current V, W
                # M has shape (mp*pn, r) where M[j*pn+k, s] = V[j,s] * W[k,s]
                M = np.zeros((mp * pn, r))
                for s in range(r):
                    M[:, s] = (V[:, s] * W[:, s]).reshape(mp * pn)
                
                # For each i, solve: M^T M * U[i,:] = M^T * vec(T[i,:,:])
                Mtm = M.T @ M
                try:
                    Mtm_inv = np.linalg.inv(Mtm)
                except:
                    Mtm_inv = np.linalg.pinv(Mtm)
                
                for i in range(nm):
                    rhs = M.T @ T[i, :, :].reshape(mp * pn)
                    U[i, :] = Mtm_inv @ rhs
                
                # Update V similarly
                # For fixed j: T[i,j,k] = sum_s V[j,s] U[i,s] W[k,s]
                # vec over (i,k): T[:,j,:] = sum_s V[j,s] * (U[:,s] * W[:,s])
                M2 = np.zeros((nm * pn, r))
                for s in range(r):
                    M2[:, s] = (U[:, s] * W[:, s]).reshape(nm * pn)
                
                M2tm = M2.T @ M2
                try:
                    M2tm_inv = np.linalg.inv(M2tm)
                except:
                    M2tm_inv = np.linalg.pinv(M2tm)
                
                for j in range(mp):
                    rhs = M2.T @ T[:, j, :].reshape(nm * pn)
                    V[j, :] = M2tm_inv @ rhs
                
                # Update W
                # For fixed k: T[i,j,k] = sum_s W[k,s] U[i,s] V[j,s]
                # vec over (i,j): T[:,:,k] = sum_s W[k,s] * (U[:,s] * V[:,s])
                M3 = np.zeros((nm * mp, r))
                for s in range(r):
                    M3[:, s] = (U[:, s] * V[:, s]).reshape(nm * mp)
                
                M3tm = M3.T @ M3
                try:
                    M3tm_inv = np.linalg.inv(M3tm)
                except:
                    M3tm_inv = np.linalg.pinv(M3tm)
                
                for k in range(pn):
                    rhs = M3.T @ T[:, :, k].reshape(nm * mp)
                    W[k, :] = M3tm_inv @ rhs
            
            # Check if we have an exact solution
            # Reconstruct and check
            recon = np.zeros((nm, mp, pn))
            for s in range(r):
                recon += np.outer(np.outer(U[:, s], V[:, s]), W[:, s])
            
            error = np.max(np.abs(recon - T))
            if error < 1e-10:
                # Check if entries are close to integers or half-integers
                def is_valid_entries(mat):
                    for val in mat.flat:
                        if abs(val - round(val)) < 1e-8:
                            continue
                        if abs(val * 2 - round(val * 2)) < 1e-8:
                            continue
                        return False
                    return True
                
                if is_valid_entries(U) and is_valid_entries(V) and is_valid_entries(W):
                    # Convert to exact values
                    def exactify(mat):
                        result = []
                        for row in mat:
                            new_row = []
                            for val in row:
                                if abs(val - round(val)) < 1e-8:
                                    new_row.append(int(round(val)))
                                else:
                                    new_row.append(int(round(val * 2)) / 2)
                            result.append(new_row)
                        return result
                    
                    return exactify(U), exactify(V), exactify(W)
        
        return None
    
    # Try ranks from low to high
    for r in range(14, 19):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: use the known rank-18 Strassen+schoolbook decomposition
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
