"""Matrix multiplication 4x5x5 - structured block and randomized low-rank search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # Build schoolbook as baseline
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Fast verification using einsum
    def verify_fast(U, V, W, rank):
        """Use einsum for fast verification"""
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        
        return np.allclose(result, 1.0, atol=1e-9)
    
    # Verify schoolbook is valid
    assert verify_fast(U_sb, V_sb, W_sb, schoolbook_rank), "Schoolbook verification failed"
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Strategy 1: Structured Block Decomposition
    # Try to decompose the 4x5x5 multiplication into smaller blocks
    # E.g., split the 5x5 matrix into blocks and use smaller multiplication algorithms
    
    # For 4x5x5, we can try:
    # - Splitting the 5 dimension into 2+3
    # - Using 2x2x2, 2x3x3, 3x2x3, 3x3x3 blocks
    
    # Let's try a specific structured approach:
    # Split the 5x5 matrix into:
    # - Top-left 2x2 block
    # - Top-right 2x3 block
    # - Bottom-left 3x2 block
    # - Bottom-right 3x3 block
    
    # For each block, use the optimal known rank for that size
    # Known optimal ranks:
    # 2x2x2: rank 7 (Strassen)
    # 2x3x3: rank ? (let's use schoolbook 18 or try to find better)
    # 3x2x3: rank ?
    # 3x3x3: rank 23 (best known)
    
    # Actually, let's try a simpler structured approach:
    # Use the fact that 4x5x5 can be decomposed using smaller known decompositions
    
    # Try: 4x5x5 = (4x2x5) + (4x3x5)
    # 4x2x5: rank 4*2*5 = 40 (schoolbook) or better?
    # 4x3x5: rank 4*3*5 = 60 (schoolbook) or better?
    
    # This doesn't help. Let's try a different approach.
    
    # Strategy 2: Randomized Low-Rank Factor Recovery
    # Generate random low-rank factors, project the tensor, and try to recover integer/half-integer solutions
    
    np.random.seed(42)
    
    # Try ranks from 99 down to some lower bound
    # We'll try a few specific ranks that are known to be achievable or close to optimal
    # For 4x5x5, the lower bound is max(4*5, 5*5, 5*4) = 25
    # But realistically, we're looking for ranks close to 100 or significantly lower
    
    # Let's try a range of ranks
    ranks_to_try = list(range(99, 80, -1))  # Try ranks 99 down to 81
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        num_trials = 5
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Generate random initial factors
            U_init = np.random.randn(nm, rank)
            V_init = np.random.randn(mp, rank)
            W_init = np.random.randn(pn, rank)
            
            # Normalize to prevent overflow
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            # Project the tensor onto the low-rank manifold
            # The tensor T[i,j,k] = 1 for all (i,j,k) in the support
            # We want to find U, V, W such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # Use alternating minimization
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            for iteration in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # The constraint is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                        # This is a linear system in U[i*m+j, :]
                        # Let x = U[i*m+j, :]
                        # sum_r x[r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                        # This is: (V[j*p+k,r] * W[k*n+i,r])_k dot x = 1 for all k
                        # Let A[k, r] = V[j*p+k,r] * W[k*n+i,r]
                        # Then A @ x = 1
                        
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        try:
                            x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # The constraint is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                        # Let y = V[j*p+k, :]
                        # sum_r U[i*m+j,r] * y[r] * W[k*n+i,r] = 1 for all i
                        # Let B[i, r] = U[i*m+j,r] * W[k*n+i,r]
                        # Then B @ y = 1
                        
                        B = np.zeros((n, rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            y = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                            V[j * p + k, :] = y
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # The constraint is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                        # Let z = W[k*n+i, :]
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * z[r] = 1 for all j
                        # Let C[j, r] = U[i*m+j,r] * V[j*p+k,r]
                        # Then C @ z = 1
                        
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        try:
                            z = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                            W[k * n + i, :] = z
                        except:
                            pass
            
            # Check if the solution is valid
            if verify_fast(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Found a valid decomposition for this rank, try lower ranks
        
        if best_rank < rank:
            break  # Found a better rank, continue to lower ranks
    
    # If no improvement found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
