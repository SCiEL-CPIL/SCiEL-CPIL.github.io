"""Matrix multiplication 4x4 x 4x7 - rank-minimal tensor decomposition."""
import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Factor sizes: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # Convention: W[k*n + i, c]
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the tensor in 3D form: T[i, j, k]
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Also store as flat index mapping for verification
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def verify(U, V, W, tol=1e-9):
        """Check if U @ V @ W reconstruction matches the tensor."""
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        # More efficiently: check the bilinear form
        # C = A @ B where A is n x m, B is m x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Our decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
        # Wait, let me think more carefully.
        # 
        # Standard: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T[i,j,k] = 1 (the "multiplication tensor")
        # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # With our indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        #
        # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        #
        # To verify efficiently, we can check: for each c, the contribution is
        # (outer product structure). Let's just do it directly for small sizes.
        rank = U.shape[1]
        # Reshape for batch computation
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each c: U[:,c] is (nm,), V[:,c] is (mp,), W[:,c] is (pn,)
        # Contribution to T[i,j,k] from c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient check: compute the reconstructed tensor
        # Rec[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        rec = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u_c = U[:, c]  # (nm,)
            v_c = V[:, c]  # (mp,)
            w_c = W[:, c]  # (pn,)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        rec[i, j, k] += u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        
        diff = np.max(np.abs(rec - T))
        return diff < tol, diff
    
    def verify_exact(U, V, W):
        """Check with exact rational arithmetic for small ranks."""
        rank = U.shape[1]
        # Convert to fractions
        Uf = [[Fraction(U[i, c]).limit_denominator(10**6) for c in range(rank)] for i in range(nm)]
        Vf = [[Fraction(V[i, c]).limit_denominator(10**6) for c in range(rank)] for i in range(mp)]
        Wf = [[Fraction(W[i, c]).limit_denominator(10**6) for c in range(rank)] for i in range(pn)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def try_rank(rank, max_iters=200, tol=1e-10, seed=None):
        """Try to find a decomposition of given rank using ALS."""
        rng = np.random.RandomState(seed)
        
        best_diff = np.inf
        best_U = best_V = best_W = None
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Random initialization
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Normalize initial columns
            for c in range(rank):
                norm = np.sqrt(np.sum(U[:,c]**2) + np.sum(V[:,c]**2) + np.sum(W[:,c]**2))
                if norm > 1e-10:
                    U[:,c] /= norm
                    V[:,c] /= norm
                    W[:,c] /= norm
            
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE - 10:
                    break
                    
                # Update U: for each i*m+j, U[i*m+j, c] = sum over k where V[j*p+k,c]*W[k*n+i,c] contributes
                # Actually, the constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system in U[i*m+j, :] for fixed (i,j)
                # For fixed (i,j): U[i*m+j, c] * (sum_k' ... ) -- no, it's per (i,j,k)
                
                # Let me reformulate. For fixed (i,j), the constraints are:
                # For each k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is: U[i*m+j, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) -- no
                # It's: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # So if we define M_k[c] = V[j*p+k, c] * W[k*n+i, c], then:
                # U[i*m+j, :] @ M_k = 1 for each k
                # This is: sum_c U[i*m+j, c] * M_k[c] = 1
                # In matrix form: U[i*m+j, :] @ M_k^T = 1, or M_k @ U[i*m+j, :]^T = 1
                
                # For fixed (i,j), we have p equations in rank unknowns (U[i*m+j, :]):
                # For k=0..p-1: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let A_{k,c} = V[j*p+k, c] * W[k*n+i, c]
                # Then A @ U[i*m+j, :]^T = ones(p)
                
                # If rank >= p, this is overdetermined; use least squares
                # If rank < p, may not be exactly solvable
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A matrix: (p, rank)
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ u = ones(p)
                        try:
                            u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=1e-10)
                        except:
                            u = np.zeros(rank)
                        U[i*m+j, :] = u
                
                # Update V: for fixed (j,k), constraints are:
                # For each i: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let B_{i,c} = U[i*m+j, c] * W[k*n+i, c]
                # Then B @ V[j*p+k, :]^T = ones(n)
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, rank))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(n), rcond=1e-10)
                        except:
                            v = np.zeros(rank)
                        V[j*p+k, :] = v
                
                # Update W: for fixed (k,i), constraints are:
                # For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let C_{j,c} = U[i*m+j, c] * V[j*p+k, c]
                # Then C @ W[k*n+i, :]^T = ones(m)
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, residuals, rank_C, sv = np.linalg.lstsq(C, np.ones(m), rcond=1e-10)
                        except:
                            w = np.zeros(rank)
                        W[k*n+i, :] = w
                
                # Check convergence
                diff, _ = verify(U, V, W)
                if diff < best_diff:
                    best_diff = diff
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if diff < tol:
                    break
            
            if best_diff < tol:
                break
        
        if best_diff < tol:
            return best_U, best_V, best_W, best_diff
        return None, None, None, best_diff
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But for matrix mult, better bounds exist. Let's try from 40 upward.
    # Known: 4x4x4 has rank 49 (Strassen-like improvements give less, but exact min is known to be ~49 or lower)
    # For 4x4x7, let's try a range
    
    best_result = None
    best_rank = nm * mp * pn  # schoolbook as fallback
    
    # Schoolbook as guaranteed fallback
    rank_school = n * m * p
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_result = (U_s, V_s, W_s)
    best_rank = rank_school
    
    # Try lower ranks
    for rank in range(40, min(rank_school, 110)):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        U, V, W, diff = try_rank(rank, max_iters=150, tol=1e-10, seed=rank*1000+42)
        
        if U is not None and diff < 1e-9:
            best_result = (U, V, W)
            best_rank = rank
            break  # Found minimal rank in this range
        
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # If we found something better, try to go lower
    if best_rank < rank_school:
        for rank in range(30, best_rank):
            if time.time() - start_time > DEADLINE - 15:
                break
            U, V, W, diff = try_rank(rank, max_iters=100, tol=1e-10, seed=rank*777+13)
            if U is not None and diff < 1e-9:
                best_result = (U, V, W)
                best_rank = rank
                break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(U[i, c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, c]) for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Final verification
    diff, _ = verify(U, V, W, tol=1e-8)
    if diff > 1e-6:
        # Fall back to schoolbook
        U_list = [[float(U_s[i, c]) for c in range(rank_school)] for i in range(nm)]
        V_list = [[float(V_s[i, c]) for c in range(rank_school)] for i in range(mp)]
        W_list = [[float(W_s[i, c]) for c in range(rank_school)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
