"""Optimized tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return np.allclose(tensor, 1.0, atol=1e-10)

def _schoolbook():
    """Return schoolbook decomposition, rank n*m*p."""
    n, m, p = DIMENSIONS
    r = n*m*p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_4x4_factors():
    """Return (U4, V4, W4) for 4x4*4x4 Strassen, rank 49."""
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    
    a_blocks = {0:[(0,0),(1,1)],1:[(1,0),(1,1)],2:[(0,0)],3:[(1,1)],4:[(0,0),(0,1)],5:[(1,0),(0,0)],6:[(0,1),(1,1)]}
    a_coeffs = {0:{0:1,1:1},1:{0:1,1:1},2:{0:1},3:{0:1},4:{0:1,1:1},5:{0:1,1:-1},6:{0:1,1:-1}}
    b_blocks = {0:[(0,0),(1,1)],1:[(0,0)],2:[(0,1),(1,1)],3:[(1,0),(0,0)],4:[(1,1)],5:[(0,0),(0,1)],6:[(1,0),(1,1)]}
    b_coeffs = {0:{0:1,1:1},1:{0:1},2:{0:1,1:-1},3:{0:1,1:-1},4:{0:1},5:{0:1,1:1},6:{0:1,1:1}}
    c_blocks = {0:[(0,0),(1,1)],1:[(1,0)],2:[(0,1)],3:[(0,0),(1,0)],4:[(0,0),(0,1)],5:[(1,1)],6:[(0,0)]}
    c_coeffs = {0:{0:1,1:1},1:{0:1},2:{0:1},3:{0:1,1:1},4:{0:-1,1:1},5:{0:1},6:{0:1}}
    
    for x in range(7):
        for y in range(7):
            col = x*7+y
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        uc = U2[ii*2+jj, y]
                        vc = V2[jj*2+kk, y]
                        wc = W2[kk*2+ii, y]
                        if uc != 0 and vc != 0 and wc != 0:
                            for idx, (bi, bj) in enumerate(a_blocks[x]):
                                U4[(2*bi+ii)*4+(2*bj+jj), col] += a_coeffs[x][idx] * uc
                            for idx, (bj, bk) in enumerate(b_blocks[x]):
                                V4[(2*bj+jj)*4+(2*bk+kk), col] += b_coeffs[x][idx] * vc
                            for idx, (bi, bk) in enumerate(c_blocks[x]):
                                W4[(2*bk+kk)*4+(2*bi+ii), col] += c_coeffs[x][idx] * wc
    return U4, V4, W4

def _peeling_search(U_fixed, V_fixed, max_rank, deadline, start_time):
    """
    Given fixed U and V matrices, try to find W by solving linear systems.
    For each rank r from min down to max_rank, check if the system is consistent.
    Returns (W, rank) if found, else (None, None).
    """
    n, m, p = DIMENSIONS
    nm = n*m
    mp = m*p
    pn = p*n
    
    # Build the tensor equations. For each (i,j,k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # For fixed U, V, this is a system in W. Each equation involves at most
    # r unknowns (the W entries that appear with nonzero U*V coefficients).
    #
    # We can set up: for each (i,j,k), the equation is:
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i, c] = 1
    #
    # This is a linear system A @ w = b where w is the flattened W.
    # But the system is overdetermined in general. We need to find if there
    # exists a W that satisfies all equations.
    #
    # Alternative approach: iterate. For each candidate rank r, we can try to
    # determine W column by column using a greedy/peeling approach.
    
    # Actually, let's use a different strategy: 
    # The tensor equation can be written as:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # If we fix U and V, this is a linear system in the entries of W.
    # The unknowns are W[k*n+i, c] for all (k,i,c).
    # 
    # Let's build the full linear system.
    # Number of equations: n*m*p = 4*4*8 = 128
    # Number of unknowns: p*n*r = 8*4*r = 32*r
    
    # For a given rank r, build the system
    for r in range(max_rank, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Build coefficient matrix A (128 x 32*r) and RHS b (128,)
        A = np.zeros((n*m*p, pn*r))
        b = np.ones(n*m*p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    for c in range(r):
                        coeff = U_fixed[i*m+j, c] * V_fixed[j*p+k, c]
                        if coeff != 0:
                            A[eq_idx, c*pn + k*n+i] = coeff
        
        # Solve the linear system. We need A @ w = b to be consistent.
        # Use least squares and check residual.
        try:
            w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            residual = np.linalg.norm(A @ w - b)
            if residual < 1e-8:
                W = w.reshape(pn, r)
                return W, r
        except np.linalg.LinAlgError:
            continue
    
    return None, None

def _try_random_patterns(deadline, start_time, max_trials=50):
    """
    Try random sparse patterns for U and V, then solve for W.
    """
    n, m, p = DIMENSIONS
    nm = n*m
    mp = m*p
    pn = p*n
    
    best_result = None
    best_rank = float('inf')
    
    rng = np.random.RandomState(42)
    
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
        
        # Try different ranks
        for r in [48, 40, 36, 32, 28, 24, 20, 16]:
            if time.time() > deadline - 10:
                break
            
            # Generate random sparse U and V
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            # Each column of U has ~2-4 nonzeros
            for c in range(r):
                nnz_u = rng.randint(2, 5)
                rows = rng.choice(nm, size=nnz_u, replace=False)
                U[rows, c] = rng.choice([-1, 1], size=nnz_u)
                
                nnz_v = rng.randint(2, 5)
                rows_v = rng.choice(mp, size=nnz_v, replace=False)
                V[rows_v, c] = rng.choice([-1, 1], size=nnz_v)
            
            W, found_rank = _peeling_search(U, V, r, deadline, start_time)
            if W is not None:
                # Verify
                U_test = U
                V_test = V
                W_test = W
                if _verify(U_test, V_test, W_test):
                    if found_rank < best_rank:
                        best_rank = found_rank
                        best_result = (U_test.copy(), V_test.copy(), W_test.copy())
                        print(f"  Found valid decomposition at rank {found_rank}")
                        # Don't break, keep searching for lower rank
    
    if best_result is not None:
        return best_result
    return None

def _construct_strassen_tiled():
    """Construct the tiled Strassen decomposition for 4x4*4x8."""
    U4, V4, W4 = _strassen_4x4_factors()
    r4 = U4.shape[1]
    total_rank = 2 * r4
    
    n, m, p = DIMENSIONS
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Left half (k=0..3)
    U_full[:, :r4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*8+k, :r4] = V4[j*4+k]
    for i in range(4):
        for k in range(4):
            W_full[k*4+i, :r4] = W4[k*4+i]
    
    # Right half (k=4..7)
    U_full[:, r4:] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*8+(k+4), r4:] = V4[j*4+k]
    for i in range(4):
        for k in range(4):
            W_full[(k+4)*4+i, r4:] = W4[k*4+i]
    
    return U_full, V_full, W_full

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying 4x4 by 4x8 matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Start with the known Strassen-based decomposition (rank 98)
    U_strassen, V_strassen, W_strassen = _construct_strassen_tiled()
    if _verify(U_strassen, V_strassen, W_strassen):
        print(f"Strassen tiled decomposition verified, rank {U_strassen.shape[1]}")
        best_rank = U_strassen.shape[1]
        best_result = (U_strassen.copy(), V_strassen.copy(), W_strassen.copy())
    else:
        print("Strassen verification failed, using schoolbook")
        U_s, V_s, W_s = _schoolbook()
        best_rank = U_s.shape[1]
        best_result = (U_s.copy(), V_s.copy(), W_s.copy())
    
    # Try peeling search with random patterns to find lower rank
    result = _try_random_patterns(deadline, start_time, max_trials=30)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new):
            if W_new.shape[1] < best_rank:
                best_rank = W_new.shape[1]
                best_result = (U_new, V_new, W_new)
                print(f"Improved to rank {best_rank}")
    
    # Final verification
    U_final, V_final, W_final = best_result
    if not _verify(U_final, V_final, W_final):
        print("Final verification failed, using schoolbook")
        U_final, V_final, W_final = _schoolbook()
    
    print(f"Final rank: {U_final.shape[1]}")
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
