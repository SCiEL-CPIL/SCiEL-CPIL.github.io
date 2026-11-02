import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

def _build_target_tensor():
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T_target=None):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    if T_target is None:
        T_target = _build_target_tensor()
    
    # Reconstruct: T_hat[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficient: T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_hat, T_target, atol=1e-8)

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

def _peeling_search(T_target, max_rank, deadline):
    """
    Tensor peeling / subspace embedding approach:
    1. Try to peel off rank-1 components one at a time.
    2. For each candidate rank r, try to find U, V, W with that rank.
    
    Strategy: Fix a sparse structured U, then solve linear system for V and W.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor has 128 ones. We need rank >= 128/8 = 16 (trivial lower bound)
    # Strassen gives 98. Let's try to find something better.
    
    best = None
    best_rank = max_rank + 1
    
    # Approach: For a given rank R, fix U to be a sparse binary matrix,
    # then the problem becomes: find V (mp x R) and W (pn x R) such that
    # sum_r U[:,r] V[:,r]^T W[:,r]^T = T_target
    # This is linear in V and W given U.
    
    # Actually, it's bilinear. But if we fix U, then for each (a,b,c):
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # This is still nonlinear in V and W jointly.
    
    # Better approach: Use the fact that the multiplication tensor has structure.
    # Try different structured U patterns and solve.
    
    # Alternative: Use randomized SVD-like peeling.
    # Start with T, find a rank-1 approximation, subtract, repeat.
    
    # Let's try a different structural approach:
    # The 4x4 * 4x8 multiplication can be seen as two 4x4 * 4x4 multiplications
    # with shared U factors. The Strassen tiling gives rank 98.
    # 
    # Can we do better by sharing more structure?
    # 
    # Let's try: Use the same U for both halves but with different V, W.
    # This is what the tiling already does.
    #
    # Another idea: Use a 4x4*4x8 specific algorithm.
    # The asymptotic rank of matrix multiplication is ~n^omega. For n=4, omega~2.8,
    # so rank ~ 4^2.8 * 4^0.5 ~ 100. Strassen gives 98.
    #
    # Let's try a direct search with structured U.
    
    # Method: For rank R, generate random sparse U (nm x R) with entries in {-1,0,1},
    # then solve the resulting system.
    
    # Actually, let's use a smarter approach:
    # The tensor T can be written as sum over (i,j,k) of e_{im+j} ⊗ e_{jp+k} ⊗ e_{kn+i}
    # 
    # Group by i: for fixed i, we have m*p terms.
    # Group by (i,k): for fixed (i,k), we have m terms.
    
    # Let's try a block structure:
    # U has blocks corresponding to i values
    # V has blocks corresponding to j,k values  
    # W has blocks corresponding to k,i values
    
    # Try rank = 64, 72, 80, 88, 96, 98
    for R in range(64, min(max_rank + 1, 100), 4):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random structured U patterns
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            # Generate a structured U: random permutation of identity-like blocks
            # U should have nm=16 rows, R columns
            # Each row of U should be sparse
            
            # Try: U is a random selection of R columns from a larger basis
            # Each column of U has exactly 2 non-zero entries (like Strassen)
            
            U = np.zeros((nm, R))
            for r in range(R):
                # Pick 2 random rows
                rows = np.random.choice(nm, size=2, replace=False)
                U[rows[0], r] = 1
                U[rows[1], r] = np.random.choice([-1, 1])
            
            # Now solve for V and W.
            # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
            # 
            # This is equivalent to: for each a, the matrix M_a = V * diag(U[a,:]) * W^T should equal T[a,:,:]
            # i.e., M_a[b,c] = sum_r V[b,r]*U[a,r]*W[c,r] = T[a,b,c]
            #
            # This is still nonlinear. But we can use alternating minimization:
            # Fix U, alternate between solving for V and W.
            
            # Initialize V, W randomly
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            # Alternating least squares
            for it in range(50):
                if time.time() > deadline - 30:
                    break
                    
                # Solve for V: for each b, V[b,:] should satisfy
                # sum_r V[b,r]*U[a,r]*W[c,r] = T[a,b,c] for all a,c
                # This is: V[b,:] * (U * W^T)[a,c] ... no, that's not right.
                #
                # Let me reformulate. Define for fixed b:
                # sum_r V[b,r] * (U[:,r] * W[c,r]) = T[:,b,c]
                # Let A_b[a,c] = sum_r U[a,r]*W[c,r]  (this is U @ W^T, same for all b)
                # Then we need: V[b,:] such that sum_r V[b,r]*U[a,r]*W[c,r] = T[a,b,c]
                # This can be written as: for each (a,c), sum_r V[b,r]*U[a,r]*W[c,r] = T[a,b,c]
                # 
                # This is a system of nm*pn equations in R unknowns V[b,:].
                # But it's over-determined and nonlinear in W.
                #
                # Better: Use gradient-based alternating optimization.
                
                # Compute current T_hat
                T_hat = np.einsum('ar,br,cr->abc', U, V, W)
                E = T_target - T_hat
                
                # Gradient for V: dL/dV[b,r] = -2 * sum_{a,c} E[a,b,c]*U[a,r]*W[c,r]
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', E, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', E, U, V)
                
                # Simple gradient step
                V -= 0.01 * grad_V
                W -= 0.01 * grad_W
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r, V_r, W_r, T_target):
                if R < best_rank:
                    best_rank = R
                    best = (U_r.copy(), V_r.copy(), W_r.copy())
                    print(f"Found rank {R}")
                    break  # Found one at this rank, try lower
            
            # Also try without rounding (allow floats)
            if _verify(U, V, W, T_target):
                if R < best_rank:
                    best_rank = R
                    best = (U.copy(), V.copy(), W.copy())
                    print(f"Found rank {R} (float)")
                    break
    
    return best

def _try_direct_construction():
    """
    Try to construct a lower-rank decomposition using known algebraic structures.
    
    The 4x4*4x8 multiplication tensor can potentially be decomposed more efficiently
    than simple tiling of Strassen.
    
    Idea: Use a 4x4*4x4 decomposition of rank R, and extend it to 4x4*4x8
    by adding terms for the second half of the 8 columns.
    """
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor()
    
    # Start with Strassen tiling (rank 98) and see if we can improve.
    U_s, V_s, W_s = _construct_strassen_tiled()
    if not _verify(U_s, V_s, W_s, T_target):
        return None
    
    # Try to find a rank-1 peeling: find u, v, w such that
    # T_target - outer(u,v,w) is still a valid multiplication tensor of lower rank.
    # This is hard to check directly.
    
    # Alternative: Use the structure that C = A*B where A is 4x4, B is 4x8.
    # B = [B1 | B2] where B1, B2 are 4x4.
    # C = A*B1 + A*B2 (but this is not how matrix multiplication works; it's C = A @ B)
    # Actually C[i,k] = sum_j A[i,j]*B[j,k]
    # For k=0..3: C[i,k] = sum_j A[i,j]*B[j,k]
    # For k=4..7: C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The Strassen tiling computes A*B1 and A*B2 separately, each with rank 49.
    # Total rank = 98.
    
    # Can we share some Strassen products between the two halves?
    # The Strassen products are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For B1 (left half), B1[j,k] for k=0..3
    # For B2 (right half), B2[j,k] for k=4..7, which is B[j,k]
    
    # The Strassen products for B1 use B1's blocks.
    # The Strassen products for B2 use B2's blocks.
    # There's no obvious sharing unless B1 and B2 have special structure.
    
    # So 98 might be optimal for this tiling approach.
    
    # Let's try a completely different approach: use a 4x8 as a single matrix
    # and find a direct decomposition.
    
    # The tensor has 4*4*8 = 128 ones.
    # A trivial lower bound on rank is 128/8 = 16 (each rank-1 term can contribute at most 8 ones... actually no, a rank-1 term u⊗v⊗w where u,v,w are 0/1 vectors with weights can contribute many ones).
    
    # Actually, the maximum number of 1s a single rank-1 term can contribute is |supp(u)|*|supp(v)|*|supp(w)|.
    # If we use 0/1 vectors, max is 16*32*32 = 16384, but we only need 128 specific ones.
    
    # The issue is that we need to hit exactly the right 128 positions and zero out the rest.
    
    # Let's try a random search with small rank.
    best = None
    best_rank = 100  # Strassen gives 98
    
    for R in range(48, 99):
        if time.time() > time.time() + 100:  # Safety
            break
            
        for trial in range(3):
            # Generate random sparse U, V, W
            U = np.zeros((16, R))
            V = np.zeros((32, R))
            W = np.zeros((32, R))
            
            for r in range(R):
                # U: 2 non-zeros
                u_rows = np.random.choice(16, size=2, replace=False)
                U[u_rows[0], r] = 1
                U[u_rows[1], r] = np.random.choice([-1, 1])
                
                # V: 2 non-zeros
                v_rows = np.random.choice(32, size=2, replace=False)
                V[v_rows[0], r] = 1
                V[v_rows[1], r] = np.random.choice([-1, 1])
                
                # W: 2 non-zeros
                w_rows = np.random.choice(32, size=2, replace=False)
                W[w_rows[0], r] = 1
                W[w_rows[1], r] = np.random.choice([-1, 1])
            
            if _verify(U, V, W, T_target):
                best = (U.copy(), V.copy(), W.copy())
                best_rank = R
                break
        if best is not None:
            break
    
    return best

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying 4x4 by 4x8 matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    T_target = _build_target_tensor()
    
    # Start with Strassen (rank 98)
    U_strassen, V_strassen, W_strassen = _construct_strassen_tiled()
    
    # Verify Strassen
    if not _verify(U_strassen, V_strassen, W_strassen, T_target):
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = U_strassen.shape[1]
    best_result = (U_strassen.copy(), V_strassen.copy(), W_strassen.copy())
    
    # Try peeling search
    if time.time() < deadline - 40:
        result = _peeling_search(T_target, 97, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if W_new.shape[1] < best_rank:
                best_rank = W_new.shape[1]
                best_result = (U_new, V_new, W_new)
                print(f"Peeling found rank {best_rank}")
    
    # Try direct construction
    if time.time() < deadline - 20:
        result2 = _try_direct_construction()
        if result2 is not None:
            U_new, V_new, W_new = result2
            if W_new.shape[1] < best_rank:
                best_rank = W_new.shape[1]
                best_result = (U_new, V_new, W_new)
                print(f"Direct found rank {best_rank}")
    
    # Final verification
    U_final, V_final, W_final = best_result
    if not _verify(U_final, V_final, W_final, T_target):
        U_final, V_final, W_final = U_strassen, V_strassen, W_strassen
    
    print(f"Final rank: {U_final.shape[1]}")
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
