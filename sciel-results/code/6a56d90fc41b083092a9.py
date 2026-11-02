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
    
    T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_hat, T_target, atol=1e-8)

def _strassen_2x2_factors():
    """Return (U, V, W) for 2x2*2x2 Strassen, rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _strassen_4x4_factors():
    """Return (U4, V4, W4) for 4x4*4x4 Strassen, rank 49."""
    U2, V2, W2 = _strassen_2x2_factors()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    a_blocks = {
        0:[(0,0),(1,1)], 1:[(1,0),(1,1)], 2:[(0,0)], 3:[(1,1)],
        4:[(0,0),(0,1)], 5:[(1,0),(0,0)], 6:[(0,1),(1,1)]
    }
    a_coeffs = {
        0:{0:1,1:1}, 1:{0:1,1:1}, 2:{0:1}, 3:{0:1},
        4:{0:1,1:1}, 5:{0:1,1:-1}, 6:{0:1,1:-1}
    }
    b_blocks = {
        0:[(0,0),(1,1)], 1:[(0,0)], 2:[(0,1),(1,1)], 3:[(1,0),(0,0)],
        4:[(1,1)], 5:[(0,0),(0,1)], 6:[(1,0),(1,1)]
    }
    b_coeffs = {
        0:{0:1,1:1}, 1:{0:1}, 2:{0:1,1:-1}, 3:{0:1,1:-1},
        4:{0:1}, 5:{0:1,1:1}, 6:{0:1,1:1}
    }
    c_blocks = {
        0:[(0,0),(1,1)], 1:[(1,0)], 2:[(0,1)], 3:[(0,0),(1,0)],
        4:[(0,0),(0,1)], 5:[(1,1)], 6:[(0,0)]
    }
    c_coeffs = {
        0:{0:1,1:1}, 1:{0:1}, 2:{0:1}, 3:{0:1,1:1},
        4:{0:-1,1:1}, 5:{0:1}, 6:{0:1}
    }
    
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

def _try_shared_u_decomposition(T_target, max_rank, deadline):
    """
    Structural change: Fix U to be a shared structured matrix (same for both halves),
    then solve for V and W via linear system.
    
    The key insight: In the 4x4 * 4x8 multiplication, the U factor (input A) is the same
    for both halves of the output. We can fix U to a structured pattern and then solve
    for V and W.
    
    Strategy:
    1. Fix U to have a specific sparse structure (e.g., Strassen-like patterns).
    2. For each column r, we have U[:,r] fixed.
    3. The equations become: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    4. This is bilinear in V and W, but if we fix V, it's linear in W, and vice versa.
    5. Use alternating minimization with exact linear solves.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    best = None
    best_rank = max_rank + 1
    
    # Try different rank values
    for R in range(48, min(max_rank + 1, 100), 2):
        if time.time() > deadline - 30:
            break
        
        # Generate structured U: each column has 2-4 non-zero entries
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            
            # Create U with structured sparsity
            U = np.zeros((nm, R))
            for r in range(R):
                # Randomly choose 2-3 non-zero positions
                n_nonzero = random.choice([2, 2, 3])
                rows = np.random.choice(nm, size=n_nonzero, replace=False)
                for row in rows:
                    U[row, r] = random.choice([-1, 1])
            
            # Initialize V and W
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Alternating linear solves
            converged = False
            for it in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Solve for V: for each b, find V[b,:] such that
                # sum_r V[b,r] * (U[:,r] * W[c,r]) = T[:,b,c] for all c
                # This is: V[b,:] @ (U * W^T) = T[:,b,:]
                # Where (U * W^T)[a,c] = sum_r U[a,r]*W[c,r]
                
                UW = U @ W.T  # shape (nm, pn)
                
                # For each b, solve V[b,:] @ UW = T[:,b,:]
                # This is an overdetermined system: nm*pn equations, R unknowns
                # Use least squares
                for b in range(mp):
                    # Reshape T[:,b,:] to (nm*pn,)
                    rhs = T_target[:, b, :].flatten()
                    # UW is (nm, pn), we need V[b,:] @ UW = rhs
                    # This is: UW^T @ V[b,:] = rhs
                    UW_T = UW.T  # (pn, nm)
                    # Actually: sum_a V[b,r] * UW[a,c] for all a,c = rhs[a,c]
                    # Vectorize: (UW^T ⊗ I) @ V[b,:] = rhs
                    # Simpler: use np.linalg.lstsq on the reshaped system
                    
                    # Build the system: for each (a,c), sum_r V[b,r] * UW[a,c] = T[a,b,c]
                    # This is: V[b,:] @ UW = T[:,b,:]
                    # Flatten: (UW.T).reshape(pn*nm, R) @ V[b,:].reshape(R) = rhs
                    # Wait, that's not right either.
                    
                    # Let's think again. We have:
                    # sum_r V[b,r] * U[a,r] * W[c,r] = T[a,b,c]
                    # For fixed b, this is: V[b,:] @ (U * W^T)[a,c] = T[a,b,c]
                    # Where (U*W^T)[a,c] = sum_r U[a,r]*W[c,r] = UW[a,c]
                    # So: V[b,:] @ UW = T[:,b,:]
                    # This is a matrix equation: (R x 1) @ (nm x pn) = (nm x pn)
                    # Wait, dimensions don't match. V[b,:] is (R,), UW is (nm, pn)
                    # V[b,:] @ UW would be (R,) @ (nm, pn) which doesn't make sense.
                    
                    # Correct: sum_r V[b,r] * UW[a,c] = T[a,b,c]
                    # This means: for each (a,c), V[b,:] dot (UW[a,c] * ones(R)) = T[a,b,c]
                    # No, that's wrong. UW[a,c] is a scalar, not a vector.
                    
                    # Let me reformulate. Define M_b[a,c] = sum_r U[a,r] * W[c,r]
                    # Then the equation is: sum_r V[b,r] * U[a,r] * W[c,r] = T[a,b,c]
                    # This is NOT the same as V[b,:] @ M_b.
                    
                    # The correct formulation:
                    # For fixed b, we have nm*pn equations:
                    # sum_r V[b,r] * (U[a,r] * W[c,r]) = T[a,b,c] for all a,c
                    # Let A_b[a,c,r] = U[a,r] * W[c,r]
                    # Then: sum_r A_b[a,c,r] * V[b,r] = T[a,b,c]
                    # This is: (A_b flattened) @ V[b,:] = T[:,b,:] flattened
                    # A_b has shape (nm, pn, R), flatten to (nm*pn, R)
                    # Then: A_b_flat @ V[b,:] = T_flat
                    # Use least squares
                    
                    A_b = U[:, :, None] * W[None, :, :]  # (nm, pn, R)
                    A_b_flat = A_b.reshape(nm * pn, R)
                    T_flat = T_target[:, b, :].flatten()
                    
                    # Solve A_b_flat @ V[b,:] = T_flat using least squares
                    V_b, _, _, _ = np.linalg.lstsq(A_b_flat, T_flat, rcond=None)
                    V[b, :] = V_b
                
                # Solve for W: similar
                for c in range(pn):
                    A_c = U[:, :, None] * V[None, :, :]  # (nm, mp, R)
                    A_c_flat = A_c.reshape(nm * mp, R)
                    T_flat = T_target[:, :, c].flatten()
                    
                    W_c, _, _, _ = np.linalg.lstsq(A_c_flat, T_flat, rcond=None)
                    W[c, :] = W_c
                
                # Check convergence
                T_hat = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_hat - T_target)
                if err < 1e-6:
                    converged = True
                    break
            
            # Round to half-integers and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r, V_r, W_r, T_target):
                if R < best_rank:
                    best_rank = R
                    best = (U_r.copy(), V_r.copy(), W_r.copy())
                    print(f"Shared-U found rank {R}")
                    break
            elif _verify(U, V, W, T_target):
                if R < best_rank:
                    best_rank = R
                    best = (U.copy(), V.copy(), W.copy())
                    print(f"Shared-U found rank {R} (float)")
                    break
    
    return best

def _try_column_pruning(T_target, U_initial, V_initial, W_initial, deadline):
    """
    Start from the Strassen tiling (rank 98) and try to prune columns.
    A column can be pruned if its contribution is redundant.
    
    Strategy:
    1. Start with the full Strassen decomposition.
    2. For each column, check if removing it still allows exact reconstruction.
    3. If yes, remove it and repeat.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    U = U_initial.copy()
    V = V_initial.copy()
    W = W_initial.copy()
    rank = U.shape[1]
    
    # Try to prune columns
    for prune_round in range(50):
        if time.time() > deadline - 30:
            break
        
        pruned_any = False
        for col in range(rank - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            
            # Remove column col
            U_test = np.delete(U, col, axis=1)
            V_test = np.delete(V, col, axis=1)
            W_test = np.delete(W, col, axis=1)
            
            # Check if the remaining columns can still reconstruct T
            # This is hard to check directly. Instead, try to solve for the
            # missing contribution using the remaining columns.
            
            # Alternative: Check if the column is linearly dependent on others
            # in a tensor sense. This is complex.
            
            # Simpler heuristic: Try to see if the column's pattern is a
            # linear combination of other columns.
            
            # For now, skip this and use a different approach.
            pass
        
        if not pruned_any:
            break
    
    return U, V, W

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
    
    # Try shared-U decomposition (structural change)
    if time.time() < deadline - 40:
        result = _try_shared_u_decomposition(T_target, 97, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if W_new.shape[1] < best_rank:
                best_rank = W_new.shape[1]
                best_result = (U_new, V_new, W_new)
                print(f"Shared-U found rank {best_rank}")
    
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
