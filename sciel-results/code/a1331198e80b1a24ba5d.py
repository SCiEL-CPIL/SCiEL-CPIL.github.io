import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Target tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Else 0.
    # We only need to check the 128 non-zero entries and a sample of zeros?
    # No, for exactness, we should check all or rely on the structure.
    # But checking all 16*32*32 = 16384 entries is fast.
    
    # Reconstruct tensor
    # T_hat[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # Efficient reconstruction
    # We can loop over r
    T_hat = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Outer product
        # T_hat += u[:, None, None] * v[None, :, None] * w[None, None, :]
        T_hat += np.einsum('a,b,c->abc', u, v, w)
        
    # Check against target
    # Target is 1 at specific indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if abs(T_hat[a, b, c] - 1.0) > 1e-8:
                    return False
                
    # Check zeros? 
    # If the rank is low, it's unlikely to hit zeros by chance if the ones are correct?
    # No, we must check zeros to ensure it's the *multiplication* tensor and not something else.
    # However, the problem statement implies we are decomposing the multiplication tensor.
    # If we satisfy the 1s, do we automatically satisfy the 0s?
    # Not necessarily.
    # But checking all 16384 entries is fast enough.
    # Let's check all entries to be safe.
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
                
    if not np.allclose(T_hat, T_target, atol=1e-8):
        return False
    return True

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

def _gradient_descent(U, V, W, T_target, steps=1000, lr=0.01):
    """
    Perform gradient descent to minimize ||T_target - sum_r u_r v_r w_r||^2
    U, V, W are modified in place.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    R = U.shape[1]
    
    # Precompute target tensor flat or keep as 3D
    # T_target is (nm, mp, pn)
    
    # Initialize
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # We will use a simple SGD with momentum or just plain GD
    # For small R, plain GD might be okay, but momentum helps.
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    mu = 0.9
    
    for step in range(steps):
        # Compute T_hat
        T_hat = np.zeros((nm, mp, pn))
        for r in range(R):
            T_hat += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        # Compute Error
        E = T_target - T_hat
        
        # Compute Gradients
        # dL/dU[a, r] = -2 * sum_{b,c} E[a,b,c] V[b,r] W[c,r]
        # This can be computed as:
        # grad_U = -2 * np.einsum('abc,br,cr->ar', E, V, W)
        # Similarly for V and W
        
        grad_U = -2.0 * np.einsum('abc,br,cr->ar', E, V, W)
        grad_V = -2.0 * np.einsum('abc,ar,cr->br', E, U, W)
        grad_W = -2.0 * np.einsum('abc,ar,br->cr', E, U, V)
        
        # Update with momentum
        vU = mu * vU + lr * grad_U
        vV = mu * vV + lr * grad_V
        vW = mu * vW + lr * grad_W
        
        U -= vU
        V -= vV
        W -= vW
        
        # Optional: early stopping if error is small
        # if np.linalg.norm(E) < 1e-6:
        #     break
        
    return U, V, W

def _round_to_half(U, V, W):
    """Round entries to nearest 0.5."""
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return U, V, W

def _optimize_rank_reduction(U_init, V_init, W_init, target_rank, deadline, start_time):
    """
    Try to find a decomposition of rank target_rank by selecting columns from init
    and optimizing.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
                
    init_rank = U_init.shape[1]
    
    # Try a few random selections of columns
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        # Select target_rank columns from init
        # If target_rank > init_rank, we can't reduce. We assume target_rank <= init_rank.
        if target_rank > init_rank:
            continue
            
        cols = np.random.choice(init_rank, size=target_rank, replace=False)
        
        U_sub = U_init[:, cols].copy()
        V_sub = V_init[:, cols].copy()
        W_sub = W_init[:, cols].copy()
        
        # Optimize
        # Use a larger learning rate initially?
        U_opt, V_opt, W_opt = _gradient_descent(U_sub, V_sub, W_sub, T_target, steps=500, lr=0.05)
        
        # Round
        U_round, V_round, W_round = _round_to_half(U_opt, V_opt, W_opt)
        
        # Verify
        if _verify(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
    return None

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying 4x4 by 4x8 matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with Strassen (rank 98)
    U_strassen, V_strassen, W_strassen = _construct_strassen_tiled()
    
    # Verify Strassen
    if not _verify(U_strassen, V_strassen, W_strassen):
        # Fallback to schoolbook if Strassen fails (should not happen)
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
    
    # Try to reduce rank
    # Range: 90 down to 50
    for target_rank in range(90, 49, -2):
        if time.time() > deadline - 20:
            break
            
        result = _optimize_rank_reduction(U_strassen, V_strassen, W_strassen, target_rank, deadline, start_time)
        if result is not None:
            U_new, V_new, W_new = result
            if W_new.shape[1] < best_rank:
                best_rank = W_new.shape[1]
                best_result = (U_new, V_new, W_new)
                print(f"Improved to rank {best_rank}")
                # Continue searching for lower rank?
                # If we found one, we can try lower.
    
    # Final verification
    U_final, V_final, W_final = best_result
    if not _verify(U_final, V_final, W_final):
        U_final, V_final, W_final = U_strassen, V_strassen, W_strassen
        
    print(f"Final rank: {U_final.shape[1]}")
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
