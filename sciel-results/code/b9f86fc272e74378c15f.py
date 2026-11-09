import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    time_limit = 140
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Pre-compute flattened views for faster ALS
    T_flat_a = T.reshape(nm, mp * pn)
    T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    def als_solve(U, V, W, rank, max_iter=200, tol=1e-9):
        """Run ALS with pre-computed projections for speed."""
        nonlocal start_time, time_limit
        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Solve for U: T_a = U @ (V ⊙ W)^T
            # P_vw[r, :] = V[:,r] ⊗ W[:,r]
            P_vw = np.einsum('br,cr->bc', V, W).reshape(rank, mp * pn)
            U_new = np.linalg.lstsq(P_vw, T_flat_a, rcond=None)[0].T
            
            # Solve for V
            P_uw = np.einsum('ar,cr->ac', U, W).reshape(rank, nm * pn)
            V_new = np.linalg.lstsq(P_uw, T_flat_b, rcond=None)[0].T
            
            # Solve for W
            P_uv = np.einsum('ar,br->ab', U, V).reshape(rank, nm * mp)
            W_new = np.linalg.lstsq(P_uv, T_flat_c, rcond=None)[0].T
            
            U, V, W = U_new, V_new, W_new
            
            if it % 3 == 0:
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
                if err < tol:
                    return U, V, W, True
                if err > prev_err * 0.999 and it > 30:
                    break
                prev_err = err
        
        err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
        return U, V, W, err < tol
    
    # Try structured initializations that exploit the 2x3x9 structure
    # Key: m=3 is small, so we can try decompositions where V has structure
    
    for rank in range(rank_sb - 1, 29, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        found = False
        max_restarts = 6
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 10:
                break
            
            np.random.seed(12345 + restart * 997 + rank * 31)
            
            # Strategy 1: Random Gaussian
            if restart == 0:
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
            # Strategy 2: Sparse binary
            elif restart == 1:
                U = (np.random.rand(nm, rank) < 0.3).astype(float)
                V = (np.random.rand(mp, rank) < 0.3).astype(float)
                W = (np.random.rand(pn, rank) < 0.3).astype(float)
            # Strategy 3: Structured - exploit block structure
            elif restart == 2:
                # U has block structure based on i
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                # Assign columns to (i,j,k) triples with some sharing
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i*m+j, col] = 1.0
                                V[j*p+k, col] = 1.0
                                W[k*n+i, col] = 1.0
                                col += 1
                # Add random perturbation to remaining columns
                if col < rank:
                    U[:, col:] = np.random.randn(nm, rank - col) * 0.1
                    V[:, col:] = np.random.randn(mp, rank - col) * 0.1
                    W[:, col:] = np.random.randn(pn, rank - col) * 0.1
            # Strategy 4: Symmetric initialization
            elif restart == 3:
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
                # Make some entries equal to exploit symmetry
                for r in range(0, rank, 3):
                    if r + 1 < rank:
                        V[0, r+1] = V[1, r]
                        V[1, r+1] = V[2, r]
            # Strategy 5: Low-rank factorized init
            elif restart == 4:
                r1 = max(1, rank // 3)
                U = np.random.randn(nm, r1) @ np.random.randn(r1, rank) * 0.5
                V = np.random.randn(mp, r1) @ np.random.randn(r1, rank) * 0.5
                W = np.random.randn(pn, r1) @ np.random.randn(r1, rank) * 0.5
            # Strategy 6: Mixed
            else:
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
                # Scale differently
                U *= 0.5
                V *= 1.0
                W *= 0.5
            
            # Normalize
            nU = np.linalg.norm(U)
            nV = np.linalg.norm(V)
            nW = np.linalg.norm(W)
            U /= (nU + 1e-10)
            V /= (nV + 1e-10)
            W /= (nW + 1e-10)
            
            U, V, W, success = als_solve(U, V, W, rank, max_iter=150, tol=1e-8)
            
            if success:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            # Polish: run more iterations with tighter tolerance
            U, V, W, _ = als_solve(best_U, best_V, best_W, rank, max_iter=100, tol=1e-12)
            best_U, best_V, best_W = U, V, W
            continue
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
