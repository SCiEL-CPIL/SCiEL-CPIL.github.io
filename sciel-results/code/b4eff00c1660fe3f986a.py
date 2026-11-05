"""Matrix multiplication 4x5x5 - Strassen baseline + linear-algebraic rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    start_time = time.time()
    deadline = 135.0

    def build_baseline():
        rank_base = 85
        U = np.zeros((nm, rank_base), dtype=float)
        V = np.zeros((mp, rank_base), dtype=float)
        W = np.zeros((pn, rank_base), dtype=float)

        s_u = np.zeros((4, 7), dtype=float)
        s_v = np.zeros((4, 7), dtype=float)
        s_w = np.zeros((4, 7), dtype=float)
        s_u[0,0]=1; s_u[3,0]=1; s_v[0,0]=1; s_v[3,0]=1; s_w[0,0]=1; s_w[3,0]=1
        s_u[2,1]=1; s_u[3,1]=1; s_v[0,1]=1; s_w[2,1]=1; s_w[3,1]=-1
        s_u[0,2]=1; s_v[1,2]=1; s_v[3,2]=-1; s_w[1,2]=1; s_w[3,2]=1
        s_u[3,3]=1; s_v[2,3]=1; s_v[0,3]=-1; s_w[0,3]=1; s_w[2,3]=1
        s_u[0,4]=1; s_u[1,4]=1; s_v[3,4]=1; s_w[0,4]=-1; s_w[1,4]=1
        s_u[0,5]=-1; s_u[2,5]=1; s_v[0,5]=1; s_v[1,5]=1; s_w[3,5]=1
        s_u[1,6]=1; s_u[3,6]=-1; s_v[2,6]=1; s_v[3,6]=1; s_w[0,6]=1

        def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
            for inner_k in range(7):
                col = col_start + inner_k
                u_vec = s_u[:, inner_k]; v_vec = s_v[:, inner_k]; w_vec = s_w[:, inner_k]
                for (a_br, a_bc), a_coeff in a_blocks.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2+lc
                            u_val = u_vec[local_idx]
                            if u_val != 0:
                                gi = 2*a_br+lr; gj = 2*a_bc+lc
                                U[gi*5+gj, col] += a_coeff*u_val
                for (b_br, b_bc), b_coeff in b_blocks.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2+lc
                            v_val = v_vec[local_idx]
                            if v_val != 0:
                                gj = 2*b_br+lr; gk = 2*b_bc+lc
                                V[gj*5+gk, col] += b_coeff*v_val
                for (c_br, c_bc), c_coeff in c_blocks.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2+lc
                            w_val = w_vec[local_idx]
                            if w_val != 0:
                                gi = 2*c_br+lr; gk = 2*c_bc+lc
                                W[gk*4+gi, col] += c_coeff*w_val

        terms = [
            ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1},{(0,0):1},{(1,0):1,(1,1):-1}),
            ({(0,0):1},{(0,1):1,(1,1):-1},{(0,1):1,(1,1):1}),
            ({(1,1):1},{(1,0):1,(0,0):-1},{(0,0):1,(1,0):1}),
            ({(0,0):1,(0,1):1},{(1,1):1},{(0,0):-1,(0,1):1}),
            ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1},{(1,1):1}),
            ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1},{(0,0):1}),
        ]
        for t_idx,(a_c,b_c,c_c) in enumerate(terms):
            add_4x4_term(a_c, b_c, c_c, t_idx*7)

        col = 49
        for i in range(4):
            for j in range(4):
                U[i*5+j, col]=1; V[j*5+4, col]=1; W[4*4+i, col]=1; col+=1
        for i in range(4):
            for k in range(4):
                U[i*5+4, col]=1; V[4*5+k, col]=1; W[k*4+i, col]=1; col+=1
        for i in range(4):
            U[i*5+4, col]=1; V[4*5+4, col]=1; W[4*4+i, col]=1; col+=1

        return U, V, W

    U0, V0, W0 = build_baseline()

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        if r == 0:
            return False
        U3 = U_mat.reshape(n, m, r)
        V3 = V_mat.reshape(m, p, r)
        W3 = W_mat.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.zeros((n, m, p))
        for i in range(n):
            target[i, :, i] = 1.0
        return np.allclose(T, target, atol=tol, rtol=0.0)

    if not verify(U0, V0, W0):
        # Fallback schoolbook
        rank_sb = n*m*p
        U = np.zeros((nm, rank_sb)); V = np.zeros((mp, rank_sb)); W = np.zeros((pn, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,c]=1; V[j*p+k,c]=1; W[k*n+i,c]=1; c+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]

    # ---------- Structural reduction via trilinear linearization ----------
    # Key insight: fix two factors, the system is linear in the third.
    # We can attempt to project the rank-r solution onto a rank-(r-1) subspace
    # by zeroing one column and solving the remaining system.

    def try_reduce_rank(U_mat, V_mat, W_mat, drop_col, tol=1e-7):
        """Try to reproduce target with rank r-1 by dropping column drop_col and re-solving."""
        r = U_mat.shape[1]
        keep = [c for c in range(r) if c != drop_col]
        r_new = len(keep)
        U_new = U_mat[:, keep].copy()
        V_new = V_mat[:, keep].copy()
        W_new = W_mat[:, keep].copy()
        
        # Coordinate descent: fix two, solve for third
        target = np.zeros((n, m, p))
        for i in range(n):
            target[i, :, i] = 1.0
        target_flat = target.reshape(n*m*p)
        
        for _ in range(4):
            if time.time() - start_time > deadline - 5:
                return None
            # Solve for U
            V3 = V_new.reshape(m, p, r_new)
            W3 = W_new.reshape(p, n, r_new)
            A = np.zeros((n*m*p, nm*r_new))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i*m+j)*p + k
                        coeff = V3[j, k, :] * W3[k, i, :]
                        base = (i*m+j)*r_new
                        A[row, base:base+r_new] = coeff
            sol, *_ = np.linalg.lstsq(A, target_flat, rcond=None)
            U_new = sol.reshape(nm, r_new)
            
            # Solve for V
            U3 = U_new.reshape(n, m, r_new)
            A2 = np.zeros((n*m*p, mp*r_new))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i*m+j)*p + k
                        coeff = U3[i, j, :] * W3[k, i, :]
                        base = (j*p+k)*r_new
                        A2[row, base:base+r_new] = coeff
            sol2, *_ = np.linalg.lstsq(A2, target_flat, rcond=None)
            V_new = sol2.reshape(mp, r_new)
            
            # Solve for W
            V3 = V_new.reshape(m, p, r_new)
            A3 = np.zeros((n*m*p, pn*r_new))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i*m+j)*p + k
                        coeff = U3[i, j, :] * V3[j, k, :]
                        base = (k*n+i)*r_new
                        A3[row, base:base+r_new] = coeff
            sol3, *_ = np.linalg.lstsq(A3, target_flat, rcond=None)
            W_new = sol3.reshape(pn, r_new)
        
        if verify(U_new, V_new, W_new, tol=tol):
            return U_new, V_new, W_new
        return None

    # Try dropping columns from the baseline, starting with boundary columns (most likely redundant)
    # Boundary columns are 49..84; Strassen columns 0..48 are tightly coupled.
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    
    # Try each column drop from highest index down (boundary first)
    for drop in range(current_rank - 1, -1, -1):
        if time.time() - start_time > deadline - 10:
            break
        result = try_reduce_rank(current_U, current_V, current_W, drop)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank -= 1
            if current_rank < best_rank:
                best_U, best_V, best_W, best_rank = current_U.copy(), current_V.copy(), current_W.copy(), current_rank
    
    # Try a few more drops
    for _ in range(3):
        if time.time() - start_time > deadline - 10:
            break
        dropped = False
        for drop in range(current_rank - 1, -1, -1):
            result = try_reduce_rank(current_U, current_V, current_W, drop)
            if result is not None:
                current_U, current_V, current_W = result
                current_rank -= 1
                if current_rank < best_rank:
                    best_U, best_V, best_W, best_rank = current_U.copy(), current_V.copy(), current_W.copy(), current_rank
                dropped = True
                break
        if not dropped:
            break

    # Final verification
    if not verify(best_U, best_V, best_W):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
