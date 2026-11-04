import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 108.0
    n, m, p = 6, 6, 6
    nm, mp, pn = 36, 36, 36

    def check(U, V, W, n=6, m=6, p=6):
        Ug = U.reshape(n, m, -1)
        Vg = V.reshape(m, p, -1)
        Wg = W.reshape(p, n, -1)
        rec = np.einsum('ijr,jkr,kir->ijk', Ug, Vg, Wg)
        return np.allclose(rec, np.ones((n, m, p)), atol=1e-9)

    # --- Schoolbook 3x3 kernel (rank 27), always valid ---
    def build_schoolbook(n3):
        r = n3 * n3 * n3
        U = np.zeros((n3 * n3, r)); V = np.zeros((n3 * n3, r)); W = np.zeros((n3 * n3, r))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    c = i * n3 * n3 + j * n3 + k
                    U[i * n3 + j, c] = 1.0
                    V[j * n3 + k, c] = 1.0
                    W[k * n3 + i, c] = 1.0
        return U, V, W

    # --- Rank-23 kernel for 3x3 (Stothers' 7x2x2+1 construction) ---
    def build_3x3_23():
        r = 23
        U = np.zeros((9, r)); V = np.zeros((9, r)); W = np.zeros((9, r))
        idx = 0
        for x in range(2):
            for y in range(2):
                # Mx = A[x,y]+A[x+1,y+1], Ny = B[x,y]+B[x+1,y+1]
                U[x*3+y, idx] = 1.0; U[(x+1)*3+(y+1), idx] = 1.0
                V[x*3+y, idx] = 1.0; V[(x+1)*3+(y+1), idx] = 1.0
                for i in range(3):
                    for j in range(3):
                        W[j*3+i, idx] += (1.0 if (i == x or i == x+1) and (j == y or j == y+1) else 0.0)
                idx += 1
        # P8 = A[1,0]*B[0,2]
        U[1*3+0, idx] = 1.0; V[0*3+2, idx] = 1.0; W[2*3+1, idx] = 1.0; idx += 1
        # P9 = A[0,1]*B[2,0]
        U[0*3+1, idx] = 1.0; V[2*3+0, idx] = 1.0; W[0*3+1, idx] = 1.0; idx += 1
        # P10 = A[2,2]*B[0,0]
        U[2*3+2, idx] = 1.0; V[0*3+0, idx] = 1.0; W[0*3+2, idx] = 1.0; idx += 1
        # P11 = A[0,0]*B[2,2]
        U[0*3+0, idx] = 1.0; V[2*3+2, idx] = 1.0; W[2*3+0, idx] = 1.0; idx += 1
        return U, V, W

    # --- Build 6x6 via 2x2 Strassen blocks of a given 3x3 kernel ---
    def build_6x6(U3, V3, W3, r3):
        total = 7 * r3
        U6 = np.zeros((nm, total)); V6 = np.zeros((mp, total)); W6 = np.zeros((pn, total))

        def add_term(start, a_blks, b_blks, c_blks):
            for rr in range(r3):
                um = U3[:, rr].reshape(3, 3)
                vm = V3[:, rr].reshape(3, 3)
                wm = W3[:, rr].reshape(3, 3)
                for bi, cf in a_blks:
                    br, bc = bi // 2, bi % 2
                    for ii in range(3):
                        for jj in range(3):
                            v = cf * um[ii, jj]
                            if v != 0:
                                U6[(br*3+ii)*6 + (bc*3+jj), start+rr] += v
                for bi, cf in b_blks:
                    br, bc = bi // 2, bi % 2
                    for jj in range(3):
                        for kk in range(3):
                            v = cf * vm[jj, kk]
                            if v != 0:
                                V6[(br*3+jj)*6 + (bc*3+kk), start+rr] += v
                for bi, cf in c_blks:
                    br, bc = bi // 2, bi % 2
                    for kk in range(3):
                        for ii in range(3):
                            v = cf * wm[kk, ii]
                            if v != 0:
                                W6[(bc*3+kk)*6 + (br*3+ii), start+rr] += v

        add_term(0,   [(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])
        add_term(r3,  [(2,1),(3,1)], [(0,1)],        [(2,1),(3,-1)])
        add_term(2*r3,[(0,1)],       [(1,1),(3,-1)], [(1,1),(3,1)])
        add_term(3*r3,[(3,1)],       [(2,1),(0,-1)], [(0,1),(2,1)])
        add_term(4*r3,[(0,1),(1,1)], [(3,1)],        [(1,1),(0,-1)])
        add_term(5*r3,[(2,1),(0,-1)],[(0,1),(1,1)],  [(3,1)])
        add_term(6*r3,[(1,1),(3,-1)],[(2,1),(3,1)],  [(0,1)])
        return U6, V6, W6

    # Schoolbook 6x6 baseline (rank 216)
    U_s, V_s, W_s = build_schoolbook(6)

    # Try rank-23 kernel; verify before trusting it
    best_U, best_V, best_W = U_s, V_s, W_s
    try:
        U3, V3, W3 = build_3x3_23()
        if check(U3, V3, W3, 3, 3, 3):
            U6, V6, W6 = build_6x6(U3, V3, W3, 23)
            if check(U6, V6, W6):
                best_U, best_V, best_W = U6, V6, W6
    except Exception:
        pass

    # Refinement: greedy rank reduction (keep current machinery)
    if time.time() < deadline - 12.0:
        try:
            keep = list(range(best_U.shape[1]))
            rU = np.linalg.matrix_rank(best_U, tol=1e-10)
            rV = np.linalg.matrix_rank(best_V, tol=1e-10)
            rW = np.linalg.matrix_rank(best_W, tol=1e-10)
            min_r = min(rU, rV, rW)
            improved = True
            while improved and time.time() < deadline - 8.0:
                improved = False
                for r in list(keep):
                    if len(keep) <= min_r:
                        break
                    new_keep = [k for k in keep if k != r]
                    if not new_keep:
                        continue
                    U_t = best_U[:, new_keep]
                    V_t = best_V[:, new_keep]
                    W_t = best_W[:, new_keep]
                    if check(U_t, V_t, W_t):
                        keep = new_keep
                        best_U, best_V, best_W = U_t, V_t, W_t
                        improved = True
                        break
        except Exception:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
