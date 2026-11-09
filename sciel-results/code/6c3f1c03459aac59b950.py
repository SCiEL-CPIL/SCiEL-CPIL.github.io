import time
import random
import itertools
import numpy as np
from scipy.linalg import qr, solve

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS  # 3, 4, 6

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def make_strassen_2x2():
        """U (4,7) idx U[i*2+j], V (4,7) idx V[j*2+k], W (4,7) idx W[k*2+i].
        Satisfies sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1 for all i,j,k in {0,1}."""
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # M1=(A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2=(A10+A11)B00
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        # M3=A00(B01-B11)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # M4=A11(B10-B00)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        # M5=(A00+A01)B11
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # M6=(A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # M7=(A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def make_block(n, m, p):
        """(n,m,p)->(p,n) block. n==m==p==2 => Strassen rank 7, else schoolbook rank n*m*p."""
        if n == 2 and m == 2 and p == 2:
            return make_strassen_2x2()
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    def assemble_blockwise():
        """Block decomposition of 3x4x6. A rows {0,1}(2) + {2}(1); A cols {0,1}+{2,3};
        B rows {0,1}+{2,3}; B cols {0,1}+{2,3}+{4,5}.
        (2,2,2) blocks use Strassen (rank 7), (1,2,2) blocks use schoolbook (rank 4).
        Total rank = 6*7 + 6*4 = 66."""
        U = np.zeros((n * m, 0)); V = np.zeros((m * p, 0)); W = np.zeros((p * n, 0))
        arow = [(0, 2), (2, 3)]
        acol = [(0, 2), (2, 4)]
        brow = [(0, 2), (2, 4)]
        bcol = [(0, 2), (2, 4), (4, 6)]
        for (ir0, ir1) in arow:
            for (jc0, jc1) in acol:
                for (kr0, kr1) in bcol:
                    bn = ir1 - ir0; bm = jc1 - jc0; bp = kr1 - kr0
                    bU, bV, bW = make_block(bn, bm, bp)
                    for c in range(bU.shape[1]):
                        U[:, c] = 0.0; V[:, c] = 0.0; W[:, c] = 0.0
                        U[ir0 * m + jc0: ir1 * m + jc1, c] = bU[:, c]
                        V[jc0 * p + kr0: jc1 * p + kr1, c] = bV[:, c]
                        W[kr0 * n + ir0: kr1 * n + ir1, c] = bW[:, c]
                        U = np.hstack([U, U[:, c:c+1]])
                        V = np.hstack([V, V[:, c:c+1]])
                        W = np.hstack([W, W[:, c:c+1]])
        return U, V, W

    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(W[k * n + i, :], U[i * m + j, :] * V[j * p + k, :])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    def reduce_rank(U, V, W, tol=1e-9):
        """Remove columns r for which the tensor U_r⊗V_r⊗W_r is a linear combination
        of the others (equivalently, the vector u_r⊗v_r has a rank-1 completion w_r
        in span{u_s⊗w_s}). This is exact for integer/half-integer entries."""
        rank = U.shape[1]
        keep = list(range(rank))
        changed = True
        while changed:
            changed = False
            r = U.shape[1]
            for idx in range(r):
                if time.time() - start_time > deadline - 15:
                    break
                u = U[:, idx]; v = V[:, idx]; w = W[:, idx]
                if np.allclose(u, 0) or np.allclose(v, 0) or np.allclose(w, 0):
                    keep.remove(idx); changed = True; break
                uv = np.kron(u, v)
                # build A (nm*pn) x (rank-1) from remaining columns
                others = [x for x in range(r) if x != idx]
                A = np.column_stack([np.kron(U[:, x], W[:, x]) for x in others])
                # solve A c = uv
                try:
                    c, res, rk, sv = np.linalg.lstsq(A, uv, rcond=None)
                except Exception:
                    continue
                w_hat = np.zeros(p * n)
                for t, x in enumerate(others):
                    w_hat += c[t] * W[:, x]
                if np.linalg.norm(w_hat - w) < tol * max(1.0, np.linalg.norm(w)):
                    keep.remove(idx); changed = True; break
            U = U[:, keep]; V = V[:, keep]; W = W[:, keep]
            keep = list(range(U.shape[1]))
        return U, V, W

    # --- Build the best known decomposition ---
    best = None
    U, V, W = assemble_blockwise()
    if verify(U, V, W):
        best = (U, V, W)

    # Attempt exact rank reduction (structural refinement).
    if time.time() - start_time < deadline - 20:
        try:
            rU, rV, rW = reduce_rank(U.copy(), V.copy(), W.copy())
            if verify(rU, rV, rW):
                if best is None or rU.shape[1] < best[0].shape[1]:
                    best = (rU, rV, rW)
        except Exception:
            pass

    # Fallback to schoolbook if nothing valid.
    if best is None:
        sU, sV, sW = make_schoolbook()
        best = (sU, sV, sW)

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
