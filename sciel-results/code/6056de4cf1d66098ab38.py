"""Matrix multiplication 2x5 x 5x5 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n*m, m*p, p*n  # 10, 25, 10

def _build_tensor():
    """T[a,b,c] where a=(i,j), b=(j',k), c=(k',i'); T = delta_{j,j'}*delta_{k,k'}."""
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for jp in range(m):   # j'
                for k in range(p):
                    b = jp*p + k
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            c = kp*n + ip
                            if jp == j and kp == k:
                                T[a, b, c] = 1.0
    return T

TENSOR = _build_tensor()

def _verify(U, V, W, tol=1e-9):
    """Check sum_r U[a,r]V[b,r]W[c,r] == T[a,b,c] for all a,b,c."""
    # U: NM x R, V: MP x R, W: PN x R
    R = U.shape[1]
    ok = True
    for a in range(NM):
        for b in range(MP):
            # (U[a] * V[b]) dot W[c] for each c
            uv = np.outer(U[a], V[b])  # R x R? No.
            # sum_r U[a,r] V[b,r] W[c,r] = W @ (U[a] * V[b])
            prod = U[a] * V[b]  # elementwise, length R
            for c in range(PN):
                val = np.dot(prod, W[c])
                if abs(val - TENSOR[a,b,c]) > tol:
                    return False
    return True

def _als(R, iters=200, seed=0, tol=1e-8):
    """Alternating least squares for rank-R decomposition."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((NM, R))
    V = rng.standard_normal((MP, R))
    W = rng.standard_normal((PN, R))
    # Normalize
    U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    
    for it in range(iters):
        # Update U: for each a, solve min sum_{b,c} (sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
        # = min || W^T diag... ||. Rewrite: for fixed a, define for each (b,c): coeff vector over r = V[b]*W[c] (elementwise).
        # So the system is: M_a U[a]^T = t_a where M_a is (MP*PN) x R with rows (V[b]*W[c]), t_a[a's] = T[a,b,c].
        # Build M_a once (independent of a!): M[b,c,r] = V[b,r]*W[c,r].
        M = np.einsum('br,cr->bcr', V, W).reshape(MP*PN, R)
        t = TENSOR.reshape(NM, MP*PN)
        # For each a: solve M U[a]^T = t[a]
        U_new = np.zeros((NM, R))
        for a in range(NM):
            # least squares
            sol, *_ = np.linalg.lstsq(M, t[a], rcond=None)
            U_new[a] = sol
        U = U_new
        # Update V
        M2 = np.einsum('ar,cr->acr', U, W).reshape(NM*PN, R)
        t2 = TENSOR.transpose(0,2,1).reshape(NM*PN, MP)  # careful with index
        # T[a,b,c]; for fixed b, we want over (a,c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # coeff over r for (a,c) = U[a]*W[c]. t over (a,c) = T[a,b,c].
        V_new = np.zeros((MP, R))
        for b in range(MP):
            M2b = np.einsum('ar,cr->acr', U, W).reshape(NM*PN, R)
            t2b = TENSOR[:, b, :].reshape(NM*PN)
            sol, *_ = np.linalg.lstsq(M2b, t2b, rcond=None)
            V_new[b] = sol
        V = V_new
        # Update W
        W_new = np.zeros((PN, R))
        for c in range(PN):
            Mc = np.einsum('ar,br->abr', U, V).reshape(NM*MP, R)
            tc = TENSOR[:, :, c].reshape(NM*MP)
            sol, *_ = np.linalg.lstsq(Mc, tc, rcond=True)
            W_new[c] = sol
        W = W_new
    return U, V, W

def _round_half(U, V, W):
    """Round to nearest half-integer."""
    return (2*U).round()/2.0, (2*V).round()/2.0, (2*W).round()/2.0

def construct():
    t0 = time.time()
    DEADLINE = 150.0
    SAFETY = 5.0
    best = None
    best_rank = None
    
    # Start from schoolbook as a guaranteed fallback
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(NM)]
        V = [[0]*rank for _ in range(MP)]
        W = [[0]*rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U0, V0, W0, r0 = schoolbook()
    best = (U0, V0, W0)
    best_rank = r0
    
    # Try to find lower rank via ALS + rounding
    for R in range(40, 50):  # try below 50
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        for seed in range(0, 6):
            if time.time() - t0 > DEADLINE - SAFETY:
                break
            try:
                U, V, W = _als(R, iters=60, seed=seed)
                U2, V2, W2 = _round_half(U, V, W)
                if _verify(U2, V2, W2, tol=1e-9):
                    # Convert to lists of Python numbers
                    Uf = [[(int(2*x)/2 if abs(2*x - round(2*x)) < 1e-6 else float(x)) for x in row] for row in U2]
                    Vf = [[(int(2*x)/2 if abs(2*x - round(2*x)) < 1e-6 else float(x)) for x in row] for row in V2]
                    Wf = [[(int(2*x)/2 if abs(2*x - round(2*x)) < 1e-6 else float(x)) for x in row] for row in W2]
                    # Re-verify with exact fractions
                    from fractions import Fraction
                    def to_frac(x):
                        hx = round(2*x)
                        return Fraction(hx, 2)
                    Uq = [[to_frac(x) for x in row] for row in U2]
                    Vq = [[to_frac(x) for x in row] for row in V2]
                    Wq = [[to_frac(x) for x in row] for row in W2]
                    # Exact verify
                    ok = True
                    for a in range(NM):
                        for b in range(MP):
                            prod = [Uq[a][r]*Vq[b][r] for r in range(R)]
                            for c in range(PN):
                                val = sum(prod[r]*Wq[c][r] for r in range(R))
                                if val != TENSOR[a,b,c]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        best = (Uf, Vf, Wf)
                        best_rank = R
                        break
            except Exception:
                pass
    
    # Return best
    U, V, W = best
    # Ensure integer/half-integer representation as floats (or ints)
    def clean(x):
        hx = round(2*x)
        if abs(hx) < 1e-9:
            return 0
        if abs(hx - round(hx)) < 1e-6:
            return int(hx)/2
        return float(x)
    Uf = [[clean(x) for x in row] for row in U]
    Vf = [[clean(x) for x in row] for row in V]
    Wf = [[clean(x) for x in row] for row in W]
    return {"U": Uf, "V": Vf, "W": Wf}
