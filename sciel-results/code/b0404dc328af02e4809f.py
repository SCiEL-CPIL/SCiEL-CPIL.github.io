"""Matrix multiplication 3x3x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _build_tensor(n, m, p):
    """Build the multiplication tensor T with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _als_decompose(T, r, n, m, p, n_iters=200, seed=0):
    """Alternating least squares to find rank-r factors U, V, W reconstructing T."""
    rng = np.random.default_rng(seed)
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    U = rng.normal(size=(n*m, r))
    V = rng.normal(size=(m*p, r))
    W = rng.normal(size=(p*n, r))
    # Normalize columns to avoid scale issues
    for mat in (U, V, W):
        for c in range(r):
            nrm = np.linalg.norm(mat[:, c])
            if nrm > 0:
                mat[:, c] /= nrm
    for it in range(n_iters):
        # Solve for U: for each row a, U[a,:] solves sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
        # Reshape: fix V,W -> linear in U. Build A_U where A_U[a, (b,d), c] = V[b,c]*W[d,c]
        VW = np.einsum('bc,dc->bdc', V, W)  # (m*p, p*n, r)
        # For row a: T[a] = sum_c U[a,c] * VW[:,:,c]  =>  vec: (VW flattened over b,d) @ U[a,:]
        VW_flat = VW.transpose(2, 0, 1).reshape(r, -1)  # (r, m*p*p*n)
        T_flat = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
        # U[a] = argmin ||VW_flat @ U[a] - T_flat[a]||
        # Least squares: U = T_flat^T @ VW_flat^+  (solve r x r system per batch)
        # Use normal equations: (VW_flat^T VW_flat) U^T = VW_flat^T T_flat^T
        G = VW_flat.T @ VW_flat  # (r, r)
        rhs = VW_flat.T @ T_flat.T  # (r, n*m)
        try:
            U_new = np.linalg.solve(G, rhs).T
        except np.linalg.LinAlgError:
            U_new = np.linalg.lstsq(VW_flat, T_flat.T, rcond=None)[0].T
        U = U_new
        # Solve for V
        UW = np.einsum('ac,dc->adc', U, W)  # (n*m, p*n, r)
        UW_flat = UW.transpose(2, 0, 1).reshape(r, -1)
        G = UW_flat.T @ UW_flat
        rhs = UW_flat.T @ T_flat.T
        try:
            V_new = np.linalg.solve(G, rhs).T
        except np.linalg.LinAlgError:
            V_new = np.linalg.lstsq(UW_flat, T_flat.T, rcond=None)[0].T
        V = V_new
        # Solve for W
        UV = np.einsum('ac,bc->abc', U, V)  # (n*m, m*p, r)
        UV_flat = UV.transpose(2, 0, 1).reshape(r, -1)
        G = UV_flat.T @ UV_flat
        rhs = UV_flat.T @ T_flat.T
        try:
            W_new = np.linalg.solve(G, rhs).T
        except np.linalg.LinAlgError:
            W_new = np.linalg.lstsq(UV_flat, T_flat.T, rcond=None)[0].T
        W = W_new
    return U, V, W

def _residual(T, U, V, W):
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.max(np.abs(recon - T))

def _round_half(x):
    """Round to nearest half-integer."""
    return np.round(2.0 * x) / 2.0

def _check_exact(T, U, V, W):
    """Check if half-integer rounded factors exactly reconstruct T."""
    Ur = _round_half(U)
    Vr = _round_half(V)
    Wr = _round_half(W)
    recon = np.einsum('ac,bc,dc->abd', Ur, Vr, Wr)
    return np.max(np.abs(recon - T)) < 1e-9, Ur, Vr, Wr

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    # Lower bound: max of the three matrix dimensions
    lo = max(n*m, m*p, p*n)
    best = None
    # Try ranks from lower bound upward
    for r in range(lo, n*m*p + 1):
        if time.time() > deadline - 5.0:
            break
        # Multiple random restarts for this rank
        max_restarts = 6
        for seed in range(max_restarts):
            if time.time() > deadline - 3.0:
                break
            U, V, W = _als_decompose(T, r, n, m, p, n_iters=150, seed=seed)
            res = _residual(T, U, V, W)
            if res < 1e-7:
                ok, Ur, Vr, Wr = _check_exact(T, U, V, W)
                if ok:
                    best = (r, Ur, Vr, Wr)
                    break
        if best is not None:
            break
    if best is None:
        # Fallback: schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    r, Ur, Vr, Wr = best
    U = Ur.tolist()
    V = Vr.tolist()
    W = Wr.tolist()
    return {"U": U, "V": V, "W": W}
