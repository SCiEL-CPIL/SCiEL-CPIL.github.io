import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    # Use 4 Strassen 2x2x2 blocks positioned to cover corners
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (1, 0, 2),
        (1, 2, 2),
    ]

    total_rank = 4 * 7 + (n * m * p - 4 * 8)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        c = col; U[i0*m+j0, c] += 1; U[(i0+1)*m+(j0+1), c] += 1; V[j0*p+k0, c] += 1; V[(j0+1)*p+(k0+1), c] += 1; W[k0*n+i0, c] += 1; W[(k0+1)*n+(i0+1), c] += 1; col += 1
        c = col; U[(i0+1)*m+j0, c] += 1; U[(i0+1)*m+(j0+1), c] += 1; V[j0*p+k0, c] += 1; W[k0*n+(i0+1), c] += 1; W[(k0+1)*n+(i0+1), c] += -1; col += 1
        c = col; U[i0*m+j0, c] += 1; V[j0*p+(k0+1), c] += 1; V[(j0+1)*p+(k0+1), c] += -1; W[(k0+1)*n+i0, c] += 1; W[(k0+1)*n+(i0+1), c] += 1; col += 1
        c = col; U[(i0+1)*m+(j0+1), c] += 1; V[(j0+1)*p+k0, c] += 1; V[j0*p+k0, c] += -1; W[k0*n+i0, c] += 1; W[k0*n+(i0+1), c] += 1; col += 1
        c = col; U[i0*m+j0, c] += 1; U[i0*m+(j0+1), c] += 1; V[(j0+1)*p+(k0+1), c] += 1; W[k0*n+i0, c] += -1; W[(k0+1)*n+i0, c] += 1; col += 1
        c = col; U[(i0+1)*m+j0, c] += 1; U[i0*m+j0, c] += -1; V[j0*p+k0, c] += 1; V[j0*p+(k0+1), c] += 1; W[(k0+1)*n+(i0+1), c] += 1; col += 1
        c = col; U[i0*m+(j0+1), c] += 1; U[(i0+1)*m+(j0+1), c] += -1; V[(j0+1)*p+k0, c] += 1; V[(j0+1)*p+(k0+1), c] += 1; W[k0*n+i0, c] += 1; col += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)

    covered = np.zeros((n, m, p), dtype=bool)
    for i0, j0, k0 in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] += 1
                    V[j*p + k, col] += 1
                    W[k*n + i, col] += 1
                    col += 1

    assert col == total_rank, (col, total_rank)

    # Verify the decomposition is correct
    def verify(Um, Vm, Wm):
        T = np.zeros((n, m, p))
        for c in range(Um.shape[1]):
            u = Um[:, c].reshape(n, m)
            v = Vm[:, c].reshape(m, p)
            w = Wm[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if u[i,j] == 0: continue
                    for k in range(p):
                        if v[j,k] == 0 or w[k,i] == 0: continue
                        T[i,j,k] += u[i,j]*v[j,k]*w[k,i]
        # Expected: T[i,j,k] = 1 if i,j,k all in range (trivial matmul tensor)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-10)

    if not verify(U, V, W):
        # Fallback to pure schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        for c in range(n*m*p):
            i, j, k = c // (m*p), (c // p) % m, c % p
            U[i*m+j, c] = 1
            V[j*p+k, c] = 1
            W[k*n+i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Greedy merge: try to combine pairs of columns into one rank-1 term
    def is_rank1(T, tol=1e-10):
        # Check if tensor T[i,j,k] = a[i]*b[j]*c[k]
        # Use matricization: T_(i)(j,k)
        mat = T.reshape(n, m*p)
        # Find rank
        try:
            s = np.linalg.svd(mat, compute_uv=False)
        except:
            return False, None
        if len(s) == 0 or s[0] < tol:
            return True, (np.zeros(n), np.zeros(m), np.zeros(p))
        if len(s) > 1 and s[1]/s[0] > 1e-8:
            return False, None
        # Rank 1: T[i,j,k] = sigma * u[i] * v[j,k]
        U_svd, s, Vt = np.linalg.svd(mat, full_matrices=False)
        u = U_svd[:, 0] * np.sqrt(s[0])
        vt = Vt[0, :] * np.sqrt(s[0])
        v = vt.reshape(m, p)
        # v[j,k] = b[j]*c[k]
        # Find b, c
        nz_j = np.where(np.abs(v).max(axis=1) > tol)[0]
        if len(nz_j) == 0:
            return False, None
        j0 = nz_j[0]
        c = v[j0, :].copy()
        b = np.zeros(m)
        b[j0] = 1.0
        for j in range(m):
            if np.abs(v[j,:]).max() < tol:
                continue
            # v[j,:] = b[j] * c
            nz_c = np.where(np.abs(c) > tol)[0]
            if len(nz_c) == 0:
                return False, None
            ratios = v[j, nz_c] / c[nz_c]
            if not np.allclose(ratios, ratios[0], atol=tol, rtol=tol):
                return False, None
            b[j] = ratios[0]
        return True, (u, b, c)

    def try_merge(Um, Vm, Wm, cols):
        T = np.zeros((n, m, p))
        for c in cols:
            u = Um[:, c].reshape(n, m)
            v = Vm[:, c].reshape(m, p)
            w = Wm[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if u[i,j] == 0: continue
                    for k in range(p):
                        if v[j,k] == 0 or w[k,i] == 0: continue
                        T[i,j,k] += u[i,j]*v[j,k]*w[k,i]
        r1, res = is_rank1(T)
        if not r1:
            return False, None, None, None
        a, b, c = res
        new_u = np.zeros(n*m)
        new_v = np.zeros(m*p)
        new_w = np.zeros(p*n)
        for i in range(n):
            for j in range(m):
                new_u[i*m+j] = a[i] * b[j]
        for j in range(m):
            for k in range(p):
                new_v[j*p+k] = 1.0
                # We need U*V*W = a[i]*b[j]*c[k]
                # Set U[i*m+j]=a[i]*b[j], V[j*p+k]=1, W[k*n+i]=c[k]
                # But V depends on j,k not i. Let's use:
                # U[i*m+j] = a[i]*b[j], V[j*p+k] = c[k], W[k*n+i] = 1
                # Check: a[i]*b[j] * c[k] * 1 = a[i]*b[j]*c[k] ✓
        for j in range(m):
            for k in range(p):
                new_v[j*p+k] = c[k]
        for k in range(p):
            for i in range(n):
                new_w[k*n+i] = 1.0
        return True, new_u, new_v, new_w

    # Greedy merge pairs
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    active = list(range(total_rank))
    merged = set()

    for _ in range(30):
        if time.time() > deadline - 10:
            break
        found = False
        new_active = []
        i = 0
        while i < len(active):
            c1 = active[i]
            if c1 in merged:
                i += 1
                continue
            merged_any = False
            for j in range(i+1, len(active)):
                c2 = active[j]
                if c2 in merged:
                    continue
                ok, nu, nv, nw = try_merge(current_U, current_V, current_W, [c1, c2])
                if ok:
                    current_U[:, c1] = nu
                    current_V[:, c1] = nv
                    current_W[:, c1] = nw
                    merged.add(c2)
                    merged_any = True
                    found = True
                    break
            if not merged_any:
                new_active.append(c1)
            i += 1
        active = new_active
        if not found:
            break

    if merged:
        keep = [c for c in range(total_rank) if c not in merged]
        U_final = current_U[:, keep]
        V_final = current_V[:, keep]
        W_final = current_W[:, keep]
    else:
        U_final = U
        V_final = V
        W_final = W

    # Final verification
    if not verify(U_final, V_final, W_final):
        U_final = U
        V_final = V
        W_final = W

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
