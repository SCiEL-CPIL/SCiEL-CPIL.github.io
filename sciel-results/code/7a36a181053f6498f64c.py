import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    prods = [
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
    ]
    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub: U[i*2+j, m_idx] += c
        for (j, k, c) in Vb: V[j*2+k, m_idx] += c
        for (i, k, c) in Wb: W[k*2+i, m_idx] += c
    return U, V, W

def _build_rank77():
    n, m, p = 4, 4, 6
    U_s, V_s, W_s = _strassen_2x2()
    r444, r442 = 49, 28
    R = r444 + r442
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    # Part 1: 4x4*4x4 Strassen via Kronecker
    for t in range(7):
        for s in range(7):
            col = 7*t + s
            for i in range(4):
                I, i0 = i//2, i%2
                for j in range(4):
                    J, j0 = j//2, j%2
                    val = U_s[2*I+J, t]*U_s[2*i0+j0, s]
                    if val != 0: U[i*4+j, col] += val
            for j in range(4):
                J, j0 = j//2, j%2
                for k in range(4):
                    K, k0 = k//2, k%2
                    val = V_s[2*J+K, t]*V_s[2*j0+k0, s]
                    if val != 0: V[j*6+k, col] += val
            for k in range(4):
                K, k0 = k//2, k%2
                for i in range(4):
                    I, i0 = i//2, i%2
                    val = W_s[2*K+I, t]*W_s[2*k0+i0, s]
                    if val != 0: W[k*4+i, col] += val
    # Part 2: 4x4*4x2 blocks
    blocks = [(0,0,0,4,0,4),(0,2,2,4,0,4),(2,0,0,4,2,4),(2,2,2,4,2,4)]
    for bi,(ar,ac,br,bc,cr,cc) in enumerate(blocks):
        for s in range(7):
            col = r444 + bi*7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j, s]
                    if val != 0: U[(ar+i)*4+(ac+j), col] += val
            for j in range(2):
                for k in range(2):
                    val = V_s[j*2+k, s]
                    if val != 0: V[(br+j)*6+(bc+k), col] += val
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i, s]
                    if val != 0: W[(cc+k)*4+(cr+i), col] += val
    return U, V, W

def _residual(U, V, W, n, m, p):
    # sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1 for all (i,j,k)
    s = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                for r in range(U.shape[1]):
                    acc += U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
                s += (acc-1.0)**2
    return s

def _drop_zero_cols(U, V, W, n, m, p, tol=1e-12):
    R = U.shape[1]
    keep = []
    for r in range(R):
        if np.abs(U[:, r]).max() > tol or np.abs(V[:, r]).max() > tol or np.abs(W[:, r]).max() > tol:
            keep.append(r)
    keep = np.array(keep)
    return U[:, keep], V[:, keep], W[:, keep]

def _merge_cols(U, V, W, n, m, p, tol=1e-9):
    # For each pair (a,b), if there exists a nonzero vector in null of V[:,a]
    # (and in null of U[:,a] OR W[:,a]) that also has zero dot with the
    # corresponding column of the other factor, then we can project out.
    # Simpler robust rule: if column a of V is a scalar multiple of column b of V,
    # and the corresponding U and W columns are also scalar multiples, drop one.
    R = U.shape[1]
    cols = list(range(R))
    changed = True
    while changed:
        changed = False
        newcols = []
        skip = set()
        for a in cols:
            if a in skip:
                continue
            newcols.append(a)
            for b in cols:
                if b in skip or b == a or b < a:
                    continue
                va, vb = V[:, a], V[:, b]
                ua, ub = U[:, a], U[:, b]
                wa, wb = W[:, a], W[:, b]
                # check if b is scalar multiple of a in all three
                if np.linalg.norm(vb) > tol and np.linalg.norm(ua) > tol:
                    # find scalar c such that vb ~ c*va
                    c = np.dot(vb, va)/np.dot(va, va)
                    if np.linalg.norm(vb - c*va) < tol*np.linalg.norm(va) and \
                       np.linalg.norm(ub - c*ua) < tol*np.linalg.norm(ua) and \
                       np.linalg.norm(wb - c*wa) < tol*np.linalg.norm(wa):
                        skip.add(b)
                        changed = True
        cols = newcols
    cols = np.array(cols)
    return U[:, cols], V[:, cols], W[:, cols]

def _nullspace_merge(U, V, W, n, m, p, tol=1e-8):
    # Try to find a column b whose V-column lies in nullspace of some linear
    # combination, allowing us to eliminate. We attempt: for each b, compute
    # null space of U[:, a] for each a; if a null vector v satisfies
    # v @ U[:,b] == 0 and v @ W[:,b] == 0 (in the relevant sense), we can
    # subtract. This is complex; instead do a targeted Gram-based merge:
    # if columns a and b of V are linearly dependent (rank 1), merge.
    R = U.shape[1]
    cols = list(range(R))
    changed = True
    while changed:
        changed = False
        newcols = []
        skip = set()
        for a in cols:
            if a in skip:
                continue
            newcols.append(a)
            for b in cols:
                if b in skip or b == a or b < a:
                    continue
                # check rank of stacked V columns a,b
                M = np.column_stack([V[:, a], V[:, b]])
                if np.linalg.matrix_rank(M, tol=tol) == 1:
                    # V dependent; check U and W also dependent with same scalar
                    # solve c: Vb = c*Va
                    va, vb = V[:, a], V[:, b]
                    denom = np.dot(va, va)
                    if denom < tol:
                        continue
                    c = np.dot(vb, va)/denom
                    ua, ub = U[:, a], U[:, b]
                    wa, wb = W[:, a], W[:, b]
                    if np.linalg.norm(ub - c*ua) < tol and np.linalg.norm(wb - c*wa) < tol:
                        skip.add(b)
                        changed = True
        cols = newcols
    cols = np.array(cols)
    return U[:, cols], V[:, cols], W[:, cols]

def _to_list(mat):
    out = []
    for row in mat:
        r = []
        for x in row:
            xi = round(x)
            if abs(x - xi) < 1e-9:
                r.append(int(xi))
            else:
                # try half-integer
                xh = round(2*x)/2
                if abs(x - xh) < 1e-9:
                    r.append(xh)
                else:
                    r.append(float(x))
        out.append(r)
    return out

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    n, m, p = 4, 4, 6

    U, V, W = _build_rank77()
    U, V, W = _drop_zero_cols(U, V, W, n, m, p)
    U, V, W = _merge_cols(U, V, W, n, m, p)
    U, V, W = _nullspace_merge(U, V, W, n, m, p)

    best_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())

    # Iterative: repeatedly try merges and small null-space eliminations
    it = 0
    while time.time() < deadline - 20.0:
        it += 1
        U2, V2, W2 = _drop_zero_cols(U, V, W, n, m, p)
        U2, V2, W2 = _merge_cols(U2, V2, W2, n, m, p)
        U2, V2, W2 = _nullspace_merge(U2, V2, W2, n, m, p)
        if U2.shape[1] < best_rank:
            best_rank = U2.shape[1]
            best = (U2.copy(), V2.copy(), W2.copy())
            U, V, W = U2, V2, W2
        else:
            # no progress; stop to avoid infinite loop
            break

    U, V, W = best

    # final verification
    res = _residual(U, V, W, n, m, p)
    if res > 1e-6:
        # fall back to fresh valid build
        U, V, W = _build_rank77()

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
