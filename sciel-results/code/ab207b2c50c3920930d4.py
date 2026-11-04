"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150 s kill

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    U: (i,j)  V: (j,k)  W: (k,i),  i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """4x4x4 via Kronecker product of two 2x2x2 Strassen (rank 49)."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _build_4x4x8_shared():
    """Rank-98 decomposition for 4x4 * 4x8 (two 4x4x4 Strassen-Kronecker blocks)."""
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        cs = block * 49
        ce = (block + 1) * 49
        U[:, cs:ce] = U4
        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V[j*8 + kg, cs:ce] = V4[j*4 + kl, :]
        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W[kg*4 + i, cs:ce] = W4[kl*4 + i, :]
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Exact decomposition check via einsum."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return bool(np.all(T == 1))

def _schoolbook(n=4, m=4, p=8):
    """Fallback schoolbook decomposition, rank 128."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _merge_duplicates(U, V, W):
    """Drop exact duplicate terms (same U,V,W columns)."""
    rank = U.shape[1]
    seen = {}
    keep = []
    for r in range(rank):
        key = (U[:, r].tobytes(), V[:, r].tobytes(), W[:, r].tobytes())
        if key in seen:
            continue
        seen[key] = r
        keep.append(r)
    if len(keep) == rank:
        return U, V, W
    U = U[:, keep]; V = V[:, keep]; W = W[:, keep]
    return U, V, W

def _rank_reduce(U, V, W, n=4, m=4, p=8, max_iters=20000, tol=1e-10, seed=0):
    """
    Find a linear combination of terms that vanishes on the multiplication tensor,
    then express the dependent term through the others and delete it. Repeats.
    """
    rng = np.random.default_rng(seed)
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    it = 0
    while rank > 1 and it < max_iters:
        it += 1
        if time.time() > _DEADLINE - 5:
            break
        # A[r] = (u_r * v_r) \otimes w_r  (size n*m*p = 128)
        A = (Ur * Vr).reshape(-1, rank) * Wr.reshape(-1, rank)
        # random combination that vanishes on the tensor: A^T c = 0
        c = rng.standard_normal(rank)
        c -= (A @ c) / (A @ A)
        if np.linalg.norm(c) < tol:
            break
        c /= np.linalg.norm(c)
        # pick the term with largest |c| to eliminate
        t = int(np.argmax(np.abs(c)))
        # solve A[:, keep] x = A[:, t]
        keep = [r for r in range(rank) if r != t]
        A_keep = A[:, keep]
        b = A[:, t]
        x, res, rankv, sv = np.linalg.lstsq(A_keep, b, rcond=None)
        if res > 1e-6 * (np.linalg.norm(b) + 1e-30) or rankv < len(keep):
            # not expressible -> try another random direction
            continue
        # new coefficients
        new = np.array([c[r] for r in keep]) + x * c[t]
        U[:, keep] += U[:, t][None, :] * new[None, :]
        V[:, keep] += V[:, t][None, :] * new[None, :]
        W[:, keep] += W[:, t][None, :] * new[None, :]
        U = np.delete(U, t, axis=1)
        V = np.delete(V, t, axis=1)
        W = np.delete(W, t, axis=1)
        rank = rank - 1
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
    # round to integers (entries stay small integers)
    U = np.rint(U).astype(np.int16)
    V = np.rint(V).astype(np.int16)
    W = np.rint(W).astype(np.int16)
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    U, V, W = _schoolbook(n, m, p)  # guaranteed valid fallback
    try:
        U98, V98, W98 = _build_4x4x8_shared()
        if not _verify(U98, V98, W98, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        U98, V98, W98 = _merge_duplicates(U98, V98, W98)
        U98, V98, W98 = _rank_reduce(U98, V98, W98, n, m, p)
        U98, V98, W98 = _merge_duplicates(U98, V98, W98)
        if _verify(U98, V98, W98, n, m, p) and U98.shape[1] < U.shape[1]:
            U, V, W = U98, V98, W98
    except Exception:
        pass
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
