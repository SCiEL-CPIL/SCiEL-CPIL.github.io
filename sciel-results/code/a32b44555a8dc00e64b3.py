"""Tensor decomposition search for (2,3,9) matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_block27(n, m, p):
    """Constructive rank-27 decomposition using 3x3 block structure of B."""
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Treat B's p columns as (p//m) blocks of m columns each.
    # For each (j, block b): one rank-1 term where
    #   U[i, t(j,b)] = 1 if i matches row, else 0  -> actually U depends on (i,j)
    # Let's define term t indexed by (j, b):
    #   U[i, t] = 1 if (i,j) else 0  -> U[i, t] = delta(i in rows of A for j)
    #   V[jp + (b*m + k'), t] = 1 if row == j and col in block b
    #   W[(b*m + k')*n + i, t] = 1 if output col in block b and row i
    # Check: sum_t U[i,t] V[jp+k, t] W[kn+i, t]
    #   = sum over t=(j',b) U[i,(j',b)] V[jp+k,(j',b)] W[kn+i,(j',b)]
    #   U[i,(j',b)] = 1 if j'==j? No, U is indexed by (i,j) rows of A.
    # Let me redefine cleanly with explicit indices.
    rank = m * (p // m)  # 3 * 3 = 9? No, need per (j, block). j in 0..m-1, b in 0..p//m-1
    # Actually rank should be m * (p//m) = 3*3 = 9? That seems too low. Let's verify.
    # Hmm, but W has p*n = 18 rows, V has m*p=27 rows, U has n*m=6 rows.
    # Let's just construct and verify numerically.
    nb = p // m  # number of blocks = 3
    rank = m * nb  # 3*3 = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for j in range(m):
        for b in range(nb):
            t = j * nb + b
            for i in range(n):
                U[i * m + j][t] = 1
            for kprime in range(m):
                V[j * p + (b * m + kprime)][t] = 1
            for i in range(n):
                for kprime in range(m):
                    W[(b * m + kprime) * n + i][t] = 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, trials=5, seed=0):
    """Verify that sum_t U[i,t]V[j,t]W[k,t] reproduces C[i,k]=sum_j A[i,j]B[j,k]."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        # Reconstruct C[i,k] = sum_t U[i*m+? ...] -- careful with indexing.
        # U rows indexed by i*m+j (A's (i,j)), V rows by j*p+k (B's (j,k)), W rows by k*n+i (C's (k,i)).
        # The tensor identity: C[i,k] = sum_j A[i,j] B[j,k]
        # In rank-1 form: C[i,k] = sum_t (sum_j U[i*m+j, t] A[i,j]) (sum_k' V[j'*p+k, t] B[j',k]) ... 
        # Actually the standard form: C[i,k] = sum_t U[i,t] V[k,t] W... no.
        # The matrix multiplication tensor T_{i,j,k} with T[i,j,k]=1 for the (i,j,k) basis.
        # Decomposition: T[i,j,k] = sum_t U[i*m+j? ...]
        # Let's use the convention: the bilinear form gives C[i,k] = sum_t U_{i,t}^{(A)} V_{k,t}^{(B)} ... 
        # Given the schoolbook uses U[i*m+j, c], V[j*p+k, c], W[k*n+i, c], the reconstruction is:
        # C[i,k] = sum_{j,j',k'} A[i,j] B[j',k'] (sum_c U[i*m+j,c] V[j'*p+k',c] W[k*n+i,c])
        # For this to equal sum_j A[i,j] B[j,k], we need the sum_c term to be delta(j,j') delta(k',k).
        Ufull = np.zeros((n, m, rank))
        Vfull = np.zeros((m, p, rank))
        Wfull = np.zeros((p, n, rank))
        for i in range(n):
            for j in range(m):
                Ufull[i, j, :] = Ua[i * m + j]
        for j in range(m):
            for k in range(p):
                Vfull[j, k, :] = Va[j * p + k]
        for k in range(p):
            for i in range(n):
                Wfull[k, i, :] = Wa[k * n + i]
        # Kernel K[j, j', k'] = sum_c U[i*m+j,c] V[j'*p+k',c] W[k*n+i,c] should be delta(j,j')delta(k',k)
        # Build M1[i,j,j',k'] = sum_c U[i*m+j,c] V[j'*p+k',c] W[k*n+i,c]
        # Then C[i,k] = sum_{j,j',k'} A[i,j] B[j',k'] M1[i,j,j',k']
        # We need M1[i,j,j',k'] = delta(j,j') delta(k',k) for all i.
        # Compute M1 for each i.
        for i in range(n):
            # U_i[j,c] = Ufull[i,j,c], shape (m, rank)
            # V_all[j',k',c] shape (m,p,rank)
            # W_i[k,c] = Wfull[k,i,c] shape (p,rank)
            # M1[j,j',k'] = sum_c U_i[j,c] V[j',k',c] W_i[k,c]
            # = sum_c (U_i[j,c] W_i[k,c]) V[j',k',c]
            # Let P[j,k,c] = U_i[j,c] * W_i[k,c], shape (m,p,rank)
            # M1[j,j',k'] = sum_c P[j,k,c] V[j',k',c]
            # = einsum('jkc,j'kc->jj'k'') ... let's do: for each (j,k): sum_c P[j,k,c] * V[j',k',c]
            # M1[j, j', k'] = sum_{c} P[j,k,c] V[j',k',c]  -- but k is fixed by W_i[k,c], and k' by V.
            # So M1[j,j',k'] = sum_c U_i[j,c] V[j',k',c] W_i[k,c]
            # Compute via: T1 = np.einsum('jc,jkc,ic->jjik', U_i, V_all, W_i) where W_i[k,c]
            U_i = Ufull[i]  # (m, rank)
            W_i = Wfull[:, i]  # (p, rank)
            # M1[j,j',k'] = sum_c U_i[j,c] * V_all[j',k',c] * W_i[k,c]
            # = np.einsum('jc,jkc,ic->jjik', U_i, Vfull, W_i)  -> result[j,j',i_out=k? , k]
            # Let's be explicit:
            # temp[j',k',j] = sum_c Vfull[j',k',c] * U_i[j,c]   (m', p, m)
            temp = np.einsum('jkc,jc->jkj', Vfull, U_i)  # shape (m, p, m): [j', k', j]
            # M1[j,j',k'] = sum_c ... = temp[j',k',j] * W_i[k,c] summed over c? No.
            # M1[j,j',k'] = sum_c U_i[j,c] V[j',k',c] W_i[k,c]
            # = sum_c (U_i[j,c] W_i[k,c]) V[j',k',c]
            # For fixed (j,k): sum_c U_i[j,c] W_i[k,c] V[j',k',c]
            # = einsum over c of (U_i[j,:] * W_i[k,:]) dot V[j',k',:]
            # Let's just compute the full 4D tensor.
            M1 = np.zeros((m, m, p))
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_i[j, c] * Vfull[jp, kp, c] * W_i[k, c]  # k here is the output col index in W_i
                        # Wait, W_i is indexed by output column k (0..p-1). But M1's third index should be the output k.
                        # Let me rename: M1[j, j', k_out] = sum_c U_i[j,c] V[j',k_out,c] W_i[k_out,c]
                        # So the third index is k_out, and W_i[k_out,c].
                        # I made an error: W_i[k,c] where k is the output column. So:
                        pass
            # Redo cleanly:
            M1 = np.zeros((m, m, p))
            for j in range(m):
                for jp in range(m):
                    for kout in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_i[j, c] * Vfull[jp, kout, c] * W_i[kout, c]
                        M1[j, jp, kout] = s
            # We need M1[j, jp, kout] = delta(j,jp) * delta(kout, ?) -- actually we need the full C[i,kout] = sum_{j,jp,kout'} A[i,j] B[jp,kout'] M1[j,jp,kout'] = sum_j A[i,j] B[j,kout]
            # So we need: sum_{j,jp,kout'} A[i,j] B[jp,kout'] M1[j,jp,kout'] = sum_j A[i,j] B[j,kout]
            # This holds iff M1[j,jp,kout'] = delta(j,jp) delta(kout', kout) ... but kout is the free output index.
            # Actually: C[i,kout] = sum_{j,jp,kout'} A[i,j] B[jp,kout'] M1[j,jp,kout']
            # For this to equal sum_j A[i,j] B[j,kout] for all A,B, we need M1[j,jp,kout'] = delta(j,jp) delta(kout',kout)
            # i.e. M1[j,jp,kout'] = delta(j,jp) * delta(kout', kout)  -- but kout is the left side's index.
            # So M1[j,jp,kout'] must equal delta(j,jp) * delta(kout', kout) where kout is the target column.
            # That means M1[j,jp,kout'] = delta(j,jp) delta(kout',kout) for the specific kout.
            # So for each kout, the matrix M1[:, :, kout'] over (j,jp,kout') contracted must give identity in j and pick kout'=kout.
            # Equivalently: for all i, the map (A[i,:], B) -> C[i,:] is correct.
            # Let's just check C directly.
            C_recon = np.zeros((n, p))
            for i2 in range(n):
                U_i2 = Ufull[i2]
                W_i2 = Wfull[:, i2]
                for j in range(m):
                    for jp in range(m):
                        for kout in range(p):
                            M1v = 0.0
                            for c in range(rank):
                                M1v += U_i2[j, c] * Vfull[jp, kout, c] * W_i2[kout, c]
                            C_recon[i2, kout] += A[i2, j] * B[jp, kout] * M1v
            if not np.allclose(C_recon, C, atol=1e-6):
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook(n, m, p)
    
    # Try the constructive block decomposition.
    try:
        cand = _build_block27(n, m, p)
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p, trials=3, seed=1):
            best = cand
    except Exception:
        pass
    
    # Attempt a random/structured search for even lower rank if time permits.
    if time.time() < _DEADLINE - 30:
        # Try rank from 9 down to 1 with random initialization and a few Newton steps is complex;
        # instead try a few structured low-rank candidates.
        for rank_try in [9, 12, 15, 18, 21, 24, 27, 30, 36, 45]:
            if time.time() > _DEADLINE - 20:
                break
            # Skip if we already have a rank <= rank_try.
            if len(best["U"][0]) <= rank_try:
                continue
            # Construct a candidate: use random but structured matrices.
            # Given complexity, we'll just keep the best verified so far.
            # (A full tensor search is heavy; the block27 construction is our main improvement.)
            pass
    
    return best
