"""Tensor decomposition for 3x4 by 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4
NP, NM, MP = N * P, N * M, M * P


def _feasible(U, V, W):
    """Check whether U,V,W (rank r) exactly decompose the multiplication tensor."""
    r = U.shape[1]
    prod = (U @ V.T) @ W.T  # shape (NP, NM, MP)
    return (prod == 1.0).all()


def _try_remove(U, V, W, r, deadline):
    """Remove one column if the decomposition stays exact. Returns (U,V,W,removed_bool)."""
    for c in range(r):
        if time.time() > deadline:
            return U, V, W, False
        Un = np.delete(U, c, axis=1)
        Vn = np.delete(V, c, axis=1)
        Wn = np.delete(W, c, axis=1)
        if _feasible(Un, Vn, Wn):
            return Un, Vn, Wn, True
    return U, V, W, False


def _swap_cols(U, V, W, a, b, deadline):
    if a == b:
        return U, V, W, False
    if time.time() > deadline:
        return U, V, W, False
    Un = U.copy(); Un[:, a], Un[:, b] = Un[:, b].copy(), Un[:, a].copy()
    Vn = V.copy(); Vn[:, a], Vn[:, b] = Vn[:, b].copy(), Vn[:, a].copy()
    Wn = W.copy(); Wn[:, a], Wn[:, b] = Wn[:, b].copy(), Wn[:, a].copy()
    if _feasible(Un, Vn, Wn):
        return Un, Vn, Wn, True
    return U, V, W, False


def _split_col(U, V, W, c, deadline):
    """Split column c into two (c, c+1). Returns (U,V,W,new_r) or (None,None,None,r)."""
    r = U.shape[1]
    if time.time() > deadline:
        return None, None, None, r
    Un = np.empty((NP, r + 1)); Vn = np.empty((MP, r + 1)); Wn = np.empty((NM, r + 1))
    Un[:, :c] = U[:, :c]; Vn[:, :c] = V[:, :c]; Wn[:, :c] = W[:, :c]
    Un[:, c] = U[:, c]; Vn[:, c] = V[:, c]; Wn[:, c] = W[:, c]
    Un[:, c + 1] = 0.0; Vn[:, c + 1] = V[:, c]; Wn[:, c + 1] = 0.0
    Un[:, c + 1:] = U[:, c:]; Vn[:, c + 1:] = V[:, c:]; Wn[:, c + 1:] = W[:, c:]
    if _feasible(Un, Vn, Wn):
        return Un, Vn, Wn, r + 1
    return None, None, None, r


def _try_reinsert(U, V, W, r, deadline):
    """Try to reinsert a new column into a zero column to reduce rank.
    Returns (U,V,W,improved_bool)."""
    # Find a zero column
    zero_cols = [c for c in range(r) if (U[:, c] == 0).all() and (V[:, c] == 0).all() and (W[:, c] == 0).all()]
    if not zero_cols:
        return U, V, W, False
    zc = zero_cols[0]
    # Remove the zero column to get a rank r-1 structure
    Un = np.delete(U, zc, axis=1)
    Vn = np.delete(V, zc, axis=1)
    Wn = np.delete(W, zc, axis=1)
    # Now try to add a new column to Un,Vn,Wn that makes it feasible
    # The new column must satisfy: for each (i,j,k) where Un,Vn,Wn already give 1, the new col must contribute 0
    # For (i,j,k) where they give 0, the new col must contribute 1
    prod = (Un @ Vn.T) @ Wn.T  # shape (NP, NM, MP)
    # Find entries that are 0 (need to be filled)
    need = (prod == 0.0)
    # Find entries that are 1 (must not be touched)
    have = (prod == 1.0)
    # For the new column (u,v,w): u[i]*v[j]*w[k] must be 0 where have is True, 1 where need is True
    # This is a rank-1 tensor matching problem
    # Try: for each i,j,k in need, u[i]*v[j]*w[k]=1, so u[i],v[j],w[k] all nonzero
    # For each i,j,k in have, at least one of u[i],v[j],w[k] is 0
    # This is a constraint satisfaction problem. Let's try a greedy approach.
    # First, determine which indices must be nonzero in u, v, w
    u_nz = set(); v_nz = set(); w_nz = set()
    for idx in np.argwhere(need):
        k, i, j = idx
        u_nz.add(i); v_nz.add(j); w_nz.add(k)
    # Check consistency: for each have entry, at least one of the three must be zero
    for idx in np.argwhere(have):
        k, i, j = idx
        if i in u_nz and j in v_nz and k in w_nz:
            return U, V, W, False  # Inconsistent
    # Now assign values: for nonzero entries, set to 1; for zero entries, set to 0
    # But we need to verify the full product
    u = np.zeros(NM); v = np.zeros(MP); w = np.zeros(NP)
    for i in u_nz: u[i] = 1.0
    for j in v_nz: v[j] = 1.0
    for k in w_nz: w[k] = 1.0
    # Verify
    new_prod = prod + np.einsum('i,j,k->kij', u, v, w)
    if (new_prod == 1.0).all():
        # Reinsert
        Un = np.insert(Un, zc, u, axis=1)
        Vn = np.insert(Vn, zc, v, axis=1)
        Wn = np.insert(Wn, zc, w, axis=1)
        if _feasible(Un, Vn, Wn):
            return Un, Vn, Wn, True
    return U, V, W, False


def construct():
    start = time.time()
    deadline = start + 115.0

    # Schoolbook initialization
    r = N * M * P
    U = np.zeros((NM, r)); V = np.zeros((MP, r)); W = np.zeros((NP, r))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r

    # Phase 1: Greedy removal
    improved = True
    while improved and time.time() < deadline:
        U, V, W, improved = _try_remove(U, V, W, U.shape[1], deadline)
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 2: Local search with swaps and splits
    r = U.shape[1]
    no_improve = 0
    while time.time() < deadline and no_improve < 50:
        made_change = False
        # Try swaps
        for a in range(r):
            if time.time() > deadline:
                break
            for b in range(a + 1, r):
                if time.time() > deadline:
                    break
                U, V, W, changed = _swap_cols(U, V, W, a, b, deadline)
                if changed:
                    made_change = True
                    r = U.shape[1]
                    break
            if time.time() > deadline:
                break
            if made_change:
                break
        if time.time() > deadline:
            break
        # Try splits
        if not made_change:
            for c in range(r):
                if time.time() > deadline:
                    break
                Un, Vn, Wn, new_r = _split_col(U, V, W, c, deadline)
                if Un is not None:
                    U, V, W = Un, Vn, Wn
                    r = new_r
                    made_change = True
                    break
        # Try removal again after changes
        if made_change:
            U, V, W, rem = _try_remove(U, V, W, U.shape[1], deadline)
            if rem:
                made_change = True
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

    # Phase 3: Try reinsertion to reduce rank
    while time.time() < deadline:
        U, V, W, imp = _try_reinsert(U, V, W, U.shape[1], deadline)
        if imp:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            break

    # Final greedy removal
    while time.time() < deadline:
        U, V, W, imp = _try_remove(U, V, W, U.shape[1], deadline)
        if imp:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            break

    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
