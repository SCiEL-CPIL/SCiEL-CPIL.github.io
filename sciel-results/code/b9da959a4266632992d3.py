"""Matrix multiplication 2x4x7 - rank-minimal decomposition via randomized search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
DEADLINE = 148.0  # safety margin under the 150s evaluator kill

def _time_left():
    return DEADLINE - time.time()

def _strassen_solution():
    """Guaranteed rank-50 decomposition (Strassen blocks + direct tail)."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
        col += 1
        U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
        col += 1
        U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    add_strassen(0, 0, 0); add_strassen(0, 0, 2)
    add_strassen(0, 2, 0); add_strassen(0, 2, 2)
    add_strassen(0, 0, 4); add_strassen(0, 2, 4)
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _merge_rank1(U, V, W):
    """Merge any two rank-1 terms whose U and V vectors are proportional."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    used = [False] * r
    newU, newV, newW = [], [], []
    for a in range(r):
        if used[a]:
            continue
        ua, va, wa = U[:, a].copy(), V[:, a].copy(), W[:, a].copy()
        for b in range(a + 1, r):
            if used[b]:
                continue
            ub, vb, wb = U[:, b], V[:, b], W[:, b]
            na, nb = np.linalg.norm(ua), np.linalg.norm(ub)
            if na < 1e-12 or nb < 1e-12:
                continue
            cos = abs(np.dot(ua, ub)) / (na * nb)
            if cos < 0.999999:
                continue
            mva = np.linalg.norm(va); mvb = np.linalg.norm(vb)
            if mva < 1e-12 or mvb < 1e-12:
                continue
            cosv = abs(np.dot(va, vb)) / (mva * mvb)
            if cosv < 0.999999:
                continue
            used[b] = True
            # Scale factor: ua = s * ub, so s = dot(ua,ub)/norm(ub)^2
            s = np.dot(ua, ub) / (nb * nb)
            wa = wa + s * wb
            newU.append(ua); newV.append(va); newW.append(wa)
            break
        else:
            newU.append(ua); newV.append(va); newW.append(wa)
    if not newU:
        return U, V, W
    U = np.column_stack(newU); V = np.column_stack(newV); W = np.column_stack(newW)
    return U, V, W

def _reduce_rank(U, V, W, T, max_iter=50):
    """Try to reduce rank by finding linear dependencies among rank-1 terms.
    
    If sum of coefficients c_r * U[:,r] = 0 and sum c_r * V[:,r] = 0,
    then we can eliminate one term if the W constraint is also satisfied.
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T2 = T.reshape(n*m, m*p, p*n)
    r = U.shape[1]
    
    for _ in range(max_iter):
        # Check for null space in U
        _, sU, VtU = np.linalg.svd(U, full_matrices=False)
        # Check for null space in V
        _, sV, VtV = np.linalg.svd(V, full_matrices=False)
        
        # Find vectors c such that c^T U ≈ 0 and c^T V ≈ 0
        # This means c is in the null space of both U^T and V^T
        # i.e., in the intersection of null spaces
        # Use SVD: null space of U^T is spanned by last (r - rank_U) rows of VtU
        # But we need common null space vectors
        
        # Alternative: find c such that ||c^T U|| and ||c^T V|| are small
        # and also c^T W satisfies the tensor constraint
        # Simpler: just check if any pair can be merged (already done)
        # Try: for each pair, check if their W vectors are also proportional
        # (we already check U and V proportionality)
        
        # Try a different approach: remove one term at a time and see if we can
        # adjust the remaining terms to compensate
        found = False
        for elim in range(r):
            # Try removing term 'elim'
            U_rem = np.delete(U, elim, axis=1)
            V_rem = np.delete(V, elim, axis=1)
            W_rem = np.delete(W, elim, axis=1)
            r_rem = r - 1
            
            # Solve for adjustment to W_rem to compensate for removed term
            # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # We need: T = sum_{r≠elim} U[:,r] ⊗ V[:,r] ⊗ (W[:,r] + delta_W[:,r])
            # So: U[:,elim] ⊗ V[:,elim] ⊗ W[:,elim] = sum_{r≠elim} U[:,r] ⊗ V[:,r] ⊗ delta_W[:,r]
            
            # This is a linear system in delta_W
            # Flatten: for each (i,j,k) in the tensor:
            # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # After removal: T[i,j,k] = sum_{r≠elim} U[i,r]*V[j,r]*(W[k,r] + dW[k,r])
            # So: sum_{r≠elim} U[i,r]*V[j,r]*dW[k,r] = U[i,elim]*V[j,elim]*W[k,elim]
            
            # This is: for each k, sum_{r≠elim} U[i,r]*V[j,r]*dW[k,r] = U[i,elim]*V[j,elim]*W[k,elim]
            # Let A[k, (i,j), r] = U[i,r]*V[j,r] for r≠elim
            # Then A[k, :, :] @ dW[k, :] = U[:,elim]*V[:,elim]*W[k,elim]... 
            # This is getting complex. Let's just verify and try a simpler approach.
            
            # Simple check: can we just zero out the eliminated term's contribution?
            # No, that changes the tensor.
            
            # Skip this for now - the merge is the main tool
            pass
        
        break  # Only do one pass for now
    
    return U, V, W

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    T2 = T.reshape(n*m, m*p, p*n)
    return bool(np.allclose(T2, _build_tensor(), atol=1e-8))

def _find_decomposition(T, rank, rng, structured=False):
    """Randomized: pick random U,V, solve for W, verify.
    If structured=True, use patterns based on the tensor structure."""
    n, m, p = DIMENSIONS
    if structured:
        # Use structured initialization based on the tensor's sparsity pattern
        # The tensor has non-zero entries at T[i*m+j, j*p+k, k*n+i] = 1
        # Try to use patterns that align with this structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        # Assign random subsets of rows to each rank term
        for r in range(rank):
            # Pick random row indices for U
            u_rows = rng.choice(n*m, size=max(1, n*m//3), replace=False)
            for idx in u_rows:
                U[idx, r] = rng.choice([-1, 1])
            # Pick random row indices for V
            v_rows = rng.choice(m*p, size=max(1, m*p//3), replace=False)
            for idx in v_rows:
                V[idx, r] = rng.choice([-1, 1])
    else:
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
    
    U, V = _merge_rank1(U, V, np.zeros((p*n, rank)))
    if U.shape[1] < rank:
        return None
    
    T2 = T.reshape(n*m, m*p, p*n)
    T2 = np.moveaxis(T2, 2, 0)  # (p*n, n*m, m*p)
    rhs = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
    rhs = np.moveaxis(rhs, 2, 0)         # (rank, n*m, m*p)
    A = rhs.reshape(rank, -1)
    b = T2.reshape(-1)
    try:
        Wflat, res, rankA, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    W = Wflat.reshape(rank, p*n)
    if not _verify(U, V, W):
        return None
    U, V, W = _merge_rank1(U, V, W)
    return U, V, W

def _structured_search(T, max_rank, rng, time_budget):
    """Try structured decompositions using the tensor's sparsity pattern."""
    n, m, p = DIMENSIONS
    best = None
    best_rank = max_rank + 1
    
    # Strategy: exploit the fact that the tensor has a very specific structure
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # This means each "fiber" has exactly one non-zero entry
    
    # Try: for each k, the slice T[:, :, k*n+i] has non-zeros at (i*m+j, j*p+k)
    # for i=0..n-1, j=0..m-1
    
    # A rank-r decomposition means we can write the multiplication as
    # sum_{r=1}^R U[:,r] * V[:,r] * W[:,r]
    
    # For 2x4x7, the schoolbook rank is 56. Strassen gives 50.
    # Let's try to find if we can do better by exploiting the small n=2.
    
    # Key insight: since n=2, we have T[i*m+j, j*p+k, k*n+i]
    # For fixed j,k: entries at (0*m+j, j*p+k, k*2+0) and (1*m+j, j*p+k, k*2+1)
    # = (j, j*p+k, 2k) and (m+j, j*p+k, 2k+1)
    
    # Try a block structure: group by j (4 groups of size 7x2)
    # For each j, we have a 2x7x2 sub-tensor (i, k, output)
    # The 2x7x2 matrix mult has rank at most 14 (schoolbook)
    # But we can share across j's
    
    # Alternative: try to find a low-rank solution by solving a larger system
    # with constraints from the tensor structure
    
    for rank in range(min(max_rank, 48), 30, -1):
        if time.time() > time_budget:
            break
        # Try multiple random initializations with structure
        for attempt in range(3):
            if time.time() > time_budget:
                break
            res = _find_decomposition(T, rank, rng, structured=True)
            if res is not None:
                U, V, W = res
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best = (U, V, W)
                break
        if best is not None and best[0].shape[1] < rank:
            break
    
    return best, best_rank

def _to_lists(U, V, W):
    U = np.array(U); V = np.array(V); W = np.array(W)
    r = U.shape[1]
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }

def construct():
    n, m, p = DIMENSIONS
    random.seed(12345)
    rng = np.random.default_rng(12345)
    T = _build_tensor()

    # Guaranteed fallback.
    best = _strassen_solution()
    best_rank = 50

    # Rank-1 merge of the fallback (free improvement if any).
    Uf, Vf, Wf = _merge_rank1(best["U"], best["V"], best["W"])
    if _verify(Uf, Vf, Wf) and Uf.shape[1] < best_rank:
        best_rank = Uf.shape[1]
        best = _to_lists(Uf, Vf, Wf)

    # Structured search for lower ranks.
    time_budget = time.time() + _time_left() * 0.5
    res, res_rank = _structured_search(T, best_rank - 1, rng, time_budget)
    if res is not None and res_rank < best_rank:
        best_rank = res_rank
        best = _to_lists(res[0], res[1], res[2])

    # Randomized search for lower ranks.
    for rank in range(best_rank - 1, 30, -1):
        if _time_left() < 5.0:
            break
        # Number of attempts scaled by remaining time.
        attempts = max(1, int(_time_left() // 1.5))
        for _ in range(attempts):
            if _time_left() < 3.0:
                break
            res = _find_decomposition(T, rank, rng)
            if res is not None:
                U, V, W = res
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best = _to_lists(U, V, W)
                break

    # Final polish: try to reduce rank further via merge
    if _time_left() > 2.0:
        Uf, Vf, Wf = _merge_rank1(best["U"], best["V"], best["W"])
        if _verify(Uf, Vf, Wf) and Uf.shape[1] < best_rank:
            best_rank = Uf.shape[1]
            best = _to_lists(Uf, Vf, Wf)

    return best
# EVOLVE-BLOCK-END
