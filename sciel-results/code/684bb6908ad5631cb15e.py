"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

_START = time.time()
_DEADLINE = _START + 120.0


def _remaining():
    return _DEADLINE - time.time()


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term([(i, j, 1)], [(j, k, 1)], [(k, i, 1)])

    def add_strassen(i_start, j_start, k_start):
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        # M1 = (A00 + A11)(B00 + B11)
        add_term([(i0, j0, 1), (i1, j1, 1)],
                 [(j0, k0, 1), (j1, k1, 1)],
                 [(k0, i0, 1), (k1, i1, 1)])
        # M2 = (A10 + A11) B00
        add_term([(i1, j0, 1), (i1, j1, 1)],
                 [(j0, k0, 1)],
                 [(k0, i1, 1), (k1, i1, -1)])
        # M3 = A00 (B01 - B11)
        add_term([(i0, j0, 1)],
                 [(j0, k1, 1), (j1, k1, -1)],
                 [(k1, i0, 1), (k1, i1, 1)])
        # M4 = A11 (B10 - B00)
        add_term([(i1, j1, 1)],
                 [(j1, k0, 1), (j0, k0, -1)],
                 [(k0, i0, 1), (k0, i1, 1)])
        # M5 = (A00 + A01) B11
        add_term([(i0, j0, 1), (i0, j1, 1)],
                 [(j1, k1, 1)],
                 [(k0, i0, -1), (k1, i0, 1)])
        # M6 = (A10 - A00)(B00 + B01)
        add_term([(i1, j0, 1), (i0, j0, -1)],
                 [(j0, k0, 1), (j0, k1, 1)],
                 [(k1, i1, 1)])
        # M7 = (A01 - A11)(B10 + B11)
        add_term([(i0, j1, 1), (i1, j1, -1)],
                 [(j1, k0, 1), (j1, k1, 1)],
                 [(k0, i0, 1)])

    # Partition m = (2,2,1), p = (2,2,1); n = 2.
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    add_schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 4, 5)

    def _decompose_terms():
        """Return (U, V, W) as lists of lists from current term set (exact floats)."""
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for c in range(rank):
            a, b, w = terms[c]
            for (i, j, val) in a:
                U[i * m + j][c] = val
            for (j, k, val) in b:
                V[j * p + k][c] = val
            for (k, i, val) in w:
                W[k * n + i][c] = val
        return U, V, W

    def _tensor_from(U, V, W):
        rank = len(W[0]) if W else 0
        T = {}
        for c in range(rank):
            for ii in range(n * m):
                u = U[ii][c]
                if u == 0.0:
                    continue
                r, s = divmod(ii, m)
                for jj in range(m * p):
                    v = V[jj][c]
                    if v == 0.0:
                        continue
                    t, uidx = divmod(jj, p)
                    if s != t:
                        continue
                    for kk in range(p * n):
                        w = W[kk][c]
                        if w == 0.0:
                            continue
                        key = (r, uidx, divmod(kk, n)[1])
                        T[key] = T.get(key, 0.0) + u * v * w
        return T

    def _exact(T, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T.get((i, j, k), 0.0) - 1.0) > tol:
                        return False
        for (i, j, k), val in T.items():
            if i < 0 or i >= n or j < 0 or j >= m or k < 0 or k >= p:
                if abs(val) > tol:
                    return False
        return True

    def _svd_reduce(U, V, W, tol=1e-10):
        """
        Use SVD to find linear dependencies among rank vectors.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank_orig = U_arr.shape[1]
        if rank_orig == 0:
            return U, V, W
        
        U_rank = np.linalg.matrix_rank(U_arr, tol=tol)
        V_rank = np.linalg.matrix_rank(V_arr, tol=tol)
        W_rank = np.linalg.matrix_rank(W_arr, tol=tol)
        
        def find_redundant_columns(mat, tol=1e-10):
            mat_arr = np.array(mat, dtype=float)
            n_cols = mat_arr.shape[1]
            redundant = set()
            
            for c in range(n_cols):
                other_cols = [j for j in range(n_cols) if j != c]
                if not other_cols:
                    continue
                
                other_mat = mat_arr[:, other_cols]
                col_c = mat_arr[:, c]
                
                try:
                    if np.linalg.matrix_rank(other_mat, tol=tol) < len(other_cols):
                        x, residuals, _, _ = np.linalg.lstsq(other_mat, col_c, rcond=tol)
                        if residuals.size > 0 and residuals[0] > tol * (np.linalg.norm(col_c) + 1e-30):
                            continue
                    else:
                        x = np.linalg.solve(other_mat, col_c)
                        if np.linalg.norm(other_mat @ x - col_c) > tol * (np.linalg.norm(col_c) + 1e-30):
                            continue
                    
                    redundant.add(c)
                except np.linalg.LinAlgError:
                    continue
            
            return redundant
        
        U_redundant = find_redundant_columns(U, tol)
        V_redundant = find_redundant_columns(V, tol)
        W_redundant = find_redundant_columns(W, tol)
        
        if U_redundant or V_redundant or W_redundant:
            common = U_redundant & V_redundant & W_redundant
            if common:
                keep = [i for i in range(rank_orig) if i not in common]
                U_new = [[U[i][j] for j in keep] for i in range(len(U))]
                V_new = [[V[i][j] for j in keep] for i in range(len(V))]
                W_new = [[W[i][j] for j in keep] for i in range(len(W))]
                
                if _exact(_tensor_from(U_new, V_new, W_new)):
                    return U_new, V_new, W_new
        
        return None

    def _pairwise_merge_reduce(U, V, W, tol=1e-10):
        """
        Try to merge pairs of terms by checking if their combined contribution
        can be expressed with fewer terms.
        """
        rank = len(U[0]) if U else 0
        if rank < 2:
            return None
        
        def is_proportional(a, b, tol=1e-10):
            a_arr = np.array(a, dtype=float)
            b_arr = np.array(b, dtype=float)
            if np.allclose(a_arr, 0, atol=tol):
                return np.allclose(b_arr, 0, atol=tol)
            if np.allclose(b_arr, 0, atol=tol):
                return False
            
            mask = np.abs(a_arr) > tol
            if not np.any(mask):
                return False
            
            scales = b_arr[mask] / a_arr[mask]
            if not np.allclose(scales, scales[0], atol=tol, rtol=tol):
                return False
            
            scale = scales[0]
            return np.allclose(b_arr, scale * a_arr, atol=tol, rtol=tol)
        
        merged = False
        for c1 in range(rank):
            for c2 in range(c1 + 1, rank):
                u1 = [U[i][c1] for i in range(len(U))]
                u2 = [U[i][c2] for i in range(len(U))]
                v1 = [V[i][c1] for i in range(len(V))]
                v2 = [V[i][c2] for i in range(len(V))]
                w1 = [W[i][c1] for i in range(len(W))]
                w2 = [W[i][c2] for i in range(len(W))]
                
                if is_proportional(u1, u2, tol) and is_proportional(v1, v2, tol) and is_proportional(w1, w2, tol):
                    keep = [i for i in range(rank) if i != c2]
                    U_new = [[U[i][j] for j in keep] for i in range(len(U))]
                    V_new = [[V[i][j] for j in keep] for i in range(len(V))]
                    W_new = [[W[i][j] for j in keep] for i in range(len(W))]
                    
                    if _exact(_tensor_from(U_new, V_new, W_new)):
                        return U_new, V_new, W_new
                    merged = True
                    break
            if merged:
                break
        
        return None

    def _try_cyclic_embedding():
        """
        Try a cyclic group code embedding using DFT over Z_5.
        Map indices j, k to Z_5 and use Fourier transform structure.
        This may yield a lower rank decomposition.
        """
        from itertools import product
        
        # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # with the constraint that the middle index matches.
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 if we're computing C[i,k] via term (i,j,k)
        
        # Try to use the fact that 5 is prime to get a structured decomposition.
        # Use the identity: sum_{t=0}^{4} omega^{t*j} * omega^{-t*k} = 5*delta_{j,k}
        # where omega = e^{2*pi*i/5}
        
        # But we need real/rational coefficients, so let's try integer-based cyclic codes.
        
        # Alternative: try to find a decomposition where the U, V, W matrices
        # have a circulant-like structure based on Z_5.
        
        # For now, try a simple approach: see if we can reduce the rank by
        # exploiting symmetries in the 5x5 block.
        
        # The current decomposition has rank 46. Let's see if we can do better
        # by using a different partitioning.
        
        # Try partitioning m and p into (1,1,1,1,1) and using schoolbook everywhere:
        # That gives 2*5*5 = 50 terms.
        
        # Try (2,2,1) for both m and p:
        # Blocks: (2,2), (2,1), (1,2), (1,1)
        # (2,2) block: 7 terms (Strassen)
        # (2,1) block: 2 terms
        # (1,2) block: 2 terms  
        # (1,1) block: 1 term
        # Total per n-row: 7+2+2+1 = 12
        # For n=2: 24 terms? No wait, the Strassen is for 2x2x2.
        
        # Let me reconsider. The decomposition is:
        # For each pair (i_row, k_col) in C, we need sum_j A[i_row,j]*B[j,k_col]
        # The tensor has entries T[i,j,k] = delta_{j,j} (trivially 1) for the
        # matrix multiplication structure.
        
        # Actually, the tensor for matrix multiplication C = A*B where
        # A is n x m, B is m x p, C is n x p has:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because C[i,k] = sum_j A[i,j] B[j,k]
        
        # So the tensor is just the all-ones tensor of shape (n, m, p) = (2,5,5).
        # Wait, that's not right either. The tensor is:
        # T_{i,j,k} such that C[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k]
        # So T_{i,j,k} = 1 if and only if the middle index matches, which it always does.
        # So T is indeed the all-ones tensor of shape (2,5,5).
        
        # The rank of the all-ones tensor of shape (n,m,p) is at most min(n*m, m*p, p*n)
        # but can be lower with structured decompositions.
        
        # For (2,5,5): the trivial rank is 2*5*5 = 50.
        # Strassen on 2x2 blocks reduces it.
        
        # Let's try a different approach: use the fact that we can write
        # the all-ones matrix of size 5x5 as a sum of rank-1 matrices.
        # The all-ones 5x5 matrix has rank 1, so J = u*v^T where u = (1,1,1,1,1)^T
        # and v = (1,1,1,1,1)^T.
        
        # But we need a tensor decomposition, not a matrix decomposition.
        
        # Let me try: T[i,j,k] = 1 = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # One simple decomposition: for each (j,k) pair, use one term:
        # U[i,j,c_{j,k}] = 1, V[j,k,c_{j,k}] = 1, W[k,i,c_{j,k}] = 1
        # This gives rank = m*p = 25, but wait, that's not right because
        # U depends on i and j, V on j and k, W on k and i.
        
        # Actually, for the all-ones tensor, we can use:
        # T[i,j,k] = 1 = (1)*(1)*(1)
        # So rank 1 would suffice if we could use U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
        # for all i,j,k. But that's not a valid decomposition because the
        # decomposition must be of the form sum_c u_c ⊗ v_c ⊗ w_c where
        # u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
        
        # For rank 1: T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
        # We need this to equal 1 for all i,j,k.
        # Let U[i*m+j] = a_{i,j}, V[j*p+k] = b_{j,k}, W[k*n+i] = c_{k,i}
        # Then a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
        # This is possible: set a_{i,j} = 1, b_{j,k} = 1, c_{k,i} = 1.
        # But wait, this would mean T is the tensor product of three vectors,
        # which is indeed rank 1. But the matrix multiplication tensor is NOT
        # the all-ones tensor!
        
        # I was confused. The matrix multiplication tensor M_{n,m,p} has:
        # M[i,j,k] = 1 if and only if the indices form a valid "path" i->j->k
        # In matrix multiplication C = A*B: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is defined such that the bilinear form is:
        # <M, A ⊗ B> = sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
        # Wait, that's not quite right either.
        
        # The standard definition: the matrix multiplication tensor M_{n,m,p}
        # is a 3-tensor with entries M[i,j,k] = 1 for all i,j,k in the appropriate
        # ranges, because the multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has M[i,j,k] = 1 for all valid (i,j,k).
        
        # Hmm, but that would make it the all-ones tensor, which has rank 1
        # as a tensor (since it's the outer product of three all-ones vectors).
        # That can't be right because matrix multiplication tensor rank is known
        # to be at least n*m*p/2 or something.
        
        # Let me look this up mentally. The matrix multiplication tensor
        # M_{n,m,p} is defined by:
        # M[i,j,k] = 1 if and only if j is the "middle" index connecting i and k.
        # In the standard basis, the tensor is:
        # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{valid}
        # For matrix multiplication, ALL triples (i,j,k) are valid because
        # C[i,k] gets a contribution from A[i,j]*B[j,k] for every j.
        # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # But then the rank would be 1, which contradicts known results.
        
        # I think I'm confusing the tensor with the multiplication map.
        # The multiplication map is: mu: R^{n x m} ⊗ R^{m x p} -> R^{n x p}
        # mu(A ⊗ B) = A * B
        # This is a trilinear map, and its tensor rank is what we're looking for.
        # The tensor associated with this map has entries:
        # T[i,k,j] = 1 if the (i,k) entry of AB comes from A[i,j]*B[j,k]
        # So T[i,k,j] = 1 for all i,j,k.
        
        # But when we write the tensor in the basis where the three indices are
        # (i,j) for A, (j,k) for B, and (k,i) for C (with the trace convention),
        # we get:
        # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * 1
        # Wait, no. The tensor is:
        # T[a, b, c] where a = (i,j), b = (j',k), c = (k',i')
        # T[a,b,c] = 1 if j = j' and k = k' and i = i' (for the contribution to C[i',k'])
        # Actually, C[i',k'] = sum_j A[i',j] B[j,k']
        # So the contribution of A[i,j] and B[j',k] to C[i',k'] is:
        # delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        # So T[(i,j),(j',k),(k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        
        # This means the tensor is NOT the all-ones tensor. It's a "structured"
        # tensor where only certain triples are nonzero.
        
        # In the trace convention W[k*n+i, r], the W index is (k,i) not (i,k).
        # So c = (k,i) and the condition is k = k' and i = i'.
        
        # So T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i'
        # Otherwise 0.
        
        # This is actually the "diagonal" tensor, which has a specific structure.
        
        # The rank of this tensor is at least max(n*m, m*p, p*n) / something.
        # For (2,5,5): n*m=10, m*p=25, p*n=10.
        # The trivial rank is 25 (one term for each (j,k) pair).
        
        # Wait, let me reconsider. For each (j,k) pair, we can have a term:
        # U[(i,j), c_{j,k}] = 1 for all i
        # V[(j,k), c_{j,k}] = 1
        # W[(k,i), c_{j,k}] = 1 for all i
        # But this doesn't work because U and W depend on i.
        
        # Actually, for a fixed (j,k), the contribution to C[i,i] (diagonal) is
        # A[i,j] * B[j,k]. But we need C[i,k] = sum_j A[i,j] B[j,k].
        
        # I think the correct tensor is:
        # T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i'
        # This means for each (j,k) pair, we have a "slice" that contributes to
        # all i with the same i on both sides.
        
        # So the tensor can be written as:
        # T = sum_{j,k} sum_i e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
        # = sum_{j,k} (sum_i e_{(i,j)} ⊗ e_{(k,i)}) ⊗ e_{(j,k)}
        
        # For fixed (j,k): sum_i e_{(i,j)} ⊗ e_{(k,i)} is a matrix of size (n*m) x (p*n)
        # that has 1s at positions ((i,j), (k,i)) for i=0,...,n-1.
        
        # This matrix has rank n (it's a permutation-like matrix).
        
        # So the total tensor rank is at most sum_{j,k} n = m*p*n = 50.
        # But we can do better by combining terms.
        
        # The current decomposition achieves rank 46. Can we do better?
        
        # Let's try to use the structure more effectively.
        
        # For n=2, m=5, p=5:
        # The tensor is: T[(i,j),(j',k),(k',i')] = delta_{j,j'} delta_{k,k'} delta_{i,i'}
        
        # We can group by j: for each j, we have a sub-tensor of size (2,1,5) x (1,5,1) x (5,2,1)
        # which is essentially a 2x5 matrix multiplication for fixed j.
        
        # Hmm, this is getting complicated. Let me just try to improve the
        # current decomposition by trying different block partitions.
        
        # Try partitioning m = (2,2,1) and p = (1,2,2):
        terms.clear()
        add_strassen(0, 0, 0)      # m-block 0-1, p-block 0-1
        add_schoolbook(0, 2, 0, 2, 2, 3)  # m-block 0-1, p-block 2
        add_schoolbook(0, 2, 0, 2, 3, 5)  # m-block 0-1, p-block 3-4
        add_schoolbook(0, 2, 2, 3, 0, 2)  # m-block 2, p-block 0-1
        add_strassen(0, 2, 3)      # m-block 2, p-block 3-4 (but m-block 2 is size 1)
        # Wait, Strassen requires 2x2 blocks. Let me be more careful.
        
        # Let me just keep the current decomposition and try to reduce it.
        terms.clear()
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)
        
        return None  # No improvement found

    U0, V0, W0 = _decompose_terms()
    if not _exact(_tensor_from(U0, V0, W0)):
        return {"U": U0, "V": V0, "W": W0}

    best_result = (U0, V0, W0)
    best_rank = len(U0[0])
    
    if _remaining() > 2.0:
        reduced = _svd_reduce(U0, V0, W0)
        if reduced is not None:
            rU, rV, rW = reduced
            if len(rW[0]) < best_rank and _exact(_tensor_from(rU, rV, rW)):
                best_result = (rU, rV, rW)
                best_rank = len(rW[0])
    
    if _remaining() > 2.0:
        reduced = _pairwise_merge_reduce(best_result[0], best_result[1], best_result[2])
        if reduced is not None:
            rU, rV, rW = reduced
            if len(rW[0]) < best_rank and _exact(_tensor_from(rU, rV, rW)):
                best_result = (rU, rV, rW)
                best_rank = len(rW[0])
    
    for _ in range(5):
        if _remaining() < 1.0:
            break
        
        rU, rV, rW = best_result
        reduced = _pairwise_merge_reduce(rU, rV, rW)
        if reduced is not None:
            nrU, nrV, nrW = reduced
            if len(nrW[0]) < best_rank and _exact(_tensor_from(nrU, nrV, nrW)):
                best_result = (nrU, nrV, nrW)
                best_rank = len(nrW[0])
            else:
                break
        else:
            break
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
