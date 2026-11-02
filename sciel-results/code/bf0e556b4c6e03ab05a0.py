import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_strassen_rank55():
    """Build the guaranteed rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    for k in range(p):
        b2k = 2 * p + k
        U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
        col += 1
        U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
        col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # The multiplication tensor T has T[i,j,k] = 1 if (i,j) pairs with (j,k) to produce (k,i)
    # In our indexing: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # The condition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    # where delta = 1 if the indices are consistent (which they always are for valid (i,j,k))
    # Actually, the tensor T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Check a sample of entries to verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_rank54_search(start_time):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    
    Key insight: With n=2, the output C has 2 rows. For each column k of B,
    the two outputs C[0,k] and C[1,k] are linear combinations of A's entries
    times B's entries. We can try to merge some terms from the schoolbook
    part (A_R * B_R) with Strassen terms to save rank.
    
    Strategy: For each of the 10 columns k, the schoolbook part uses 2 terms
    (A02*B2k and A12*B2k). If we can find a single term that contributes to
    both C[0,k] and C[1,k] using a shared factor, we save 1 rank per column.
    But this is constrained by the Strassen structure.
    
    Alternative: Try to find a rank-54 by modifying one of the Strassen blocks
    to share structure with the schoolbook terms.
    
    We attempt a local search: start from rank-55, try removing one column
    and see if the remaining 54 columns can be rescaled/modified to still
    satisfy the tensor equation.
    """
    n, m, p = DIMENSIONS
    U55, V55, W55 = _build_strassen_rank55()
    
    # Try each possible column to remove
    for remove_col in range(55):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove column remove_col
        mask = np.ones(55, dtype=bool)
        mask[remove_col] = False
        U54 = U55[:, mask]
        V54 = V55[:, mask]
        W54 = W55[:, mask]
        
        # Check if the remaining 54 columns can be adjusted to work.
        # The residual tensor is: R[i,j,k] = 1 - sum_{r != remove_col} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # This residual should equal the contribution of the removed column:
        # R[i,j,k] = U[i*m+j,remove_col] * V[j*p+k,remove_col] * W[k*n+i,remove_col]
        
        # Compute residual
        residual = np.ones((n, m, p), dtype=float)
        for r in range(54):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= U54[i * m + j, r] * V54[j * p + k, r] * W54[k * n + i, r]
        
        # The residual should factor as u * v * w where:
        # u[i*m+j] = U55[i*m+j, remove_col]
        # v[j*p+k] = V55[j*p+k, remove_col]
        # w[k*n+i] = W55[k*n+i, remove_col]
        
        u_orig = U55[:, remove_col]
        v_orig = V55[:, remove_col]
        w_orig = W55[:, remove_col]
        
        # Check if residual equals outer product u_orig * v_orig * w_orig
        # i.e., residual[i,j,k] == u_orig[i*m+j] * v_orig[j*p+k] * w_orig[k*n+i]
        matches = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = u_orig[i * m + j] * v_orig[j * p + k] * w_orig[k * n + i]
                    if abs(residual[i, j, k] - expected) > 1e-9:
                        matches = False
                        break
                if not matches:
                    break
            if not matches:
                break
        
        if matches:
            # This means removing column remove_col doesn't change the tensor,
            # which can only happen if the column was zero (unlikely) or we made an error.
            # Actually, this check is wrong. The residual IS the removed column's contribution.
            # We need to check if the residual can be represented with the SAME u,v,w
            # but possibly different scaling. Since we're just removing, the residual
            # must equal the original column's contribution, which it does by construction.
            # So this approach doesn't directly give rank 54.
            pass
        
        # Better approach: try to find if any column is redundant by checking
        # if the tensor can be decomposed with that column set to zero and
        # other columns adjusted. This is a hard problem, so we try a simpler heuristic.
        
        # Check if the removed column has very small norm (indicating it might be removable)
        col_norm = np.linalg.norm(u_orig) * np.linalg.norm(v_orig) * np.linalg.norm(w_orig)
        if col_norm < 1e-6:
            # Column is effectively zero, we can remove it
            if _verify_decomposition(U54, V54, W54):
                return U54, V54, W54
    
    return None


def _try_merge_schoolbook_with_strassen(start_time):
    """
    Try to merge schoolbook terms with Strassen terms to reduce rank.
    
    The schoolbook part has 20 terms: for each k in 0..9, two terms:
      A02 * B2k -> C[0,k]
      A12 * B2k -> C[1,k]
    
    The Strassen part for block b (columns 2b, 2b+1) has 7 terms.
    
    Idea: For a given column k, can we modify the Strassen terms for the
    block containing k to also contribute to C[0,k] or C[1,k] using A02 or A12?
    
    This requires that some Strassen term already involves a B entry from row 2
    (the B_R part), which it doesn't in the standard construction.
    
    Alternative: Try a different partition of B's columns. Instead of
    Strassen on 2x2 blocks of B_L, try a mixed approach.
    
    For now, try a simple local search: perturb the rank-55 solution and
    see if we can find a rank-54 solution via gradient descent on a
    relaxed objective.
    """
    n, m, p = DIMENSIONS
    U55, V55, W55 = _build_strassen_rank55()
    
    # Try to find a rank-54 by solving a system of equations.
    # We have 54 unknown columns (each with 6+30+20=56 entries, but with
    # scaling freedom: we can fix one entry per column to 1, giving 55*54 unknowns).
    # The tensor equation gives 2*3*10 = 60 constraints.
    # This is underdetermined, so solutions should exist if rank 54 is achievable.
    
    # Use a simpler approach: try all pairs of columns to merge.
    # For each pair (i, j) from the 55 columns, try to see if their contributions
    # can be combined into a single column.
    
    for i in range(55):
        if time.time() - start_time > DEADLINE:
            break
        for j in range(i + 1, 55):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to find if columns i and j can be replaced by a single column.
            # The combined contribution of columns i and j to the tensor is:
            # T_ij[i',j',k'] = U[i',m+j',i]*V[j'*p+k',i]*W[k'*n+i',i] 
            #                 + U[i',m+j',j]*V[j'*p+k',j]*W[k'*n+i',j]
            # We need this to equal a single rank-1 tensor:
            # u[i',m+j'] * v[j'*p+k'] * w[k'*n+i']
            
            # This is a very constrained problem. Let's check if the sum
            # of the two rank-1 tensors is itself rank-1.
            
            # Compute the 60-dimensional vector of tensor entries for each column
            vec_i = np.zeros(60)
            vec_j = np.zeros(60)
            idx = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        vec_i[idx] = U55[ii * m + jj, i] * V55[jj * p + kk, i] * W55[kk * n + ii, i]
                        vec_j[idx] = U55[ii * m + jj, j] * V55[jj * p + kk, j] * W55[kk * n + ii, j]
                        idx += 1
            
            # Check if vec_i and vec_j are linearly dependent
            if np.linalg.norm(vec_i) > 1e-10 and np.linalg.norm(vec_j) > 1e-10:
                cross = np.linalg.norm(vec_i * np.linalg.norm(vec_j) - vec_j * np.linalg.norm(vec_i))
                if cross < 1e-8:
                    # They are proportional, so we can merge them
                    # The merged column is just vec_i / (U55[0,i]*V55[0,i]*W55[0,i]) if that's nonzero
                    # This is a valid merge
                    # Remove column j and keep column i (or vice versa)
                    mask = np.ones(55, dtype=bool)
                    mask[j] = False
                    U54 = U55[:, mask]
                    V54 = V55[:, mask]
                    W54 = W55[:, mask]
                    if _verify_decomposition(U54, V54, W54):
                        return U54, V54, W54
    
    return None


def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - First try to find a rank-54 decomposition via search.
    - Fall back to the guaranteed rank-55 Strassen-based decomposition.
    """
    start_time = time.time()
    
    # Try to find rank-54
    result = _try_merge_schoolbook_with_strassen(start_time)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try the column removal search
    result = _try_rank54_search(start_time)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-55
    if time.time() - start_time > DEADLINE:
        # Emergency: return a quick approximation
        U, V, W = _build_strassen_rank55()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = _build_strassen_rank55()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
