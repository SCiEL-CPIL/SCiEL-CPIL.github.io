import time
import sys
import numpy as np
from collections import defaultdict
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all 0s
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s) > tol:
                        return False
        return True

    # 1. Construct Rank 42 (Strassen blocks) using exact fractions
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=object)
    V42 = np.zeros((mp, rank42), dtype=object)
    W42 = np.zeros((pn, rank42), dtype=object)
    for i in range(nm):
        for j in range(rank42):
            U42[i, j] = Fraction(0)
    for i in range(mp):
        for j in range(rank42):
            V42[i, j] = Fraction(0)
    for i in range(pn):
        for j in range(rank42):
            W42[i, j] = Fraction(0)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        U42[a00, offset] += 1; U42[a11, offset] += 1
        V42[b00, offset] += 1; V42[b11, offset] += 1
        W42[w00, offset] += 1; W42[w11, offset] += 1

        U42[a10, offset+1] += 1; U42[a11, offset+1] += 1
        V42[b00, offset+1] += 1
        W42[w10, offset+1] += 1; W42[w11, offset+1] -= 1

        U42[a00, offset+2] += 1
        V42[b01, offset+2] += 1; V42[b11, offset+2] -= 1
        W42[w01, offset+2] += 1; W42[w11, offset+2] += 1

        U42[a11, offset+3] += 1
        V42[b10, offset+3] += 1; V42[b00, offset+3] -= 1
        W42[w00, offset+3] += 1; W42[w10, offset+3] += 1

        U42[a00, offset+4] += 1; U42[a01, offset+4] += 1
        V42[b11, offset+4] += 1
        W42[w00, offset+4] -= 1; W42[w01, offset+4] += 1

        U42[a10, offset+5] += 1; U42[a00, offset+5] -= 1
        V42[b00, offset+5] += 1; V42[b01, offset+5] += 1
        W42[w11, offset+5] += 1

        U42[a01, offset+6] += 1; U42[a11, offset+6] -= 1
        V42[b10, offset+6] += 1; V42[b11, offset+6] += 1
        W42[w00, offset+6] += 1

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    # Convert to float for verification
    U42_float = np.array([[float(x) for x in row] for row in U42], dtype=float)
    V42_float = np.array([[float(x) for x in row] for row in V42], dtype=float)
    W42_float = np.array([[float(x) for x in row] for row in W42], dtype=float)

    best_U, best_V, best_W = U42_float.copy(), V42_float.copy(), W42_float.copy()
    best_rank = 42

    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Use linear algebra to find the minimal rank
    # Each term r defines a rank-1 tensor: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Flatten each term into a vector of length nm*mp*pn
    # Then find the rank of the matrix whose columns are these flattened vectors
    
    def flatten_term(U_col, V_col, W_col):
        """Flatten a rank-1 tensor into a vector."""
        return np.outer(np.outer(U_col, V_col), W_col).flatten()
    
    # Build the matrix of flattened terms
    terms = []
    for r in range(best_rank):
        terms.append(flatten_term(best_U[:, r], best_V[:, r], best_W[:, r]))
    
    terms_matrix = np.column_stack(terms)  # shape: (nm*mp*pn, best_rank)
    
    # Find the rank using SVD
    try:
        _, s_vals, Vt = np.linalg.svd(terms_matrix, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        significant = np.sum(s_vals > tol)
    except:
        significant = best_rank
    
    if significant < best_rank:
        # We can potentially reduce rank
        # Use the right singular vectors to find a basis
        # But we need to reconstruct the actual factor matrices
        
        # Alternative: use QR decomposition to find independent columns
        Q, R = np.linalg.qr(terms_matrix)
        diag_R = np.diag(R)
        independent_mask = np.abs(diag_R) > tol
        n_independent = np.sum(independent_mask)
        
        if n_independent < best_rank:
            # Keep only independent columns
            # But this may not preserve the exact decomposition
            # We need a more careful approach
            pass
    
    # More practical: iterative elimination using linear dependencies
    # Try to eliminate one column at a time by expressing it as a combination of others
    
    def try_eliminate_column(U_mat, V_mat, W_mat, col_to_eliminate, tol=1e-10):
        """Try to eliminate col_to_eliminate by expressing it as combination of others."""
        r = U_mat.shape[1]
        if r <= 1:
            return None
        
        # Build the system: find coefficients c_j such that
        # U[:,col] ⊗ V[:,col] ⊗ W[:,col] = sum_j c_j * U[:,j] ⊗ V[:,j] ⊗ W[:,j]
        # for j != col_to_eliminate
        
        # This is a linear system in the coefficients
        # Flatten all terms
        target = flatten_term(U_mat[:, col_to_eliminate], V_mat[:, col_to_eliminate], W_mat[:, col_to_eliminate])
        
        # Build matrix of other terms
        other_cols = [j for j in range(r) if j != col_to_eliminate]
        if len(other_cols) == 0:
            return None
        
        A = np.column_stack([flatten_term(U_mat[:, j], V_mat[:, j], W_mat[:, j]) for j in other_cols])
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=tol)
        except:
            return None
        
        # Check if the approximation is good
        reconstructed = A @ coeffs
        if np.linalg.norm(reconstructed - target) > tol * max(1.0, np.linalg.norm(target)):
            return None
        
        # If we can eliminate, return the modified matrices
        new_U = np.delete(U_mat, col_to_eliminate, axis=1)
        new_V = np.delete(V_mat, col_to_eliminate, axis=1)
        
        # Adjust W: for each remaining column j, add c_j * W[:,col_to_eliminate]
        # But we need to be careful: the new W[:,j] should be W[:,j] + c_j * W[:,col_to_eliminate]
        # Wait, that's not right. Let me think again.
        
        # We have: term_col = sum_j c_j * term_j
        # term_col = U_col ⊗ V_col ⊗ W_col
        # term_j = U_j ⊗ V_j ⊗ W_j
        # This doesn't directly give us a way to modify the factors.
        
        # The issue is that the tensor decomposition is not unique in a simple way.
        # We can't just add coefficients to W because the U and V parts are different.
        
        # This approach won't work directly. Let's try a different strategy.
        return None
    
    # Better approach: use the fact that if two terms have proportional U and V vectors,
    # we can merge them. Let's be more systematic.
    
    def merge_all_prop(U_mat, V_mat, W_mat, tol=1e-10):
        """Merge all columns with proportional U and V vectors."""
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        r = current_U.shape[1]
        
        keep = [True] * r
        merged = False
        
        for i in range(r):
            if not keep[i]:
                continue
            for j in range(i + 1, r):
                if not keep[j]:
                    continue
                
                # Check if U_i and U_j are proportional
                nz_i = np.nonzero(np.abs(current_U[:, i]) > tol)[0]
                nz_j = np.nonzero(np.abs(current_U[:, j]) > tol)[0]
                
                if len(nz_i) == 0 and len(nz_j) == 0:
                    s = 1.0
                elif len(nz_i) == 0 or len(nz_j) == 0:
                    continue
                elif not np.array_equal(nz_i, nz_j):
                    continue
                else:
                    ratios = current_U[nz_i, i] / current_U[nz_i, j]
                    if np.std(ratios) > tol:
                        continue
                    s = ratios[0]
                    if abs(s) < tol:
                        continue
                
                # Check if V_i and V_j are proportional
                nz_v_i = np.nonzero(np.abs(current_V[:, i]) > tol)[0]
                nz_v_j = np.nonzero(np.abs(current_V[:, j]) > tol)[0]
                
                if len(nz_v_i) == 0 and len(nz_v_j) == 0:
                    t = 1.0
                elif len(nz_v_i) == 0 or len(nz_v_j) == 0:
                    continue
                elif not np.array_equal(nz_v_i, nz_v_j):
                    continue
                else:
                    ratios_v = current_V[nz_v_i, i] / current_V[nz_v_i, j]
                    if np.std(ratios_v) > tol:
                        continue
                    t = ratios_v[0]
                    if abs(t) < tol:
                        continue
                
                # Merge j into i: term_j = (1/s)(1/t) term_i * (W_j / W_i) -- not quite
                # Actually: U_j = (1/s) U_i, V_j = (1/t) V_i
                # term_j = U_j ⊗ V_j ⊗ W_j = (1/s)(1/t) U_i ⊗ V_i ⊗ W_j
                # So we can absorb term_j into term_i by adding (1/s)(1/t) W_j to W_i
                scale = 1.0 / (s * t)
                current_W[:, i] += scale * current_W[:, j]
                keep[j] = False
                merged = True
        
        if not merged:
            return current_U, current_V, current_W, r
        
        kept_indices = [i for i in range(r) if keep[i]]
        return current_U[:, kept_indices], current_V[:, kept_indices], current_W[:, kept_indices], len(kept_indices)
    
    # Apply merging iteratively
    changed = True
    while changed:
        if time.time() > deadline:
            break
        new_U, new_V, new_W, new_rank = merge_all_prop(best_U, best_V, best_W)
        if new_rank < best_rank:
            if verify(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                changed = True
            else:
                changed = False
        else:
            changed = False
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
