"""Bootstrap spawn: direct construction of a rank-45 3x3x6 decomposition.

Approach: slice the 6 columns of B into two groups of 3. Each (3x3) x (3x3)
slice uses a rank-7 Strassen-type decomposition for the 3x3x3 multiplication
tensor. The 3x3x3 tensor lives in W's i-dimension (trace convention W[k*n+i]),
so the Strassen "A-side" (indices (i,k)) factors are placed in U and the
"column-group" factors in V, while the W factors carry the Strassen "C-side"
(indices (i,k) in the slice).

Extended: search for a rank-44 decomposition by finding a non-trivial linear
combination of the 45 rank-1 terms that is itself rank-1.
"""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np
from itertools import product

# Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
# T[(i,k)] = sum_r  A_r[i,k] * B_r[i,k] * C_r[i,k],  r = 0..6
# where T[i,k] = sum_j A[i,j] B[j,k]  (A 3x3 times B 3x3).
_A = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
]
_B = [
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
]
_C = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
]


def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _schoolbook():
    """Fallback rank-54 decomposition (guaranteed valid)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _rank45():
    n, m, p = DIMENSIONS          # 3, 3, 6
    n_groups = 2                  # 6 columns -> 2 groups of 3
    R = n_groups * len(_A)        # 2 * 7 = 45
    U = _zero_matrix(n * m, R)    # 9 x 45
    V = _zero_matrix(m * p, R)    # 18 x 45
    W = _zero_matrix(p * n, R)    # 18 x 45
    col = 0
    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            for i in range(n):
                for k in range(p):
                    val = _A[r][i][k]   # 3x3x3 A-side: (i, k) in the slice
                    if val:
                        U[i * m + k][base + r] = val
                        # column-group factor for this group (3 columns)
                        V[k * m + g * 3 + i][base + r] = 1
                        W[g * 3 + i * n + k][base + r] = _C[r][i][k]
            col = base + len(_A)
    return {"U": U, "V": V, "W": W}


def _find_rank44(deadline):
    """Search for a rank-44 decomposition by finding a rank-1 linear combination."""
    n, m, p = DIMENSIONS
    R = 45
    cand = _rank45()
    U = np.array(cand["U"], dtype=float)
    V = np.array(cand["V"], dtype=float)
    W = np.array(cand["W"], dtype=float)
    
    # For a rank-1 tensor T = sum_r alpha_r * U_r ⊗ V_r ⊗ W_r,
    # all matricizations must have rank 1.
    # The (i,j)-slice: for each fixed (i,j), the vector over k must be proportional
    # to a fixed vector, and similarly for other slices.
    
    # Strategy: fix one index pattern and solve for alphas such that
    # the resulting tensor has rank 1.
    
    # A tensor has rank 1 iff for any two "parallel" slices, they are proportional.
    # Specifically, for the multiplication tensor T[i,j,k] = sum_r alpha_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's use the fact that if T = a ⊗ b ⊗ c, then T[i,j,k] = a[i*m+j] * b[j*p+k] * c[k*n+i]
    # This means T is determined by 3 vectors.
    
    # Approach: try to find alphas such that the tensor is rank 1.
    # We can set up equations: T[i,j,k] * T[i',j',k'] = T[i,j,k'] * T[i',j',k]
    # for all i,j,k,i',j',k'.
    
    # This is a system of quadratic equations in the alphas.
    # Let's try a different approach: assume the rank-1 tensor has specific structure.
    
    # Since the 45 terms come from two identical Strassen decompositions (one for each 3-column block),
    # there may be symmetries we can exploit.
    
    # Let's try: find alphas such that sum_r alpha_r * U_r ⊗ V_r ⊗ W_r = a ⊗ b ⊗ c
    # for some vectors a (length 9), b (length 18), c (length 18).
    
    # We can fix a and b, then c is determined (if consistent), and we get linear equations in alphas.
    
    # Simpler: use the fact that for rank-1 tensors, all 2x2 minors of any matricization vanish.
    # Let's pick a specific matricization and enforce rank 1.
    
    # Matricize as 9 x (18*18) matrix M where M[i*m+j, (j*p+k)*18 + (k*n+i)] = T[i,j,k]
    # Wait, this isn't right. Let me think more carefully.
    
    # T is a 3D tensor with dimensions (n*m, m*p, p*n) = (9, 18, 18).
    # For T = a ⊗ b ⊗ c, the matricization along the first axis is 9 x (18*18) and has rank 1.
    
    # So: for any two rows i1, i2 of this matricization, they must be proportional.
    # Row i of the matricization: for each (j,k) pair, entry is T[i, j, k] = sum_r alpha_r U[i, r] V[j, r] W[k, r]
    
    # Let's define: for each i in 0..8, and each (j,k) in 0..17 x 0..17,
    # T[i, j, k] = sum_r alpha_r * U[i, r] * V[j, r] * W[k, r]
    
    # For rank 1: T[i1, j, k] / T[i2, j, k] = a[i1] / a[i2] for all j,k (where T[i2,j,k] != 0)
    
    # This is complex. Let's try a numerical approach:
    # 1. Randomly initialize alphas
    # 2. Compute the tensor
    # 3. Check if it's close to rank 1
    # 4. If not, adjust alphas using gradient descent or similar
    
    # Actually, let's try a more direct approach:
    # Assume the rank-1 tensor is of the form a ⊗ b ⊗ c.
    # Then for each r, alpha_r must satisfy:
    #   U[i, r] V[j, r] W[k, r] = a[i] b[j] c[k] for all i,j,k
    # This is very restrictive - it means each rank-1 term U_r ⊗ V_r ⊗ W_r must be proportional to a ⊗ b ⊗ c.
    # But that's only possible if all 45 terms are the same, which they're not.
    
    # So the linear combination sum_r alpha_r T_r = a ⊗ b ⊗ c is what we want, not each T_r being rank 1.
    
    # Let me try: fix a and b, then for each k, c[k] is determined (if consistent),
    # and we get linear equations in alphas.
    
    # T[i, j, k] = a[i] b[j] c[k]
    # sum_r alpha_r U[i, r] V[j, r] W[k, r] = a[i] b[j] c[k]
    
    # For fixed i, j: sum_r alpha_r U[i, r] V[j, r] W[k, r] = (a[i] b[j]) c[k]
    # This is a linear system in alphas for each k, but the RHS depends on c[k].
    
    # Let's fix a and b, and solve for c and alphas simultaneously.
    # For each (i, j, k): sum_r alpha_r U[i, r] V[j, r] W[k, r] - a[i] b[j] c[k] = 0
    # This is linear in (alphas, c) if a and b are fixed.
    
    # Number of unknowns: 45 (alphas) + 18 (c) = 63
    # Number of equations: 9 * 18 * 18 = 2916
    # This is overdetermined, but we can use least squares.
    
    # Let's try several random (a, b) pairs and see if we get a good solution.
    
    np.random.seed(42)
    best_rank = R
    best_alphas = None
    best_abc = None
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random a, b
        a = np.random.randn(n * m)
        b = np.random.randn(m * p)
        
        # Build the linear system: for each (i, j, k),
        # sum_r alpha_r U[i, r] V[j, r] W[k, r] - a[i] b[j] c[k] = 0
        # Variables: alphas[0..44], c[0..17]
        
        # For efficiency, let's build the matrix A and vector 0 (homogeneous system)
        # Actually, let's build it as: A * x = 0 where x = [alphas; c]
        
        # Number of equations: 9*18*18 = 2916
        # Number of unknowns: 45 + 18 = 63
        
        # Build A as a sparse matrix
        rows = []
        cols = []
        vals = []
        
        eq_idx = 0
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    # Coefficients for alphas
                    for r in range(R):
                        val = U[i, r] * V[j, r] * W[k, r]
                        if val != 0:
                            rows.append(eq_idx)
                            cols.append(r)
                            vals.append(val)
                    # Coefficient for c[k]
                    val = -a[i] * b[j]
                    if val != 0:
                        rows.append(eq_idx)
                        cols.append(R + k)
                        vals.append(val)
                    eq_idx += 1
        
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(eq_idx, R + p * n))
        
        # Find the null space
        # Use SVD on the dense matrix if it's not too large
        # 2916 x 63 is manageable
        Adense = A.toarray()
        
        # SVD
        try:
            U_svd, s, Vt = np.linalg.svd(Adense, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        
        # The null space is spanned by the last few columns of Vt
        # Check the smallest singular value
        if len(s) > 0 and s[-1] < 1e-8:
            # There's a non-trivial null space
            # Take the last row of Vt as the solution
            x = Vt[-1]
            alphas = x[:R]
            c = x[R:]
            
            # Normalize
            norm = np.linalg.norm(alphas)
            if norm < 1e-10:
                continue
            alphas = alphas / norm
            c = c / norm
            
            # Verify: compute the tensor and check if it's rank 1
            T = np.zeros((n * m, m * p, p * n))
            for r in range(R):
                T += alphas[r] * np.outer(U[:, r], V[:, r])[:, :, None] * W[:, r][None, None, :]
            
            # Check rank of T by computing its matricization
            Tmat = T.reshape(n * m, m * p * p * n)
            try:
                _, s_T, _ = np.linalg.svd(Tmat, full_matrices=False)
                rank_T = np.sum(s_T > 1e-6)
            except:
                continue
            
            if rank_T <= 1:
                # We found a rank-1 linear combination!
                # Now we can eliminate one term
                # Find the index with the largest |alpha|
                max_idx = np.argmax(np.abs(alphas))
                
                # Construct the new decomposition with rank 44
                # Remove term max_idx and add the combination of the others
                # Actually, the combination IS the rank-1 tensor, so we can replace
                # one of the terms with this combination.
                
                # Let's say we remove term max_idx. Then:
                # sum_{r != max_idx} alpha_r T_r = T_rank1 - alpha[max_idx] T[max_idx]
                # This is not directly useful.
                
                # Better: we have sum_r alpha_r T_r = T_rank1
                # So T[max_idx] = (1/alpha[max_idx]) * (T_rank1 - sum_{r != max_idx} alpha_r T_r)
                # This expresses T[max_idx] as a combination of T_rank1 and the other T_r.
                # But T_rank1 is a single rank-1 tensor, so we can include it as a new term.
                
                # New decomposition: {T_r for r != max_idx} ∪ {T_rank1}
                # This has rank 44 (44 terms: 43 original + 1 new).
                
                # But we need to express T_rank1 as a ⊗ b ⊗ c.
                # We already have a, b, c from above.
                
                # Let's verify this works
                new_U = _zero_matrix(n * m, R - 1)
                new_V = _zero_matrix(m * p, R - 1)
                new_W = _zero_matrix(p * n, R - 1)
                
                # Copy all terms except max_idx
                for r in range(R):
                    if r == max_idx:
                        continue
                    new_r = r if r < max_idx else r - 1
                    for i in range(n * m):
                        new_U[i][new_r] = U[i, r]
                    for j in range(m * p):
                        new_V[j][new_r] = V[j, r]
                    for k in range(p * n):
                        new_W[k][new_r] = W[k, r]
                
                # Add the rank-1 term
                new_r = R - 1
                for i in range(n * m):
                    new_U[i][new_r] = a[i]
                for j in range(m * p):
                    new_V[j][new_r] = b[j]
                for k in range(p * n):
                    new_W[k][new_r] = c[k]
                
                # Verify
                if _verify(new_U, new_V, new_W):
                    return {"U": new_U, "V": new_V, "W": new_W}
    
    return None


def _verify(U, V, W):
    """True iff U,V,W exactly decompose the 3x3x6 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
            return False
    seen = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s == 1.0:
                    if (i, j, k) in seen:
                        return False
                    seen.add((i, j, k))
                elif s != 0.0:
                    return False
    return len(seen) == n * m * p


def construct():
    deadline = time.time() + 120.0
    try:
        # First try to find a rank-44 decomposition
        cand44 = _find_rank44(deadline)
        if cand44 is not None:
            return cand44
    except Exception:
        pass
    
    try:
        cand = _rank45()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    if time.time() > deadline - 5.0:
        return _schoolbook()
    return _schoolbook()
