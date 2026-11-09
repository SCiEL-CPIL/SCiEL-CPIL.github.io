"""LLL lattice reduction for rank-minimal 3x4x4 matrix multiplication tensor."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta[i,k]*delta[j,i]... 
    Actually T[i,j,k] = 1 if output index (k*n+i) gets input A[i,j]*B[j,k].
    T is indexed: A index = i*m+j, B index = j*p+k, C index = k*n+i
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4():
    """Strassen 2x4x4: rank 28."""
    rank = 28
    U = np.zeros((8, rank))   # A: 2x4
    V = np.zeros((16, rank))  # B: 4x4
    W = np.zeros((8, rank))   # C: 4x2 (k*2+i)
    
    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,1,1)], 1: [(1,2,-1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(0,3,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,0,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,1,-1)], 1: [(0,2,1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [], 1: [(1,2,1),(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        
        for j in range(2):
            for k in range(2):
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1
    
    return U, V, W

def _build_schoolbook_1x4x4():
    """Schoolbook 1x4x4: rank 16."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    for j in range(4):
        for k in range(4):
            idx = j * 4 + k
            U[j, idx] = 1
            V[idx, idx] = 1
            W[k, idx] = 1
    return U, V, W

def _combine_3x4x4():
    """Combine 2x4x4 (Strassen, rank 28) + 1x4x4 (schoolbook, rank 16) = rank 44."""
    U2, V2, W2 = _build_strassen_2x4x4()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # 2x4x4 part (cols 0-27)
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
    
    # 1x4x4 part (cols 28-43)
    for j in range(4):
        U[8+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _lll_reduce(basis, deadline):
    """
    Perform LLL-style lattice reduction on a set of integer vectors.
    basis: (dim, n_cols) integer matrix.
    Returns reduced basis columns sorted by norm.
    Uses a simplified LLL: repeated Gaussian elimination + swapping.
    """
    if time.time() > deadline:
        return basis
    
    n_cols = basis.shape[1]
    if n_cols == 0:
        return basis
    
    # Work with float for numerical stability, but track integer structure
    B = basis.astype(float).copy()
    dim = B.shape[0]
    
    # Simple LLL: iterate through columns, reduce each against previous ones
    for i in range(1, n_cols):
        if time.time() > deadline:
            break
        for j in range(i):
            # Reduce B[:,i] by B[:,j]
            norm_j = np.linalg.norm(B[:, j])
            if norm_j < 1e-12:
                continue
            mu = np.dot(B[:, i], B[:, j]) / (norm_j * norm_j)
            # Round to nearest integer for lattice reduction
            mu_rounded = np.round(mu)
            B[:, i] -= mu_rounded * B[:, j]
    
    # Sort by norm
    norms = np.linalg.norm(B, axis=0)
    order = np.argsort(norms)
    B = B[:, order]
    
    # Convert back to integer if close
    B_int = np.round(B)
    if np.allclose(B, B_int, atol=1e-6):
        B = B_int.astype(int)
    else:
        B = np.round(B).astype(int)
    
    return B

def _try_rank_reduction_lattice(U, V, W, n, m, p, deadline):
    """
    Use lattice reduction to find rank reductions.
    
    Key insight: The tensor T = sum_c U_c ⊗ V_c ⊗ W_c.
    If we fix V and W, the U columns must satisfy linear constraints.
    More generally, we look for integer linear relations among the rank-1 terms.
    
    Strategy: 
    1. Build the "tensor coefficient matrix" where each rank-1 term is a vector.
    2. Look for short integer vectors in the kernel space.
    3. If a relation exists, we can merge terms.
    """
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    # Build tensor coefficient vectors
    # Each rank-1 term c gives a vector of size nm*mp*pn
    # term_c[a,b,c_out] = U[a,c] * V[b,c] * W[c_out,c]
    
    terms = np.zeros((nm * mp * pn, current_rank))
    for c in range(current_rank):
        term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        terms[:, c] = term.flatten()
    
    # Check if the terms are linearly independent
    # If not, there's a dependency we can exploit
    try:
        # Use SVD to find the rank
        U_svd, S, Vt = np.linalg.svd(terms, full_matrices=False)
        effective_rank = np.sum(S > 1e-8 * S[0])
        
        if effective_rank < current_rank:
            # There are dependencies
            # The null space vectors give linear relations
            # We can try to use these to reduce rank
            # For now, just check if we can remove one term
            null_dim = current_rank - effective_rank
            if null_dim > 0:
                # Find the smallest singular value direction
                # This corresponds to a near-dependency
                # Try removing the term with smallest contribution
                term_norms = np.linalg.norm(terms, axis=0)
                min_idx = np.argmin(term_norms)
                
                # Try removing this term
                mask = [i for i in range(current_rank) if i != min_idx]
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                
                if _validate(U_test, V_test, W_test, n, m, p):
                    return U_test, V_test, W_test
    except Exception:
        pass
    
    # Alternative: try all single-term removals
    for c_remove in range(current_rank):
        if time.time() > deadline:
            break
        mask = [i for i in range(current_rank) if i != c_remove]
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        if _validate(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    
    return U, V, W

def _try_structured_reformulation(U, V, W, n, m, p, deadline):
    """
    Try to reformulate the decomposition by exploiting block structure.
    
    The 3x4x4 tensor can be seen as 3 blocks of 4x4x4 (one per output row i).
    Each block uses the same B matrix but different A rows and C columns.
    
    We try to find a shared structure that allows rank reduction.
    """
    current_rank = U.shape[1]
    
    # Check if we can find a linear combination of columns that gives a valid decomposition
    # with one fewer term
    
    # Try: for each pair of columns (c1, c2), check if removing c1 and adjusting c2
    # still gives a valid decomposition. This is a 2-term merge.
    # This is expensive, so only try a few pairs.
    
    # Simpler: try removing each column
    for c_remove in range(current_rank):
        if time.time() > deadline:
            break
        mask = [i for i in range(current_rank) if i != c_remove]
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        if _validate(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    
    return U, V, W

def _try_bini_4x4x4():
    """
    Try to construct a rank-17 decomposition for 4x4x4 using Bini's algorithm.
    If successful, combine with the 3rd row's 4x4x4 to get rank 34 for 3x4x4.
    
    Bini's algorithm for 4x4x4 uses 17 multiplications.
    The construction is complex; we'll try a known formulation.
    """
    # Bini's 4x4x4 algorithm (rank 17)
    # This is a well-known result but the explicit construction is intricate.
    # We'll use a simplified approach: start with schoolbook (rank 16) for 4x4x4
    # and see if we can extend it.
    
    # Actually, rank 17 > rank 16, so schoolbook is already better for 4x4x4.
    # The 3x4x4 problem is different from 3 * 4x4x4.
    
    # Let's just return the schoolbook 4x4x4 as a building block
    rank = 16
    U = np.zeros((16, rank))  # A: 4x4
    V = np.zeros((16, rank))  # B: 4x4
    W = np.zeros((16, rank))  # C: 4x4
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = i * 16 + j * 4 + k
                if idx < rank:
                    U[i * 4 + j, idx] = 1
                    V[j * 4 + k, idx] = 1
                    W[k * 4 + i, idx] = 1
    
    return U, V, W

def _try_direct_rank43_search(U, V, W, n, m, p, deadline):
    """
    Direct search for rank-43 decomposition by trying to express one term
    as a linear combination of the others.
    
    For each column c, check if U[:,c] ⊗ V[:,c] ⊗ W[:,c] can be written as
    sum_{c' != c} alpha_{c'} * U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']
    
    This is a system of nm*mp*pn equations in current_rank-1 unknowns.
    """
    current_rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    for c_target in range(current_rank):
        if time.time() > deadline:
            break
        
        # Build the system: find alpha such that
        # U_c ⊗ V_c ⊗ W_c = sum_{c' != c_target} alpha_{c'} * U_{c'} ⊗ V_{c'} ⊗ W_{c'}
        
        # This is overdetermined (nm*mp*pn equations, current_rank-1 unknowns)
        # Use least squares and check if residual is ~0
        
        target_term = np.einsum('i,j,k->ijk', U[:, c_target], V[:, c_target], W[:, c_target])
        target_flat = target_term.flatten()
        
        # Build coefficient matrix
        other_cols = [c for c in range(current_rank) if c != c_target]
        A_mat = np.zeros((nm * mp * pn, len(other_cols)))
        for idx, c in enumerate(other_cols):
            term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            A_mat[:, idx] = term.flatten()
        
        # Solve least squares
        try:
            alpha, residuals, rank_svd, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
            residual = np.linalg.norm(A_mat @ alpha - target_flat)
            
            if residual < 1e-8:
                # We found a dependency! Remove column c_target
                mask = other_cols
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                
                if _validate(U_test, V_test, W_test, n, m, p):
                    return U_test, V_test, W_test
        except Exception:
            continue
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with the known rank-44 Strassen-based decomposition
    try:
        U, V, W = _combine_3x4x4()
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook
            U_s, V_s, W_s = _schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
        
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Strategy 1: Direct rank-43 search (express one term as combo of others)
        if time.time() < deadline:
            U_red, V_red, W_red = _try_direct_rank43_search(
                best_U, best_V, best_W, n, m, p, deadline
            )
            if U_red.shape[1] < best_rank and _validate(U_red, V_red, W_red, n, m, p):
                best_rank = U_red.shape[1]
                best_U, best_V, best_W = U_red, V_red, W_red
                print(f"Direct search reduced rank to {best_rank}")
        
        # Strategy 2: Lattice-based reduction
        if time.time() < deadline and best_rank > 40:
            U_red, V_red, W_red = _try_rank_reduction_lattice(
                best_U, best_V, best_W, n, m, p, deadline
            )
            if U_red.shape[1] < best_rank and _validate(U_red, V_red, W_red, n, m, p):
                best_rank = U_red.shape[1]
                best_U, best_V, best_W = U_red, V_red, W_red
                print(f"Lattice reduction rank to {best_rank}")
        
        # Strategy 3: Structured reformulation
        if time.time() < deadline and best_rank > 40:
            U_red, V_red, W_red = _try_structured_reformulation(
                best_U, best_V, best_W, n, m, p, deadline
            )
            if U_red.shape[1] < best_rank and _validate(U_red, V_red, W_red, n, m, p):
                best_rank = U_red.shape[1]
                best_U, best_V, best_W = U_red, V_red, W_red
                print(f"Structured reformulation rank to {best_rank}")
        
        # Strategy 4: Iterative single-term removal
        if time.time() < deadline:
            for _ in range(5):  # Try up to 5 more reductions
                if time.time() > deadline:
                    break
                U_red, V_red, W_red = _try_rank_reduction_lattice(
                    best_U, best_V, best_W, n, m, p, deadline
                )
                if U_red.shape[1] < best_rank and _validate(U_red, V_red, W_red, n, m, p):
                    best_rank = U_red.shape[1]
                    best_U, best_V, best_W = U_red, V_red, W_red
                    print(f"Iterative reduction rank to {best_rank}")
                else:
                    break
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    except Exception as e:
        print(f"Error: {e}")
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
