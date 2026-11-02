"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x10 matrix.

Strategy:
- The multiplication tensor M_{2,2,10} can be decomposed by partitioning the 10 output
  columns into 5 blocks of 2. Each block is a 2x2x2 tensor of rank 7 (Strassen).
- Total rank: 5 * 7 = 35.
- We also attempt to find a lower-rank decomposition (e.g., rank 34 or 33) by exploiting
  global linear dependencies across blocks. The key idea is that the U-matrix (which
  depends only on the 2x2 input A) is shared across all blocks, so there may be linear
  dependencies in the combined system that allow us to reduce the number of independent
  products.

- We use a randomized search: start with the rank-35 Strassen decomposition, then try to
  find a linear combination of the 35 rank-1 terms that reproduces the same tensor with
  fewer terms. This is equivalent to finding a lower-rank decomposition in the span of
  the original terms.

- Additionally, we try a direct algebraic approach: set up the tensor equations for rank r
  and solve for the coefficients. For small rank reductions (35 -> 34 -> 33), we can use
  the fact that the system is underdetermined and find a solution by fixing some variables
  and solving linear equations.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # safety margin before 150s kill


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' of shapes (4, rank), (20, rank), (20, rank).
    """
    n, m, p = DIMENSIONS

    # Deadline guard
    if time.time() > _DEADLINE:
        U, V, W = _construct_rank35(n, m, p)
        return {"U": U, "V": V, "W": W}

    # First, build the known rank-35 solution
    U35, V35, W35 = _construct_rank35(n, m, p)
    if _verify_decomposition(n, m, p, U35, V35, W35):
        best_rank = 35
        best = {"U": U35, "V": V35, "W": W35}
    else:
        # Fallback to schoolbook
        U40, V40, W40 = _construct_schoolbook(n, m, p)
        return {"U": U40, "V": V40, "W": W40}

    # Try to reduce rank from 35
    for target_rank in [34, 33, 32, 31]:
        if time.time() > _DEADLINE - 10:
            break
        result = _try_rank_reduction(n, m, p, U35, V35, W35, target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_decomposition(n, m, p, U_r, V_r, W_r):
                best_rank = target_rank
                best = {"U": U_r, "V": V_r, "W": W_r}
                break  # Found lower rank, stop

    return best


def _construct_rank35(n, m, p):
    """
    Construct the rank-35 Strassen-based decomposition.
    5 blocks of 2 columns, each using Strassen (rank 7).
    """
    rank = 35

    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen's 7 products for 2x2 matmul:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]

    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1],  # P7: b-d
    ]

    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1],   # P7: g+h
    ]

    w_contrib = [
        [1, 0, 0, 1],   # P1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # P2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # P3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # P4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # P5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # P6: C11 += 1
        [1, 0, 0, 0],   # P7: C00 += 1
    ]

    for b in range(5):
        for r in range(7):
            col = b * 7 + r

            # U: index i*m + j
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]

            # V: index j*p + k
            V[2 * b][col] = strassen_B[r][0]
            V[2 * b + 1][col] = strassen_B[r][1]
            V[10 + 2 * b][col] = strassen_B[r][2]
            V[10 + 2 * b + 1][col] = strassen_B[r][3]

            # W: index k*n + i
            W[4 * b][col] = w_contrib[r][0]
            W[4 * b + 2][col] = w_contrib[r][1]
            W[4 * b + 1][col] = w_contrib[r][2]
            W[4 * b + 3][col] = w_contrib[r][3]

    return U, V, W


def _construct_schoolbook(n, m, p):
    """Construct the rank-40 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1

    return U, V, W


def _try_rank_reduction(n, m, p, U35, V35, W35, target_rank):
    """
    Try to find a rank-target_rank decomposition by finding a subset/combination
    of the 35 rank-1 terms that still spans the tensor.
    
    Approach: The tensor T is represented as sum_{r=0}^{34} u_r ⊗ v_r ⊗ w_r.
    We want to find target_rank linear combinations that reproduce T.
    
    Method: Try to eliminate one term at a time by expressing it as a linear
    combination of others. This is equivalent to finding linear dependencies
    in the tensor decomposition.
    
    We use a randomized approach: pick a random (target_rank)-subset of the 35
    terms and check if the remaining terms can be expressed in their span.
    
    More practically: we set up the linear system where we try to express
    term 34 as a linear combination of terms 0..33. If the system is consistent,
    we can eliminate one term.
    """
    if time.time() > _DEADLINE - 5:
        return None

    U35 = np.array(U35, dtype=float)
    V35 = np.array(V35, dtype=float)
    W35 = np.array(W35, dtype=float)

    # The tensor T[i,j,k] = sum_r U35[i*m+j, r] * V35[j*p+k, r] * W35[k*n+i, r]
    # We want to find a subset of columns that still gives the same tensor.
    
    # Strategy: Try to find a linear dependency among the 35 rank-1 tensors.
    # Each rank-1 tensor t_r = u_r ⊗ v_r ⊗ w_r is a vector in R^{n*m*p} = R^{40}.
    # If these 40-dim vectors are linearly dependent, we can reduce the rank.
    
    # Build the 40x35 matrix of tensor entries
    # T_r[i*m+j, j*p+k, k*n+i] ... actually let's flatten
    # Tensor entry (i,j,k) for term r: U35[i*m+j, r] * V35[j*p+k, r] * W35[k*n+i, r]
    
    # Flatten: index = i*(m*p) + j*p + k
    tensor_terms = np.zeros((n * m * p, 35))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(35):
                    tensor_terms[idx, r] = U35[i * m + j, r] * V35[j * p + k, r] * W35[k * n + i, r]

    # Check if the 35 columns are linearly independent
    # If rank(tensor_terms) < 35, there's a dependency
    rank_tr = np.linalg.matrix_rank(tensor_terms)
    
    if rank_tr < 35:
        # There's a linear dependency. Find it.
        # The null space of tensor_terms.T (35 x 40) gives dependencies
        # Actually, we need: sum_r c_r * t_r = 0
        # This means tensor_terms @ c = 0
        # Find null space of tensor_terms (40 x 35)
        U_svd, s, Vt = np.linalg.svd(tensor_terms, full_matrices=False)
        # Null space vectors are rows of Vt where s is small
        # For 40x35, the null space has dimension 35 - rank
        null_dim = 35 - rank_tr
        if null_dim > 0:
            # The last null_dim rows of Vt span the null space
            # We can eliminate terms corresponding to non-zero components
            # For simplicity, try to eliminate the last term
            # Find c in null space with c[34] != 0
            for row_idx in range(35 - null_dim, 35):
                c = Vt[row_idx, :]
                if abs(c[34]) > 1e-10:
                    # We can express term 34 as combination of others
                    # c[34] * t_34 = -sum_{r<34} c[r] * t_r
                    # So t_34 = -sum_{r<34} (c[r]/c[34]) * t_r
                    # This means we can drop term 34 and adjust the others
                    # But this doesn't directly reduce the rank of the decomposition
                    # because we still need all 34 other terms.
                    # The rank reduction happens if we can find a dependency
                    # that allows us to drop a term entirely.
                    pass

    # Alternative approach: directly try to find a rank-r decomposition
    # by solving the tensor equations.
    # For rank r, we need:
    # T[i,j,k] = sum_{s=0}^{r-1} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # for all (i,j,k) in {0,1}x{0,1}x{0,1,...,9}
    
    # This is 40 equations with (4r + 20r + 20r) = 44r unknowns.
    # For r=34: 40 equations, 1496 unknowns. Very underdetermined.
    # We can fix most variables and solve for the rest.
    
    # Practical approach: use the rank-35 solution as a starting point and
    # try to zero out one column by absorbing it into others.
    
    # Try each column to eliminate
    for elim_col in range(35):
        if time.time() > _DEADLINE - 5:
            return None
        
        # Try to express column elim_col as a linear combination of other columns
        # in the tensor space
        # We need: sum_{s != elim_col} alpha_s * (U_s ⊗ V_s ⊗ W_s) = U_elim ⊗ V_elim ⊗ W_elim
        
        # This is a system of 40 equations in (35-1) = 34 unknowns (alpha_s).
        # Generally overdetermined, so it won't work for arbitrary columns.
        
        # But we can try: for each (i,j,k), the equation is:
        # sum_{s != elim_col} alpha_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = U[i*m+j,elim_col] * V[j*p+k,elim_col] * W[k*n+i,elim_col]
        
        # This is linear in alpha_s! Let's set it up.
        A_mat = np.zeros((40, 34))
        b_vec = np.zeros(40)
        
        other_cols = [c for c in range(35) if c != elim_col]
        for eq_idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
            for s_idx, s in enumerate(other_cols):
                A_mat[eq_idx, s_idx] = U35[i * m + j, s] * V35[j * p + k, s] * W35[k * n + i, s]
            b_vec[eq_idx] = U35[i * m + j, elim_col] * V35[j * p + k, elim_col] * W35[k * n + i, elim_col]
        
        # Solve least squares
        try:
            alpha, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            # Check if the solution is exact
            residual = np.linalg.norm(A_mat @ alpha - b_vec)
            if residual < 1e-10:
                # We found an exact representation!
                # Build the new rank-34 decomposition
                U_new = np.zeros((n * m, 34))
                V_new = np.zeros((m * p, 34))
                W_new = np.zeros((p * n, 34))
                for s_idx, s in enumerate(other_cols):
                    U_new[:, s_idx] = U35[:, s] * alpha[s_idx]
                    V_new[:, s_idx] = V35[:, s]
                    W_new[:, s_idx] = W35[:, s]
                
                # Convert to lists
                U_list = U_new.tolist()
                V_list = V_new.tolist()
                W_list = W_new.tolist()
                return U_list, V_list, W_list
        except:
            pass

    return None


def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    """
    if not U or not V or not W:
        return False

    rank = len(U[0]) if U else 0
    if rank == 0:
        return False

    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False

    # Direct tensor equation check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-10:
                    return False

    # Random verification
    rng = np.random.RandomState(42)
    for trial in range(5):
        A = rng.randint(-5, 6, size=(n, m)).astype(float)
        B = rng.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B

        U_reshaped = U.reshape(n, m, rank)
        A_u = np.einsum('ij,ijr->ir', A, U_reshaped)

        V_reshaped = V.reshape(m, p, rank)
        B_v = np.einsum('kl,klr->lr', B, V_reshaped)

        W_reshaped = W.reshape(p, n, rank)
        C_computed = np.einsum('ir,lr,lir->il', A_u, B_v, W_reshaped)

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False

    return True
