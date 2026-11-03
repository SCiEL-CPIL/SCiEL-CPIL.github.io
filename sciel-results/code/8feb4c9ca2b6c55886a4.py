"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural idea:
  Split A (2x3) into A0 (2x1, j=0) and A2 (2x2, j=1,2).
  - A0 * B0 (2x1 * 1x10): naive, 2*1*10 = 20 multiplications.
  - A2 * B2 (2x2 * 2x10): split B2 into 5 blocks of 2x2, apply Strassen (7 mults each) = 35.
  Total rank = 55.

Refinement:
  Try to reduce rank below 55 by:
  1. Merging Strassen terms across different B-blocks
  2. Local optimization of coefficients
  3. Random linear combinations of existing terms
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(1000):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _try_merge_strassen_terms(U55, V55, W55, n, m, p, target_rank, deadline):
    """
    Try to merge pairs of Strassen terms from different B-blocks.
    
    The idea: if two rank-1 terms have similar U or V patterns, they might
    be combinable into a single term with adjusted W.
    
    We try all pairs of terms and check if their difference can be expressed
    as a single rank-1 term.
    """
    rank55 = 55
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    # For each pair of terms, try to see if we can eliminate one by adjusting others
    # Strategy: pick a subset of terms to keep, then solve for adjusted coefficients
    
    # First, try dropping one term at a time and see if the remaining can be adjusted
    for drop_idx in range(rank55):
        if time.time() > deadline:
            return None
        
        # Create a matrix where each column is one of the 55 terms (except drop_idx)
        keep_idx = [i for i in range(rank55) if i != drop_idx]
        U_keep = U55[:, keep_idx]  # shape (n*m, 54)
        V_keep = V55[:, keep_idx]  # shape (m*p, 54)
        W_keep = W55[:, keep_idx]  # shape (p*n, 54)
        
        # The tensor to reconstruct: sum of all 55 terms minus the dropped one
        # T_target = T_all - T_drop
        # We want: sum_{r in keep} c_r * (U_keep[:,r] ⊗ V_keep[:,r] ⊗ W_keep[:,r]) = T_target
        
        # This is a linear system in the coefficients c_r
        # Flatten the tensor equation: for each (a,b,c) index:
        # sum_r c_r * U_keep[a,r] * V_keep[b,r] * W_keep[c,r] = T_target[a,b,c]
        
        # This is a 3D tensor equation. We can write it as:
        # For each (a,b): sum_r c_r * U_keep[a,r] * V_keep[b,r] * W_keep[c,r] = T_target[a,b,c]
        # This is linear in c_r for each (a,b,c).
        
        # Build the linear system: A @ c = b
        # A[i, r] = U_keep[a,r] * V_keep[b,r] * W_keep[c,r] for the i-th equation
        # But we have (n*m)*(m*p)*(p*n) equations, which is too many.
        
        # Instead, use a least-squares approach with random sampling
        # Sample a subset of (a,b,c) indices to form the linear system
        
        n_eq = min(500, (n*m) * (m*p) * (p*n))
        random.seed(42)
        a_indices = [random.randint(0, n*m-1) for _ in range(n_eq)]
        b_indices = [random.randint(0, m*p-1) for _ in range(n_eq)]
        c_indices = [random.randint(0, p*n-1) for _ in range(n_eq)]
        
        # Build A matrix: n_eq x 54
        A = np.zeros((n_eq, 54), dtype=np.float64)
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            for r in range(54):
                A[eq, r] = U_keep[a, r] * V_keep[b, r] * W_keep[c, r]
        
        # Build b vector: T_target at sampled indices
        # T_target = sum of all 55 terms - dropped term
        # = T_all - T_drop
        # T_all[a,b,c] = 1 if (a,b,c) is a valid (i,j,k) mapping, else 0
        # T_drop[a,b,c] = U55[a,drop_idx] * V55[b,drop_idx] * W55[c,drop_idx]
        
        b_vec = np.zeros(n_eq, dtype=np.float64)
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            # Check if (a,b,c) is a valid multiplication tensor entry
            # a = i*m+j, b = j*p+k, c = k*n+i
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                b_vec[eq] = 1.0
            # Subtract the dropped term
            b_vec[eq] -= U55[a, drop_idx] * V55[b, drop_idx] * W55[c, drop_idx]
        
        # Solve least squares: A @ c = b_vec
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except Exception:
            continue
        
        # Check if the solution is valid: residuals should be small
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Check if coefficients are close to integers or half-integers
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        # Construct the new decomposition
        U_new = np.zeros((n*m, 54), dtype=np.float64)
        V_new = np.zeros((m*p, 54), dtype=np.float64)
        W_new = np.zeros((p*n, 54), dtype=np.float64)
        for r in range(54):
            U_new[:, r] = coeffs[r] * U_keep[:, r]
            V_new[:, r] = coeffs[r] * V_keep[:, r]
            W_new[:, r] = coeffs[r] * W_keep[:, r]
        
        # Verify
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _targeted_search(n, m, p, rank, deadline, values=None):
    """
    Targeted search for a rank-`rank` decomposition using structured
    initialization based on the Strassen-55 solution.

    Strategies:
    1. Try to drop terms and solve for adjusted coefficients
    2. Random linear combinations of existing terms
    3. Merge terms across different B-blocks
    """
    if values is None:
        values = [0, 0.5, 1, -0.5, -1]

    # Start from Strassen-55
    U55, V55, W55 = _build_strassen55(n, m, p)
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)

    # Strategy 1: Try to reduce rank by dropping terms and adjusting
    if rank <= 54:
        result = _try_merge_strassen_terms(U55, V55, W55, n, m, p, rank, deadline)
        if result is not None:
            return result

    # Strategy 2: Random linear combinations
    for trial in range(300):
        if time.time() > deadline:
            break

        # Generate random coefficients for `rank` combinations
        idx = random.sample(range(55), rank)
        U_trial = U55[:, idx].copy()
        V_trial = V55[:, idx].copy()
        W_trial = W55[:, idx].copy()

        # Check if this subset still gives the correct tensor
        if _verify(U_trial.tolist(), V_trial.tolist(), W_trial.tolist(), n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()

    # Strategy 3: Start with rank terms, each being a random linear combination
    for trial in range(300):
        if time.time() > deadline:
            break

        U_trial = np.zeros((n * m, rank), dtype=np.float64)
        V_trial = np.zeros((m * p, rank), dtype=np.float64)
        W_trial = np.zeros((p * n, rank), dtype=np.float64)

        for r in range(rank):
            # Pick 1-3 Strassen terms to combine
            n_terms = random.randint(1, 3)
            idx = random.sample(range(55), n_terms)
            coeffs = [random.choice([1, -1]) for _ in range(n_terms)]
            for c, i in zip(coeffs, idx):
                U_trial[:, r] += c * U55[:, i]
                V_trial[:, r] += c * V55[:, i]
                W_trial[:, r] += c * W55[:, i]

        if _verify(U_trial.tolist(), V_trial.tolist(), W_trial.tolist(), n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()

    # Strategy 4: Try to find a rank-r decomposition by solving a linear system
    # Pick r random "basis" terms and try to express the full tensor as a combination
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Pick r random indices
        idx = random.sample(range(55), min(r, 55))
        U_basis = U55[:, idx]
        V_basis = V55[:, idx]
        W_basis = W55[:, idx]
        
        # Try to find coefficients such that sum c_r * (U_r ⊗ V_r ⊗ W_r) = T
        # This is a linear system in c_r
        # Sample equations
        n_eq = min(500, (n*m) * (m*p) * (p*n))
        a_indices = [random.randint(0, n*m-1) for _ in range(n_eq)]
        b_indices = [random.randint(0, m*p-1) for _ in range(n_eq)]
        c_indices = [random.randint(0, p*n-1) for _ in range(n_eq)]
        
        A = np.zeros((n_eq, len(idx)), dtype=np.float64)
        b_vec = np.zeros(n_eq, dtype=np.float64)
        
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            # Check if (a,b,c) is a valid multiplication tensor entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
            
            for r in range(len(idx)):
                A[eq, r] = U_basis[a, r] * V_basis[b, r] * W_basis[c, r]
            b_vec[eq] = target
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except Exception:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Check if coefficients are valid
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        U_new = U_basis * coeffs
        V_new = V_basis * coeffs
        W_new = W_basis * coeffs
        
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()

    return None


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.

    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2 columns.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices for outputs
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _random_search_rank(n, m, p, rank, values, trials):
    """Random search for a decomposition of the given rank."""
    for _ in range(trials):
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
        if _verify(U, V, W, n, m, p) and _is_valid_entries(U, V, W):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try targeted search for ranks below 55
    values = [0, 0.5, 1, -0.5, -1]

    # Try ranks from 54 down to 50
    for r in range(54, 49, -1):
        if time.time() > deadline - 15:
            break

        # Try the targeted search
        result = _targeted_search(n, m, p, r, deadline - 10)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best, "V": V_best, "W": W_best}
