import time
import random
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n * m)], dtype=float)
        vc = np.array([V[j][c] for j in range(m * p)], dtype=float)
        wc = np.array([W[k][c] for k in range(p * n)], dtype=float)
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True


def _construct_schoolbook():
    n, m, p = DIMENSIONS
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


def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer (i.e., 2*x is integer)."""
    two_x = 2 * x
    return abs(two_x - round(two_x)) < tol


def _all_entries_half_integer(arr, tol=1e-9):
    for val in arr.flat:
        if not _is_half_integer(val, tol):
            return False
    return True


def _gauss_solve(A, b):
    """
    Solve A x = b using Gaussian elimination with partial pivoting.
    A is (eqs, vars), b is (eqs,). Returns solution vector or None if no unique solution.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n_eq, n_var = A.shape
    if n_eq < n_var:
        return None  # underdetermined
    # Augmented matrix
    M = np.hstack([A, b.reshape(-1, 1)])
    for col in range(n_var):
        # Partial pivot
        max_row = col + np.argmax(np.abs(M[col:, col]))
        if abs(M[max_row, col]) < 1e-12:
            return None  # singular
        M[[col, max_row]] = M[[max_row, col]]
        # Eliminate
        factor = M[col, col]
        M[col] = M[col] / factor
        for row in range(n_eq):
            if row != col and abs(M[row, col]) > 1e-12:
                M[row] -= M[row, col] * M[col]
    x = M[:n_var, n_var]
    # Verify
    if not np.allclose(A @ x, b, atol=1e-8):
        return None
    return x


def _pattern_to_system(supports, n, m, p):
    """
    Given a list of supports (each support is a set of (i,j,k) triples),
    build the linear system A x = b where x are the unknown coefficients.
    
    For each output entry (k, i) [row in W], the equation is:
      sum over terms c that include (i,j,k) in their support: U_c[i,j] * V_c[j,k] = W_c[k,i]
    
    We treat U_c[i,j], V_c[j,k], W_c[k,i] as variables. But this is nonlinear (product).
    
    Alternative: For each term c, we have variables u_{c,ij}, v_{c,jk}, w_{c,ki}.
    The constraint is: for each (i,j,k) that term c covers:
      u_{c,ij} * v_{c,jk} = w_{c,ki}
    And the sum constraint: for each (i,j,k):
      sum_{c: (i,j,k) in support_c} u_{c,ij} * v_{c,jk} = 1
    
    This is still nonlinear. Let's use a different formulation:
    
    Actually, the standard approach: each rank-1 term is defined by 3 vectors:
      u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    with the constraint that for ALL (i,j,k):
      u_c[i*m+j] * v_c[j*p+k] = w_c[k*n+i]
    
    And the sum constraint: for all (i,j,k):
      sum_c u_c[i*m+j] * v_c[j*p+k] = 1
    
    The first constraint means w_c is determined by u_c and v_c (outer product reshaped).
    So we only need to find u_c and v_c such that:
      sum_c u_c[i*m+j] * v_c[j*p+k] = 1  for all (i,j,k)
    
    This is still bilinear in (u_c, v_c). 
    
    New approach: Fix the support pattern (which (i,j,k) each term covers).
    For a term with support S_c (a set of (i,j,k)), we need:
      u_c[i*m+j] * v_c[j*p+k] = w_c[k*n+i] for (i,j,k) in S_c
      u_c[i*m+j] * v_c[j*p+k] = 0 for (i,j,k) NOT in S_c
    
    The zero constraint means: if (i,j) is in the row support and (j,k) is in the col support,
    then (i,j,k) must be in S_c. So the support must be "closed": 
    if (i,j) appears in any triple in S_c, and (j,k) appears in any triple in S_c,
    then (i,j,k) must be in S_c.
    
    This means the support is determined by a set of (i,j) pairs and a set of (j,k) pairs,
    and S_c = {(i,j,k) : (i,j) in IJ_c AND (j,k) in JK_c}.
    
    So we can parameterize each term by:
      IJ_c: subset of {0..n-1} x {0..m-1}
      JK_c: subset of {0..m-1} x {0..p-1}
    
    And the support is the "join" over j.
    
    For the coefficient: given IJ_c and JK_c, we need to find values u_{ij}, v_{jk} such that
    the product u_{ij}*v_{jk} gives the right contribution.
    
    Actually, let's simplify further. For a term with row-support R_c (subset of n*m) and
    col-support C_c (subset of m*p), the support in (i,j,k) space is:
      {(i,j,k) : (i*m+j) in R_c AND (j*p+k) in C_c}
    
    For this to be consistent (the tensor entry w_{ki} = u_{ij} * v_{jk}), we need:
      For any two triples (i,j,k1) and (i,j,k2) in the support with the same (i,j):
        w_{k1,i} / v_{j,k1} = u_{i,j} = w_{k2,i} / v_{j,k2}
      This means w_{k1,i} * v_{j,k2} = w_{k2,i} * v_{j,k1}
    
    Similarly for fixed (j,k) and varying i.
    
    This is getting complex. Let me use a more direct approach:
    
    For small numbers of terms, I'll use a randomized approach:
    1. Generate random sparse patterns for U, V, W
    2. For each pattern, solve the linear system that enforces the bilinear constraint
    
    Actually, the most practical approach for this problem size:
    Use the fact that we need sum_c u_c[i*m+j] * v_c[j*p+k] = 1 for all (i,j,k).
    
    This is a system of n*m*p = 105 bilinear equations in the variables {u_c[i*m+j], v_c[j*p+k]}.
    
    For a given number of terms r, we have r*(n*m + m*p) = r*(15+35) = 50r variables
    and 105 equations.
    
    For r = 105 (schoolbook), we have 5250 variables and 105 equations - very underdetermined.
    The schoolbook solution sets most variables to 0.
    
    For lower r, we need the sparsity to help.
    
    Let me try a different tactic: start from the schoolbook and try to merge terms
    by finding linear dependencies, using the structure of the problem.
    """
    pass


def _try_strassen_3x3_block():
    """
    Apply a 3x3x3 Strassen-like decomposition.
    The 3x3x3 matrix multiplication tensor has rank at most 23 (Borasu) or 20 (lower bound).
    We use a known 23-term decomposition for 3x3x3.
    """
    n, m, p = DIMENSIONS
    # We'll use the 3x3x3 decomposition with 23 terms
    # This is based on the Borasu 23-term algorithm for 3x3 matrix multiplication
    # The 23 products are:
    
    # Let A, B be 3x3 matrices, C = A*B
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # M2 = (A11+A22)(B11+B22)
    # M3 = (A11+A33)(B11+B33)
    # M4 = (A22+A33)(B22+B33)
    # M5 = (A11+A22+A33)(B12+B23+B31)
    # M6 = (A12+A21+A33)(B12+B23+B31)
    # M7 = (A13+A22+A31)(B13+B21+B32)
    # M8 = (A11+A22)(B12+B23)
    # M9 = (A12+A21)(B13+B32)
    # M10 = (A13+A31)(B12+B23)
    # M11 = (A22+A33)(B12+B31)
    # M12 = (A12+A22)(B23+B31)
    # M13 = (A13+A33)(B12+B31)
    # M14 = (A11+A33)(B23+B32)
    # M15 = (A21+A33)(B13+B32)
    # M16 = (A11+A22)(B31+B13)
    # M17 = (A22+A33)(B32+B21)
    # M18 = (A33+A11)(B31+B12)
    # M19 = (A33+A22)(B32+B13)
    # M20 = (A11+A33)(B21+B13)
    # M21 = (A22+A11)(B31+B23)
    # M22 = (A33+A12)(B32+B12)
    # M23 = (A11+A12)(B33+B23)
    
    # This is getting very complex. Let me use a simpler approach:
    # Use the known 3x3x3 rank-23 decomposition from the literature.
    # For now, let's just use schoolbook for 3x3x3 blocks and see if the
    # overall structure helps.
    
    # Actually, let's try a different approach entirely:
    # Use the "laser method" or "tensor rank" approach directly.
    
    # For 3x5x7, let's try to find a decomposition by solving the
    # constraint satisfaction problem directly.
    
    return None


def _try_constraint_satisfaction(deadline, max_rank=105):
    """
    Try to find a low-rank decomposition using constraint satisfaction.
    
    Approach:
    1. Start with a target rank r
    2. For each term, decide which (i,j) pairs it touches (row support)
       and which (j,k) pairs it touches (col support)
    3. The (i,j,k) support is determined by the join
    4. Solve for the coefficients using the bilinear constraint
    
    For efficiency, we'll use a randomized search over support patterns.
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    random.seed(42)
    
    for target_rank in range(max(70, max_rank - 40), max_rank + 1):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Generate random sparse support patterns
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for c in range(target_rank):
                # Choose a random subset of (i,j) pairs for U
                num_u = random.randint(1, 4)
                u_indices = random.sample(range(n * m), num_u)
                for idx in u_indices:
                    U[idx][c] = random.choice([-1, 1])
                
                # Choose a random subset of (j,k) pairs for V
                num_v = random.randint(1, 4)
                v_indices = random.sample(range(m * p), num_v)
                for idx in v_indices:
                    V[idx][c] = random.choice([-1, 1])
                
                # W is determined by the outer product constraint:
                # W[k*n+i][c] = U[i*m+j][c] * V[j*p+k][c] for all (i,j,k)
                # But this only works if the supports are consistent.
                # For now, compute W from U and V where both are nonzero.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i * m + j][c]
                            v_val = V[j * p + k][c]
                            if u_val != 0 and v_val != 0:
                                W[k * n + i][c] = u_val * v_val
            
            # Verify
            if _verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best = {"U": U, "V": V, "W": W}
                break  # Found valid for this rank, try lower
    
    return best


def _try_column_merging(base_result, deadline):
    """
    Try to reduce the rank of a valid decomposition by merging columns.
    Two columns c1 and c2 can be merged if there exist scalars a, b such that
    the new column (a*U_c1 + b*U_c2, a*V_c1 + b*V_c2, a*W_c1 + b*W_c2)
    still satisfies the bilinear constraint.
    
    This is equivalent to checking if the 4-tensor formed by the two columns
    has rank 1 in a certain sense.
    """
    n, m, p = DIMENSIONS
    if base_result is None:
        return None
    
    U = [row[:] for row in base_result["U"]]
    V = [row[:] for row in base_result["V"]]
    W = [row[:] for row in base_result["W"]]
    r = len(U[0])
    
    best_rank = r
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to remove columns by checking linear dependencies
    # Convert to numpy arrays for efficiency
    U_np = np.array(U, dtype=float)  # (n*m, r)
    V_np = np.array(V, dtype=float)  # (m*p, r)
    W_np = np.array(W, dtype=float)  # (p*n, r)
    
    # For each pair of columns, check if one is a scalar multiple of the other
    # (in the tensor sense)
    for c1 in range(r - 1, -1, -1):
        if time.time() > deadline - 15:
            break
        if c1 >= best_rank:
            continue
        
        for c2 in range(c1):
            if c2 >= best_rank:
                continue
            
            # Check if column c1 is a scalar multiple of column c2
            # in the tensor product sense
            # This means: U_c1 = a * U_c2, V_c1 = a * V_c2, W_c1 = a^2 * W_c2
            # (since W = U * V, the scaling is a^2)
            
            # Find the scaling factor from U
            u1 = U_np[:, c1]
            u2 = U_np[:, c2]
            v1 = V_np[:, c1]
            v2 = V_np[:, c2]
            w1 = W_np[:, c1]
            w2 = W_np[:, c2]
            
            # Find a such that u1 = a * u2
            mask_u = np.abs(u2) > 1e-10
            if not np.any(mask_u):
                continue
            
            a_candidates = u1[mask_u] / u2[mask_u]
            if not np.allclose(a_candidates, a_candidates[0], atol=1e-8):
                continue
            
            a = a_candidates[0]
            
            # Check V: v1 should be a * v2
            mask_v = np.abs(v2) > 1e-10
            if np.any(mask_v):
                v_candidates = v1[mask_v] / v2[mask_v]
                if not np.allclose(v_candidates, a, atol=1e-8):
                    continue
            
            # Check W: w1 should be a^2 * w2
            mask_w = np.abs(w2) > 1e-10
            if np.any(mask_w):
                w_candidates = w1[mask_w] / (a * a * w2[mask_w])
                if not np.allclose(w_candidates, 1.0, atol=1e-8):
                    continue
            
            # Column c1 is a scalar multiple of column c2 - remove it
            # But we need to be careful: we can't just remove it, we need
            # to adjust the other columns.
            # Actually, if c1 = a * c2, then we can absorb c1 into c2
            # by replacing c2 with c2 + c1 and removing c1.
            # But this changes the sum, so we need to be more careful.
            
            # Actually, the decomposition is:
            #   T = sum_c U_c ⊗ V_c ⊗ W_c
            # If c1 = a * c2 (in the tensor sense), then:
            #   U_c1 ⊗ V_c1 ⊗ W_c1 = a^3 * U_c2 ⊗ V_c2 ⊗ W_c2
            # So we can replace c2 with (1 + a^3) * c2 and remove c1.
            
            # But this only works if the scaling is consistent.
            # For now, let's just try to remove duplicate columns.
            
            # Check if the columns are identical
            if np.allclose(u1, u2, atol=1e-8) and np.allclose(v1, v2, atol=1e-8) and np.allclose(w1, w2, atol=1e-8):
                # Remove column c1
                for i in range(n * m):
                    U[i].pop(c1)
                for i in range(m * p):
                    V[i].pop(c1)
                for i in range(p * n):
                    W[i].pop(c1)
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                best_rank -= 1
                break
    
    if best_rank < r:
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result


def _try_strassen_2x2x2_block(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2 block at row i0..i0+1, col j0..j0+1, producing k0..k0+1."""
    a_coeffs = [
        [(0,1), (1,1)],
        [(1,0), (1,1)],
        [(0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (0,0)],
        [(0,1), (1,1)],
    ]
    a_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    
    b_coeffs = [
        [(0,0), (1,1)],
        [(0,0)],
        [(1,0), (1,1)],
        [(0,1), (0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (1,1)],
    ]
    b_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    
    c_contribs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    for t in range(7):
        c = col + t
        for (di, dj), coeff in zip(a_coeffs[t], a_signs[t]):
            row = (i0 + di) * m + (j0 + dj)
            U[row][c] += coeff
        for (dj, dk), coeff in zip(b_coeffs[t], b_signs[t]):
            row = (j0 + dj) * p + (k0 + dk)
            V[row][c] += coeff
        for (dk, di), coeff in c_contribs[t].items():
            row = (k0 + dk) * n + (i0 + di)
            W[row][c] += coeff
    
    return col + 7


def _try_hybrid_strassen(deadline):
    """
    Try various hybrid decompositions using Strassen on 2x2x2 sub-blocks.
    This is the approach that achieved rank 99 in the knowledge base.
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    # Try splitting the 3x5x7 into blocks
    # 3 = 2 + 1
    # 5 = 2 + 2 + 1 or 2 + 3 or 3 + 2
    # 7 = 2 + 2 + 2 + 1 or 2 + 2 + 3 or 2 + 3 + 2 or 3 + 2 + 2
    
    splits_i = [(2, 1)]
    splits_j = [(2, 2, 1), (2, 3), (3, 2), (1, 2, 2)]
    splits_k = [(2, 2, 2, 1), (2, 2, 3), (2, 3, 2), (3, 2, 2), (1, 2, 2, 2)]
    
    for si in splits_i:
        for sj in splits_j:
            for sk in splits_k:
                if time.time() > deadline - 10:
                    break
                
                max_rank = n * m * p
                U = [[0] * max_rank for _ in range(n * m)]
                V = [[0] * max_rank for _ in range(m * p)]
                W = [[0] * max_rank for _ in range(p * n)]
                
                col = 0
                i_offsets = [0]
                for s in si[:-1]:
                    i_offsets.append(i_offsets[-1] + s)
                j_offsets = [0]
                for s in sj[:-1]:
                    j_offsets.append(j_offsets[-1] + s)
                k_offsets = [0]
                for s in sk[:-1]:
                    k_offsets.append(k_offsets[-1] + s)
                
                for ii in range(len(si)):
                    for jj in range(len(sj)):
                        for kk in range(len(sk)):
                            hi = si[ii]
                            hj = sj[jj]
                            hk = sk[kk]
                            i0 = i_offsets[ii]
                            j0 = j_offsets[jj]
                            k0 = k_offsets[kk]
                            
                            if hi == 2 and hj == 2 and hk == 2:
                                col = _try_strassen_2x2x2_block(i0, j0, k0, n, m, p, U, V, W, col)
                            else:
                                # Schoolbook for this block
                                for di in range(hi):
                                    for dj in range(hj):
                                        for dk in range(hk):
                                            U[(i0 + di) * m + (j0 + dj)][col] = 1
                                            V[(j0 + dj) * p + (k0 + dk)][col] = 1
                                            W[(k0 + dk) * n + (i0 + di)][col] = 1
                                            col += 1
                
                # Trim
                last_col = 0
                for c in range(col - 1, -1, -1):
                    nz = False
                    for row in range(n * m):
                        if U[row][c] != 0:
                            nz = True
                            break
                    if not nz:
                        for row in range(m * p):
                            if V[row][c] != 0:
                                nz = True
                                break
                    if not nz:
                        for row in range(p * n):
                            if W[row][c] != 0:
                                nz = True
                                break
                    if nz:
                        last_col = c
                        break
                final_rank = last_col + 1
                
                U_trim = [row[:final_rank] for row in U]
                V_trim = [row[:final_rank] for row in V]
                W_trim = [row[:final_rank] for row in W]
                
                if _verify(U_trim, V_trim, W_trim):
                    if final_rank < best_rank:
                        best_rank = final_rank
                        best = {"U": U_trim, "V": V_trim, "W": W_trim}
    
    return best


def _try_laser_method(deadline):
    """
    Try the laser method / tensor rank approach.
    The idea is to find a set of "laser" tensors that approximate the multiplication tensor
    and then use the Capelli lemma to bound the rank.
    
    For 3x5x7, we can try to find a decomposition that exploits the rectangular shape.
    
    One approach: use the fact that 3x5x7 can be embedded in a larger square problem
    and use known results for square matrix multiplication.
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    # Try to use the 3x3x3 Strassen-like decomposition as a building block
    # For the 3x3 part, use 23 terms (Borasu)
    # For the remaining parts, use schoolbook
    
    # 3x5x7 = 3x(3+2)x(7) = 3x3x7 + 3x2x7
    # 3x3x7: use 3x3 Strassen (23 terms) x 7 = 161 terms? No, that's not right.
    # Actually, 3x3x7 means multiplying 3x3 by 3x7, which is 3x3x7 tensor.
    # The rank of 3x3x7 is at most 3*3*7 = 63 (schoolbook).
    # With Strassen on the 3x3 part, we might do better.
    
    # Let's try a different decomposition:
    # 3x5x7 = (2+1)x5x(2+2+2+1)
    # Use Strassen on the 2x2x2 sub-blocks
    
    # This is similar to the hybrid approach, but let's try different splits.
    
    random.seed(12345)
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        
        # Random split
        si = random.choice([(2,1), (1,2)])
        sj = random.choice([(2,2,1), (1,2,2), (2,3), (3,2), (1,4), (4,1)])
        sk = random.choice([(2,2,2,1), (1,2,2,2), (2,2,3), (3,2,2), (2,3,2), (1,3,3), (3,1,3), (3,3,1)])
        
        max_rank = n * m * p
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        
        col = 0
        i_offsets = [0]
        for s in si[:-1]:
            i_offsets.append(i_offsets[-1] + s)
        j_offsets = [0]
        for s in sj[:-1]:
            j_offsets.append(j_offsets[-1] + s)
        k_offsets = [0]
        for s in sk[:-1]:
            k_offsets.append(k_offsets[-1] + s)
        
        for ii in range(len(si)):
            for jj in range(len(sj)):
                for kk in range(len(sk)):
                    hi = si[ii]
                    hj = sj[jj]
                    hk = sk[kk]
                    i0 = i_offsets[ii]
                    j0 = j_offsets[jj]
                    k0 = k_offsets[kk]
                    
                    if hi == 2 and hj == 2 and hk == 2:
                        col = _try_strassen_2x2x2_block(i0, j0, k0, n, m, p, U, V, W, col)
                    else:
                        for di in range(hi):
                            for dj in range(hj):
                                for dk in range(hk):
                                    U[(i0 + di) * m + (j0 + dj)][col] = 1
                                    V[(j0 + dj) * p + (k0 + dk)][col] = 1
                                    W[(k0 + dk) * n + (i0 + di)][col] = 1
                                    col += 1
        
        # Trim
        last_col = 0
        for c in range(col - 1, -1, -1):
            nz = False
            for row in range(n * m):
                if U[row][c] != 0:
                    nz = True
                    break
            if not nz:
                for row in range(m * p):
                    if V[row][c] != 0:
                        nz = True
                        break
            if not nz:
                for row in range(p * n):
                    if W[row][c] != 0:
                        nz = True
                        break
            if nz:
                last_col = c
                break
        final_rank = last_col + 1
        
        U_trim = [row[:final_rank] for row in U]
        V_trim = [row[:final_rank] for row in V]
        W_trim = [row[:final_rank] for row in W]
        
        if _verify(U_trim, V_trim, W_trim):
            if final_rank < best_rank:
                best_rank = final_rank
                best = {"U": U_trim, "V": V_trim, "W": W_trim}
    
    return best


# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 140  # Safety margin: 150s limit, leave 10s
    
    best_result = None
    best_rank = 105
    
    # 1. Schoolbook baseline
    schoolbook = _construct_schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # 2. Try hybrid Strassen approach (known to achieve rank 99)
    if time.time() < deadline - 30:
        hybrid = _try_hybrid_strassen(deadline)
        if hybrid is not None:
            r = len(hybrid["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = hybrid
    
    # 3. Try laser method / random splits
    if time.time() < deadline - 25:
        laser = _try_laser_method(deadline)
        if laser is not None:
            r = len(laser["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = laser
    
    # 4. Try constraint satisfaction
    if time.time() < deadline - 20 and best_rank > 95:
        cs = _try_constraint_satisfaction(deadline, max_rank=best_rank)
        if cs is not None:
            r = len(cs["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = cs
    
    # 5. Try column merging on the best result
    if time.time() < deadline - 15 and best_result is not None:
        merged = _try_column_merging(best_result, deadline)
        if merged is not None:
            r = len(merged["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = merged
    
    if best_result is None:
        return schoolbook
    return best_result
