"""Matrix multiplication 2x4 x 4x4 tensor decomposition - attempting rank 27 via null space."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # First, build the rank-28 Strassen decomposition as a baseline
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    
                    # U side: A[i, 2*a_blk + j]
                    # U flat index: i * 4 + (2*a_blk + j)
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[2*a_blk + j, 2*b + k]
                    # V flat index: (2*a_blk + j) * 4 + (2*b + k)
                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, 2*b + k]
                    # W flat index: (2*b + k) * 2 + i
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # Use numpy for speed
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Check non-zero entries: T[i, j, k] = 1 if i==i', j==j', k==k' in appropriate mapping
        # Actually, the tensor T has T[i, j, k] = 1 for the "diagonal" of the multiplication
        # Indices:
        # U: i (0..n-1), j (0..m-1) -> flat i*m+j
        # V: j (0..m-1), k (0..p-1) -> flat j*p+k
        # W: k (0..p-1), i (0..n-1) -> flat k*n+i
        
        # Reconstruct tensor T_recon of shape (n, m, p)
        # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficiently compute using einsum or loops
        # Since dimensions are small, loops are fine, but numpy is faster.
        
        # Check 1: Diagonal entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term in the tensor for multiplication A*B=C is
                    # C[i, k] += A[i, j] * B[j, k]
                    # So the tensor component T[i, j, k] (with indices i, j, k)
                    # corresponds to U[i, j], V[j, k], W[k, i]
                    # Wait, the standard tensor is T = sum e_i \otimes e_j \otimes e_k
                    # where the map is (A, B) -> C
                    # C[i, k] = sum_j A[i, j] B[j, k]
                    # So the basis vectors are e_i (for A), e_j (for B), e_k (for C)?
                    # No, C is 2x4. Indices i (0,1), k (0..3).
                    # A is 2x4. Indices i (0,1), j (0..3).
                    # B is 4x4. Indices j (0..3), k (0..3).
                    # So the tensor is sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i} ?
                    # Let's check the indices in validate:
                    # U[i*m+j], V[j*p+k], W[k*n+i]
                    # This matches C[i, k] getting contribution from A[i, j] and B[j, k].
                    # So for each (i, j, k), the coefficient should be 1.
                    # All other combinations should be 0.
                    
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check 2: Zero entries
        # Any (i, j, k) tuple is a valid term in the sum for C[i, k].
        # The tensor T is defined such that T(A, B) = C.
        # The tensor itself is an element of U \otimes V \otimes W.
        # The "support" of the tensor is the set of (i, j, k) where the coefficient is non-zero.
        # For matrix multiplication, the coefficient is 1 for ALL i, j, k?
        # No.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor T maps (A, B) to C.
        # T = sum_{i, j, k} E_{i, j} \otimes E_{j, k} \otimes E_{k, i} is incorrect notation.
        # Correctly: T = sum_{i, j, k} u_{i, j} \otimes v_{j, k} \otimes w_{k, i}
        # where u_{i, j} is the basis vector for A[i, j], etc.
        # So for EVERY i, j, k, there is a term 1 * u_{i, j} \otimes v_{j, k} \otimes w_{k, i}.
        # So the "diagonal" check above covers ALL entries of the tensor.
        # Are there any entries that should be 0?
        # The tensor T is a specific vector in the space.
        # If we decompose it as sum_r U_r \otimes V_r \otimes W_r,
        # then the coefficient of u_{i, j} \otimes v_{j, k} \otimes w_{k, i} in the sum is
        # sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # This must be 1 for all i, j, k.
        # Are there other basis vectors?
        # The space is spanned by u_{i, j} \otimes v_{j', k} \otimes w_{k', i'}.
        # The tensor T only has non-zero components for the "matching" indices?
        # No, T is a linear map.
        # T(A, B) = C.
        # The component of T in the direction u_{i, j} \otimes v_{j', k} \otimes w_{k', i'} is
        # 1 if (i, j, k, i') matches the multiplication pattern?
        # Actually, the tensor T is defined by its action.
        # T = sum_{i, j, k} E_{i, j}^A \otimes E_{j, k}^B \otimes E_{k, i}^C.
        # So the coefficient of E_{i, j}^A \otimes E_{j', k}^B \otimes E_{k', i'}^C is
        # 1 if j == j' and i == i' and k == k'?
        # No.
        # The term E_{i, j}^A \otimes E_{j', k}^B \otimes E_{k', i'}^C
        # contributes to C[i', k'] from A[i, j] and B[j', k].
        # This is non-zero in the tensor definition ONLY if the indices match the multiplication rule?
        # No, the tensor T is a fixed object.
        # T = sum_{i, j, k} e_{i, j} \otimes e_{j, k} \otimes e_{k, i}.
        # So the coefficient of e_{i, j} \otimes e_{j', k} \otimes e_{k', i'} is
        # 1 if (i, j, k) = (i', j', k')? No.
        # The sum is over i, j, k.
        # So the term e_{i, j} \otimes e_{j, k} \otimes e_{k, i} appears with coefficient 1.
        # Any other combination e_{a, b} \otimes e_{c, d} \otimes e_{e, f}
        # appears with coefficient 0 unless it matches one of the terms in the sum.
        # The terms in the sum are exactly those where the indices form a cycle i->j->k->i?
        # No, the indices are i (row of A), j (col of A / row of B), k (col of B / col of C).
        # And C has row i, col k.
        # So the term is e_{i, j} \otimes e_{j, k} \otimes e_{k, i}.
        # So for a general basis vector e_{a, b} \otimes e_{c, d} \otimes e_{e, f},
        # the coefficient is 1 if b == c and a == e and b == d? No.
        # It is 1 if (a, b, c, d, e, f) corresponds to (i, j, j, k, k, i).
        # So b must equal c.
        # And a must equal e.
        # And d must equal f? No, d is col of B, f is col of C.
        # In the term e_{i, j} \otimes e_{j, k} \otimes e_{k, i}:
        # A index: (i, j)
        # B index: (j, k)
        # C index: (k, i) -- Wait, C is 2x4. Indices (row, col) = (i, k).
        # So C basis vector is e_{i, k}.
        # So the term is e_{i, j}^A \otimes e_{j, k}^B \otimes e_{i, k}^C.
        # So the condition for non-zero coefficient is:
        # A index (a, b), B index (c, d), C index (e, f).
        # Coefficient is 1 if b == c and a == e and d == f.
        # Otherwise 0.
        
        # So we must check:
        # 1. For all i, j, k: sum_r U[i, j, r] V[j, k, r] W[i, k, r] == 1
        #    (Note: W index is k*n+i? No, W is p x n. Flat index k*n+i.
        #     C index is (i, k). So W[k*n+i] corresponds to C[i, k].
        #     So the check above `W[k*n+i]` is correct for C[i, k].)
        
        # 2. For all other combinations (a, b, c, d, e, f) where NOT (b==c and a==e and d==f):
        #    sum_r U[a, b, r] V[c, d, r] W[e, f, r] == 0
        #    Note: W index for C[e, f] is f*n+e.
        #    So check sum_r U[a*m+b, r] V[c*p+d, r] W[f*n+e, r] == 0.

        # We already checked the 1s. Now check the 0s.
        # There are 8*16*8 = 1024 total entries.
        # 32 are 1s. 992 are 0s.
        
        for a in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b_idx // p
                    k1 = b_idx % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    
                    # Check if this is a "1" entry
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue
                    
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b_idx][r] * W[c_idx][r]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Try to find rank-27 decomposition using null space of the 28 terms
    def try_null_space_reduction():
        U28, V28, W28 = best_result["U"], best_result["V"], best_result["W"]
        rank28 = len(U28[0])
        
        # Construct the 28 flattened tensors
        # Size: n*m * m*p * p*n = 8 * 16 * 8 = 1024
        T_flat = []
        for r in range(rank28):
            vec = np.zeros(n*m * m*p * p*n)
            for a in range(n*m):
                for b_idx in range(m*p):
                    for c_idx in range(p*n):
                        val = U28[a][r] * V28[b_idx][r] * W28[c_idx][r]
                        idx = a * (m*p * p*n) + b_idx * (p*n) + c_idx
                        vec[idx] = val
            T_flat.append(vec)
        
        T_matrix = np.array(T_flat).T # Shape (1024, 28)
        
        # Find null space
        # Use SVD or QR with pivoting
        # Since 28 is small, we can use np.linalg.matrix_rank
        rank_T = np.linalg.matrix_rank(T_matrix, tol=1e-9)
        
        if rank_T < rank28:
            # Find a null vector
            # Solve T_matrix x = 0
            # Using SVD
            U_svd, s, Vh = np.linalg.svd(T_matrix)
            # Null space vectors are rows of Vh corresponding to zero singular values
            # Find indices where s is close to 0
            null_indices = np.where(s < 1e-9)[0]
            if len(null_indices) > 0:
                # Take the first null vector
                c = Vh[null_indices[0], :]
                
                # Find an index k where c[k] is not close to 0
                k = -1
                for idx in range(rank28):
                    if abs(c[idx]) > 1e-6:
                        k = idx
                        break
                
                if k != -1:
                    # lambda = -1 / c[k]
                    lam = -1.0 / c[k]
                    
                    # New coefficients
                    new_coeffs = []
                    for r in range(rank28):
                        if r == k:
                            continue
                        alpha = 1.0 + lam * c[r]
                        new_coeffs.append(alpha)
                    
                    # Check if new coefficients are "nice" (integers or half-integers)
                    # If not, we might not be able to use them directly if the evaluator
                    # requires integer/half-integer entries.
                    # However, the problem statement says "integer/half-integer entries".
                    # If the null vector has rational entries, the alphas will be rational.
                    # We can scale the entire decomposition?
                    # No, scaling U, V, W by factors that multiply to 1 is possible?
                    # If we multiply column r of U by alpha_r, column r of V by 1, column r of W by 1,
                    # the term scales by alpha_r.
                    # If alpha_r are not half-integers, the entries in U will not be half-integers.
                    # We need to check if we can find a null vector with small integer coefficients.
                    
                    # For now, let's just try to construct the result and validate.
                    # If validation passes and entries are close to half-integers, we accept.
                    
                    U_new = []
                    V_new = []
                    W_new = []
                    
                    for a in range(n*m):
                        row = []
                        for r in range(rank28):
                            if r == k:
                                continue
                            row.append(U28[a][r] * new_coeffs[r- (1 if r>k else 0)]) 
                            # Note: new_coeffs index mapping
                        U_new.append(row)
                        
                    # Correct mapping for new_coeffs
                    new_U_rows = []
                    new_V_rows = []
                    new_W_rows = []
                    
                    coeff_map = {}
                    idx_new = 0
                    for r in range(rank28):
                        if r != k:
                            coeff_map[r] = new_coeffs[idx_new]
                            idx_new += 1
                            
                    for a in range(n*m):
                        row = []
                        for r in range(rank28):
                            if r != k:
                                row.append(U28[a][r] * coeff_map[r])
                        new_U_rows.append(row)
                        
                    for b_idx in range(m*p):
                        row = []
                        for r in range(rank28):
                            if r != k:
                                row.append(V28[b_idx][r] * 1.0) # V and W not scaled?
                                # Wait, T_r = U_r \otimes V_r \otimes W_r.
                                # We want sum alpha_r T_r.
                                # So we can scale U_r by alpha_r, or V_r, or W_r, or distribute.
                                # To keep entries simple, maybe scale U_r by alpha_r?
                                # If alpha_r is rational, U_r entries become rational.
                                # If we require half-integers, we might fail.
                                # But let's try.
                        new_V_rows.append(row)
                        
                    for c_idx in range(p*n):
                        row = []
                        for r in range(rank28):
                            if r != k:
                                row.append(W28[c_idx][r] * 1.0)
                        new_W_rows.append(row)
                        
                    # Wait, if we scale U_r by alpha_r, the term becomes alpha_r U_r \otimes V_r \otimes W_r.
                    # This is correct.
                    
                    if validate(new_U_rows, new_V_rows, new_W_rows, rank28 - 1):
                        return {"U": new_U_rows, "V": new_V_rows, "W": new_W_rows}
        
        return None

    best_result = build_strassen_28()
    best_rank = 28
    
    # Try null space reduction
    reduced = try_null_space_reduction()
    if reduced:
        best_result = reduced
        best_rank = 27
        
    return best_result

# EVOLVE-BLOCK-END
