"""
Matrix multiplication 3x4 * 4x4 tensor decomposition.
Strategy:
1. Build a high-quality initial decomposition using a mixed structural approach:
   - Use Strassen for all 2x2 * 2x2 sub-blocks (rank 7 each).
   - Use a specialized optimized 1x2 * 2x2 decomposition for the bottom row blocks.
     Instead of pure schoolbook (rank 4), we exploit the linear structure.
     Actually, for 1x2 * 2x2, the tensor is of size 2 x 4 x 2 (A: 1x2, B: 2x2, C: 2x2).
     Wait, the block is A[i, j] (i fixed, j in {0,1}) * B[j, k] (j in {0,1}, k in {0,1}) -> C[i, k] (k in {0,1}).
     This is essentially two independent 1x1 * 2x1 multiplications? No.
     C[i, k] = A[i, 0] B[0, k] + A[i, 1] B[1, k].
     This is rank 2? No, for each k, it's a dot product.
     The tensor T_{j,k,l} (where j is A-col, k is B-col, l is C-col) has T_{j,k,l} = 1 if j == k else 0? No.
     A has indices (i, j). Since i is fixed, A is a vector of length 2.
     B is 2x2. C is a vector of length 2.
     C_l = sum_j A_j B_{j,l}.
     The tensor entries are T_{j, k, l} where j is index in A, k is index in B (row), l is index in C.
     Wait, B is 2x2. Indices (j, k).
     The term is A_j * B_{j,k} * W_{k,l}? No, W is for C.
     Standard form: C_{i,k} = sum_j A_{i,j} B_{j,k}.
     For fixed i, C_k = A_0 B_{0,k} + A_1 B_{1,k}.
     This can be done with rank 2?
     Term 1: A_0, B_{0,:}, C_0? No.
     Let's look at the tensor for 1x2 * 2x2 -> 1x2.
     Input A: dim 2. Input B: dim 4 (2x2). Output C: dim 2.
     T_{j, (r,k), l} = 1 if j==r else 0? And output index l corresponds to k.
     So T_{j, r*2+k, k} = 1 if j==r else 0.
     This tensor is rank 2?
     Decomposition:
     Term 0: A_0, B_{0,0}, C_0. (j=0, r=0, k=0 -> l=0)
     Term 1: A_0, B_{0,1}, C_1. (j=0, r=0, k=1 -> l=1)
     Term 2: A_1, B_{1,0}, C_0. (j=1, r=1, k=0 -> l=0)
     Term 3: A_1, B_{1,1}, C_1. (j=1, r=1, k=1 -> l=1)
     Rank 4 is schoolbook.
     Is there a lower rank?
     The tensor is equivalent to the identity tensor? No.
     It's the tensor for two independent scalar multiplications? No, because B shares rows.
     Actually, this is just the identity map from A to A (if B=I)? No.
     If B is arbitrary, it's a linear map from A (dim 2) to C (dim 2) given by B?
     No, A is a row vector. C = A B.
     This is just matrix-vector multiplication 1x2 * 2x2.
     The rank of the tensor T_{j, r, k} (where C_k = sum_j A_j B_{j,k})
     Note that B_{j,k} is the entry.
     The tensor is T_{j, (j', k), l} = delta_{j, j'} delta_{k, l}.
     This is the tensor for the identity operation on the "k" index, conditioned on "j".
     It is the direct sum of two rank-1 tensors?
     T = T_0 + T_1
     T_0: j=0, j'=0. C_k = A_0 B_{0,k}.
     T_1: j=1, j'=1. C_k = A_1 B_{1,k}.
     For T_0, the support is j=0, j'=0. The k and l indices are linked by identity.
     So T_0 is rank 1?
     U: e_0. V: sum_k e_{0,k}. W: sum_k e_{k,k}.
     Product: A_0 * (sum_k B_{0,k}) * (sum_l C_l delta_{k,l})? No.
     If we use rank 1: A_0 * B_{0,0} * C_0. This only gives the (0,0) term.
     We need C_1 to get A_0 B_{0,1}.
     So T_0 requires at least rank 2 (for the 2 outputs).
     Similarly T_1 requires rank 2.
     Total rank 4.
     So schoolbook is optimal for 1x2 * 2x2.
     
     Therefore, the structural rank is 4*7 + 4*4 = 28 + 16 = 44.
     
2. We will attempt to reduce this rank by finding linear dependencies among the terms.
   Specifically, we will look for sets of terms that can be combined or eliminated.
   We will use a more sophisticated linear algebra approach to find dependencies in the space
   spanned by the tensor components.
   
   However, a stronger structural change is to use a **different block partitioning**.
   Instead of 2x2 blocks, what if we use 3x3?
   A is 3x4. B is 4x4.
   We can split B into 2x2, 2x1, 1x2, 1x1?
   
   Let's try a **Rank Reduction via Algebraic Geometry / Gröbner Bases** approach?
   Too complex.
   
   Let's try **Randomized Local Search** for Rank 43.
   We initialize U, V, W with random values (integers) for Rank 43.
   We define the error as the sum of squared differences between the computed tensor and the target.
   We perform a hill-climbing search.
   
   However, the landscape is non-convex.
   
   Alternative: **Use the known optimal rank for 2x2x2 (7) and 2x2x4 (12?)**.
   Actually, 2x2x4 can be done with rank 12?
   2x2 * 2x4.
   Strassen for 2x2 * 2x2 gives rank 7.
   For 2x2 * 2x4, we can split the 2x4 into two 2x2.
   Rank 7+7=14.
   Is there a better way?
   The rank of 2x2x4 is known to be 12?
   If rank(2x2x4) = 12, then:
   A_top (2x4) * B (4x4).
   Split B into B_left (4x2) and B_right (4x2).
   C_top_left (2x2) = A_top * B_left.
   C_top_right (2x2) = A_top * B_right.
   If we can do 2x4 * 4x2 in rank 12?
   Note 2x4 * 4x2 is equivalent to 2x2 * 2x2 if we permute?
   A (2x4), B (4x2).
   This is a 2x4x2 tensor.
   The rank of 2x4x2 is the same as 2x2x4.
   If rank(2x2x4) = 12, then we can do the top part (2x4 * 4x4) in rank 12+12 = 24?
   No, because A is shared.
   A_top * B = A_top * [B_left, B_right] = [A_top B_left, A_top B_right].
   The decomposition of A_top * B_left and A_top * B_right might share terms in U.
   If we use a decomposition of 2x4x4, what is its rank?
   2x4 * 4x4.
   This is 2x4x4 tensor.
   Rank of 2x4x4?
   Schoolbook: 2*4*4 = 32.
   Strassen based:
   Split B into four 2x2 blocks.
   A is 2x4.
   A = [A1, A2] where A1, A2 are 2x2.
   B = [[B11, B12], [B21, B22]].
   C = [A1 B11 + A2 B21, A1 B12 + A2 B22].
   This is sum of 4 products of 2x2.
   Using Strassen for each? No, A1 is shared.
   
   Actually, there is a known result: Rank of 2x4x4 is 18? Or 20?
   If we can get 18, then total rank for top part is 18.
   Bottom part: 1x4 * 4x4.
   Rank of 1x4x4?
   Schoolbook: 16.
   Can we do better?
   1x4 * 4x4.
   A is 1x4.
   C = A B.
   This is just a linear combination of rows of B.
   The tensor is T_{j, k, l} = 1 if j==k?
   A_j B_{j,l} -> C_l.
   The tensor is T_{j, (r,l), l'} = delta_{j,r} delta_{l,l'}.
   This is the identity tensor on the l index, repeated for each j?
   No, it's the tensor for the map A -> C.
   It is rank 4?
   Term j: A_j, B_{j,:}, C_j? No.
   C_l = sum_j A_j B_{j,l}.
   For each l, we need a term.
   But B_{j,l} couples j and l.
   If we use rank 4:
   Term l: A_{?}, B_{?,l}, C_l.
   We need sum_j A_j B_{j,l}.
   If we set U to be A, V to be B_{:,l}, W to be C_l.
   Then sum_l (A . B_{:,l}) C_l.
   This requires U to be the same for all l? No, U is a vector.
   In the decomposition C = sum_r U_r A_r V_r B_r W_r C_r.
   If we fix r=l, U_r = 1 (scalar), V_r = B_{:,l}, W_r = C_l.
   Then sum_l 1 * B_{:,l} * C_l.
   This gives sum_l B_{:,l} C_l.
   But we want sum_j A_j B_{j,l} C_l.
   The A_j is missing.
   So we need U_r to depend on j.
   But r is the rank index.
   So we need distinct terms for each j?
   If we use terms for each j and l?
   Term (j,l): A_j, B_{j,l}, C_l.
   Rank 4*4=16.
   Can we reduce?
   If we use rank 8?
   
   Let's stick to the 44 rank and try to eliminate dependencies numerically.
   The previous attempt failed or scored 1.0.
   Maybe the dependencies are not simple linear dependencies of columns, but involve **reindexing** or **permutation** of the tensor basis?
   
   Let's try a **Genetic Algorithm** on the entries of U, V, W for Rank 43.
   We will use the 44-rank solution as a starting point, zero out one column, and try to adjust the others to fix the error.
   This is a "Null Space" search.
   We remove column r, creating a deficit D.
   We try to express D as a linear combination of the remaining columns?
   D = sum_{s != r} c_s (U_s, V_s, W_s).
   This is the same linear dependency check.
   If it fails, maybe we can **perturb** the remaining columns slightly to absorb the deficit?
   But entries must be integer/half-integer.
   
   Let's try **Rank 43 with Random Integers**.
   We will run a short local search.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    def _verify_decomp(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > tol:
                            return False
        return True

    def _error_metric(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        err += (actual - expected) ** 2
        return err

    def _strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        return U, V, W

    def _schoolbook_block(rows_a, cols_a, rows_b, cols_b):
        rank = rows_a * cols_a * cols_b
        U = np.zeros((rows_a * cols_a, rank))
        V = np.zeros((cols_a * cols_b, rank))
        W = np.zeros((cols_b * rows_a, rank))
        col = 0
        for i in range(rows_a):
            for j in range(cols_a):
                for k in range(cols_b):
                    U[i * cols_a + j, col] = 1.0
                    V[j * cols_b + k, col] = 1.0
                    W[k * rows_a + i, col] = 1.0
                    col += 1
        return U, V, W

    # Build 44-rank decomposition
    U_full = np.zeros((n * m, 44))
    V_full = np.zeros((m * p, 44))
    W_full = np.zeros((p * n, 44))
    
    col_offset = 0
    # 4 Strassen blocks
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            U_s, V_s, W_s = _strassen_2x2x2()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        val = U_s[i*2+j, r]
                        if val != 0:
                            U_full[(i)*4+(j+j_off), col_offset+r] += val
                for j in range(2):
                    for k in range(2):
                        val = V_s[j*2+k, r]
                        if val != 0:
                            V_full[(j+i_off)*4+k, col_offset+r] += val
                for k in range(2):
                    for i in range(2):
                        val = W_s[k*2+i, r]
                        if val != 0:
                            W_full[k*3+i, col_offset+r] += val
            col_offset += 7
            
    # 4 Schoolbook blocks (1x2 * 2x2)
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            U_sb, V_sb, W_sb = _schoolbook_block(1, 2, 2, 2)
            for r in range(4):
                for i in range(1):
                    for j in range(2):
                        val = U_sb[i*2+j, r]
                        if val != 0:
                            U_full[(2)*4+(j+j_off), col_offset+r] += val
                for j in range(2):
                    for k in range(2):
                        val = V_sb[j*2+k, r]
                        if val != 0:
                            V_full[(j)*4+(k+k_off), col_offset+r] += val
                for k in range(2):
                    for i in range(1):
                        val = W_sb[k*1+i, r]
                        if val != 0:
                            W_full[(k+k_off)*3+(2), col_offset+r] += val
            col_offset += 4

    if not _verify_decomp(U_full, V_full, W_full):
        # Fallback
        U_full, V_full, W_full = np.zeros((n*m, 48)), np.zeros((m*p, 48)), np.zeros((p*n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i*m+j, col] = 1.0
                    V_full[j*p+k, col] = 1.0
                    W_full[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Attempt to reduce rank to 43
    # We try removing each column and checking if the remaining system is consistent
    # via least squares. If the residual is non-zero, we try to minimize the error
    # by adjusting the remaining columns slightly? No, must be exact.
    
    # Instead, we try a randomized search for a rank 43 solution.
    # We initialize with the 44 solution, remove one column, and then
    # perform a local search to find integer/half-integer values for the
    # remaining columns that minimize the error.
    
    best_U, best_V, best_W = U_full, V_full, W_full
    best_rank = 44
    best_error = 0.0

    for target_rank in [43, 42, 41]:
        if time.time() > deadline - 10:
            break
            
        for _ in range(20): # 20 restarts
            if time.time() > deadline - 5:
                break
            
            U = U_full.copy()
            V = V_full.copy()
            W = W_full.copy()
            
            # Remove (44 - target_rank) random columns
            cols_to_remove = random.sample(range(44), 44 - target_rank)
            for c in sorted(cols_to_remove, reverse=True):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
            
            # Local Search
            err = _error_metric(U, V, W)
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                if err == 0:
                    break
                    
                # Pick a random matrix and entry
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    mat = U; other1 = V; other2 = W
                elif mat_choice == 'V':
                    mat = V; other1 = U; other2 = W
                else:
                    mat = W; other1 = U; other2 = V
                
                r_idx, c_idx = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
                old_val = mat[r_idx, c_idx]
                
                # Try small integer perturbations
                improved = False
                for delta in [-2, -1, 1, 2]:
                    mat[r_idx, c_idx] = old_val + delta
                    new_err = _error_metric(U, V, W)
                    if new_err < err:
                        err = new_err
                        improved = True
                        break
                    else:
                        mat[r_idx, c_idx] = old_val
                
                if not improved:
                    # Try larger jump or random value
                    new_val = random.randint(-3, 3)
                    mat[r_idx, c_idx] = new_val
                    new_err = _error_metric(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        mat[r_idx, c_idx] = old_val
            
            if err < 1e-9:
                # Snap to clean values
                def snap(x):
                    nx = round(x * 2) / 2
                    if abs(x - nx) < 1e-9: return nx
                    return x
                U = np.apply_along_axis(snap, 1, U)
                V = np.apply_along_axis(snap, 1, V)
                W = np.apply_along_axis(snap, 1, W)
                if _verify_decomp(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    best_error = 0.0
                    break # Found optimal for this target rank
            
            if best_error > err and err < 1e-3:
                # Keep best approximation if not exact? No, must be exact.
                pass

    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result

    return {"U": _to_clean_list(best_U), "V": _to_clean_list(best_V), "W": _to_clean_list(best_W)}
