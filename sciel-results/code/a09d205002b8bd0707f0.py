"""
Strategy: Hard-coded Strassen + Global Sparsity Reduction.
1. Directly construct the 6x6 matrix multiplication tensor using Strassen's 7-product 
   algorithm for the 2x2 block structure, where each block multiplication is the 
   3x3 schoolbook (rank 27). This yields a guaranteed rank of 7 * 27 = 189.
2. Instead of searching for a better 3x3 decomposition (which is hard and slow), 
   we perform a global optimization on the 189-term 6x6 decomposition.
3. We attempt to remove terms that are linearly dependent or have near-zero 
   coefficients, or we try to merge terms if possible. 
4. Given the complexity of 6x6, the primary goal is to reliably return rank 189 
   (which is an improvement over the seed's potential failure or default 216), 
   and then opportunistically try to reduce it further.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _construct_strassen_6x6():
    """
    Constructs the 6x6 matrix multiplication decomposition using Strassen's algorithm.
    Rank = 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2 blocks of 3x3 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the linear combinations for A, B, and C for each of the 7 terms.
    # Each term corresponds to a 3x3 matrix multiplication.
    # For each term, we iterate over the 27 schoolbook products of the 3x3 sub-matrices.
    
    # Block indices:
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # Helper to get global index for a block (bi, bj) and local index (li, lj)
    # bi, bj in {0, 1}
    def get_global_idx(block_i, block_j, local_i, local_j):
        return (block_i * 3 + local_i) * 6 + (block_j * 3 + local_j)
    
    # Define the 7 Strassen terms
    # Each term is a tuple: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping (block_i, block_j) to coefficient
    # B_coeffs: dict mapping (block_i, block_j) to coefficient
    # C_coeffs: dict mapping (block_i, block_j) to coefficient
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # Wait, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
        # Let's re-verify the C contributions.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C11(+1), C22(+1)
        # M2: C21(+1), C22(-1)
        # M3: C12(+1), C22(+1)
        # M4: C11(+1), C21(+1)
        # M5: C11(-1), C12(+1)
        # M6: C22(+1)
        # M7: C11(+1)
        
        # Let's re-list them correctly.
    ]
    
    # Corrected Strassen Terms:
    # Term 0 (M1): A={(0,0):1, (1,1):1}, B={(0,0):1, (1,1):1}, C={(0,0):1, (1,1):1}
    # Term 1 (M2): A={(1,0):1, (1,1):1}, B={(0,0):1}, C={(1,0):1, (1,1):-1}
    # Term 2 (M3): A={(0,0):1}, B={(0,1):1, (1,1):-1}, C={(0,1):1, (1,1):1}
    # Term 3 (M4): A={(1,1):1}, B={(1,0):1, (0,0):-1}, C={(0,0):1, (1,0):1}
    # Term 4 (M5): A={(0,0):1, (0,1):1}, B={(1,1):1}, C={(0,0):-1, (0,1):1}
    # Term 5 (M6): A={(1,0):1, (0,0):-1}, B={(0,0):1, (0,1):1}, C={(1,1):1}
    # Term 6 (M7): A={(0,1):1, (1,1):-1}, B={(1,0):1, (1,1):1}, C={(0,0):1}
    
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # For each Strassen term (7 terms), we have a 3x3 multiplication.
    # We use the schoolbook decomposition for the 3x3 multiplication.
    # The schoolbook decomposition for 3x3 has 27 terms.
    # For each of the 27 terms, we update the global U, V, W.
    
    for s_idx, (A_dict, B_dict, C_dict) in enumerate(terms):
        for i3 in range(3):
            for j3 in range(3):
                for k3 in range(3):
                    col = s_idx * 27 + i3 * 9 + j3 * 3 + k3
                    
                    # U contribution
                    # The "virtual" A matrix for this Strassen term is a linear combination of blocks.
                    # For a specific (i3, j3) in the 3x3 virtual matrix, the value is sum_{block} coeff * A_block[i3, j3].
                    # However, in the tensor decomposition, U[global_A_idx, col] corresponds to the coefficient of A[global_A_idx] in the "virtual" A.
                    # So, for each block in A_dict, we add the coefficient to U[global_idx, col].
                    for (bi, bj), coeff in A_dict.items():
                        global_a_idx = get_global_idx(bi, bj, i3, j3)
                        U[global_a_idx, col] += coeff
                    
                    # V contribution
                    for (bi, bj), coeff in B_dict.items():
                        global_b_idx = get_global_idx(bi, bj, j3, k3)
                        V[global_b_idx, col] += coeff
                    
                    # W contribution
                    # Note: The output C is indexed by [k, i] in the convention W[k*n+i].
                    # In the 3x3 sub-problem, the output index is [k3, i3].
                    # The "virtual" C matrix is a linear combination of blocks.
                    for (bi, bj), coeff in C_dict.items():
                        global_c_idx = get_global_idx(bi, bj, k3, i3)
                        W[global_c_idx, col] += coeff
                        
    return U, V, W, rank

def _verify_6x6(U, V, W, n, m, p):
    rank = U.shape[1]
    # Fast verification using numpy
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We can reshape U, V, W and use einsum or broadcasting.
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_res = U.reshape(n, m, rank)
    V_res = V.reshape(m, p, rank)
    W_res = W.reshape(p, n, rank)
    
    # Compute T
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Note: V is indexed by j, k. W is indexed by k, i.
    # Let's use einsum.
    # U[i, j, c] V[j, k, c] W[k, i, c] -> sum over c
    # Result T[i, j, k]
    
    # We need to be careful with indices.
    # U: n, m, r
    # V: m, p, r
    # W: p, n, r
    
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    
    # Let's compute this step by step to avoid huge intermediate tensors if possible, 
    # but 6x6x6 is small (216 elements).
    
    # Product UV[i, j, k, c] = U[i, j, c] * V[j, k, c]
    # This is (n, m, p, r). 6*6*6*189 = 40824 elements. Very small.
    
    UV = U_res[:, :, np.newaxis, :] * V_res[np.newaxis, :, :, :] # Shape: (n, m, p, r)
    # Now multiply by W[k, i, c]. W is (p, n, r).
    # We need W[k, i, c] to match UV's indices (n, m, p, r).
    # UV has indices (i, j, k, c).
    # W needs to be (i, j, k, c) ? No, W depends on k, i, c.
    # We can expand W to (n, m, p, r) by broadcasting.
    # W_expanded[i, j, k, c] = W[k, i, c]
    
    # W is (p, n, r). We want (n, m, p, r).
    # Axis 0 of W is k (p). Axis 1 is i (n). Axis 2 is c (r).
    # We want to map k -> axis 2 of result, i -> axis 0 of result.
    # Result shape: (n, m, p, r)
    # W_transposed shape: (n, p, r) -> (n, 1, p, r)
    
    W_trans = W_res.T # (n, p, r)
    W_exp = W_trans[:, np.newaxis, :, :] # (n, 1, p, r)
    
    T = np.einsum('ijkc,ikc->ijk', UV, W_exp)
    
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _try_reduce_rank(U, V, W, rank, deadline):
    """
    Attempts to reduce the rank by removing terms.
    This is a heuristic.
    """
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    # Try removing columns one by one.
    # If removing a column keeps the tensor valid, keep it removed.
    # This is O(rank^2 * verify_cost).
    # Verify cost is ~ 216 * rank operations.
    # Rank is 189.
    # 189 * 189 * 216 * 189 ~ 140 million ops. Might be too slow in Python loops.
    # We will use the fast numpy verification.
    
    n, m, p = 6, 6, 6
    
    for attempt in range(rank):
        if time.time() > deadline:
            break
            
        # Try to remove a random column
        # We'll iterate through columns and try to remove them if possible.
        # To make it faster, we can shuffle columns first.
        if attempt == 0:
            indices = list(range(rank))
            np.random.shuffle(indices)
        else:
            # Just try the next one in the shuffled order? 
            # Actually, let's just iterate through the current columns.
            pass
            
        # Let's just try to remove the last column if it's zero or redundant.
        # A simpler heuristic: check if any column is zero.
        # Or check if removing column c makes the tensor invalid.
        
        # We'll try to remove columns from the end backwards, checking validity.
        # If valid, keep it removed.
        
        c = best_rank - 1
        if c < 0:
            break
            
        U_test = best_U[:, :c]
        V_test = best_V[:, :c]
        W_test = best_W[:, :c]
        
        if _verify_6x6(U_test, V_test, W_test, n, m, p):
            best_U = U_test
            best_V = V_test
            best_W = W_test
            best_rank = c
        else:
            # If removing the last column fails, try removing a random other column?
            # For now, we just stop or try a few random ones.
            # To save time, we will just try the last few.
            if attempt < 10:
                # Try a random column
                c_rand = np.random.randint(0, best_rank)
                mask = np.ones(best_rank, dtype=bool)
                mask[c_rand] = False
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                if _verify_6x6(U_test, V_test, W_test, n, m, p):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank -= 1

    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Construct Strassen 6x6 (Rank 189)
    U, V, W, rank = _construct_strassen_6x6()
    
    # 2. Verify
    if not _verify_6x6(U, V, W, n, m, p):
        # Fallback to schoolbook (Rank 216) just in case Strassen construction has a bug
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = n * m * p
    else:
        # 3. Try to reduce rank
        U, V, W, rank = _try_reduce_rank(U, V, W, rank, deadline)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
