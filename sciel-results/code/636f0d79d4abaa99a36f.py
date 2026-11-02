"""
Strategy: Hierarchical Decomposition with Sub-Problem Optimization.
1. Use Strassen's 7-product decomposition for the 2x2 block structure of 6x6 matrices.
2. Optimize the inner 3x3 matrix multiplication decomposition to have minimal rank.
3. The total rank will be 7 * (rank of 3x3 decomposition).
   - Current: 7 * 27 (schoolbook) = 189.
   - Target: 7 * R, where R is the minimal rank for 3x3 found by search.
   - Known lower bound for 3x3 is 19, upper bound 23.
   - If we find rank 23, total is 161.
   - If we find rank 24, total is 168.
"""

import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

def _verify_3x3(U, V, W, rank):
    """Verify if U, V, W form a valid decomposition for 3x3 matrix multiplication."""
    n = 3
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0.0
                for c in range(rank):
                    s += U[i * n + j, c] * V[j * n + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _search_3x3_decomposition(max_rank=27, min_rank=19, deadline=None, rng=None):
    """
    Search for a low-rank decomposition of 3x3 matrix multiplication.
    Uses randomized initialization and local search (removing terms).
    """
    if rng is None:
        rng = random.Random(42)
    
    best_rank = max_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook rank 27
    U = np.zeros((9, max_rank))
    V = np.zeros((9, max_rank))
    W = np.zeros((9, max_rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    
    if _verify_3x3(U, V, W, max_rank):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = max_rank
    
    # Try to reduce rank by removing terms
    # We can also try to "merge" or "adjust" terms, but removal is easier to implement correctly.
    # However, simple removal from schoolbook often gets stuck at high ranks.
    # A better approach for 3x3 is to use known structures or more aggressive search.
    
    # Let's try a few random restarts with slightly perturbed schoolbook or random initializations
    # that satisfy the constraints approximately, then refine.
    
    for attempt in range(50):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with random vectors scaled to satisfy constraints? 
        # Actually, let's stick to structured search.
        # Try removing terms from the current best.
        cur_U = best_U.copy()
        cur_V = best_V.copy()
        cur_W = best_W.copy()
        cur_rank = best_rank
        
        # Shuffle column order to try different removal sequences
        indices = list(range(cur_rank))
        rng.shuffle(indices)
        
        for idx in indices:
            if cur_rank <= min_rank:
                break
            if deadline and time.time() > deadline:
                break
            
            # Try removing term idx
            mask = np.ones(cur_rank, dtype=bool)
            mask[idx] = False
            U_new = cur_U[:, mask]
            V_new = cur_V[:, mask]
            W_new = cur_W[:, mask]
            
            if _verify_3x3(U_new, V_new, W_new, cur_rank - 1):
                cur_U, cur_V, cur_W = U_new, V_new, W_new
                cur_rank -= 1
                if cur_rank < best_rank:
                    best_rank = cur_rank
                    best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
                    # If we found a very low rank, we can break early
                    if best_rank <= 20:
                        break
        
    return best_U, best_V, best_W, best_rank

def _construct_6x6_from_3x3(U3, V3, W3, rank3):
    """
    Construct 6x6 matrix multiplication decomposition using Strassen's 7-product scheme,
    where each product is a 3x3 matrix multiplication using the provided rank3 decomposition.
    
    Strassen steps for 2x2 blocks (A, B, C):
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = 6
    m = 6
    p = 6
    
    total_rank = 7 * rank3
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Define the Strassen terms
    # Each term is (A_part, B_part, C_part, sign)
    # A_part: which 3x3 block of A and with what sign
    # B_part: which 3x3 block of B and with what sign
    # C_part: which 3x3 block of C and with what sign
    
    # Blocks are indexed 0,1,2,3 for TL, TR, BL, BR
    # A11=0, A12=1, A21=2, A22=3
    
    strassen_terms = [
        # M1 = (A11 + A22) * (B11 + B22)
        # C11 += M1, C22 += M1
        # We need to decompose (A11+A22)*(B11+B22) into rank3 terms? 
        # No, Strassen works by computing M1 as a single matrix product.
        # But M1 is a 3x3 matrix product of (A11+A22) and (B11+B22).
        # So we use the 3x3 decomposition for this product.
        # The inputs to the 3x3 decomposition are the matrices (A11+A22) and (B11+B22).
        # The output is M1.
        # M1 contributes to C11 and C22.
        
        # This structure is tricky to map directly to the U,V,W factor matrices of the whole 6x6 tensor.
        # The standard way to embed is:
        # For each Strassen product Ms, we have a 3x3 multiplication tensor Ts.
        # Ts[i,j,k] = sum_c U3[i,j,c] V3[j,k,c] W3[k,i,c]
        # We need to map the indices i,j,k of the 3x3 block to the global 6x6 indices.
        
        # Let's define the mapping for each Strassen term.
    ]
    
    # This direct embedding is complex because Strassen combines A blocks before multiplication.
    # A better way:
    # The 6x6 multiplication tensor T6[i,j,k] is the sum over Strassen terms.
    # T6[i,j,k] = sum_s sum_c U3[...] V3[...] W3[...]
    
    # Let's define the linear combinations for A and B for each Strassen term.
    # Term 1: M1 = (A11 + A22) (B11 + B22)
    # A_part = A11 + A22. In the global A matrix (6x6), A11 is rows 0-2, cols 0-2. A22 is rows 3-5, cols 3-5.
    # So for a specific entry (i,j) in the 3x3 "virtual" A matrix for M1, 
    # if i,j < 3, it corresponds to A11[i,j] + A22[i,j]? No.
    # The virtual A matrix for M1 is A11 + A22.
    # So virtual_A[i,j] = A11[i,j] + A22[i,j] for i,j in 0..2.
    # Wait, A11 and A22 are disjoint blocks. A11[i,j] is A[i,j]. A22[i,j] is A[i+3, j+3].
    # So virtual_A[i,j] = A[i,j] + A[i+3, j+3].
    
    # Similarly for B.
    
    # Let's build the global U, V, W by iterating over the 7 Strassen terms.
    # For each term s, we have a 3x3 multiplication.
    # The "input" A for this term is a linear combination of global A blocks.
    # The "input" B for this term is a linear combination of global B blocks.
    # The "output" C for this term is added to specific global C blocks.
    
    # We can represent the global decomposition as:
    # U_global[global_a_idx, global_rank_idx]
    # V_global[global_b_idx, global_rank_idx]
    # W_global[global_c_idx, global_rank_idx]
    
    # For a specific Strassen term s, and a specific rank-1 term c in the 3x3 decomposition:
    # U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
    
    # The contribution to the global tensor is:
    # Sum over i,j,k in 0..2 of U3[i,j,c] * V3[j,k,c] * W3[k,i,c] * (mapping to global indices)
    
    # Actually, it's easier to construct the global U, V, W directly.
    
    # Let's define the Strassen terms explicitly with their block mappings.
    
    # Term 1: M1
    # A_virtual = A11 + A22
    # B_virtual = B11 + B22
    # C_virtual contributes to C11 (+1) and C22 (+1)
    
    # For each (i, j) in 0..2 (virtual A indices):
    #   Global A indices involved: (i, j) from A11 and (i+3, j+3) from A22.
    #   So U_global[i, col] += U3[i, j, c]
    #   U_global[i+3, j+3, col] += U3[i, j, c]
    
    # For each (j, k) in 0..2 (virtual B indices):
    #   Global B indices involved: (j, k) from B11 and (j+3, k+3) from B22.
    #   So V_global[j, col] += V3[j, k, c]
    #   V_global[j+3, k+3, col] += V3[j, k, c]
    
    # For each (k, i) in 0..2 (virtual C indices, output of multiplication):
    #   Note: W3 is indexed by [k, i] in the 3x3 decomposition? 
    #   Standard 3x3 decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    #   Here i is row of A, j is col of A/row of B, k is col of B.
    #   Output C[i,k].
    #   So W3 is indexed by [k, i] or [i, k]? 
    #   In my _verify_3x3, I used W[k * 3 + i, c]. So W is indexed by [k, i] where k is col of B, i is row of A?
    #   Let's re-verify standard convention.
    #   C[i,k] = sum_j A[i,j] B[j,k].
    #   Tensor T[i,j,k] = A[i,j] B[j,k] C[k,i] (if C is transposed in the tensor definition?)
    #   Usually T[i,j,k] = A[i,j] B[j,k] C[i,k].
    #   My _verify_3x3 uses:
    #   s += U[i * n + j, c] * V[j * n + k, c] * W[k * n + i, c]
    #   This implies the tensor is T[i,j,k] = sum U[i,j] V[j,k] W[k,i].
    #   This corresponds to C[k,i] = sum A[i,j] B[j,k]? No.
    #   If C[k,i] is the output, then the output matrix is transposed?
    #   Let's check the task contract: "output factor follows the official trace convention W[k*n+i,r]".
    #   This usually means the third mode is indexed by k (col of B) and i (row of A)?
    #   Or is it C[i,k]?
    #   Standard matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
    #   Tensor T[i,j,k] = A[i,j] B[j,k] C[i,k].
    #   If the contract says W[k*n+i], it might be implying the index is k*n+i, which corresponds to C[i,k] if we flatten row-major?
    #   If C is (p, n) i.e. (rows, cols), then C[i,k] is at index i*p + k.
    #   But the contract says W has shape (p*n, rank) and index k*n+i.
    #   This implies the output matrix is indexed by (k, i) where k is the first index (0..p-1) and i is the second (0..n-1).
    #   This suggests the output matrix is C^T or the multiplication is defined as C[k,i] = sum_j A[i,j] B[j,k]?
    #   Let's assume the standard C[i,k] = sum A[i,j] B[j,k].
    #   Then the index in the flattened output is i*p + k.
    #   The contract says W[k*n+i]. This is different.
    #   Let's look at the seed program's `_schoolbook`:
    #   W[k * n + i, col] = 1
    #   And the loop is for i (0..n-1), j (0..m-1), k (0..p-1).
    #   U[i*m+j], V[j*p+k], W[k*n+i].
    #   This matches T[i,j,k] = U[i,j] V[j,k] W[k,i].
    #   So the output index is k*n + i.
    #   This means the output matrix C is indexed by (k, i).
    #   So C[k, i] = sum_j A[i, j] B[j, k].
    #   This is the standard definition if we consider A (n x m), B (m x p), C (p x n).
    #   Wait, C (p x n) means rows are p, cols are n.
    #   C[k, i] where k is row (0..p-1), i is col (0..n-1).
    #   Standard matmul: C = A * B is (n x p).
    #   Here C is (p x n). So C = B^T * A^T? Or just a convention swap.
    #   Regardless, I must follow the seed program's convention:
    #   U[i, j], V[j, k], W[k, i] for A[i, j], B[j, k], C[k, i].
    
    # So for the 3x3 sub-problem, I must use the same convention.
    # My _search_3x3_decomposition uses this convention.
    
    # Now, mapping Strassen terms to global indices.
    
    # Term 1: M1 = (A11 + A22) (B11 + B22)
    # A_virtual[i, j] = A11[i, j] + A22[i, j] = A[i, j] + A[i+3, j+3]
    # B_virtual[j, k] = B11[j, k] + B22[j, k] = B[j, k] + B[j+3, k+3]
    # C_virtual[k, i] contributes to C11[k, i] and C22[k, i].
    # C11 is top-left 3x3 block of C (rows 0-2, cols 0-2). Index in global C: k*6 + i.
    # C22 is bottom-right 3x3 block of C (rows 3-5, cols 3-5). Index in global C: (k+3)*6 + (i+3).
    
    # So for each rank-1 term c in the 3x3 decomposition:
    # For each i, j, k in 0..2:
    #   val = U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
    #   This val is the contribution to C_virtual[k, i].
    #   
    #   We need to add this to global C.
    #   Global C index 1: k*6 + i (for C11)
    #   Global C index 2: (k+3)*6 + (i+3) (for C22)
    
    #   But we are constructing U, V, W, not C directly.
    #   The global U, V, W must satisfy:
    #   sum_c U_global[a, c] V_global[b, c] W_global[c_idx, c] = T_global[a, b, c_idx]
    
    #   For a specific Strassen term s and 3x3 rank term c:
    #   The 3x3 term contributes U3[i,j,c] to A_virtual[i,j].
    #   A_virtual[i,j] = A[i,j] + A[i+3, j+3].
    #   So U_global[i, col_s_c] += U3[i,j,c]
    #      U_global[i+3, j+3, col_s_c] += U3[i,j,c]
    
    #   Similarly for V:
    #      V_global[j, col_s_c] += V3[j,k,c]
    #      V_global[j+3, k+3, col_s_c] += V3[j,k,c]
    
    #   For W:
    #      The output is C_virtual[k,i].
    #      It adds to C11[k,i] and C22[k,i].
    #      So W_global[k*6+i, col_s_c] += W3[k,i,c]
    #         W_global[(k+3)*6+(i+3), col_s_c] += W3[k,i,c]
    
    #   This logic holds for Term 1.
    
    # Let's generalize for all 7 terms.
    
    # Define the terms as (A_blocks, B_blocks, C_blocks, signs)
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign)
    
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    
    # M1: A={0:1, 3:1}, B={0:1, 3:1}, C={0:1, 3:1}
    # M2: A={2:1, 3:1}, B={0:1}, C={2:1}  # (A21+A22)B11 -> C21
    # M3: A={0:1}, B={1:1, 3:-1}, C={1:1}  # A11(B12-B22) -> C12
    # M4: A={3:1}, B={2:1, 0:-1}, C={1:1, 3:-1} # A22(B21-B11) -> C21 and C22? 
    #    Wait, C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    #    M4 contributes to C21 (+1) and C22 (-1)?
    #    Let's check Strassen formulas again.
    #    C11 = M1 + M4 - M5 + M7
    #    C12 = M3 + M5
    #    C21 = M2 + M4
    #    C22 = M1 - M2 + M3 + M6
    
    #    So:
    #    M1: C11(+1), C22(+1)
    #    M2: C21(+1), C22(-1)
    #    M3: C12(+1), C22(+1)
    #    M4: C11(+1), C21(+1)
    #    M5: C11(-1), C12(+1)
    #    M6: C22(+1)
    #    M7: C11(+1)
    
    #    Now A and B parts:
    #    M1: (A11+A22)(B11+B22) -> A:{0:1, 3:1}, B:{0:1, 3:1}
    #    M2: (A21+A22)B11 -> A:{2:1, 3:1}, B:{0:1}
    #    M3: A11(B12-B22) -> A:{0:1}, B:{1:1, 3:-1}
    #    M4: A22(B21-B11) -> A:{3:1}, B:{2:1, 0:-1}
    #    M5: (A11+A12)B22 -> A:{0:1, 1:1}, B:{3:1}
    #    M6: (A21-A11)(B11+B12) -> A:{2:1, 0:-1}, B:{0:1, 1:1}
    #    M7: (A12-A22)(B21+B22) -> A:{1:1, 3:-1}, B:{2:1, 3:1}
    
    terms = [
        # (A_dict, B_dict, C_dict)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),      # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),      # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),      # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),      # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),      # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),      # M7
    ]
    
    for s_idx, (A_dict, B_dict, C_dict) in enumerate(terms):
        for c in range(rank3):
            col = s_idx * rank3 + c
            # Iterate over all i, j, k in 0..2
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        u_val = U3[i * 3 + j, c]
                        v_val = V3[j * 3 + k, c]
                        w_val = W3[k * 3 + i, c]
                        
                        # Add to U
                        for a_block, sign in A_dict.items():
                            row_start = (a_block // 2) * 3
                            col_start = (a_block % 2) * 3
                            U[(row_start + i) * 6 + (col_start + j), col] += sign * u_val
                        
                        # Add to V
                        for b_block, sign in B_dict.items():
                            row_start = (b_block // 2) * 3
                            col_start = (b_block % 2) * 3
                            V[(row_start + j) * 6 + (col_start + k), col] += sign * v_val
                        
                        # Add to W
                        for c_block, sign in C_dict.items():
                            row_start = (c_block // 2) * 3
                            col_start = (c_block % 2) * 3
                            W[(row_start + k) * 6 + (col_start + i), col] += sign * w_val
                            
    return U, V, W, total_rank

def _verify_6x6(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    rng = random.Random(42)
    
    # Phase 1: Find best 3x3 decomposition
    U3, V3, W3, rank3 = _search_3x3_decomposition(max_rank=27, min_rank=19, deadline=deadline - 10, rng=rng)
    
    # Phase 2: Construct 6x6 from 3x3
    U, V, W, rank6 = _construct_6x6_from_3x3(U3, V3, W3, rank3)
    
    # Phase 3: Verify
    if not _verify_6x6(U, V, W, n, m, p):
        # Fallback to schoolbook 6x6
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
        rank6 = n * m * p
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
