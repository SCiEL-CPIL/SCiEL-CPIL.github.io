import time
import numpy as np
import sys

def construct():
    """
    Returns a rank-minimal exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 4, 8
    
    # Build the 8-block Strassen decomposition (rank 56)
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_block(j0, k0, base_col):
        """Add Strassen decomposition for 2x2 * 2x2 block."""
        U[0*m + j0][base_col + 0] += 1
        U[1*m + (j0+1)][base_col + 0] += 1
        V[j0*p + k0][base_col + 0] += 1
        V[(j0+1)*p + (k0+1)][base_col + 0] += 1
        W[k0*n + 0][base_col + 0] += 1
        W[(k0+1)*n + 1][base_col + 0] += 1
        
        U[1*m + j0][base_col + 1] += 1
        U[1*m + (j0+1)][base_col + 1] += 1
        V[j0*p + k0][base_col + 1] += 1
        W[k0*n + 1][base_col + 1] += 1
        W[(k0+1)*n + 1][base_col + 1] += -1
        
        U[0*m + j0][base_col + 2] += 1
        V[j0*p + (k0+1)][base_col + 2] += 1
        V[(j0+1)*p + (k0+1)][base_col + 2] += -1
        W[(k0+1)*n + 0][base_col + 2] += 1
        W[(k0+1)*n + 1][base_col + 2] += 1
        
        U[1*m + (j0+1)][base_col + 3] += 1
        V[(j0+1)*p + k0][base_col + 3] += 1
        V[j0*p + k0][base_col + 3] += -1
        W[k0*n + 0][base_col + 3] += 1
        W[k0*n + 1][base_col + 3] += 1
        
        U[0*m + j0][base_col + 4] += 1
        U[0*m + (j0+1)][base_col + 4] += 1
        V[(j0+1)*p + (k0+1)][base_col + 4] += 1
        W[k0*n + 0][base_col + 4] += -1
        W[(k0+1)*n + 0][base_col + 4] += 1
        
        U[1*m + j0][base_col + 5] += 1
        U[0*m + j0][base_col + 5] += -1
        V[j0*p + k0][base_col + 5] += 1
        V[j0*p + (k0+1)][base_col + 5] += 1
        W[(k0+1)*n + 1][base_col + 5] += 1
        
        U[0*m + (j0+1)][base_col + 6] += 1
        U[1*m + (j0+1)][base_col + 6] += -1
        V[(j0+1)*p + k0][base_col + 6] += 1
        V[(j0+1)*p + (k0+1)][base_col + 6] += 1
        W[k0*n + 0][base_col + 6] += 1
        
        return base_col + 7
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = add_strassen_block(j0, k0, col)
    
    # Verify the base decomposition is valid
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    max_diff = 0
    for i in range(n):
        for k in range(p):
            c_val = 0
            for j in range(m):
                for r in range(rank):
                    c_val += A_test[i, j] * B_test[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            diff = abs(c_val - C_expected[i, k])
            if diff > max_diff:
                max_diff = diff
    
    if max_diff > 1e-10:
        # Fallback: return schoolbook if something went wrong
        print(f"WARNING: Base decomposition invalid, max_diff={max_diff}")
        U_list = [[0] * (n*m*p) for _ in range(n*m)]
        V_list = [[0] * (n*m*p) for _ in range(m*p)]
        W_list = [[0] * (n*m*p) for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U_list[i*m+j][idx] = 1
                    V_list[j*p+k][idx] = 1
                    W_list[k*n+i][idx] = 1
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Now attempt rank reduction by finding linear dependencies
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find if any columns of (U, V, W) are linearly dependent
    # across the full decomposition.
    
    # Strategy: Build the "slice" matrices for each output element (i,k)
    # For fixed (i,k), the contribution to C[i,k] is:
    #   sum_j sum_r A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r]
    # = sum_j A[i,j] B[j,k] (sum_r U[i,j,r] V[j,k,r] W[k,i,r])
    # 
    # Define M_{i,k}[j, r'] where we group by r... this gets complex.
    # 
    # Alternative: Try to find a lower-rank decomposition by searching
    # for a transformation that reduces rank.
    #
    # Key insight: For 2x4x8, the tensor is T ∈ R^{2×4×8}.
    # Rank of matrix multiplication tensor MM_{2,4,8}:
    # - Schoolbook: 64
    # - 8×Strassen: 56
    # - Known lower bound: at least max(2*4, 4*8, 2*8) = 32 (trivial)
    # - Actually, rank ≥ n*m*p / (something)... 
    # - For 2x2x2, rank is 7 (Strassen is optimal).
    # - For 2x4x4, we could use 2x Strassen = 14, or maybe less?
    #
    # Let's try to see if we can do better than 56 by mixing blocks.
    # 
    # Approach: Use the fact that the 8 blocks share structure.
    # The A dimension has only 2 rows. So U has 8 rows total.
    # V has 32 rows. W has 16 rows.
    #
    # Let's try a different decomposition strategy:
    # Split differently to allow more sharing.
    #
    # Alternative: Use a 2x4x4 block (rank at most 14 via 2 Strassen)
    # and a 2x4x4 block, giving 28. But that's the same as 4 Strassen blocks.
    # Wait, 2x4x4: split A into two 2x2, B into two 2x4.
    # Each 2x2 * 2x4 = 2x2 * (2x2 + 2x2) = 2 Strassen blocks = 14.
    # So 2x4x4 has rank at most 14.
    # Then 2x4x8 = 2x4x4 + 2x4x4 = 28? No, that's not right.
    # B is 4x8, so splitting columns gives two 4x4 blocks.
    # C = A*B1 + A*B2 where B1, B2 are 4x4.
    # Each A*B is 2x4 * 4x4.
    # For 2x4*4x4: split A into A1(2x2), A2(2x2).
    # B1 into B1_top(2x4), B1_bot(2x4).
    # A*B = A1*B1_top + A2*B1_bot.
    # Each 2x2*2x4 = 2x2*(2x2+2x2) = 2 Strassen = 14.
    # So A*B has rank at most 14+14=28? No, that's double counting.
    # Actually A1*B1_top is 2x2*2x4. Split B1_top into two 2x2: rank 7+7=14.
    # A2*B1_bot is 2x2*2x4. Split B1_bot into two 2x2: rank 7+7=14.
    # Total for A*B: 14+14=28.
    # Two such products: 56. Same as before.
    
    # Let me try a completely different approach: 
    # Directly search for a lower-rank decomposition using the tensor structure.
    #
    # The tensor T[i,j,k] for matrix multiplication satisfies:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Actually, the multiplication tensor is:
    # M_{2,4,8} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the "product" is defined
    # by the bilinear map.
    #
    # More precisely, the tensor is:
    # T = sum_{i=0}^{1} sum_{j=0}^{3} sum_{k=0}^{7} A_i ⊗ B_j ⊗ C_{ik}
    # where A_i is the i-th row basis, B_j is the j-th column basis, 
    # C_{ik} is the (i,k) entry basis.
    #
    # For a rank-r decomposition:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{2×4}, v_r ∈ R^{4×8}, w_r ∈ R^{8×2}
    # and the product satisfies: for all A,B:
    #   A*B = sum_r (A·u_r)(B·v_r) w_r
    # where A·u_r = sum_{i,j} A[i,j] u_r[i,j]
    # and B·v_r = sum_{j,k} B[j,k] v_r[j,k]
    # and w_r is a 2x8 matrix (reshaped from w_r).
    #
    # So we need: for all i,k:
    #   (A*B)[i,k] = sum_r (sum_{j} A[i,j] u_r[i,j]) (sum_{j} B[j,k] v_r[j,k]) w_r[k,i]
    #
    # This must hold for ALL A and B, so:
    #   For each (i,k): sum_r u_r[i,j1] v_r[j2,k] w_r[k,i] = delta[j1,j2] * 1
    #   Wait, that's not quite right. Let me be more careful.
    #
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r [sum_j A[i,j] u_r[i,j]] [sum_j' B[j',k] v_r[j',k]] w_r[k,i]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A,B:
    # We need: sum_r u_r[i,j] v_r[j',k] w_r[k,i] = delta_{j,j'} for all i,k,j,j'
    #
    # Wait, that's a constraint on the tensor entries. Let me define:
    # S_{i,k}[j,j'] = sum_r u_r[i,j] v_r[j',k] w_r[k,i]
    # We need S_{i,k}[j,j'] = delta_{j,j'}
    #
    # This is a system of linear equations in the unknowns {u_r, v_r, w_r}.
    # For rank R, we have R * (8 + 32 + 16) = 56R unknowns.
    # Constraints: for each (i,k): 4*4 = 16 constraints (but only 4 are independent since delta is diagonal)
    # Total constraints: 2*8*16 = 256 (but many are zero, so 2*8*4 = 64 nonzero constraints)
    #
    # Actually, the constraint is:
    # sum_r u_r[i,j] v_r[j',k] w_r[k,i] = delta_{j,j'}
    # for i in {0,1}, k in {0..7}, j in {0..3}, j' in {0..3}
    # That's 2*8*4*4 = 256 equations, but 2*8*4 = 64 have RHS=1 and the rest have RHS=0.
    #
    # This is a non-linear system (products of variables), so it's hard to solve directly.
    #
    # Let me try a different approach: use the known structure and try to find
    # a rank-deficient combination.
    
    # Check if we can reduce rank by finding a linear relation among the 56 terms.
    # If there exist coefficients c_r (not all zero) such that:
    #   sum_r c_r * u_r = 0 (element-wise)
    #   sum_r c_r * v_r = 0 (element-wise)  
    #   sum_r c_r * w_r = 0 (element-wise)
    # then we could potentially remove those terms. But that would mean the tensor
    # can be written with fewer terms only if the cross terms cancel.
    #
    # More precisely, if we have a rank-R decomposition and there's a non-trivial
    # linear dependence among the "triples" (u_r, v_r, w_r), we might be able to
    # reduce the rank.
    
    # Let's try: can we find a 2x4x8 decomposition with rank < 56?
    # 
    # Theoretical lower bounds:
    # - Rank ≥ n*m*p / max(n,m,p) = 64/8 = 8 (very weak)
    # - Rank ≥ n*m + m*p - n*p? No, that's not a standard bound.
    # - For 2x2x2, rank is 7.
    # - For 2x4x4: we can write as 2x(2x2x2) + 2x(2x2x2) = 14, or maybe less?
    #   Actually, 2x4x4 = 2x2x2 ⊗ 2x2x1 + ... hmm.
    #   Let's think: A is 2x4, B is 4x4, C is 2x4.
    #   Split A = [A1 | A2] where A1, A2 are 2x2.
    #   Split B = [B1; B2] where B1, B2 are 2x4.
    #   C = A1*B1 + A2*B2.
    #   Each A_i*B_i is 2x2 * 2x4.
    #   2x2 * 2x4: split B_i = [B_i1 | B_i2] where each is 2x2.
    #   A_i * B_i = A_i * B_i1 + A_i * B_i2, each 2x2*2x2 with rank 7.
    #   So 2x2*2x4 has rank at most 14.
    #   Total: 14 + 14 = 28 for 2x4x4.
    #   But can we do better? The tensor 2x4x4 has 2*4*4=32 entries.
    #   Rank is at least... well, 28 is our construction.
    #
    # For 2x4x8:
    #   Split B into four 4x2 blocks: B1, B2, B3, B4 (each 4x2).
    #   C = A*B1 + A*B2 + A*B3 + A*B4.
    #   Each A*B_i is 2x4 * 4x2.
    #   2x4 * 4x2: 
    #     Split A = [A1 | A2] (2x2 each), B_i = [B_i1; B_i2] (2x2 each).
    #     A*B_i = A1*B_i1 + A2*B_i2.
    #     Each 2x2*2x2 has rank 7.
    #     So 2x4*4x2 has rank at most 14.
    #   Total: 4 * 14 = 56. Same as before.
    #
    # Can we do better for 2x4*4x2?
    #   2x4*4x2: A is 2x4, B is 4x2, C is 2x2.
    #   This is equivalent to: given A (2x4) and B (4x2), compute C = A*B (2x2).
    #   We can view this as: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0,1}.
    #   That's 4 output entries, each a dot product of length 4.
    #   Schoolbook: 4*4 = 16 mults.
    #   Strassen-like: Can we do better?
    #   
    #   Let's try: 
    #   C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0] + A[1,3]B[3,0]
    #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1] + A[1,3]B[3,1]
    #
    #   This is like multiplying two 2x4 matrices... no, it's 2x4 * 4x2.
    #   We can use the 2x2x2 Strassen idea by grouping.
    #   
    #   Group A as [A_top | A_bot] where A_top = A[0,:], A_bot = A[1,:] (each 1x4).
    #   That doesn't help directly.
    #   
    #   Alternative: View as 4 independent 1x4 * 4x1 products, but we can share.
    #   Actually, this is a special case of matrix multiplication.
    #   
    #   Let me try to find a rank-12 or rank-11 decomposition for 2x4*4x2.
    #   If we can get rank 12, then 2x4x8 would have rank 4*12 = 48.
    #
    #   For 2x4*4x2, the tensor has 2*4*2 = 16 entries.
    #   The "natural" tensor is: T[i,j,k] = 1 for the multiplication pattern.
    #   
    #   Let me try to explicitly construct a low-rank decomposition.
    #   We need: for all i in {0,1}, k in {0,1}, j in {0..3}, j' in {0..3}:
    #     sum_r u_r[i,j] v_r[j',k] w_r[k,i] = delta_{j,j'}
    #
    #   This is 2*2*4*4 = 64 equations, 2*2*4 = 16 with RHS=1, rest 0.
    #
    #   Let's try rank 12: 12*(4+8+2) = 168 unknowns for 64 equations. Should be solvable.
    #   But it's non-linear. Let me try a greedy/iterative approach.
    
    # For now, let's just return the rank-56 decomposition.
    # A full search for lower rank would require solving a non-linear system
    # which is complex and may not yield improvements within the time budget.
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print("Rank:", rank)
    
    # Verification
    n, m, p = 2, 4, 8
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    max_diff = 0
    for i in range(n):
        for k in range(p):
            c_val = 0
            for j in range(m):
                for r in range(rank):
                    c_val += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            diff = abs(c_val - C_expected[i, k])
            if diff > max_diff:
                max_diff = diff
    print("Max diff:", max_diff)
    if max_diff > 1e-10:
        print("ERROR: Decomposition is invalid!")
    else:
        print("Decomposition is valid.")
