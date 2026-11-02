"""
Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.

Strategy:
1. Start from the schoolbook decomposition (rank 48).
2. Use Strassen's identity on 2x2 blocks to reduce rank.
   - Partition the 4x4 matrix B into four 2x2 blocks.
   - Partition A (3x4) into a top 2x4 block and a bottom 1x4 block.
   - The top 2x4 * 4x4 part can be computed using Strassen-like reductions
     on overlapping 2x2 sub-blocks.
   - The bottom 1x4 * 4x4 part is just a row vector times a matrix, which
     can be done with 4 multiplications (schoolbook for 1x4 * 4x4).
3. Additionally, try to find further rank reductions by exploiting the
   specific structure of the 3x4x4 tensor.

The key insight is that the 2x2x2 tensor has rank 7 (Strassen) instead of 8.
We can embed this into the 3x4x4 problem.

For the 3x4x4 case:
- A is 3x4, B is 4x4, C = A*B is 3x4.
- We can partition A into A_top (2x4) and A_bottom (1x4).
- We can partition B into 2x2 blocks.
- The top 2x4 * 4x4 multiplication can use Strassen's method on 2x2 blocks.
- The bottom 1x4 * 4x4 is just 16 scalar products.

Using Strassen's method on the 2x2x2 sub-tensors within the 2x4x4 part:
- The 2x4x4 tensor can be decomposed into two 2x2x4 tensors (left and right halves of B).
- Each 2x2x4 can be further decomposed using Strassen-like identities.

Actually, let's use a more direct approach:
- The 2x2x2 tensor has rank 7.
- We can embed this into the 3x4x4 problem by considering overlapping 2x2x2 sub-tensors.

Let me construct a rank-47 decomposition using Strassen's method on one 2x2x2 block.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    rank0 = n * m * p  # 48
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, rank0), dtype=float)
    V = np.zeros((m * p, rank0), dtype=float)
    W = np.zeros((p * n, rank0), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank0
    
    # Helper to verify exactness
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Helper to check if entries are integers or half-integers
    def is_valid_entries(U_mat, V_mat, W_mat):
        for mat in [U_mat, V_mat, W_mat]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Apply Strassen's method to reduce rank by 1
    # We'll replace 8 schoolbook terms for a 2x2x2 sub-tensor with 7 Strassen terms.
    
    # Consider the sub-tensor for i in {0,1}, j in {0,1}, k in {0,1}
    # This corresponds to 8 schoolbook terms: (i,j,k) for i in {0,1}, j in {0,1}, k in {0,1}
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11+A22)(B11+B22) -> u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,1) in appropriate basis
    # M2 = (A21+A22)B11 -> u=(0,0,1,1), v=(1,0,0,0), w=(0,1,0,0)
    # M3 = A11(B12-B22) -> u=(1,0,0,0), v=(0,1,0,-1), w=(0,0,1,0)
    # M4 = A22(B21-B11) -> u=(0,0,0,1), v=(0,0,1,-1), w=(0,1,0,0)
    # M5 = (A11+A12)B22 -> u=(1,1,0,0), v=(0,0,0,1), w=(0,0,0,1)
    # M6 = (A21-A11)(B11+B12) -> u=(0,0,1,-1), v=(1,1,0,0), w=(0,1,0,0)
    # M7 = (A12-A22)(B21+B22) -> u=(0,1,0,-1), v=(0,0,1,1), w=(0,0,1,0)
    
    # Wait, I need to be more careful with the indexing.
    # Let's use a different approach: directly construct the Strassen decomposition.
    
    # For the 2x2x2 tensor, Strassen's decomposition has 7 terms.
    # Each term c has vectors u_c, v_c, w_c such that:
    # T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # Let me define the 7 Strassen terms for 2x2x2:
    # The 2x2x2 tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1}
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M, we need to express it as u * v * w where:
    # u[i,j] is the coefficient for A[i,j]
    # v[j,k] is the coefficient for B[j,k]
    # w[k,i] is the coefficient for C[k,i] (note: output index order)
    
    # Let's define the 7 terms:
    strassen_terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # u: A00=1, A11=1 -> u[0,0]=1, u[1,1]=1
    # v: B00=1, B11=1 -> v[0,0]=1, v[1,1]=1
    # w: C00=1, C11=1 -> w[0,0]=1, w[1,1]=1
    strassen_terms.append({
        'u': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # u[i,j]
        'v': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # v[j,k]
        'w': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}}   # w[k,i]
    })
    
    # M2 = (A10+A11)B00
    # u: A10=1, A11=1 -> u[1,0]=1, u[1,1]=1
    # v: B00=1 -> v[0,0]=1
    # w: C10=1 -> w[0,1]=1
    strassen_terms.append({
        'u': {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},
        'v': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}},
        'w': {0: {0: 0, 1: 1}, 1: {0: 0, 1: 0}}
    })
    
    # M3 = A00(B01-B11)
    # u: A00=1 -> u[0,0]=1
    # v: B01=1, B11=-1 -> v[0,1]=1, v[1,1]=-1
    # w: C01=1 -> w[1,0]=1
    strassen_terms.append({
        'u': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}},
        'v': {0: {0: 0, 1: 1}, 1: {0: 0, 1: -1}},
        'w': {0: {0: 0, 1: 0}, 1: {0: 1, 1: 0}}
    })
    
    # M4 = A11(B10-B00)
    # u: A11=1 -> u[1,1]=1
    # v: B10=1, B00=-1 -> v[1,0]=1, v[0,0]=-1
    # w: C10=1, C00=1 -> w[0,1]=1, w[0,0]=1
    strassen_terms.append({
        'u': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},
        'v': {0: {0: -1, 1: 0}, 1: {0: 1, 1: 0}},
        'w': {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}}
    })
    
    # M5 = (A00+A01)B11
    # u: A00=1, A01=1 -> u[0,0]=1, u[0,1]=1
    # v: B11=1 -> v[1,1]=1
    # w: C00=-1, C11=1 -> w[0,0]=-1, w[1,1]=1
    strassen_terms.append({
        'u': {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}},
        'v': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},
        'w': {0: {0: -1, 1: 0}, 1: {0: 0, 1: 1}}
    })
    
    # M6 = (A10-A00)(B00+B01)
    # u: A10=1, A00=-1 -> u[1,0]=1, u[0,0]=-1
    # v: B00=1, B01=1 -> v[0,0]=1, v[0,1]=1
    # w: C11=1 -> w[1,1]=1
    strassen_terms.append({
        'u': {0: {0: -1, 1: 0}, 1: {0: 1, 1: 0}},
        'v': {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}},
        'w': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}}
    })
    
    # M7 = (A01-A11)(B10+B11)
    # u: A01=1, A11=-1 -> u[0,1]=1, u[1,1]=-1
    # v: B10=1, B11=1 -> v[1,0]=1, v[1,1]=1
    # w: C00=1 -> w[0,0]=1
    strassen_terms.append({
        'u': {0: {0: 0, 1: 1}, 1: {0: 0, 1: -1}},
        'v': {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},
        'w': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}}
    })
    
    # Now let's verify Strassen's decomposition for 2x2x2
    def verify_strassen_2x2x2(terms, tol=1e-9):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = 0
                    for term in terms:
                        u_val = term['u'][i][j]
                        v_val = term['v'][j][k]
                        w_val = term['w'][k][i]
                        val += u_val * v_val * w_val
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Verify
    if not verify_strassen_2x2x2(strassen_terms):
        print("Strassen verification failed!")
    
    # Now embed into 3x4x4
    # We'll use the first 8 schoolbook terms for (i in {0,1}, j in {0,1}, k in {0,1})
    # and replace them with 7 Strassen terms.
    
    # The 8 schoolbook terms are at columns 0-7 (i=0,j=0,k=0), (i=0,j=0,k=1), etc.
    # Let's identify which columns correspond to (i in {0,1}, j in {0,1}, k in {0,1})
    
    schoolbook_cols = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                col_idx = i * m * p + j * p + k
                schoolbook_cols.append(col_idx)
    
    # Create new decomposition with rank 47
    new_rank = rank0 - 1
    new_U = np.zeros((n * m, new_rank), dtype=float)
    new_V = np.zeros((m * p, new_rank), dtype=float)
    new_W = np.zeros((p * n, new_rank), dtype=float)
    
    # Copy all schoolbook terms except the 8 we're replacing
    new_col = 0
    for col_idx in range(rank0):
        if col_idx not in schoolbook_cols:
            new_U[:, new_col] = U[:, col_idx]
            new_V[:, new_col] = V[:, col_idx]
            new_W[:, new_col] = W[:, col_idx]
            new_col += 1
    
    # Add the 7 Strassen terms
    for term in strassen_terms:
        for i in range(2):
            for j in range(2):
                if term['u'][i][j] != 0:
                    new_U[i * m + j, new_col] += term['u'][i][j]
        for j in range(2):
            for k in range(2):
                if term['v'][j][k] != 0:
                    new_V[j * p + k, new_col] += term['v'][j][k]
        for k in range(2):
            for i in range(2):
                if term['w'][k][i] != 0:
                    new_W[k * n + i, new_col] += term['w'][k][i]
        new_col += 1
    
    # Verify the new decomposition
    if verify(new_U, new_V, new_W) and is_valid_entries(new_U, new_V, new_W):
        best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
        best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
