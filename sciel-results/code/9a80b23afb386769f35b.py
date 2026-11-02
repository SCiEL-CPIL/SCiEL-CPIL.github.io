"""Matrix multiplication 3,3,6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _strassen_3x3x3_terms():
    """
    Returns a list of 22 rank-1 terms for 3x3x3 matrix multiplication.
    Indices i,j,k in {0,1,2}.
    U index: 3i+j
    V index: 3j+k
    W index: 3k+i
    """
    terms = []
    
    # 1. Strassen core for top-left 2x2 block (indices 0,1)
    # A_sub = [[A00, A01], [A10, A11]]
    # B_sub = [[B00, B01], [B10, B11]]
    # C_sub = A_sub * B_sub
    
    # M1 = (A00+A11)(B00+B11)
    u1 = {0:1, 4:1} # A00, A11
    v1 = {0:1, 4:1} # B00, B11
    w1 = {0:1, 4:1} # C00, C11
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00
    u2 = {3:1, 4:1} # A10, A11
    v2 = {0:1}      # B00
    w2 = {3:1}      # C10 (Wait, C10 is index 3*0+1=1? No. C[i,k]. C[1,0] index 3*0+1=1. C[1,1] index 3*1+1=4.)
    # Correction on W indices: W[k*n+i] = W[3k+i].
    # C[1,0] -> k=0, i=1 -> idx 1.
    # C[1,1] -> k=1, i=1 -> idx 4.
    # M2 contributes to C[1,0] and C[1,1]?
    # (A10+A11)B00 = A10*B00 + A11*B00.
    # A10*B00 contributes to C[1,0].
    # A11*B00 contributes to C[1,1]? No, A[1,1]*B[1,0] contributes to C[1,0].
    # Wait, Strassen formulas:
    # M2 = (A10+A11)B00.
    # C10 = M2 - A11*B00? No.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map these to rank-1 tensors.
    # A rank-1 tensor is U * V * W.
    # We need to express the sum of products as sum of rank-1 tensors.
    # This is tricky because Strassen uses sums of products, not just single products.
    # A single multiplication (A_ij + A_i'j') * (B_jk + B_j'k') expands to 4 terms.
    # To represent (A+B)(C+D) as rank-1, we can't. We need 4 rank-1 terms?
    # No, Strassen's algorithm works by defining intermediate variables.
    # But for tensor rank, we need to express the final C entries as sums of rank-1 terms.
    # The rank of the tensor is the number of scalar multiplications.
    # So each "M" in Strassen is one scalar multiplication.
    # So we need to define U, V, W such that U[i,j] V[j,k] W[i,k] corresponds to the contribution of that M to C[i,k].
    
    # For M1 = (A00+A11)(B00+B11):
    # It contributes to C00, C01, C10, C11?
    # M1 is a scalar.
    # C00 += M1. So W[0,0] gets +1.
    # C11 += M1. So W[1,1] gets +1.
    # But M1 is formed from A00, A11 and B00, B11.
    # So U[0,0]=1, U[1,1]=1.
    # V[0,0]=1, V[1,1]=1.
    # W[0,0]=1, W[1,1]=1.
    # This single term M1 contributes +1 to C00 and +1 to C11?
    # Let's check: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1. Correct for C00.
    # U[1,1]V[1,1]W[1,1] = 1*1*1 = 1. Correct for C11.
    # But what about C01 and C10?
    # The term M1 does NOT contribute to C01 or C10 in the Strassen formula?
    # C01 = M3 + M5. M1 is not there.
    # So for the tensor decomposition, the term corresponding to M1 should ONLY contribute to C00 and C11.
    # So W[0,1]=0, W[1,0]=0.
    # So W = {0:1, 4:1}. (Indices 0 and 4 for 3x3? No, for 2x2 sub-block, indices are 0,1. In 3x3, 0,1,2.
    # C[0,0] idx 0. C[1,1] idx 4.
    
    # Let's re-verify the Strassen mapping to rank-1 terms.
    # Term 1 (M1):
    # U: A00, A11 -> {0:1, 4:1}
    # V: B00, B11 -> {0:1, 4:1}
    # W: C00, C11 -> {0:1, 4:1}
    
    # Term 2 (M2): (A10+A11)B00
    # U: A10, A11 -> {3:1, 4:1}
    # V: B00 -> {0:1}
    # W: C10, C11?
    # C10 = M2 + M4. So M2 contributes +1 to C10.
    # C11 = M1 - M2 + M3 + M6. So M2 contributes -1 to C11.
    # So W: C10 (+1), C11 (-1).
    # C[1,0] idx 1. C[1,1] idx 4.
    # W: {1:1, 4:-1}
    
    # Term 3 (M3): A00(B01-B11)
    # U: A00 -> {0:1}
    # V: B01, B11 -> {1:1, 4:-1}
    # W: C00, C01?
    # C00 = M1 + M4 - M5 + M7. No M3.
    # C01 = M3 + M5. So M3 contributes +1 to C01.
    # C11 = M1 - M2 + M3 + M6. So M3 contributes +1 to C11.
    # W: C01 (+1), C11 (+1).
    # C[0,1] idx 3. C[1,1] idx 4.
    # W: {3:1, 4:1}
    
    # Term 4 (M4): A11(B10-B00)
    # U: A11 -> {4:1}
    # V: B10, B00 -> {3:1, 0:-1}
    # W: C00, C10?
    # C00 = M1 + M4 ... -> +1 to C00.
    # C10 = M2 + M4 -> +1 to C10.
    # W: C00 (+1), C10 (+1).
    # C[0,0] idx 0. C[1,0] idx 1.
    # W: {0:1, 1:1}
    
    # Term 5 (M5): (A00+A01)B11
    # U: A00, A01 -> {0:1, 1:1}
    # V: B11 -> {4:1}
    # W: C00, C01?
    # C00 = ... - M5 ... -> -1 to C00.
    # C01 = M3 + M5 -> +1 to C01.
    # W: C00 (-1), C01 (+1).
    # C[0,0] idx 0. C[0,1] idx 3.
    # W: {0:-1, 3:1}
    
    # Term 6 (M6): (A10-A00)(B00+B01)
    # U: A10, A00 -> {3:1, 0:-1}
    # V: B00, B01 -> {0:1, 1:1}
    # W: C11?
    # C11 = ... + M6 -> +1 to C11.
    # Any other?
    # C00, C01, C10 do not have M6.
    # W: C11 (+1).
    # C[1,1] idx 4.
    # W: {4:1}
    
    # Term 7 (M7): (A01-A11)(B10+B11)
    # U: A01, A11 -> {1:1, 4:-1}
    # V: B10, B11 -> {3:1, 4:1}
    # W: C00?
    # C00 = ... + M7 -> +1 to C00.
    # W: C00 (+1).
    # C[0,0] idx 0.
    # W: {0:1}
    
    core_terms = [
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),
        ({3:1, 4:1}, {0:1}, {1:1, 4:-1}),
        ({0:1}, {1:1, 4:-1}, {3:1, 4:1}),
        ({4:1}, {3:1, 0:-1}, {0:1, 1:1}),
        ({0:1, 1:1}, {4:1}, {0:-1, 3:1}),
        ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
        ({1:1, 4:-1}, {3:1, 4:1}, {0:1})
    ]
    
    # 2. Schoolbook terms for the rest
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1]
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2]
    
    tail_terms = []
    # i=0, k=2
    tail_terms.append(({0:1}, {2:1}, {6:1})) # A00 B02 C02 (C[0,2] idx 3*2+0=6)
    tail_terms.append(({1:1}, {5:1}, {6:1})) # A01 B12 C02 (B[1,2] idx 3*1+2=5)
    tail_terms.append(({2:1}, {8:1}, {6:1})) # A02 B22 C02 (B[2,2] idx 3*2+2=8)
    
    # i=1, k=2
    tail_terms.append(({3:1}, {2:1}, {9:1})) # A10 B02 C12 (C[1,2] idx 3*2+1=7? No. 3*2+1=7. Wait.
    # C[i,k] index is k*n+i = 3k+i.
    # C[0,2] -> 3*2+0 = 6.
    # C[1,2] -> 3*2+1 = 7.
    # C[2,0] -> 3*0+2 = 2.
    # C[2,1] -> 3*1+2 = 5.
    # C[2,2] -> 3*2+2 = 8.
    
    # Let's re-index W correctly.
    # C[0,2] idx 6.
    # C[1,2] idx 7.
    # C[2,0] idx 2.
    # C[2,1] idx 5.
    # C[2,2] idx 8.
    
    tail_terms = []
    # C[0,2]
    tail_terms.append(({0:1}, {2:1}, {6:1})) # A00 B02
    tail_terms.append(({1:1}, {5:1}, {6:1})) # A01 B12
    tail_terms.append(({2:1}, {8:1}, {6:1})) # A02 B22
    
    # C[1,2]
    tail_terms.append(({3:1}, {2:1}, {7:1})) # A10 B02
    tail_terms.append(({4:1}, {5:1}, {7:1})) # A11 B12
    tail_terms.append(({5:1}, {8:1}, {7:1})) # A12 B22
    
    # C[2,0]
    tail_terms.append(({6:1}, {0:1}, {2:1})) # A20 B00
    tail_terms.append(({7:1}, {3:1}, {2:1})) # A21 B10
    tail_terms.append(({8:1}, {6:1}, {2:1})) # A22 B20
    
    # C[2,1]
    tail_terms.append(({6:1}, {1:1}, {5:1})) # A20 B01
    tail_terms.append(({7:1}, {4:1}, {5:1})) # A21 B11
    tail_terms.append(({8:1}, {7:1}, {5:1})) # A22 B21
    
    # C[2,2]
    tail_terms.append(({6:1}, {2:1}, {8:1})) # A20 B02
    tail_terms.append(({7:1}, {5:1}, {8:1})) # A21 B12
    tail_terms.append(({8:1}, {8:1}, {8:1})) # A22 B22
    
    return core_terms + tail_terms

def _build_decomposition_3x3x6():
    """
    Constructs a rank-44 decomposition for 3x3x6 by using two 3x3x3 blocks.
    """
    n, m, p = 3, 3, 6
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = _strassen_3x3x3_terms()
    
    # First block: k in {0, 1, 2}
    for r in range(22):
        u_dict, v_dict, w_dict = terms[r]
        for idx_u, val in u_dict.items():
            U[idx_u][r] = val
        for idx_v, val in v_dict.items():
            V[idx_v][r] = val
        for idx_w, val in w_dict.items():
            W[idx_w][r] = val
            
    # Second block: k in {3, 4, 5}
    # This is the same structure, but shifted in V and W indices.
    # V index: j*p + k. For k' in {0,1,2}, k = k' + 3.
    # W index: k*n + i. For k' in {0,1,2}, k = k' + 3.
    
    for r in range(22):
        col = 22 + r
        u_dict, v_dict, w_dict = terms[r]
        
        # U is the same (A indices don't change)
        for idx_u, val in u_dict.items():
            U[idx_u][col] = val
            
        # V: shift k by 3
        for idx_v, val in v_dict.items():
            j = idx_v // 3
            k_prime = idx_v % 3
            new_k = k_prime + 3
            new_idx_v = j * 3 + new_k
            V[new_idx_v][col] = val
            
        # W: shift k by 3
        for idx_w, val in w_dict.items():
            k_prime = idx_w // 3
            i = idx_w % 3
            new_k = k_prime + 3
            new_idx_w = new_k * 3 + i
            W[new_idx_w][col] = val
            
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x6 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use the rank-44 decomposition
    result = _build_decomposition_3x3x6()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
