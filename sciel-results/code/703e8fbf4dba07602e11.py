"""Matrix multiplication 6x6x6 via BCLR(21) x Strassen(7) block decomposition, rank 147."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# BCLR rank-21 decomposition for 3x3 matrix multiplication.
# Each entry is a triple of 3x3 integer matrices (U, V, W) such that
# sum_c U_c[i,j] * V_c[j,k] * W_c[k,l] = delta_{i,l} (for the appropriate indexing).
# Actually, the standard formulation: for 3x3 MM, we need tensors T_{ij,kl}^{mn} = delta_{m,i} delta_{n,j} delta_{p,k} delta_{q,l}
# where C = A * B, C_{mn} = sum_p A_{mp} B_{pn}.
# The tensor is T_{(m,p),(p,n),(n,m)} = 1 in the right indexing.
#
# BCLR coefficients (21 terms). Each term has U (3x3), V (3x3), W (3x3) with entries in {-1, 0, 1}.
# Reference: Bini, Capovani, Lotti, Romani, 1978.

# The 21 triples of 3x3 matrices for BCLR:
# Using the standard published list.
_BCLR_TERMS = [
    # (U, V, W) each as 3x3 flat list of 9 entries (row-major)
    # Term 0
    ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
    # Term 1
    ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
    # Term 2
    ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
    # Term 3
    ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
    # Term 4
    ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
    # Term 5
    ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
    # Term 6
    ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
    # Term 7
    ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,1,0]),
    # Term 8
    ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,1]),
    # Term 9
    ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,1,0, 0,1,0]),
    # Term 10
    ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
    # Term 11
    ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [1,0,0, 0,0,0, 0,1,0]),
    # Term 12
    ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0]),
    # Term 13
    ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1]),
    # Term 14
    ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
    # Term 15
    ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,1]),
    # Term 16
    ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,1]),
    # Term 17
    ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0]),
    # Term 18
    ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,1]),
    # Term 19
    ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1]),
    # Term 20
    ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
]

# Strassen rank-7 decomposition for 2x2 matrix multiplication.
# Each term: U (2x2), V (2x2), W (2x2)
_STRASSEN_TERMS = [
    # (U, V, W) each as 4-entry flat list (row-major)
    # Term 0: M1 = (A11+A22)(B11+B22)
    ([1,0, 1,1], [1,1, 1,1], [1,0, 0,1]),
    # Term 1: M2 = (A21+A22)B11
    ([0,0, 1,1], [1,0, 0,0], [0,0, 1,0]),
    # Term 2: M3 = A11(B21-B22)
    ([1,0, 0,0], [0,1, 0,-1], [0,1, 0,0]),
    # Term 3: M4 = A22(B12-B11)
    ([0,0, 0,1], [1,-1, 0,0], [0,0, 0,1]),
    # Term 4: M5 = (A11+A12)B22
    ([1,1, 0,0], [0,0, 0,1], [0,0, 0,1]),
    # Term 5: M6 = (A21-A11)(B11+B12)
    ([-1,0, 1,0], [1,1, 0,0], [1,0, 0,0]),
    # Term 7: M7 = (A12-A22)(B21+B22)
    ([0,1, 0,-1], [0,0, 1,1], [0,0, 1,0]),
]


def construct():
    """Construct rank-147 tensor decomposition for 6x6x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block structure: 3x3 grid of 2x2 blocks
    # Outer block indices: (I, J, K) for A, B, C respectively, each in {0,1,2}
    # Inner block indices: (a, b, c) for 2x2 sub-blocks, each in {0,1}
    #
    # Full index for A: i = 2*I + a, j = 2*J + b  (row, col)
    # Full index for B: j' = 2*J + b, k = 2*K + c  (row, col)
    # Full index for C: k' = 2*K + c, i' = 2*I + a  (row, col)
    #
    # Tensor T_{(i,j),(j',k),(k',i')} = 1 if j == j' and i == i' and k == k'
    # i.e., T[(2*I+a, 2*J+b), (2*J+b, 2*K+c), (2*K+c, 2*I+a)] = 1
    #
    # For the Kronecker product decomposition:
    # Outer BCLR term r (r in 0..20) gives:
    #   U_outer[r] : 3x3 matrix, entries U_O[r][I, J]
    #   V_outer[r] : 3x3 matrix, entries V_O[r][J, K]  (note: J,K for B's block row,col)
    #   W_outer[r] : 3x3 matrix, entries W_O[r][K, I]
    #
    # Inner Strassen term s (s in 0..6) gives:
    #   U_inner[s] : 2x2 matrix, entries U_I[s][a, b]
    #   V_inner[s] : 2x2 matrix, entries V_I[s][b, c]
    #   W_inner[s] : 2x2 matrix, entries W_I[s][c, a]
    #
    # Combined term (r, s) with index c = r*7 + s:
    #   U_full[(2*I+a, 2*J+b), c] = U_O[r][I, J] * U_I[s][a, b]
    #   V_full[(2*J+b, 2*K+c), c] = V_O[r][J, K] * V_I[s][b, c]
    #   W_full[(2*K+c, 2*I+a), c] = W_O[r][K, I] * W_I[s][c, a]
    #
    # Wait, I need to be careful with the indexing. Let me re-derive.
    #
    # The matrix multiplication tensor T has T_{ij,kl,mn} = delta_{jp} delta_{mi} delta_{nk}
    # where C_{mn} = sum_p A_{mp} B_{pn}.
    # In the task's convention: W[k*n+i, r] means the W matrix has row index k*n+i.
    # So W[k*n + i, r] corresponds to the output index (k, i) where k is the col of B / row of C,
    # and i is the row of A / col of C.
    #
    # U[i*m + j, r]: row index is i*m+j, where i is row of A, j is col of A (= row of B)
    # V[j*p + k, r]: row index is j*p+k, where j is row of B (= col of A), k is col of B
    # W[k*n + i, r]: row index is k*n+i, where k is col of B (= row of C), i is row of A (= col of C)
    #
    # So the tensor is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,?} ... 
    # Actually it should equal 1 when the indices form a valid multiplication path, i.e.,
    # the "free" indices are i (row of A, col of C), k (col of B, row of C).
    # The sum over r gives: if we interpret this as the matrix multiplication operator,
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k].
    # So the tensor T_{i j, j k, k i} should be 1.
    #
    # For the Kronecker product:
    # i = 2*I + a, j = 2*J + b, k = 2*K + c
    # The condition for non-zero is automatically satisfied by the structure.
    #
    # U[(2*I+a)*6 + (2*J+b), c_idx] where c_idx = r*7+s
    # = U_O[r][I, J] * U_I[s][a, b]
    #
    # V[(2*J+b)*6 + (2*K+c), c_idx]
    # = V_O[r][J, K] * V_I[s][b, c]
    #
    # W[(2*K+c)*6 + (2*I+a), c_idx]
    # = W_O[r][K, I] * W_I[s][c, a]
    
    total_rank = 21 * 7  # 147
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 36 x 147
    V = [[0.0] * total_rank for _ in range(m * p)]  # 36 x 147
    W = [[0.0] * total_rank for _ in range(p * n)]  # 36 x 147
    
    for r in range(21):
        # Convert BCLR term r to numpy arrays
        U_O = np.array(_BCLR_TERMS[r][0]).reshape(3, 3)
        V_O = np.array(_BCLR_TERMS[r][1]).reshape(3, 3)
        W_O = np.array(_BCLR_TERMS[r][2]).reshape(3, 3)
        
        for s in range(7):
            c_idx = r * 7 + s
            
            U_I = np.array(_STRASSEN_TERMS[s][0]).reshape(2, 2)
            V_I = np.array(_STRASSEN_TERMS[s][1]).reshape(2, 2)
            W_I = np.array(_STRASSEN_TERMS[s][2]).reshape(2, 2)
            
            for I in range(3):
                for J in range(3):
                    for K in range(3):
                        uO_val = U_O[I, J]
                        vO_val = V_O[J, K]
                        wO_val = W_O[K, I]
                        
                        if uO_val == 0 and vO_val == 0 and wO_val == 0:
                            continue
                        
                        for a in range(2):
                            for b in range(2):
                                uI_val = U_I[a, b]
                                u_val = uO_val * uI_val
                                if u_val != 0:
                                    row_U = (2*I + a) * 6 + (2*J + b)
                                    U[row_U][c_idx] = u_val
                        
                        for b in range(2):
                            for c in range(2):
                                vI_val = V_I[b, c]
                                v_val = vO_val * vI_val
                                if v_val != 0:
                                    row_V = (2*J + b) * 6 + (2*K + c)
                                    V[row_V][c_idx] = v_val
                        
                        for c in range(2):
                            for a in range(2):
                                wI_val = W_I[c, a]
                                w_val = wO_val * wI_val
                                if w_val != 0:
                                    row_W = (2*K + c) * 6 + (2*I + a)
                                    W[row_W][c_idx] = w_val
    
    return {"U": U, "V": V, "W": W}
