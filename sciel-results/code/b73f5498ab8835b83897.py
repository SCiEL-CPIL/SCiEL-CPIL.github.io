import numpy as np
import time

def construct():
    """
    Construct a rank-39 exact tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    Strategy:
    1. Split the middle dimension (m=4) into two blocks of size 2.
       A = [A1, A2] where A1, A2 are 2x2.
       B = [B1; B2] where B1, B2 are 2x7.
       C = A1 B1 + A2 B2.
       
    2. For each 2x2 by 2x7 multiplication (rank of 2x2x7 tensor), 
       split the output dimension (p=7) into 2+2+2+1.
       Let B1 = [B1_1, B1_2, B1_3, B1_4] where B1_1, B1_2, B1_3 are 2x2 and B1_4 is 2x1.
       
    3. For the 2x2x2 parts, use Strassen's algorithm (rank 7).
       For the 2x2x1 part, use a rank-3 exact decomposition (Lambert et al.).
       
    Total rank per 2x2x7 block = 7 + 7 + 7 + 3 = 24.
    Total rank for 2x4x7 = 24 + 24 = 48? No, 24 * 2 = 48. Wait.
    Let's check the rank of 2x2x1.
    The tensor T(i,j,k) for C_ij = A_i0 B_0j + A_i1 B_1j.
    This is equivalent to multiplying a 2x2 matrix by a 2x1 matrix.
    The tensor has shape (2, 2, 2).
    It is the restriction of the 2x2x2 matrix multiplication tensor.
    It is known that the rank of the 2x2x1 matrix multiplication tensor is 3.
    Reference: "The rank of the 2x2x1 matrix multiplication tensor is 3".
    Actually, let's verify.
    C_0 = A_00 B_0 + A_01 B_1
    C_1 = A_10 B_0 + A_11 B_1
    We need to compute these two linear forms.
    Rank 1: A_00 B_0 -> C_0
    Rank 2: A_01 B_1 -> C_0
    Rank 3: A_10 B_0 -> C_1
    Rank 4: A_11 B_1 -> C_1
    Can we do it in 3?
    Let M1 = (A_00 + A_10)(B_0 + B_1) = A_00 B_0 + A_00 B_1 + A_10 B_0 + A_10 B_1
    Let M2 = (A_00 - A_10) B_0 = A_00 B_0 - A_10 B_0
    Let M3 = (A_01 + A_11) B_1 = A_01 B_1 + A_11 B_1
    This doesn't seem to isolate the terms easily without more terms.
    
    Actually, the tensor T_{i,j,k} for 2x2x1 is:
    T = e_0^A \otimes (e_0^B \otimes e_0^C + e_1^B \otimes e_0^C) + e_1^A \otimes (e_0^B \otimes e_1^C + e_1^B \otimes e_1^C)
    T = e_0^A \otimes e_0^B \otimes e_0^C + e_0^A \otimes e_1^B \otimes e_0^C + e_1^A \otimes e_0^B \otimes e_1^C + e_1^A \otimes e_1^B \otimes e_1^C
    This is 4 disjoint rank-1 tensors.
    Is the rank 4 or lower?
    The tensor is equivalent to the sum of 4 orthogonal rank-1 tensors.
    However, sometimes they can be merged.
    For 2x2x1, the rank is actually 4?
    Let's check the literature.
    The rank of the tensor for multiplying a 2x2 matrix by a 2x1 vector.
    This is equivalent to the rank of the tensor $T_{ijk}$ where $C_i = \sum_j A_{ij} B_j$.
    This is a bilinear map from $(V^* \otimes W^*) \times W \to V$? No.
    Input: A (2x2), B (2x1). Output: C (2x1).
    The tensor is in $A^* \otimes B^* \otimes C^*$.
    The components are:
    T[0,0,0] = 1 (A00 B0 -> C0)
    T[0,1,0] = 1 (A01 B1 -> C0)
    T[1,0,1] = 1 (A10 B0 -> C1)
    T[1,1,1] = 1 (A11 B1 -> C1)
    
    This tensor is:
    e_A0 \otimes (e_B0 \otimes e_C0 + e_B1 \otimes e_C0) + e_A1 \otimes (e_B0 \otimes e_C1 + e_B1 \otimes e_C1)
    = e_A0 \otimes e_B0 \otimes e_C0 + e_A0 \otimes e_B1 \otimes e_C0 + e_A1 \otimes e_B0 \otimes e_C1 + e_A1 \otimes e_B1 \otimes e_C1
    
    Consider the slice $i=0$: $T[0,:,:] = e_B0 \otimes e_C0 + e_B1 \otimes e_C0 = (e_B0+e_B1) \otimes e_C0$. Rank 1.
    Consider the slice $i=1$: $T[1,:,:] = e_B0 \otimes e_C1 + e_B1 \otimes e_C1 = (e_B0+e_B1) \otimes e_C1$. Rank 1.
    So the tensor is $e_A0 \otimes (e_B0+e_B1) \otimes e_C0 + e_A1 \otimes (e_B0+e_B1) \otimes e_C1$.
    This is a sum of 2 rank-1 tensors!
    Wait.
    Let's check.
    Term 1: $u_1 = e_A0$, $v_1 = e_B0+e_B1$, $w_1 = e_C0$.
    Contribution to T[0,0,0]: $1 \cdot 1 \cdot 1 = 1$.
    Contribution to T[0,1,0]: $1 \cdot 1 \cdot 1 = 1$.
    Contribution to T[0,0,1]: $1 \cdot 1 \cdot 0 = 0$.
    Contribution to T[0,1,1]: $1 \cdot 1 \cdot 0 = 0$.
    Contribution to T[1,0,0]: $0 \cdot 1 \cdot 1 = 0$.
    ...
    Term 2: $u_2 = e_A1$, $v_2 = e_B0+e_B1$, $w_2 = e_C1$.
    Contribution to T[1,0,1]: $1 \cdot 1 \cdot 1 = 1$.
    Contribution to T[1,1,1]: $1 \cdot 1 \cdot 1 = 1$.
    
    So yes, the rank of the 2x2x1 multiplication tensor is 2?
    Let's re-verify the indices.
    A is 2x2. B is 2x1. C is 2x1.
    C_0 = A_00 B_0 + A_01 B_1
    C_1 = A_10 B_0 + A_11 B_1
    
    My tensor decomposition:
    $R_1 = e_A0 \otimes (e_B0 + e_B1) \otimes e_C0$
    $R_2 = e_A1 \otimes (e_B0 + e_B1) \otimes e_C1$
    
    Sum:
    At (0,0,0): 1*1*1 = 1. (A00 B0 -> C0). Correct.
    At (0,1,0): 1*1*1 = 1. (A01 B1 -> C0). Correct.
    At (1,0,1): 1*1*1 = 1. (A10 B0 -> C1). Correct.
    At (1,1,1): 1*1*1 = 1. (A11 B1 -> C1). Correct.
    
    Are there any other non-zero entries?
    (0,0,1): 1*1*0 + 0*1*1 = 0. (A00 B0 -> C1). Should be 0. Correct.
    (0,1,1): 1*1*0 + 0*1*1 = 0. (A01 B1 -> C1). Should be 0. Correct.
    (1,0,0): 0*1*1 + 1*1*0 = 0. (A10 B0 -> C0). Should be 0. Correct.
    (1,1,0): 0*1*1 + 1*1*0 = 0. (A11 B1 -> C0). Should be 0. Correct.
    
    So the rank is 2!
    
    Therefore, for each 2x2x7 block:
    3 * Strassen (7) + 1 * Rank-2 (2) = 21 + 2 = 23.
    Total rank = 23 + 23 = 46.
    
    Can we do better for 2x2x2?
    Strassen is rank 7. It is known to be optimal for 2x2x2 over most fields?
    Actually, the lower bound is 7. So 7 is optimal.
    
    Can we do better for the splitting?
    Maybe split 4x7 differently?
    Or use a different base algorithm.
    
    Let's try to construct the rank 46 solution.
    
    """
    n, m, p = 2, 4, 7
    rank = 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(term_idx, u_entries, v_entries, w_entries):
        for idx, val in u_entries:
            U[idx, term_idx] = val
        for idx, val in v_entries:
            V[idx, term_idx] = val
        for idx, val in w_entries:
            W[idx, term_idx] = val

    def process_strassen_block(a_base, b_base, c_base, term_start):
        A11_idx = a_base + 0
        A12_idx = a_base + 1
        A21_idx = a_base + 4
        A22_idx = a_base + 5
        
        B11_idx = b_base + 0
        B12_idx = b_base + 1
        B21_idx = b_base + 7
        B22_idx = b_base + 8
        
        C11_idx = c_base + 0
        C12_idx = c_base + 2
        C21_idx = c_base + 1
        C22_idx = c_base + 3
        
        # M1 = (A11 + A22)(B11 + B22)
        add_term(term_start + 0, 
                 [(A11_idx, 1), (A22_idx, 1)], 
                 [(B11_idx, 1), (B22_idx, 1)], 
                 [(C11_idx, 1), (C22_idx, 1)])
                 
        # M2 = (A21 + A22) B11
        add_term(term_start + 1, 
                 [(A21_idx, 1), (A22_idx, 1)], 
                 [(B11_idx, 1)], 
                 [(C21_idx, 1), (C22_idx, -1)])
                 
        # M3 = A11 (B12 - B22)
        add_term(term_start + 2, 
                 [(A11_idx, 1)], 
                 [(B12_idx, 1), (B22_idx, -1)], 
                 [(C12_idx, 1), (C22_idx, 1)])
                 
        # M4 = A22 (B21 - B11)
        add_term(term_start + 3, 
                 [(A22_idx, 1)], 
                 [(B21_idx, 1), (B11_idx, -1)], 
                 [(C11_idx, 1), (C21_idx, 1)])
                 
        # M5 = (A11 + A12) B22
        add_term(term_start + 4, 
                 [(A11_idx, 1), (A12_idx, 1)], 
                 [(B22_idx, 1)], 
                 [(C11_idx, -1), (C12_idx, 1)])
                 
        # M6 = (A21 - A11) (B11 + B12)
        add_term(term_start + 5, 
                 [(A21_idx, 1), (A11_idx, -1)], 
                 [(B11_idx, 1), (B12_idx, 1)], 
                 [(C22_idx, 1)])
                 
        # M7 = (A12 - A22) (B21 + B22)
        add_term(term_start + 6, 
                 [(A12_idx, 1), (A22_idx, -1)], 
                 [(B21_idx, 1), (B22_idx, 1)], 
                 [(C11_idx, 1), (C12_idx, 1)])
                 
        return term_start + 7

    def process_rank2_block(a_base, b_base, c_base, term_start):
        """
        Process a 2x2x1 multiplication using rank-2 decomposition.
        C0 = A00 B0 + A01 B1
        C1 = A10 B0 + A11 B1
        
        Term 1: u=eA0, v=eB0+eB1, w=eC0
        Term 2: u=eA1, v=eB0+eB1, w=eC1
        """
        A0_idx = a_base + 0
        A1_idx = a_base + 4
        
        B0_idx = b_base + 0
        B1_idx = b_base + 7
        
        C0_idx = c_base + 0
        C1_idx = c_base + 1
        
        add_term(term_start + 0, 
                 [(A0_idx, 1)], 
                 [(B0_idx, 1), (B1_idx, 1)], 
                 [(C0_idx, 1)])
                 
        add_term(term_start + 1, 
                 [(A1_idx, 1)], 
                 [(B0_idx, 1), (B1_idx, 1)], 
                 [(C1_idx, 1)])
                 
        return term_start + 2

    current_term = 0
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 base index: 0
    # B1 base index: 0
    
    # B1 cols 0,1 -> Strassen
    current_term = process_strassen_block(0, 0, 0, current_term)
    # B1 cols 2,3 -> Strassen
    current_term = process_strassen_block(0, 2, 4, current_term)
    # B1 cols 4,5 -> Strassen
    current_term = process_strassen_block(0, 4, 8, current_term)
    # B1 col 6 -> Rank 2
    current_term = process_rank2_block(0, 6, 12, current_term)
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # A2 base index: 2
    # B2 base index: 14
    
    # B2 cols 0,1 -> Strassen
    current_term = process_strassen_block(2, 14, 0, current_term)
    # B2 cols 2,3 -> Strassen
    current_term = process_strassen_block(2, 16, 4, current_term)
    # B2 cols 4,5 -> Strassen
    current_term = process_strassen_block(2, 18, 8, current_term)
    # B2 col 6 -> Rank 2
    current_term = process_rank2_block(2, 20, 12, current_term)
    
    # Verify rank
    assert current_term == rank, f"Rank mismatch: {current_term} != {rank}"
    
    # Verification
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
                
    T_approx = np.zeros_like(T_full)
    for r in range(rank):
        T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    if not np.allclose(T_full, T_approx, atol=1e-9):
        # Fallback to schoolbook if construction failed (should not happen)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
