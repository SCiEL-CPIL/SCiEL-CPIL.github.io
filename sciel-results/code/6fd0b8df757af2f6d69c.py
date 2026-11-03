import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,4,5} into two M_{2,2,5} tensors by splitting m=4 into 2+2.
    # Each M_{2,2,5} is decomposed into M_{2,2,2} and M_{2,2,3} by splitting p=5 into 2+3.
    # M_{2,2,2} uses Strassen (rank 7).
    # M_{2,2,3} uses Schoolbook (rank 12) for safety, or potentially rank 9/10.
    # Let's use Schoolbook for M_{2,2,3} to ensure validity. Rank 12.
    # Total rank = 2 * (7 + 12) = 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, val=1):
        nonlocal col_offset
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U[i * m + j][col_offset] = val
        V[j * p + k][col_offset] = val
        W[k * n + i][col_offset] = val
        col_offset += 1

    # We need to map the logical indices to the physical indices.
    # Physical m is 4. We split into m1 (0,1) and m2 (2,3).
    # Physical p is 5. We split into p1 (0,1) and p2 (2,3,4).
    
    # For each half of m (0, 1):
    for m_part in range(2):
        m_base = m_part * 2
        
        # Part 1: p in [0, 1] (2x2x2) using Strassen
        # Indices: i in 0..1, j in 0..1 (local), k in 0..1 (local)
        # Global j = m_base + j_local
        # Global k = k_local
        
        # Strassen terms for 2x2x2
        # M1 = A00(B00+B11)
        # M2 = A01(B00+B10)
        # M3 = A10(B01+B11)
        # M4 = A11(B01+B10)
        # M5 = (A00+A11)(B00+B11)
        # M6 = (A01+A11)(B10-B11)
        # M7 = (A00+A10)(B01-B00)
        
        # We need to expand these into rank-1 terms in U,V,W.
        # This is tricky because Strassen uses sums.
        # We must express the 7 multiplications as 7 rank-1 terms in the tensor product space.
        # Actually, the Strassen decomposition IS a rank-7 decomposition.
        # We just need the U, V, W vectors for each of the 7 terms.
        
        # Let's define the 7 terms for 2x2x2.
        # Term 1: A00 * (B00 + B11)
        # This is not a single rank-1 term in the standard basis?
        # Wait, the tensor decomposition T = sum u_t v_t w_t.
        # Strassen gives 7 bilinear forms.
        # Each bilinear form corresponds to a rank-1 tensor?
        # Yes, $M_1 = A_{00} (B_{00} + B_{11})$ is a product of a linear form in A and a linear form in B.
        # So it corresponds to a rank-1 term in the tensor.
        
        # We need to fill U, V, W for these 7 terms.
        # For term t:
        # U is the vector of coefficients for A.
        # V is the vector of coefficients for B.
        # W is the vector of coefficients for C (output).
        
        # Let's construct the 7 terms explicitly.
        # Local indices: i, j, k in {0,1}.
        
        # Term 1: A00 * (B00 + B11)
        # A part: only A00 is 1.
        # B part: B00 is 1, B11 is 1.
        # C part: This term contributes to C00 and C11?
        # No, Strassen combines them later.
        # The rank-1 term is $u \otimes v \otimes w$.
        # $w$ is the output vector.
        # For $M_1 = A_{00} B_{00} + A_{00} B_{11}$, this is NOT a single rank-1 term in the multiplication tensor?
        # The multiplication tensor is $\sum A_{ij} B_{jk} C_{ik}$.
        # A rank-1 term is $u_i v_j w_k C_{ik}$? No.
        # A rank-1 term is $u \otimes v \otimes w$ such that $\sum_{i,j,k} u_i v_j w_k T_{ijk} = 1$?
        # No, the decomposition is $T = \sum_{t} u^{(t)} \otimes v^{(t)} \otimes w^{(t)}$.
        # This means $T_{ijk} = \sum_t u^{(t)}_i v^{(t)}_j w^{(t)}_k$.
        
        # Strassen's algorithm computes 7 products $M_t$.
        # $M_t = (\sum_i u^{(t)}_i A_i) (\sum_j v^{(t)}_j B_j)$.
        # Then $C_k = \sum_t w^{(t)}_k M_t$.
        # So $C_k = \sum_t w^{(t)}_k (\sum_i u^{(t)}_i A_i) (\sum_j v^{(t)}_j B_j)$.
        # $C_k = \sum_{i,j} A_i B_j (\sum_t u^{(t)}_i v^{(t)}_j w^{(t)}_k)$.
        # So $T_{ijk} = \sum_t u^{(t)}_i v^{(t)}_j w^{(t)}_k$.
        
        # So we just need the $u, v, w$ vectors for the 7 Strassen terms.
        
        # Strassen 2x2x2:
        # t=0: u=[1,0,0,0] (A00), v=[1,0,0,1] (B00+B11), w=[1,0,0,1] (C00+C11)? No.
        # Let's check the outputs.
        # C00 = M5 + M4 - M6 - M7
        # C01 = M1 + M7
        # C10 = M2 + M6
        # C11 = M5 + M3 - M1 - M4
        
        # M1 = A00(B00+B11). u=[1,0,0,0], v=[1,0,0,1].
        # M2 = A01(B00+B10). u=[0,1,0,0], v=[1,0,1,0].
        # M3 = A10(B01+B11). u=[0,0,1,0], v=[0,1,0,1].
        # M4 = A11(B01+B10). u=[0,0,0,1], v=[0,1,1,0].
        # M5 = (A00+A11)(B00+B11). u=[1,0,0,1], v=[1,0,0,1].
        # M6 = (A01+A11)(B10-B11). u=[0,1,0,1], v=[0,0,1,-1].
        # M7 = (A00+A10)(B01-B00). u=[1,0,1,0], v=[-1,1,0,0].
        
        # Now w vectors:
        # C00: M5(1) + M4(1) - M6(1) - M7(1). w=[1,0,0,1,1,-1,-1]? No, w is per term.
        # w^{(1)} for M1: contributes to C01 (1) and C11 (-1). w=[0,1,0,-1].
        # w^{(2)} for M2: contributes to C10 (1). w=[0,0,1,0].
        # w^{(3)} for M3: contributes to C11 (1). w=[0,0,0,1].
        # w^{(4)} for M4: contributes to C00 (1) and C11 (-1). w=[1,0,0,-1].
        # w^{(5)} for M5: contributes to C00 (1) and C11 (1). w=[1,0,0,1].
        # w^{(6)} for M6: contributes to C10 (1) and C00 (-1). w=[-1,0,1,0].
        # w^{(7)} for M7: contributes to C01 (1) and C00 (-1). w=[-1,1,0,0].
        
        # We add these 7 terms to U, V, W.
        
        def add_strassen_222(m_base, p_base):
            nonlocal col_offset
            # u, v, w vectors for 2x2x2
            # A indices: 0..3 (i,j) -> i*2+j
            # B indices: 0..3 (j,k) -> j*2+k
            # C indices: 0..3 (k,i) -> k*2+i
            
            terms = [
                ([1,0,0,0], [1,0,0,1], [0,1,0,-1]), # M1
                ([0,1,0,0], [1,0,1,0], [0,0,1,0]),   # M2
                ([0,0,1,0], [0,1,0,1], [0,0,0,1]),   # M3
                ([0,0,0,1], [0,1,1,0], [1,0,0,-1]),  # M4
                ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M5
                ([0,1,0,1], [0,0,1,-1], [-1,0,1,0]), # M6
                ([1,0,1,0], [-1,1,0,0], [-1,1,0,0])  # M7
            ]
            
            for u, v, w in terms:
                # Map local indices to global
                # U: i*4 + (m_base + j)
                for li in range(2):
                    for lj in range(2):
                        if u[li*2+lj] != 0:
                            U[li*4 + m_base + lj][col_offset] = u[li*2+lj]
                            
                # V: (m_base + j)*5 + (p_base + k)
                for lj in range(2):
                    for lk in range(2):
                        if v[lj*2+lk] != 0:
                            V[(m_base + lj)*5 + p_base + lk][col_offset] = v[lj*2+lk]
                            
                # W: (p_base + k)*2 + i
                for lk in range(2):
                    for li in range(2):
                        if w[lk*2+li] != 0:
                            W[(p_base + lk)*2 + li][col_offset] = w[lk*2+li]
                            
                col_offset += 1

        add_strassen_222(m_base, 0)
        
        # Part 2: p in [2, 4] (2x2x3) using Schoolbook
        # Rank 12.
        # For each i, j, k, add a term.
        for li in range(2):
            for lj in range(2):
                for lk in range(3):
                    add_term(li, m_base + lj, 2 + lk, 1)
                    
    return {"U": U, "V": V, "W": W}
