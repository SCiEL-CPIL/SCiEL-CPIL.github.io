# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Laderman's 23-term algorithm for 3x3 matrix multiplication
    # Each term r: U_r is a linear form in A (9 entries), V_r is a linear form in B (9 entries)
    # W_r[k*n+i] is the coefficient for C[i,k]
    # Tensor identity: sum_r U_r[i,j] * V_r[j,k] * W_r[k*n+i] = delta[i,j,k] where
    # the multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually T[i,j,k] = delta_{ik} * delta_{jk}... no.
    # The tensor for matrix mult: T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is: T[i,j,k] = 1 if we're at position (i,j,k) in the "middle" index j
    # More precisely: sum_r U_r[i,j] V_r[j,k] W_r[k*n+i] should equal the standard matrix mult
    # For each (i,k): C[i,k] = sum_j A[i,j] B[j,k]
    # So we need: for each (i,k), sum_j A[i,j] B[j,k] = sum_r W_r[k*n+i] * (sum_j' U_r[i,j'] A[i,j']) * (sum_k' V_r[j',k'] B[j',k'])
    # Wait, that's not quite right. Let me re-read the convention.
    #
    # U has shape (n*m, rank) = (9, rank). U[i*m+j, r] is coefficient of A[i,j] in term r.
    # V has shape (m*p, rank) = (9, rank). V[j*p+k, r] is coefficient of B[j,k] in term r.
    # W has shape (p*n, rank) = (9, rank). W[k*n+i, r] is coefficient for C[i,k] in term r.
    #
    # The decomposition says: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    #
    # Wait, that's not standard tensor decomposition. Let me think again.
    # Standard: T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}  (using Einstein notation loosely)
    # But here the indices are flattened:
    # U[i*m+j, r] corresponds to the (i,j) component
    # V[j*p+k, r] corresponds to the (j,k) component  
    # W[k*n+i, r] corresponds to the (k,i) -> output (i,k) component
    #
    # The tensor identity for matrix multiplication:
    # For all i, j, k: delta_{ik} * delta_{jk} ... no.
    # The multiplication tensor M satisfies: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: M[i,j,k] = 1 if we interpret it as the coefficient, but actually
    # the tensor T[i,j,k] such that T[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k (it's just the identity in the middle index).
    #
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't work dimensionally. Let me re-read the task.
    #
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" - so W is indexed by (k*n+i) and r.
    #
    # I think the decomposition is:
    # For the matrix multiplication tensor, we want:
    # sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_{j'} U[i*m+j', r] A[i,j']) * (sum_{j''} V[j''*p+k, r] B[j'',k])
    #
    # Hmm, but that's not a standard rank decomposition. Let me think of it differently.
    #
    # Actually, I think the standard formulation is:
    # The tensor T has entries T[i,j,k] where the "multiplication tensor" for 3x3 mat mult is
    # T[i,j,k] = 1 if i,k are output indices and j is the summing index.
    # But T is a 3rd order tensor with indices (i,j,k) where:
    # - i: row of A / row of C
    # - j: col of A / row of B  
    # - k: col of B / col of C
    #
    # T[i,j,k] = 1 for all i,j,k (since every combination contributes)
    # No wait, T[i,j,k] should be such that sum_j A[i,j] T[i,j,k] B[j,k] = C[i,k]
    # So T[i,j,k] = delta_{j,j} = 1... that's just saying the tensor is all ones? No.
    #
    # I think the correct interpretation is:
    # The multiplication tensor is T[i,j,k] where we sum over j:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the "tensor" is actually just the identity in the j-index, meaning T[i,j,k] = 1 for all i,j,k
    # in the sense that each (i,j,k) triple contributes A[i,j]*B[j,k] to C[i,k].
    #
    # The rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i,j]^{(r)} V[j,k]^{(r)} W[k,i]^{(r)}
    # Where W[k,i] means the coefficient for output C[i,k] from term r.
    #
    # With flattening: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    #
    # So the condition is: for all i, j, k:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, that's not right either because the LHS should be the tensor entry, which for matrix mult
    # is 1 for each (i,j,k) triple? No, the tensor entry T[i,j,k] represents the contribution of
    # A[i,j]*B[j,k] to C[i,k]. Since C[i,k] = sum_j A[i,j]*B[j,k], each (i,j,k) with valid j
    # contributes. So T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    #
    # But then the decomposition would be: 1 = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] for all i,j,k.
    # That can't be right because we'd need the sum to be 1 for all 27 triples.
    #
    # I think I'm misunderstanding. Let me reconsider.
    #
    # The standard bilinear complexity formulation:
    # C = AB where C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition uses r products of linear forms:
    # C[i,k] = sum_{r=1}^{r} (sum_j u_{r,i,j} A[i,j]) * (sum_j v_{r,j,k} B[j,k]) * w_{r,i,k}
    # Wait, no. The standard is:
    # C[i,k] = sum_{r=1}^{r} w_{r,i,k} * P_r
    # where P_r = (linear form in A) * (linear form in B)
    #
    # So: C[i,k] = sum_{r=1}^{r} w_{r,i,k} * (sum_{i',j'} u_{r,i',j'} A[i',j']) * (sum_{j',k'} v_{r,j',k'} B[j',k'])
    #
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r w_{r,i,k} u_{r,i',j'} v_{r,j',k'} = delta_{i',i} delta_{j',j} delta_{k',k} ... no.
    #
    # Actually: sum_j A[i,j] B[j,k] = sum_{i',j',k'} delta_{i',i} delta_{k',k} A[i',j'] B[j',k] * delta_{j',j}
    # This is getting confusing. Let me just use the verification approach.
    #
    # The condition is: for random A, B, the computed C should equal A @ B.
    # Computed C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    #
    # Wait, I don't think that's right either. Let me look at this more carefully.
    #
    # I believe the correct interpretation is:
    # Each term r defines:
    #   L_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (a scalar)
    #   M_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (a scalar)
    #   The product L_r * M_r is added to C[i,k] with coefficient W[k*n+i, r]
    #
    # So: C[i,k] = sum_r W[k*n+i, r] * L_r * M_r
    #
    # And we need this to equal (A@B)[i,k] for all A, B.
    #
    # This is the standard bilinear algorithm formulation. Let me implement Laderman's 23 terms
    # and verify numerically.

    # Laderman's 23 terms for 3x3 matrix multiplication
    # Reference: Laderman, "A 23-Multiplication Algorithm for 3x3 Matrix Multiplication"
    # I'll use the formulation from various sources.
    
    # Let me define the 23 terms as (U_dict, V_dict, W_dict)
    # U_dict: {(i,j): coeff} for A[i,j]
    # V_dict: {(j,k): coeff} for B[j,k]  
    # W_dict: {(i,k): coeff} for C[i,k], stored at index k*n+i
    
    terms = []
    
    # I'll use a known correct set of Laderman's 23 terms.
    # From the literature, one standard set is:
    
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    
    # The 23 products and their contributions:
    
    # I'll use the set from: https://en.wikipedia.org/wiki/Strassen_algorithm or similar
    # Actually, let me use a verified set. Here's one from a reliable source:
    
    # Term 1: (a00+a11+a22)(b00+b11+b22) -> contributes to C00,C11,C22 with coeff 1, and to off-diagonals
    # This is getting complex. Let me use a different approach.
    
    # I'll use the Laderman algorithm from this explicit list:
    # Source: "Fast Matrix Multiplication" by Coppersmith and Winograd, or similar references
    
    # Here is a known correct set of 23 terms (verified in multiple sources):
    
    def add(u, v, w):
        terms.append((u, v, w))
        
    # Using the formulation where:
    # P1 = (a00 + a11 + a22)(b00 + b11 + b22)
    # P2 = (a01 + a10 + a22)(b01 + b10 + b22)  
    # P3 = (a02 + a12 + a20)(b02 + b12 + b20)
    # P4 = (a00 + a11)(b00 + b11)
    # P5 = (a00 + a11)(b02 + b12)
    # P6 = (a00 + a11)(b20 + b21)
    # P7 = (a01 + a10)(b00 + b11)
    # P8 = (a01 + a10)(b02 + b12)
    # P9 = (a01 + a10)(b20 + b21)
    # P10 = (a02 + a20)(b00 + b22)
    # P11 = (a02 + a20)(b01 + b21)
    # P12 = (a02 + a20)(b10 + b12)
    # P13 = (a12 + a21)(b00 + b22)
    # P14 = (a12 + a21)(b01 + b21)
    # P15 = (a12 + a21)(b10 + b12)
    # P16 = a22*b22
    # P17 = a12*b21
    # P18 = a21*b12
    # P19 = a02*b20
    # P20 = a20*b02
    # P21 = a10*b01
    # P22 = a01*b10
    # P23 = a00*b00
    
    # Wait, this doesn't look right. Let me try a different verified set.
    
    # Actually, let me use a completely different approach. I'll use the fact that
    # rank-23 is achievable and construct it using a known correct decomposition.
    
    # Here's a set I can verify:
    # From Laderman's original paper or equivalent:
    
    # Let me try this set which is commonly cited:
    
    # P1 = (a00+a11+a22)(b00+b11+b22)
    # P2 = (a01+a10+a22)(b01+b10+b22)
    # P3 = (a02+a12+a20)(b02+b12+b20)
    # P4 = (a00+a11)(b00+b11)
    # P5 = (a00+a11)(b02+b12)
    # P6 = (a00+a11)(b20+b21)
    # P7 = (a01+a10)(b00+b11)
    # P8 = (a01+a10)(b02+b12)
    # P9 = (a01+a10)(b20+b21)
    # P10 = (a02+a20)(b00+b22)
    # P11 = (a02+a20)(b01+b21)
    # P12 = (a02+a20)(b10+b12)
    # P13 = (a12+a21)(b00+b22)
    # P14 = (a12+a21)(b01+b21)
    # P15 = (a12+a21)(b10+b12)
    # P16 = a22*b22
    # P17 = a12*b21
    # P18 = a21*b12
    # P19 = a02*b20
    # P20 = a20*b02
    # P21 = a10*b01
    # P22 = a01*b10
    # P23 = a00*b00
    
    # Let me check if this gives the right answer for C00:
    # C00 = a00*b00 + a01*b10 + a02*b20
    # From P1: a00*b00 + a00*b11 + a00*b22 + a11*b00 + a11*b11 + a11*b22 + a22*b00 + a22*b11 + a22*b22
    # This is way too much. The W coefficients must cancel things out.
    
    # I think the issue is that I'm not correctly accounting for the W coefficients.
    # Each product P_r contributes to multiple C[i,k] entries with different coefficients.
    
    # Let me try a different, more reliable approach. I'll use a set of 23 terms that I
    # can verify numerically. Let me use the set from a well-known reference.
    
    # Actually, let me just use the schoolbook rank-27 as a fallback and try to find
    # a correct rank-23 by searching or using a known correct set.
    
    # For now, let me implement the schoolbook and verify it works, then try rank-23.
    
    # Schoolbook rank-27:
    terms_sb = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(i, k): 1}  # C[i,k] gets this term
                terms_sb.append((u, v, w))
    
    # Verify schoolbook:
    def verify(terms, n=3, m=3, p=3, trials=5):
        rng = np.random.default_rng(42)
        for _ in range(trials):
            A = rng.random((n, m))
            B = rng.random((m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for u_dict, v_dict, w_dict in terms:
                L = sum(coeff * A[i, j] for (i, j), coeff in u_dict.items())
                M = sum(coeff * B[j, k] for (j, k), coeff in v_dict.items())
                for (i, k), coeff in w_dict.items():
                    C_computed[i, k] += coeff * L * M
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    if verify(terms_sb):
        print("Schoolbook verified")
    else:
        print("Schoolbook FAILED")
    
    # Now let me try to implement a correct Laderman 23-term algorithm.
    # I'll use a set from a reliable source and verify it.
    
    # Here's a set that I believe is correct (from Laderman 1976 or equivalent):
    # I'll define them and verify.
    
    terms_23 = []
    
    # Let me use this specific set which is known to work:
    # From: https://github.com/... or similar verified sources
    
    # Actually, let me try to construct it using the following approach:
    # I know that rank-23 is achievable. Let me use a set that I can verify.
    
    # Here's a set from a paper:
    # P1 = (a00+a11+a22)(b00+b11+b22)
    # P2 = (a01+a10+a22)(b01+b10+b22)
    # P3 = (a02+a12+a20)(b02+b12+b20)
    # P4 = (a00+a11)(b00+b11)
    # P5 = (a00+a11)(b02+b12)
    # P6 = (a00+a11)(b20+b21)
    # P7 = (a01+a10)(b00+b11)
    # P8 = (a01+a10)(b02+b12)
    # P9 = (a01+a10)(b20+b21)
    # P10 = (a02+a20)(b00+b22)
    # P11 = (a02+a20)(b01+b21)
    # P12 = (a02+a20)(b10+b12)
    # P13 = (a12+a21)(b00+b22)
    # P14 = (a12+a21)(b01+b21)
    # P15 = (a12+a21)(b10+b12)
    # P16 = a22*b22
    # P17 = a12*b21
    # P18 = a21*b12
    # P19 = a02*b20
    # P20 = a20*b02
    # P21 = a10*b01
    # P22 = a01*b10
    # P23 = a00*b00
    
    # The W coefficients for each C[i,k] from each P_r:
    # This is the tricky part. Let me compute what the W coefficients should be.
    
    # For C00 = a00*b00 + a01*b10 + a02*b20:
    # I need to find w_{00,r} for each r such that:
    # sum_r w_{00,r} * L_r * M_r = a00*b00 + a01*b10 + a02*b20
    
    # This is a system of equations. Let me set it up.
    # For each r, L_r and M_r are known linear forms.
    # The product L_r * M_r is a bilinear form.
    # I need to find w_{00,r} such that the sum equals the target.
    
    # This is underdetermined in general, but for a correct decomposition,
    # there should be a unique solution (up to the constraints).
    
    # Let me just try the set above and solve for W coefficients.
    
    # Define the 23 L_r and M_r:
    L_forms = [
        {(0,0):1, (1,1):1, (2,2):1},  # P1
        {(0,1):1, (1,0):1, (2,2):1},  # P2
        {(0,2):1, (1,2):1, (2,0):1},  # P3
        {(0,0):1, (1,1):1},           # P4
        {(0,0):1, (1,1):1},           # P5
        {(0,0):1, (1,1):1},           # P6
        {(0,1):1, (1,0):1},           # P7
        {(0,1):1, (1,0):1},           # P8
        {(0,1):1, (1,0):1},           # P9
        {(0,2):1, (2,0):1},           # P10
        {(0,2):1, (2,0):1},           # P11
        {(0,2):1, (2,0):1},           # P12
        {(1,2):1, (2,1):1},           # P13
        {(1,2):1, (2,1):1},           # P14
        {(1,2):1, (2,1):1},           # P15
        {(2,2):1},                     # P16
        {(1,2):1},                     # P17
        {(2,1):1},                     # P18
        {(0,2):1},                     # P19
        {(2,0):1},                     # P20
        {(1,0):1},                     # P21
        {(0,1):1},                     # P22
        {(0,0):1},                     # P23
    ]
    
    M_forms = [
        {(0,0):1, (1,1):1, (2,2):1},  # P1
        {(0,1):1, (1,0):1, (2,2):1},  # P2
        {(0,2):1, (1,2):1, (2,0):1},  # P3
        {(0,0):1, (1,1):1},           # P4
        {(0,2):1, (1,2):1},           # P5
        {(2,0):1, (2,1):1},           # P6
        {(0,0):1, (1,1):1},           # P7
        {(0,2):1, (1,2):1},           # P8
        {(2,0):1, (2,1):1},           # P9
        {(0,0):1, (2,2):1},           # P10
        {(0,1):1, (2,1):1},           # P11
        {(1,0):1, (1,2):1},           # P12
        {(0,0):1, (2,2):1},           # P13
        {(0,1):1, (2,1):1},           # P14
        {(1,0):1, (1,2):1},           # P15
        {(2,2):1},                     # P16
        {(2,1):1},                     # P17
        {(1,2):1},                     # P18
        {(2,0):1},                     # P19
        {(0,2):1},                     # P20
        {(0,1):1},                     # P21
        {(1,0):1},                     # P22
        {(0,0):1},                     # P23
    ]
    
    # For each output C[i,k], I need to find w_{i,k,r} such that:
    # sum_r w_{i,k,r} * L_r * M_r = sum_j A[i,j] * B[j,k]
    
    # Let me set up the system for each (i,k).
    # The bilinear form L_r * M_r can be represented as a 9x9 matrix (or 3x3x3x3 tensor).
    # For a given (i,k), the target is: sum_j A[i,j] * B[j,k]
    
    # I'll solve for w using least squares or exact solve.
    
    # Let me represent each L_r * M_r as a vector in the space of bilinear forms.
    # A bilinear form in A (9 vars) and B (9 vars) can be represented as a 9x9 matrix.
    # The target for C[i,k] is a 9x9 matrix with 1's at positions (i,j) x (j,k) for j=0,1,2.
    
    # Let me build the system.
    
    # First, let me create the matrix representation of each L_r * M_r.
    # L_r is a linear form in A: L_r = sum_{i,j} u_{r,i,j} A[i,j]
    # M_r is a linear form in B: M_r = sum_{j,k} v_{r,j,k} B[j,k]
    # L_r * M_r = sum_{i,j,j',k'} u_{r,i,j} v_{r,j',k'} A[i,j] B[j',k']
    # So the (9,9) matrix has entry (i*3+j, j'*3+k') = u_{r,i,j} * v_{r,j',k'}
    
    # The target for C[i,k] is: sum_j A[i,j] B[j,k]
    # This is a 9x9 matrix with entry (i*3+j, j*3+k) = 1 for j=0,1,2.
    
    # I'll set up: for each (i,k), find w vector such that sum_r w_r * M_r = T_{i,k}
    # where M_r is the 9x9 matrix for L_r * M_r, and T_{i,k} is the target matrix.
    
    # This is a linear system: [M_1, M_2, ..., M_23] * w = T_{i,k}
    # where the matrix is 81 x 23 (flattening the 9x9 matrices).
    
    # Let me solve this.
    
    # Build the coefficient matrix (81 x 23)
    coeff_matrix = np.zeros((81, 23))
    for r in range(23):
        for i in range(3):
            for j in range(3):
                for j2 in range(3):
                    for k in range(3):
                        u_coeff = L_forms[r].get((i, j), 0)
                        v_coeff = M_forms[r].get((j2, k), 0)
                        if u_coeff != 0 and v_coeff != 0:
                            coeff_matrix[i*3+j, r] += u_coeff * v_coeff  # This is wrong, need to be more careful
    
    # Actually, let me redo this. The bilinear form L_r * M_r as a 9x9 matrix:
    # Entry (a, b) where a = i*3+j (index in A), b = j2*3+k (index in B)
    # = sum over all (i,j) in L_r and (j2,k) in M_r: u_{i,j} * v_{j2,k} * delta_{a, i*3+j} * delta_{b, j2*3+k}
    # = u_{i,j} * v_{j2,k} if a = i*3+j and b = j2*3+k, else 0
    # Wait, no. L_r * M_r = (sum_{i,j} u_{r,i,j} A[i,j]) * (sum_{j2,k} v_{r,j2,k} B[j2,k])
    # = sum_{i,j,j2,k} u_{r,i,j} v_{r,j2,k} A[i,j] B[j2,k]
    # So the coefficient of A[i,j] * B[j2,k] in L_r * M_r is u_{r,i,j} * v_{r,j2,k}
    
    # So the 9x9 matrix for term r has entry (i*3+j, j2*3+k) = u_{r,i,j} * v_{r,j2,k}
    
    coeff_matrix = np.zeros((81, 23))
    for r in range(23):
        for i in range(3):
            for j in range(3):
                u_coeff = L_forms[r].get((i, j), 0)
                if u_coeff == 0:
                    continue
                for j2 in range(3):
                    for k in range(3):
                        v_coeff = M_forms[r].get((j2, k), 0)
                        if v_coeff == 0:
                            continue
                        a_idx = i * 3 + j
                        b_idx = j2 * 3 + k
                        coeff_matrix[a_idx * 9 + b_idx, r] = u_coeff * v_coeff
    
    # Now for each (i,k), the target is: sum_j A[i,j] B[j,k]
    # Target matrix: entry (i*3+j, j*3+k) = 1 for j=0,1,2
    
    W_coeffs = {}
    for i in range(3):
        for k in range(3):
            target = np.zeros(81)
            for j in range(3):
                a_idx = i * 3 + j
                b_idx = j * 3 + k
                target[a_idx * 9 + b_idx] = 1.0
            
            # Solve coeff_matrix * w = target
            # This is 81 equations, 23 unknowns. Overdetermined.
            # Use least squares.
            w, residuals, rank, sv = np.linalg.lstsq(coeff_matrix, target, rcond=None)
            W_coeffs[(i, k)] = w
    
    # Build the terms
    terms_23 = []
    for r in range(23):
        u_dict = dict(L_forms[r])
        v_dict = dict(M_forms[r])
        w_dict = {}
        for i in range(3):
            for k in range(3):
                w_val = W_coeffs[(i, k)][r]
                if abs(w_val) > 1e-10:
                    w_dict[(i, k)] = w_val
        terms_23.append((u_dict, v_dict, w_dict))
    
    # Verify
    if verify(terms_23):
        print("Laderman 23 verified!")
        terms = terms_23
    else:
        print("Laderman 23 FAILED, using schoolbook")
        terms = terms_sb
    
    # Convert to matrix form
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] = val
        for (j, k), val in v.items():
            V[j * p + k][r] = val
        for (i, k), val in w.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
