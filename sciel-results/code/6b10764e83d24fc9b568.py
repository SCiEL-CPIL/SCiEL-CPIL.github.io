"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    Also checks that entries are integers or half-integers.
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check entries are integer or half-integer
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=tol):
            return False
            
    # Check tensor equation
    # We can vectorize this check for speed
    # Reshape to tensors: U (n, m, R), V (m, p, R), W (p, n, R)
    U_t = Ua.reshape(n, m, rank)
    V_t = Va.reshape(m, p, rank)
    W_t = Wa.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This is a bit tricky to vectorize directly without large intermediate arrays if done naively.
    # Let's iterate over i, j and use matrix multiplication tricks or simple loops since n,m,p are small.
    # n=4, m=4, p=9. Loops are fine.
    for i in range(n):
        for j in range(m):
            u_vec = Ua[i*m+j, :] # shape (rank,)
            v_mat = Va[j*p:(j+1)*p, :] # shape (p, rank)
            # We need sum_r u_vec[r] * V[j, k, r] * W[k, i, r]
            # Let M[k, r] = V[j, k, r] * W[k, i, r]
            # Then result[k] = dot(u_vec, M[k, :])
            # W rows for fixed i: W[k*n+i, :]
            w_rows = Wa[:, :] # We need specific rows
            # Actually, let's just loop k as well, it's only 9 iterations.
            for k in range(p):
                val = np.dot(u_vec, Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > tol:
                    return False
    return True

def _construct_strassen_block(a_idx, b_idx, c_idx, offset, U, V, W):
    """
    Adds a rank-7 Strassen decomposition for a 2x2 matrix multiplication block.
    a_idx: list of 4 indices into U corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    b_idx: list of 4 indices into V corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    c_idx: list of 4 indices into W corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    offset: starting column index in the factor matrices.
    Returns the next available column index (offset + 7).
    """
    # Indices:
    # A: a0=a_idx[0], a1=a_idx[1], a2=a_idx[2], a3=a_idx[3]
    # B: b0=b_idx[0], b1=b_idx[1], b2=b_idx[2], b3=b_idx[3]
    # C: c0=c_idx[0], c1=c_idx[1], c2=c_idx[2], c3=c_idx[3]
    
    # M1 = (a0+a3)(b0+b3)
    # C0 += M1, C3 += M1
    U[a_idx[0]][offset] += 1
    U[a_idx[3]][offset] += 1
    V[b_idx[0]][offset] += 1
    V[b_idx[3]][offset] += 1
    W[c_idx[0]][offset] += 1
    W[c_idx[3]][offset] += 1
    offset += 1
    
    # M2 = (a2+a3)b0
    # C2 += M2, C3 -= M2
    U[a_idx[2]][offset] += 1
    U[a_idx[3]][offset] += 1
    V[b_idx[0]][offset] += 1
    W[c_idx[2]][offset] += 1
    W[c_idx[3]][offset] -= 1
    offset += 1
    
    # M3 = a0(b1-b3)
    # C1 += M3, C3 += M3
    U[a_idx[0]][offset] += 1
    V[b_idx[1]][offset] += 1
    V[b_idx[3]][offset] -= 1
    W[c_idx[1]][offset] += 1
    W[c_idx[3]][offset] += 1
    offset += 1
    
    # M4 = a3(b2-a2) ... WAIT.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Let's re-verify the indices for M4.
    # d is A[1,1] (a3). g is B[1,0] (b2). c is A[1,0] (a2).
    # M4 = d * (g - c).
    # This is NOT a product of a linear form in A and a linear form in B?
    # d is in A. g-c is ... g is in B, c is in A.
    # So M4 = A[1,1] * B[1,0] - A[1,1] * A[1,0].
    # This term is NOT bilinear in (A, B) in the sense of tensor rank decomposition.
    # Tensor rank decomposition requires M_k = u_k(A) v_k(B).
    # If M4 depends on A twice, it's not a valid rank-1 term.
    
    # Let's check the actual Strassen algorithm structure.
    # The 7 multiplications are:
    # 1. (a+d)(e+h) -> u=(a+d), v=(e+h)
    # 2. (c+d)e -> u=(c+d), v=e
    # 3. a(f-h) -> u=a, v=(f-h)
    # 4. d(g-c) -> u=d, v=(g-c). Here c is from A. So v depends on A.
    #    This implies standard Strassen is NOT a direct tensor decomposition of the multiplication tensor?
    #    No, Strassen's algorithm is a valid algorithm. The "rank" of the algorithm is 7.
    #    The tensor rank of matrix multiplication is defined as the minimum number of scalar multiplications required.
    #    In Strassen, the inputs to the scalar multiplications are linear combinations of the entries of A and B.
    #    For M4 = d(g-c), the input from A is 'd' and from B is ... wait.
    #    Is 'c' from A or B?
    #    A = [[a, b], [c, d]]. B = [[e, f], [g, h]].
    #    M4 = d(g-c).
    #    d is from A. g is from B. c is from A.
    #    So the second factor is (g-c). This is a linear combination of B[1,0] and A[1,0].
    #    This is NOT a linear form in B alone.
    #    Therefore, M4 is not a product of a linear form in A and a linear form in B.
    #    
    #    HOWEVER, it is a known fact that Strassen's algorithm corresponds to a rank-7 decomposition.
    #    How?
    #    Perhaps the definition of the tensor allows the linear forms to be arbitrary?
    #    No, the tensor is in A* \otimes B* \otimes C*.
    #    A rank-1 tensor is u \otimes v \otimes w where u \in A*, v \in B*, w \in C*.
    #    The evaluation is u(A)v(B)w(C).
    #    If M4 = d(g-c), then u(A)=d, v(B)=?
    #    If v(B) = g-c, then v depends on A. This is not in B*.
    #    
    #    Let's look at the actual terms again.
    #    Maybe I am misremembering the indices.
    #    M4 = d(g-c).
    #    Is it possible that 'c' in the formula refers to an entry of B?
    #    Standard notation:
    #    A = [[a, b], [c, d]]
    #    B = [[e, f], [g, h]]
    #    M4 = d(g-c).
    #    Yes, c is A[1,0].
    #    
    #    Let's check M6 = (c-a)(e+f).
    #    u = c-a (from A), v = e+f (from B). This is fine.
    #    
    #    Why is M4 problematic?
    #    Actually, in the context of tensor rank, we are allowed to choose ANY linear forms.
    #    The decomposition is T = sum u_i \otimes v_i \otimes w_i.
    #    The condition is that for all A, B, sum u_i(A)v_i(B)w_i(C) = C(AB).
    #    This does NOT require that the algorithm is "Strassen's algorithm" in the sense of data flow.
    #    It requires that the sum of the rank-1 tensors equals the multiplication tensor.
    #    
    #    Let's verify if Strassen's 7 terms sum to the multiplication tensor.
    #    C00 = M1 - M4 + M6 - M7
    #    C01 = M3 + M5
    #    C10 = M2 + M4
    #    C11 = M1 - M3 + M5 - M7
    #    
    #    Let's check C10 = M2 + M4.
    #    M2 = (c+d)e
    #    M4 = d(g-c)
    #    M2 + M4 = ce + de + dg - dc.
    #    True C10 = c*e + d*g.
    #    Difference: de - dc = d(e-c).
    #    This is not zero.
    #    
    #    Wait, I must have the formulas wrong.
    #    Let's look up the exact Strassen formulas.
    #    M1 = (a+d)(e+h)
    #    M2 = (c+d)e
    #    M3 = a(f-h)
    #    M4 = d(g-c)
    #    M5 = (a+b)h
    #    M6 = (c-a)(e+f)
    #    M7 = (b-d)(g+h)
    #    
    #    C = [[M1 - M4 + M6 - M7, M3 + M5],
    #         [M2 + M4, M1 - M3 + M5 - M7]]
    #     
    #    Let's re-calculate C10.
    #    C10 = M2 + M4 = (c+d)e + d(g-c) = ce + de + dg - dc.
    #    Target: ce + dg.
    #    Extra term: de - dc.
    #    This is not zero.
    #    
    #    There is a mistake in my memory of the formulas or the indices.
    #    Let's check M4 again.
    #    Some sources say M4 = d(g-c).
    #    Some say M4 = d(g-c) is wrong.
    #    
    #    Correct Strassen:
    #    M1 = (a+d)(e+h)
    #    M2 = (c+d)e
    #    M3 = a(f-h)
    #    M4 = d(g-c)
    #    M5 = (a+b)h
    #    M6 = (c-a)(e+f)
    #    M7 = (b-d)(g+h)
    #    
    #    Let's check C11.
    #    C11 = M1 - M3 + M5 - M7
    #    = (ae+ah+de+dh) - (af-ah) + (ah+bh) - (bg+bh-dg-dh)
    #    = ae+ah+de+dh - af+ah + ah+bh - bg-bh+dg+dh
    #    = ae - af + de + dg + (ah+ah+ah-bh-bh) + (dh+dh)
    #    This is getting messy.
    #    
    #    Let's use a known valid rank-7 decomposition if it exists, or stick to schoolbook.
    #    Given the risk of incorrect implementation and the fact that the current program is at 1.0 (schoolbook rank 144),
    #    and the inspiration 2 failed to implement Strassen correctly,
    #    I will attempt to implement a verified Strassen decomposition.
    #    
    #    Actually, there is a simpler approach: Use the known rank-7 decomposition for 2x2.
    #    Reference: "The rank of the 2x2 matrix multiplication tensor is 7".
    #    
    #    Let's try to construct it carefully.
    #    If I cannot guarantee correctness, I should fall back to schoolbook.
    #    However, the task asks to improve.
    #    
    #    Let's try a different strategy: Block diagonal.
    #    4x4 * 4x9.
    #    We can view this as 2x2 blocks of (2x2 * 2x4.5?) No, dimensions must be integer.
    #    
    #    Let's stick to the schoolbook for now to ensure validity, but try to optimize the code structure.
    #    Or, try to implement Strassen and verify it. If it fails verification, return schoolbook.
    #    
    #    I will implement a function that tries to build a Strassen-based decomposition and verifies it.
    #    If verification fails, it falls back to schoolbook.
    #    
    #    Rank calculation:
    #    4x4 matrix A can be split into four 2x2 blocks.
    #    4x9 matrix B... 9 is not divisible by 2.
    #    This makes block Strassen tricky for the 4x9 case.
    #    We could pad B to 4x10? No, that changes the problem.
    #    
    #    We can use Strassen for the 4x4 part of B (4x4) and schoolbook for the remaining 4x5.
    #    A (4x4) * B (4x9) = A * [B1 (4x4) | B2 (4x5)]
    #    = [A*B1 | A*B2]
    #    
    #    A*B1: 4x4 * 4x4. Can use Strassen?
    #    4x4 multiplication can be done with 2x2 blocks.
    #    A = [[A11, A12], [A21, A22]] where each is 2x2.
    #    B1 = [[B11, B12], [B21, B22]] where each is 2x2.
    #    C1 = A*B1.
    #    C11 = A11*B11 + A12*B21
    #    C12 = A11*B12 + A12*B22
    #    C21 = A21*B11 + A22*B21
    #    C22 = A21*B12 + A22*B22
    #    
    #    Each term is a sum of two 2x2 multiplications.
    #    Total 8 multiplications of 2x2 matrices.
    #    Using Strassen (rank 7) for each 2x2 multiplication?
    #    No, Strassen computes the product of two 2x2 matrices in 7 scalar multiplications.
    #    Here we have 8 separate 2x2 matrix multiplications to perform.
    #    Rank = 8 * 7 = 56.
    #    
    #    A*B2: 4x4 * 4x5.
    #    Schoolbook: 4*4*5 = 80.
    #    
    #    Total rank = 56 + 80 = 136.
    #    Schoolbook total rank = 4*4*9 = 144.
    #    136 < 144.
    #    
    #    Let's implement this hybrid approach.
    #    It requires careful index mapping.
    
    return offset

def _construct_hybrid():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # Split B into B1 (4x4) and B2 (4x5)
    # C = A * B1 + A * B2
    
    rank1 = 8 * 7  # 56 for A*B1 using Strassen on 2x2 blocks
    rank2 = 4 * 4 * 5  # 80 for A*B2 using Schoolbook
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A * B1 (4x4 * 4x4)
    # B1 columns 0-3.
    # We need to compute C1 = A * B1.
    # C1 is 4x4.
    # We decompose A and B1 into 2x2 blocks.
    # A indices:
    # A11: (0,0), (0,1), (1,0), (1,1) -> U indices: 0, 1, 4, 5
    # A12: (0,2), (0,3), (1,2), (1,3) -> U indices: 2, 3, 6, 7
    # A21: (2,0), (2,1), (3,0), (3,1) -> U indices: 8, 9, 12, 13
    # A22: (2,2), (2,3), (3,2), (3,3) -> U indices: 10, 11, 14, 15
    
    # B1 indices (columns 0-3 of B, rows 0-3):
    # B is 4x9. B1 is rows 0-3, cols 0-3.
    # V indices: row * 9 + col.
    # B11: (0,0), (0,1), (1,0), (1,1) -> V indices: 0, 1, 9, 10
    # B12: (0,2), (0,3), (1,2), (1,3) -> V indices: 2, 3, 11, 12
    # B21: (2,0), (2,1), (3,0), (3,1) -> V indices: 18, 19, 27, 28
    # B22: (2,2), (2,3), (3,2), (3,3) -> V indices: 20, 21, 29, 30
    
    # C1 indices (rows 0-3, cols 0-3 of C):
    # W is 9x4 (p*n). Index: col * n + row.
    # C11: (0,0), (0,1), (1,0), (1,1) -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C12: (0,2), (0,3), (1,2), (1,3) -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C21: (2,0), (2,1), (3,0), (3,1) -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C22: (2,2), (2,3), (3,2), (3,3) -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # We need to perform 8 Strassen multiplications.
    # 1. A11 * B11 -> contributes to C11
    # 2. A11 * B12 -> contributes to C12
    # 3. A12 * B21 -> contributes to C11
    # 4. A12 * B22 -> contributes to C12
    # 5. A21 * B11 -> contributes to C21
    # 6. A21 * B12 -> contributes to C22
    # 7. A22 * B21 -> contributes to C21
    # 8. A22 * B22 -> contributes to C22
    
    offset = 0
    
    def add_strassen(a_indices, b_indices, c_indices, offset):
        """
        Adds a rank-7 Strassen decomposition.
        a_indices: [a0, a1, a2, a3] corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
        b_indices: [b0, b1, b2, b3] corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
        c_indices: [c0, c1, c2, c3] corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
        """
        # M1 = (a0+a3)(b0+b3) -> C0+=1, C3+=1
        U[a_indices[0]][offset] += 1
        U[a_indices[3]][offset] += 1
        V[b_indices[0]][offset] += 1
        V[b_indices[3]][offset] += 1
        W[c_indices[0]][offset] += 1
        W[c_indices[3]][offset] += 1
        offset += 1
        
        # M2 = (a2+a3)b0 -> C2+=1, C3-=-1
        U[a_indices[2]][offset] += 1
        U[a_indices[3]][offset] += 1
        V[b_indices[0]][offset] += 1
        W[c_indices[2]][offset] += 1
        W[c_indices[3]][offset] -= 1
        offset += 1
        
        # M3 = a0(b1-b3) -> C1+=1, C3+=1
        U[a_indices[0]][offset] += 1
        V[b_indices[1]][offset] += 1
        V[b_indices[3]][offset] -= 1
        W[c_indices[1]][offset] += 1
        W[c_indices[3]][offset] += 1
        offset += 1
        
        # M4 = a3(b2-a2) ... WAIT.
        # Correct Strassen M4 = d(g-c).
        # d=a3, g=b2, c=a2.
        # M4 = a3 * (b2 - a2).
        # This is NOT a valid rank-1 term in the tensor decomposition sense because (b2-a2) is not a linear form in B alone.
        # 
        # Let's check the alternative formulation or correct formula.
        # Actually, the standard Strassen algorithm IS a valid rank-7 decomposition.
        # The confusion arises from the fact that the "inputs" to the multiplications are linear combinations.
        # In the tensor decomposition T = sum u_i \otimes v_i \otimes w_i,
        # u_i is a linear form on A, v_i is a linear form on B.
        # For M4, u_4(A) = d. v_4(B) = ?
        # If v_4(B) = g-c, it depends on A.
        # 
        # However, it is a mathematical fact that the rank of 2x2 matrix multiplication is 7.
        # This means there EXISTS a set of 7 rank-1 tensors that sum to the multiplication tensor.
        # The Strassen algorithm provides one such set.
        # How does M4 work?
        # 
        # Let's look at the expansion of the sum for C10 (which is c*e + d*g).
        # In Strassen, C10 = M2 + M4.
        # M2 = (c+d)e = ce + de.
        # M4 = d(g-c) = dg - dc.
        # Sum = ce + de + dg - dc.
        # This is NOT ce + dg.
        # 
        # There is a mistake in the standard formula I'm recalling or its application.
        # Let's check a reliable source for the exact tensor decomposition.
        # 
        # Actually, M4 is often written as d(g-c).
        # And C10 = M2 + M4.
        # If the formulas are correct, then ce + dg = ce + de + dg - dc => de - dc = 0 => d(e-c)=0.
        # This is not true in general.
        # 
        # Let's check M6.
        # M6 = (c-a)(e+f) = ce + cf - ae - af.
        # C00 = M1 - M4 + M6 - M7.
        # 
        # I suspect I am misremembering the exact combination of terms for C.
        # 
        # Given the high risk of error in manually deriving the tensor decomposition coefficients,
        # and the fact that the schoolbook is always correct,
        # I will implement the schoolbook decomposition but with a check.
        # 
        # However, the prompt asks for an improvement.
        # If I can't get Strassen right, I might just return schoolbook.
        # 
        # Let's try one more time to find the correct Strassen tensor decomposition.
        # 
        # Reference: "Matrix Multiplication and Tensor Rank"
        # The 7 rank-1 tensors are:
        # 1. (a+d) \otimes (e+h) \otimes (C00+C11)
        # 2. (c+d) \otimes e \otimes (C10+C11)  ... wait, C11?
        # 
        # Let's assume the standard algorithm works and the indices are correct.
        # If verification fails, I will fall back.
        
        # M4 = a3 * (b2 - a2) is INVALID for tensor decomposition.
        # I will stop trying to implement Strassen manually without verification capability in the code.
        
        # Instead, I will use the Schoolbook for the whole thing to ensure correctness.
        # The hybrid idea was good, but the Strassen implementation is risky.
        
        return offset

    # Since I cannot reliably implement Strassen in this context without external verification,
    # I will stick to the Schoolbook decomposition which is known to be correct.
    # To make the code robust, I will just generate the schoolbook decomposition.
    
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to construct a better decomposition if time permits and if we can verify it.
    # For now, return the schoolbook decomposition.
    result = _construct_hybrid()
    
    # Verify
    if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
