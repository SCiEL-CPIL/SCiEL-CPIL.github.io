import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 140.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def check_time():
        return time.time() - t0 < DEADLINE

    def validate_int(U, V, W, r):
        """Validate using integer arithmetic."""
        for i in range(n):
            for j in range(m):
                Uj = U[i*m + j]
                for k in range(p):
                    Vk = V[j*p + k]
                    Wk = W[k*n + i]
                    tot = 0
                    for c in range(r):
                        tot += Uj[c] * Vk[c] * Wk[c]
                    if tot != 1:
                        return False
        return True

    # Winograd Frequency Domain Algorithm (FDTA) using 2-point transforms
    # This approach uses the cyclic convolution structure.
    # For 2x2, the 2-point DFT over a ring (or Winograd 2-point) allows rank reduction.
    # We construct a decomposition for 4x4x4 first, then extend to 4x4x9.
    
    # The 2-point Winograd algorithm for convolution of length 2:
    # a0*a0 + a1*a1 is not the convolution.
    # Convolution (a0, a1) * (b0, b1) = (a0b0, a0b1+a1b0, a1b1).
    # Winograd 2-point:
    # U = [a0+a1, a0]
    # V = [b0+b1, b1]
    # W = [1, 1]
    # This computes the cyclic convolution of length 2: (a0+a1)(b0+b1) = a0b0 + a0b1 + a1b0 + a1b1.
    # This is not the standard linear convolution.
    # However, for matrix multiplication, we can embed the product into a larger convolution or use a specific ring.
    
    # A simpler and more effective strategy for small sizes is to use the known Strassen-based
    # decomposition but optimized. The previous attempt got 114.
    # Let's try a different structural decomposition: 
    # Decompose the 4x4x9 tensor into 9 4x4x1 tensors? No, that's schoolbook.
    # Decompose into 4x4x4 and 4x4x5.
    
    # Let's implement a known low-rank algorithm for 4x4x4.
    # The rank of 4x4x4 is 49 (Strassen).
    # Can we do better? Yes, the "Winograd" 2-point algorithm for 4x4 matrices
    # can be constructed by applying 2-point transforms to rows and columns.
    # The rank of the 4x4x4 tensor using 2-point Winograd is 16? No.
    # Actually, the 2-point Winograd algorithm for N x N matrix multiplication
    # has a rank of N^2 * 2 if N is a power of 2? No.
    
    # Let's use a robust fallback: Strassen for 4x4x4 (rank 49) + Schoolbook for 4x4x5 (rank 20).
    # Total rank = 69.
    # The previous program got 114, which is worse. Why?
    # Because it used a naive Kronecker product of Strassen blocks which is not optimal.
    # A proper Strassen for 4x4x4 uses 7x7 = 49 multiplications.
    
    # Let's build the Strassen 4x4x4 decomposition explicitly.
    # Strassen 2x2x2 has rank 7.
    # 4x4x4 can be seen as 4 blocks of 2x2.
    # The standard Strassen recursion for 4x4x4 yields rank 49.
    
    def strassen_2x2_terms():
        # Returns list of (u, v, w) vectors for the 7 rank-1 terms of 2x2x2
        # u, v, w are length 4 (2x2 flattened)
        terms = []
        # M1 = (a00+a11)(b00+b11)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # c00 += M1, c11 -= M1
        # This is getting complicated to get the signs right.
        # Let's use a known correct set of coefficients for 2x2x2.
        
        # Standard Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to express each C_ij as sum of M_k.
        # Each M_k is a product of a linear form in A and a linear form in B.
        # So each M_k is a rank-1 tensor in the 2x2x2 space.
        
        # Flatten indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # M1: A: [1,0,0,1], B: [1,0,0,1]
        # M2: A: [0,0,1,1], B: [1,0,0,0]
        # M3: A: [1,0,0,0], B: [0,0,1,-1]
        # M4: A: [0,0,0,1], B: [0,1,-1,0]
        # M5: A: [1,1,0,0], B: [0,0,0,1]
        # M6: A: [0,0,1,-1], B: [1,1,0,0]
        # M7: A: [0,1,0,-1], B: [0,0,1,1]
        
        M_As = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        M_Bs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [0, 1, -1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # C coefficients:
        # C00: M1(1) + M4(1) - M5(1) - M7(1)
        # C01: M3(1) + M5(1)
        # C10: M2(1) + M4(1)
        # C11: M1(1) - M2(1) + M3(1) + M7(1)
        
        C_coeffs = [
            [1, 0, 0, 1, -1, 0, -1], # M1..M7 for C00
            [0, 0, 1, 0, 1, 0, 0],   # for C01
            [0, 1, 0, 1, 0, 0, 0],   # for C10
            [1, -1, 1, 0, 0, 0, 1]   # for C11
        ]
        
        return M_As, M_Bs, C_coeffs

    M_As, M_Bs, C_coeffs = strassen_2x2_terms()
    
    # Now build 4x4x4 Strassen decomposition
    # 4x4 matrix is 4 blocks of 2x2.
    # A is 2x2 blocks: A00, A01, A10, A11
    # B is 2x2 blocks: B00, B01, B10, B11
    # C is 2x2 blocks: C00, C01, C10, C11
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # We apply Strassen to each 2x2 block multiplication.
    # Total multiplications: 4 blocks * 7 = 28? No, that's not right.
    # Strassen for 4x4x4 uses 7*7=49 multiplications if we recurse.
    # But we can just use the 2x2 Strassen for each of the 4 sub-products?
    # No, the sub-products are 2x2. So we need 4 * 7 = 28 multiplications?
    # Wait, C00 = A00*B00 + A01*B10. This is sum of two 2x2 products.
    # Each 2x2 product has rank 7. So C00 has rank 14.
    # Total rank = 4 * 14 = 56?
    # But Strassen recursion for 4x4x4 is 49.
    # The difference is that the 49-term Strassen shares multiplications between blocks.
    # Specifically, it uses the 7 Strassen terms on the 4x4 matrices directly? No, Strassen is for 2x2.
    # To get 49, you apply Strassen to 2x2 blocks, but you group them.
    # Actually, the standard Strassen for N=4 is just 4 independent 2x2 Strassen applications?
    # No, that would be 28.
    # Why is it 49?
    # Ah, 4x4x4 can be viewed as 2x2 blocks of 2x2.
    # The tensor rank of 4x4x4 is 49.
    # The decomposition is the Kronecker product of the 2x2x2 Strassen decomposition with itself?
    # Rank(4x4x4) = Rank(2x2x2) * Rank(2x2x2) = 7 * 7 = 49.
    # Yes! The Strassen algorithm for 4x4 is the tensor product of two 2x2 Strassen algorithms.
    
    # So we can generate the 49 terms by taking all pairs of the 7 base terms.
    
    terms_4x4 = []
    for i in range(7):
        for j in range(7):
            # The A vector is the Kronecker product of M_As[i] and M_As[j]
            # But wait, the block structure matters.
            # A is 4x4. Index (r,c) where r,c in 0..3.
            # Block row br = r//2, block col bc = c//2.
            # Inner row ir = r%2, inner col ic = c%2.
            # The Kronecker product structure:
            # A_global[r,c] = A_outer[br,bc] * A_inner[ir,ic] ? No.
            # The tensor product of two 2x2 matrices A and B is A \otimes B.
            # (A \otimes B)_{(i,k),(j,l)} = A_{ij} B_{kl}.
            # So the 4x4 matrix is A \otimes B where A and B are 2x2.
            # This corresponds to the block structure where the "outer" 2x2 determines the block,
            # and the "inner" 2x2 determines the position within the block.
            
            u_vec = np.kron(M_As[i], M_As[j])
            v_vec = np.kron(M_Bs[i], M_Bs[j])
            
            # The C vector is more complex because of the addition of blocks.
            # C = A*B = (A_outer \otimes A_inner) * (B_outer \otimes B_inner)
            # = (A_outer * B_outer) \otimes (A_inner * B_inner)
            # So the C block at (br_outer, bc_outer) is C_outer[br_outer, bc_outer] * (A_inner * B_inner)
            # The C_outer is computed by the outer Strassen terms.
            # The C_inner is computed by the inner Strassen terms.
            
            # So for a pair (i,j), the contribution to C is:
            # For each outer C block (br, bc) and inner C block (ir, ic):
            # C[br*2+ir, bc*2+ic] += C_coeffs_outer[br, bc, i] * C_coeffs_inner[ir, ic, j]
            
            w_vec = np.zeros(16)
            for br in range(2):
                for bc in range(2):
                    coeff_outer = C_coeffs[br*2+bc][i]
                    if coeff_outer == 0: continue
                    for ir in range(2):
                        for ic in range(2):
                            coeff_inner = C_coeffs[ir*2+ic][j]
                            if coeff_inner == 0: continue
                            c_idx = (br*2+ir)*2 + (bc*2+ic)
                            w_vec[c_idx] += coeff_outer * coeff_inner
            
            terms_4x4.append((u_vec, v_vec, w_vec))
            
    # Now extend to 4x4x9
    # We have 49 terms for the first 4 columns of B (and corresponding C columns).
    # For the remaining 5 columns of B (indices 4..8), we can use schoolbook (rank 1 each).
    # Total rank = 49 + 4*5 = 49 + 20 = 69.
    
    r_total = 49 + 20
    U = np.zeros((16, r_total))
    V = np.zeros((36, r_total))
    W = np.zeros((36, r_total))
    
    # Fill the 49 Strassen terms
    for idx, (u, v, w) in enumerate(terms_4x4):
        U[:, idx] = u
        # V index: j*p + k. Here j is 0..3, k is 0..3.
        # v is length 16, flattened as (j*2+ri, k*2+ci)?
        # Wait, the v_vec from kron(M_Bs[i], M_Bs[j]) is length 16.
        # The 2x2 matrix B is flattened as [b00, b01, b10, b11].
        # The 4x4 matrix B is B_outer \otimes B_inner.
        # B[br*2+ri, bc*2+ci] = B_outer[br, bc] * B_inner[ri, ci].
        # So the flattened index is (br*2+ri)*4 + (bc*2+ci).
        # The kron vector is indexed by (br, bc, ri, ci) -> (br*2+ri)*2 + (bc*2+ci)? No.
        # np.kron(A, B) where A is 2x2, B is 2x2.
        # Result is 4x4.
        # Result[br*2+ri, bc*2+ci] = A[br, bc] * B[ri, ci].
        # This matches the block structure.
        # So v_vec is already in the correct 4x4 order (row-major).
        # We need to map this to the 4x9 matrix V.
        # V has shape (m*p, rank) = (4*9, rank).
        # Index in V is j*9 + k.
        # The Strassen terms only involve k=0..3.
        # So for k in 0..3, V[j*9+k] = v_vec[j*4+k].
        
        for j in range(4):
            for k in range(4):
                V[j*9+k, idx] = v_vec[j*4+k]
                
        W[:, idx] = w # w is length 16, corresponding to 4x4 C.
        # W has shape (p*n, rank) = (9*4, rank).
        # Index in W is k*4 + i.
        # The Strassen terms only involve k=0..3.
        # So for k in 0..3, W[k*4+i] = w_vec[k*4+i].
        # w_vec is flattened 4x4 C, index c_row*4 + c_col.
        # c_row = k, c_col = i.
        # So w_vec[k*4+i] is correct.
        
    # Fill the 20 schoolbook terms for k=4..8
    for k in range(4, 9):
        for i in range(4):
            for j in range(4):
                c_idx = 49 + (k-4)*16 + i*4 + j # Wait, 4*5=20 terms.
                # Actually, for each (i,j,k), we need one term.
                # i in 0..3, j in 0..3, k in 4..8.
                # Total 4*4*5 = 80 terms? No.
                # Schoolbook for 4x4x5:
                # C[i,k] = sum_j A[i,j] B[j,k].
                # For each (i,j,k), we have a term A[i,j] * B[j,k] * C[i,k].
                # So for fixed k, we have 4*4=16 terms.
                # Total 5 * 16 = 80 terms.
                # My previous calculation of 20 was wrong.
                # 49 + 80 = 129. This is worse than 114.
                
                # Let's re-evaluate.
                # Is there a better way to handle the 4x4x5 part?
                # Maybe use Strassen for 4x4x4 and 4x4x1?
                # 4x4x1 is rank 16 (schoolbook).
                # So 49 + 16*5 = 129.
                
                # The previous program got 114. How?
                # It used a "Kronecker product approach" which might have shared some terms?
                # Or maybe the 4x4x4 part was not 49?
                # If the 4x4x4 part was 49, and the 4x4x5 part was 65?
                # 49 + 65 = 114.
                # How to get 65 for 4x4x5?
                # Maybe using a mixed decomposition?
                
                pass # This loop is not executed yet, just planning
    
    # Let's try a different approach:
    # Use the 49-term Strassen for the 4x4x4 part.
    # For the 4x4x5 part, use schoolbook (80 terms).
    # Total 129.
    
    # Can we reduce the 4x4x5 part?
    # The tensor 4x4x5 has rank at least 16 (max(4,4,5)^2? No, lower bound is max(n,m,p)^2? No.
    # Lower bound for n x m x p is max(n,m,p)^2? No.
    # Lower bound is n*m, m*p, p*n. So 16, 20, 36.
    # So rank >= 36.
    # Schoolbook is 80.
    # Can we do better?
    # Maybe use Strassen on the 4x4x5 part?
    # Decompose 4x4x5 into 4x4x4 + 4x4x1.
    # 4x4x4 rank 49. 4x4x1 rank 16. Total 65.
    # So 49 (for first 4 cols) + 49 (for next 4 cols) + 16 (for last 1 col) = 114.
    # This matches the previous score!
    
    # So the strategy is:
    # 1. Use 49-term Strassen for cols 0-3.
    # 2. Use 49-term Strassen for cols 4-7.
    # 3. Use 16-term Schoolbook for col 8.
    # Total rank = 49 + 49 + 16 = 114.
    
    # Can we do better than 114?
    # Maybe a different decomposition for 4x4x4?
    # The rank of 4x4x4 is 49. Is it possible to do 4x4x9 with less than 114?
    # If we use a 4x4x9 specific algorithm?
    # Maybe use Winograd 2-point for the whole thing?
    # Winograd 2-point for 4x4x4 has rank 16? No.
    # Let's check the rank of 4x4x4 using Winograd 2-point.
    # The 2-point Winograd algorithm for N x N matrix multiplication
    # has rank N^2 * 2 if N is a power of 2?
    # For N=2, rank=7 (Strassen). Winograd 2-point is not better for N=2.
    # For N=4, Winograd 2-point might be better?
    # Actually, the "Winograd" algorithm often refers to the use of 2-point transforms
    # which reduces the number of multiplications for convolutions.
    # For matrix multiplication, the best known for 4x4 is 49.
    
    # So 114 is likely the best we can get with this block structure.
    # To beat 1.263 (which corresponds to rank ~114/1.263? No, score is seed/value.
    # Seed is 144. 144/114 = 1.263.
    # So the current best is 114.
    # To improve, we need rank < 114.
    # Is there a rank < 114 for 4x4x9?
    # Lower bound is 36.
    # Upper bound is 114.
    # Maybe we can share terms between the two 4x4x4 blocks?
    # The two blocks are independent (different k ranges).
    # So no sharing.
    
    # What if we use a different algorithm for 4x4x9?
    # Maybe treat it as 4x4x8 + 4x4x1?
    # 4x4x8 rank?
    # If we use Strassen for 4x4x4 (49) and 4x4x4 (49), total 98.
    # Plus 4x4x1 (16). Total 114.
    
    # What if we use a 4x4x4 algorithm with rank < 49?
    # Is there one?
    # The rank of 4x4x4 is 49. It is a known result.
    
    # So we are stuck at 114?
    # Unless we use a non-block algorithm.
    # For example, a direct decomposition of 4x4x9.
    # This is a hard problem.
    
    # Let's try to implement the 114-rank decomposition correctly and see if we can optimize it.
    # Maybe we can use a different 4x4x4 decomposition that has fewer non-zero entries?
    # No, the rank is the number of terms.
    
    # Let's just implement the 114-rank decomposition robustly.
    
    r_total = 49 + 49 + 16
    U = np.zeros((16, r_total))
    V = np.zeros((36, r_total))
    W = np.zeros((36, r_total))
    
    # Block 1: k=0..3
    for idx, (u, v, w) in enumerate(terms_4x4):
        U[:, idx] = u
        for j in range(4):
            for k in range(4):
                V[j*9+k, idx] = v_vec[j*4+k] # Bug: v_vec is from last iteration
        W[:, idx] = w
        
    # Fix the V assignment bug
    for idx, (u, v, w) in enumerate(terms_4x4):
        U[:, idx] = u
        for j in range(4):
            for k in range(4):
                V[j*9+k, idx] = v[j*4+k]
        W[:, idx] = w
        
    # Block 2: k=4..7
    for idx, (u, v, w) in enumerate(terms_4x4):
        c_idx = 49 + idx
        U[:, c_idx] = u
        for j in range(4):
            for k in range(4):
                V[j*9+(k+4), c_idx] = v[j*4+k]
        # W index: k*4 + i. Here k is 4..7.
        # So W[(k+4)*4+i, c_idx] = w[k*4+i]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, c_idx] = w[k*4+i]
                
    # Block 3: k=8
    for i in range(4):
        for j in range(4):
            c_idx = 98 + i*4 + j
            U[i*4+j, c_idx] = 1
            V[j*9+8, c_idx] = 1
            W[8*4+i, c_idx] = 1
            
    if not validate_int(U, V, W, r_total):
        # Fallback
        r_fb = n * m * p
        U = np.zeros((nm, r_fb))
        V = np.zeros((mp, r_fb))
        W = np.zeros((pn, r_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
