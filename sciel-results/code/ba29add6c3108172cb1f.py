"""Matrix multiplication 4x4x4 with rank-minimal decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s margin for safety
    
    n, m, p = 4, 4, 4
    
    def _verify_decomposition(U, V, W, verbose=False):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Condition: For all A, B, C = A*B iff C[i,k] = sum_r (U[:,r] . A_vec) * (V[:,r] . B_vec) * W[k*n+i, r]
        """
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank: return False
        for row in V:
            if len(row) != rank: return False
        for row in W:
            if len(row) != rank: return False
            
        # Check on a few random integer matrices to ensure exactness
        # Since entries are integers/half-integers, if it works for small random ints, it's likely exact.
        # But to be safe, we check the tensor identity directly if possible, or use many random tests.
        
        # The tensor T[i,j,k] is 1 if we are computing (A*B)_{i,k} using A_{i,j} and B_{j,k}.
        # Actually, the identity is:
        # (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
        # Decomposition: (A*B)_{i,k} = sum_{r} (sum_{i',j'} U[i'm+j',r] A[i',j']) (sum_{j',k'} V[j'p+k',r] B[j',k']) W[k*n+i,r]
        
        # This must hold for all A, B.
        # This implies that for each (i,k), the coefficient of A_{i',j'} B_{j',k'} in the expansion must match the delta function.
        # Specifically, the term A_{i,j} B_{j,k} must appear with coefficient 1 if i'=i, j'=j, k'=k and 0 otherwise?
        # No, the expansion is:
        # sum_r W[k*n+i,r] (sum_{i',j'} U[i'm+j',r] A[i',j']) (sum_{j',k'} V[j'p+k',r] B[j',k'])
        # = sum_{i',j',j',k'} W[k*n+i,r] U[i'm+j',r] V[j'p+k',r] A[i',j'] B[j',k']
        # We want this to equal sum_j A[i,j] B[j,k] (note the indices i,k are fixed for the output).
        # So for fixed i,k, the sum over r of W[k*n+i,r] U[i'm+j',r] V[j'p+k',r] must be:
        # 1 if i'=i, j'=j, k'=j (wait, j' is the inner index)
        # Let's trace indices carefully.
        # Output C[i,k].
        # Term in sum: A[i', j'] * B[j', k'].
        # We need the coefficient of A[i', j'] * B[j', k'] in the expression for C[i,k] to be:
        # 1 if i'==i and j'==j and k'==k? No.
        # The standard matrix multiplication is C[i,k] = sum_{j} A[i,j] B[j,k].
        # So the only non-zero terms in the sum over j' are those where j'=j.
        # And we sum over j.
        # So the coefficient of A[i', j'] B[j', k'] in C[i,k] is:
        # 1 if i'==i and k'==k and j' is the summation index?
        # Wait, the decomposition is a sum of rank-1 tensors.
        # T = sum_r u_r \otimes v_r \otimes w_r
        # u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}.
        # The bilinear form is <u_r, A> <v_r, B> w_r.
        # The component (i,k) of the output is sum_r <u_r, A> <v_r, B> w_r[k*n+i].
        # <u_r, A> = sum_{i',j'} U[i'm+j',r] A[i',j']
        # <v_r, B> = sum_{j',k'} V[j'p+k',r] B[j',k']
        # So C[i,k] = sum_r W[k*n+i,r] (sum_{i',j'} U[i'm+j',r] A[i',j']) (sum_{j',k'} V[j'p+k',r] B[j',k'])
        # = sum_{i',j',j',k'} (sum_r W[k*n+i,r] U[i'm+j',r] V[j'p+k',r]) A[i',j'] B[j',k']
        # We require C[i,k] = sum_{j} A[i,j] B[j,k].
        # This means that for fixed i,k, the coefficient of A[i',j'] B[j',k'] in the above sum must be:
        # 1 if i'==i, k'==k, and j' is the index being summed?
        # No, the RHS is sum_j A[i,j] B[j,k].
        # So the term A[i,j] B[j,k] has coefficient 1.
        # Any other term A[i',j'] B[j',k'] must have coefficient 0.
        # So for fixed i,k:
        # sum_r W[k*n+i,r] U[i'm+j',r] V[j'p+k',r] = 1 if i'==i, k'==k, and j'==j (for some j? No, j is summed on RHS).
        # Actually, the RHS is a specific linear combination of products A[i',j']B[j',k'].
        # It contains terms A[i,j]B[j,k] for j=0..m-1.
        # So for fixed i,k, the coefficient of A[i',j']B[j',k'] is:
        # 1 if i'==i, k'==k, and j' is any valid index? No.
        # The term is A[i,j] B[j,k]. The indices are i, j, k.
        # In the product A[i',j'] B[j',k'], the middle index must match: j' (from A) must equal j' (from B).
        # So the term is A[i',j'] B[j',k'].
        # We want this to be non-zero only if i'==i, k'==k, and the middle index is free?
        # No, the matrix multiplication is defined as:
        # (AB)_{ik} = \sum_{j} A_{ij} B_{jk}.
        # So for a fixed output (i,k), the contributing terms are A_{i0}B_{0k}, A_{i1}B_{1k}, ..., A_{i,m-1}B_{m-1,k}.
        # So in the expansion sum_{i',j',k'} coeff_{i,k}(i',j',k') A[i',j'] B[j',k'],
        # coeff_{i,k}(i',j',k') = 1 if i'==i and k'==k and j' is the summation index?
        # No, j' is a specific index in the product A[i',j']B[j',k'].
        # So coeff_{i,k}(i',j',k') = 1 if i'==i, k'==k, and j' is any index?
        # Let's check: if i'=i, k'=k, then the term is A[i,j'] B[j',k].
        # The sum over j' of these terms is exactly (AB)_{ik}.
        # So yes, for fixed i,k, the coefficient is 1 if i'==i and k'==k (regardless of j'), and 0 otherwise.
        
        # So the condition is:
        # For all i, k, i', j', k':
        # sum_r W[k*n+i,r] U[i'm+j',r] V[j'p+k',r] = 1 if i'==i and k'==k else 0
        
        # This is a system of linear equations.
        # We can verify this directly.
        
        # Dimensions: i,k in 0..3 (16 outputs)
        # i',k' in 0..3 (16 combos)
        # j' in 0..3 (4 combos)
        # Total 16*16*4 = 1024 equations.
        # Variables are the rank r terms.
        
        # Let's build the matrix M where M[eq_idx, r] = ...
        # But we have U, V, W already. We just need to check if the identity holds.
        
        # To be efficient, we can check on a basis of inputs.
        # Or just check the tensor identity directly.
        
        # Let's construct the tensor T from U,V,W and compare to the identity tensor.
        # T[i, j, k] is not the right way to think.
        # The identity is:
        # sum_r W[k*n+i,r] U[i'm+j',r] V[j'p+k',r] = delta_{i,i'} delta_{k,k'}
        
        # We can verify this for all i,k,i',k',j'.
        # 16 * 16 * 4 = 1024 checks.
        # Each check is a dot product of length rank (49).
        # 1024 * 49 ops is trivial.
        
        # Optimization: Precompute for each r, the matrices U_r (n*m) and V_r (m*p).
        # Actually, we can just loop.
        
        # Let's use NumPy for speed if available, but pure Python is fine for 1024*49.
        
        # Convert to lists of floats for calculation
        Uf = [[float(x) for x in row] for row in U]
        Vf = [[float(x) for x in row] for row in V]
        Wf = [[float(x) for x in row] for row in W]
        
        for i in range(n):
            for k in range(p):
                for i_p in range(n):
                    for k_p in range(p):
                        for j_p in range(m):
                            s = 0.0
                            for r in range(rank):
                                s += Wf[k*n+i][r] * Uf[i_p*m+j_p][r] * Vf[j_p*p+k_p][r]
                            expected = 1.0 if (i == i_p and k == k_p) else 0.0
                            if abs(s - expected) > 1e-9:
                                return False
        return True

    def _strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen (7 terms)."""
        U = []
        V = []
        W = []
        
        # A_vec = [A00, A01, A10, A11]
        # B_vec = [B00, B01, B10, B11]
        # C_vec indexing: W[k*2+i]
        
        # M1 = (A00 + A11)(B00 + B11)
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        # C00 += M1, C11 += M1
        W.append([1, 0, 0, 1]) # W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
        
        # M2 = (A10 + A11)B00
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        # C10 += M2, C11 -= M2
        W.append([0, 1, 0, -1])
        
        # M3 = A00(B01 - B11)
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        # C01 += M3, C11 += M3
        W.append([0, 0, 1, 1])
        
        # M4 = A11(B10 - B00)
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])
        # C00 += M4, C10 += M4
        W.append([1, 1, 0, 0])
        
        # M5 = (A00 + A01)B11
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        # C00 -= M5, C01 += M5
        W.append([-1, 0, 1, 0])
        
        # M6 = (A10 - A00)(B00 + B01)
        U.append([-1, 0, 1, 0])
        V.append([1, 1, 0, 0])
        # C11 += M6
        W.append([0, 0, 0, 1])
        
        # M7 = (A01 - A11)(B10 + B11)
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        # C00 += M7
        W.append([1, 0, 0, 0])
        
        return U, V, W

    def _build_4x4_strassen():
        """
        Build a rank-49 decomposition for 4x4x4.
        Strategy:
        Split 4x4 into 2x2 blocks of 2x2 matrices.
        A = [[A00, A01], [A10, A11]]
        B = [[B00, B01], [B10, B11]]
        C = A*B
        C00 = A00B00 + A01B10
        C01 = A00B01 + A01B11
        C10 = A10B00 + A11B10
        C11 = A10B01 + A11B11
        
        Each Ci,j is a sum of two 2x2 matrix multiplications.
        We can use Strassen for each 2x2 multiplication, but that would be 8 * 7 = 56 terms.
        However, we can share terms if we are clever?
        Actually, the standard recursive Strassen for 4x4 uses 7^2 = 49 terms.
        How?
        It treats the 4x4 multiplication as a 2x2 block multiplication where the "elements" are 2x2 matrices.
        But matrix multiplication of 2x2 blocks is not commutative and requires care.
        The recursive Strassen algorithm applies Strassen's 7 multiplications to the block structure.
        Let M1..M7 be the Strassen products of the blocks.
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        Each Mx is a 2x2 matrix product, which itself requires 7 scalar multiplications (if we recurse) or 4 (schoolbook).
        If we use Strassen for the inner 2x2 products, each Mx takes 7 terms.
        Total terms = 7 (outer) * 7 (inner) = 49.
        
        So we need to construct the 49-term decomposition by "expanding" each of the 7 block-level Strassen terms into 7 scalar-level Strassen terms.
        """
        
        U2, V2, W2 = _strassen_2x2()
        rank2 = 7
        
        # The outer Strassen structure defines 7 "virtual" matrix multiplications.
        # Let's denote the outer terms as T1..T7.
        # T1 = (A00+A11)(B00+B11)
        # T2 = (A10+A11)B00
        # T3 = A00(B01-B11)
        # T4 = A11(B10-B00)
        # T5 = (A00+A01)B11
        # T6 = (A10-A00)(B00+B01)
        # T7 = (A01-A11)(B10+B11)
        
        # For each Ti, we need to compute the product of two 2x2 matrices.
        # The first matrix is a linear combination of blocks of A.
        # The second matrix is a linear combination of blocks of B.
        
        # We will build U, V, W of size 49.
        # Index r in 0..48 corresponds to (outer_term, inner_term) where outer in 0..6, inner in 0..6.
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # Helper to get block indices in flattened vector
        # A is 4x4, flattened row-major: A[i][j] is at index i*4+j.
        # Block (bi, bj) of A (2x2 blocks) consists of elements:
        # A[2*bi + li][2*bj + lj] for li, lj in 0,1.
        # Indices: (2*bi+li)*4 + (2*bj+lj)
        
        # For a linear combination of blocks, e.g., A00 + A11:
        # A00 is block (0,0), A11 is block (1,1).
        # The resulting 2x2 matrix M has entries M[li][lj] = A[0*2+li][0*2+lj] + A[1*2+li][1*2+lj]
        
        # We need to map this to the U vector (size 16).
        # For a specific outer term t, the "left matrix" L is a 2x2 matrix formed by linear combos of A blocks.
        # The "right matrix" R is a 2x2 matrix formed by linear combos of B blocks.
        # Then we compute L*R using the inner Strassen decomposition.
        
        # Let's define the outer terms explicitly as (L_blocks, R_blocks, signs)
        # L_blocks: list of (block_row, block_col, sign)
        # R_blocks: list of (block_row, block_col, sign)
        
        outer_terms = [
            # T1: (A00+A11)(B00+B11)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # T2: (A10+A11)B00
            ([(1,0,1), (1,1,1)], [(0,0,1)]),
            # T3: A00(B01-B11)
            ([(0,0,1)], [(0,1,1), (1,1,-1)]),
            # T4: A11(B10-B00)
            ([(1,1,1)], [(1,0,1), (0,0,-1)]),
            # T5: (A00+A01)B11
            ([(0,0,1), (0,1,1)], [(1,1,1)]),
            # T6: (A10-A00)(B00+B01)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            # T7: (A01-A11)(B10+B11)
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ]
        
        # For each outer term t, and each inner Strassen term s (0..6):
        # The inner Strassen term s computes a scalar product of linear combos of L's entries and R's entries.
        # U2[s] gives the coefficients for L's entries.
        # V2[s] gives the coefficients for R's entries.
        # W2[s] gives the contributions to the output 2x2 block C_t.
        
        # Wait, the output of the outer term t is a 2x2 matrix C_t.
        # The final C is assembled from C_t using the outer Strassen reconstruction.
        # C00 = T1 + T4 - T5 + T7
        # C01 = T3 + T5
        # C10 = T2 + T4
        # C11 = T1 - T2 + T3 + T6
        
        # So for each outer term t, its result C_t is added to specific blocks of C with specific signs.
        
        # Reconstruction signs:
        # C00: T1(+), T4(+), T5(-), T7(+)
        # C01: T3(+), T5(+)
        # C10: T2(+), T4(+)
        # C11: T1(+), T2(-), T3(+), T6(+)
        
        recon = {
            (0,0): [(0,1), (3,1), (4,-1), (6,1)], # T1, T4, T5, T7
            (0,1): [(2,1), (4,1)],                # T3, T5
            (1,0): [(1,1), (3,1)],                # T2, T4
            (1,1): [(0,1), (1,-1), (2,1), (5,1)], # T1, T2, T3, T6
        }
        
        # Now, for each outer term t (0..6) and inner term s (0..6):
        # The scalar product is:
        # prod = (sum_{li,lj} U2[s][li*2+lj] * L[li][lj]) * (sum_{li,lj} V2[s][li*2+lj] * R[li][lj])
        
        # L[li][lj] = sum_{(br,bc,sgn) in L_blocks} sgn * A[2*br+li][2*bc+lj]
        # R[li][lj] = sum_{(br,bc,sgn) in R_blocks} sgn * B[2*br+li][2*bc+lj]
        
        # So prod = sum_{li,lj, li',lj'} U2[s][li*2+lj] V2[s][li'*2+lj'] 
        #          * (sum_{Lb} sgn_L A[2*br_L+li][2*bc_L+lj]) 
        #          * (sum_{Rb} sgn_R B[2*br_R+li'][2*bc_R+lj'])
        
        # This expands to a sum of terms A[...]*B[...].
        # We need to add these to U4, V4, W4.
        
        for t in range(7):
            L_blocks, R_blocks = outer_terms[t]
            for s in range(7):
                r_idx = t * 7 + s
                
                # Extract U2[s] and V2[s]
                # U2[s] is length 4: [coeff for L00, L01, L10, L11]
                u_s = U2[s]
                v_s = V2[s]
                
                # For each non-zero coefficient in u_s and v_s:
                for li in range(2):
                    for lj in range(2):
                        cu = u_s[li*2+lj]
                        if cu == 0: continue
                        for li_p in range(2):
                            for lj_p in range(2):
                                cv = v_s[li_p*2+lj_p]
                                if cv == 0: continue
                                
                                # Now we have a term cu * cv * L[li][lj] * R[li_p][lj_p]
                                # L[li][lj] is a linear combo of A blocks.
                                # R[li_p][lj_p] is a linear combo of B blocks.
                                
                                # We need to add to U4, V4, W4 such that:
                                # sum_r U4[i'm+j',r] A[i',j'] V4[j'p+k',r] B[j',k'] W4[k*n+i,r]
                                # contributes to the correct C[i,k].
                                
                                # The term is cu * cv * (sum_{Lb} sgn_L A[idx_A]) * (sum_{Rb} sgn_R B[idx_B])
                                # = sum_{Lb, Rb} cu * cv * sgn_L * sgn_R * A[idx_A] * B[idx_B]
                                
                                # This product A[idx_A] B[idx_B] contributes to the output C.
                                # Which output?
                                # The outer term t produces a 2x2 block C_t.
                                # The inner term s produces a specific entry of C_t, say C_t[ci, cj].
                                # W2[s] tells us which entries of the 2x2 result are affected.
                                # W2[s] is length 4: [coeff for C00, C01, C10, C11] of the block.
                                # Wait, W2[s] in _strassen_2x2 is defined as W[k*2+i].
                                # So W2[s][0] is C00, W2[s][1] is C10, W2[s][2] is C01, W2[s][3] is C11.
                                # Let's re-verify _strassen_2x2 W indexing.
                                # In _strassen_2x2:
                                # W.append([1, 0, 0, 1]) for M1.
                                # C00 = M1... so W[0] corresponds to C00.
                                # C10 = ... so W[1] corresponds to C10.
                                # C01 = ... so W[2] corresponds to C01.
                                # C11 = ... so W[3] corresponds to C11.
                                # Yes, W2[s] = [c00, c10, c01, c11].
                                
                                # So for each entry (ci, cj) of the block C_t:
                                # coeff_in_block = W2[s][cj*2+ci] ?
                                # Let's check: W2[s][0] is C00 (ci=0, cj=0). Index 0.
                                # W2[s][1] is C10 (ci=1, cj=0). Index 1.
                                # W2[s][2] is C01 (ci=0, cj=1). Index 2.
                                # W2[s][3] is C11 (ci=1, cj=1). Index 3.
                                # So index = cj*2 + ci.
                                
                                for ci in range(2):
                                    for cj in range(2):
                                        w_block = W2[s][cj*2+ci]
                                        if w_block == 0: continue
                                        
                                        # This term contributes w_block * cu * cv to C_t[ci, cj].
                                        # C_t[ci, cj] is part of the final C.
                                        # The block t corresponds to a specific position in the 4x4 C?
                                        # No, t is an outer Strassen term.
                                        # The final C is assembled from T1..T7.
                                        # T_t is a 2x2 matrix.
                                        # T_t[ci, cj] contributes to C[2*block_row + ci][2*block_col + cj]?
                                        # No, the outer Strassen computes the 2x2 block matrix C_blocks.
                                        # C_blocks[br, bc] is the 2x2 block at position (br, bc).
                                        # So T_t is not a block, it's a "virtual" product.
                                        # The reconstruction adds T_t to specific blocks.
                                        # recon[(br, bc)] contains (t, sign).
                                        # So T_t[ci, cj] * sign contributes to C[2*br+ci][2*bc+cj].
                                        
                                        # So for each (br, bc, sign) in recon:
                                        # if t is in recon[(br, bc)]:
                                        #   final_coeff = sign * w_block * cu * cv
                                        #   This contributes to C[2*br+ci][2*bc+cj].
                                        
                                        for (br, bc), pairs in recon.items():
                                            for (tt, sign) in pairs:
                                                if tt == t:
                                                    final_coeff = sign * w_block * cu * cv
                                                    if final_coeff == 0: continue
                                                    
                                                    # Now we have a contribution to C[2*br+ci][2*bc+cj].
                                                    # The term is final_coeff * A[idx_A] * B[idx_B].
                                                    # We need to add this to the decomposition.
                                                    # U4[idx_A, r_idx] += final_coeff / (something)?
                                                    # No, the decomposition is:
                                                    # C[i,k] = sum_r U[i'm+j',r] A[i',j'] V[j'p+k',r] B[j',k'] W[k*n+i,r]
                                                    
                                                    # Here, the specific term is:
                                                    # A[idx_A] * B[idx_B] * final_coeff
                                                    # This should appear in the sum for C[i,k] where i=2*br+ci, k=2*bc+cj.
                                                    
                                                    # So we set:
                                                    # U4[idx_A, r_idx] += final_coeff
                                                    # V4[idx_B, r_idx] += 1  ??? No.
                                                    # The term is U * A * V * B * W.
                                                    # We have A[idx_A] * B[idx_B].
                                                    # So we can set U4[idx_A, r_idx] = 1, V4[idx_B, r_idx] = 1, W4[k*n+i, r_idx] = final_coeff.
                                                    # Or distribute the coefficient.
                                                    # Let's set U4[idx_A, r_idx] += final_coeff, V4[idx_B, r_idx] += 1, W4[k*n+i, r_idx] += 1.
                                                    # But wait, idx_A and idx_B are specific indices.
                                                    # There might be multiple (Lb, Rb) pairs contributing to the same r_idx.
                                                    # We just accumulate.
                                                    
                                                    i_out = 2*br + ci
                                                    k_out = 2*bc + cj
                                                    w_idx = k_out * n + i_out
                                                    
                                                    # We need to identify idx_A and idx_B.
                                                    # idx_A comes from L_blocks.
                                                    # L[li][lj] = sum_{(br_L, bc_L, sgn_L) in L_blocks} sgn_L * A[2*br_L+li][2*bc_L+lj]
                                                    # So for a specific (br_L, bc_L, sgn_L):
                                                    # idx_A = (2*br_L+li)*4 + (2*bc_L+lj)
                                                    # coeff_A = sgn_L * cu
                                                    
                                                    # Similarly for B:
                                                    # idx_B = (2*br_R+li_p)*4 + (2*bc_R+lj_p)
                                                    # coeff_B = sgn_R * cv
                                                    
                                                    # The term is coeff_A * coeff_B * A[idx_A] * B[idx_B] * w_block * sign
                                                    # = (sgn_L * cu) * (sgn_R * cv) * w_block * sign * A[idx_A] * B[idx_B]
                                                    
                                                    # So we add to U4[idx_A, r_idx] the value (sgn_L * cu)
                                                    # and to V4[idx_B, r_idx] the value (sgn_R * cv)
                                                    # and to W4[w_idx, r_idx] the value (w_block * sign)
                                                    
                                                    # Wait, this logic is flawed because idx_A depends on Lb, and idx_B depends on Rb.
                                                    # And we are iterating over Lb and Rb in the inner loops?
                                                    # No, I didn't iterate over Lb and Rb in the code above.
                                                    # I need to iterate over L_blocks and R_blocks.
                                                    
                                                    pass # Placeholder, logic needs restructuring.
        
        # Restructure:
        # For each t, s:
        #   r_idx = t*7 + s
        #   For each (li, lj, cu) in non-zeros of U2[s]:
        #     For each (li_p, lj_p, cv) in non-zeros of V2[s]:
        #       For each (br_L, bc_L, sgn_L) in L_blocks:
        #         idx_A = (2*br_L+li)*4 + (2*bc_L+lj)
        #         coeff_A = sgn_L * cu
        #         For each (br_R, bc_R, sgn_R) in R_blocks:
        #           idx_B = (2*br_R+li_p)*4 + (2*bc_R+lj_p)
        #           coeff_B = sgn_R * cv
        #           For each (ci, cj, w_block) in non-zeros of W2[s]:
        #             For each (br, bc, sign) in recon where t is present:
        #               i_out = 2*br + ci
        #               k_out = 2*bc + cj
        #               w_idx = k_out * 4 + i_out
        #               coeff_W = sign * w_block
        #               
        #               # We want U[idx_A, r_idx] * V[idx_B, r_idx] * W[w_idx, r_idx] to equal coeff_A * coeff_B * coeff_W
        #               # But we are accumulating into U, V, W.
        #               # If we set U[idx_A, r_idx] += coeff_A, V[idx_B, r_idx] += coeff_B, W[w_idx, r_idx] += coeff_W
        #               # Then the product is not simply the product of the sums.
        #               # The decomposition requires that the sum of products equals the tensor.
        #               # If we add multiple terms to the same r_idx, it's not a rank-1 term anymore.
        #               # Each r_idx must be a rank-1 term: U_r, V_r, W_r.
        #               # So U_r must be a vector, V_r a vector, W_r a vector.
        #               # The contribution of term r is (U_r . A) (V_r . B) W_r.
        #               # (U_r . A) = sum_j U_r[j] A[j].
        #               # So if U_r has non-zeros at multiple indices, it's a linear combination.
        #               # That's fine.
        #               # So we can just accumulate:
        #               U4[idx_A, r_idx] += coeff_A
        #               V4[idx_B, r_idx] += coeff_B
        #               W4[w_idx, r_idx] += coeff_W
        #               
        #               Wait, is this correct?
        #               The term r contributes:
        #               (sum_{j} U4[j, r_idx] A[j]) * (sum_{k} V4[k, r_idx] B[k]) * W4[w_idx, r_idx]
        #               If we add coeff_A to U4[idx_A, r_idx] and coeff_B to V4[idx_B, r_idx],
        #               then the product includes the term coeff_A * A[idx_A] * coeff_B * B[idx_B] * W4[w_idx, r_idx].
        #               This is exactly coeff_A * coeff_B * W4[w_idx, r_idx] * A[idx_A] B[idx_B].
        #               So yes, we just accumulate.
                
        # Let's implement this correctly.
        
        U4 = [[0.0]*49 for _ in range(16)]
        V4 = [[0.0]*49 for _ in range(16)]
        W4 = [[0.0]*49 for _ in range(16)]
        
        for t in range(7):
            L_blocks, R_blocks = outer_terms[t]
            for s in range(7):
                r_idx = t * 7 + s
                u_s = U2[s]
                v_s = V2[s]
                w_s = W2[s]
                
                # Non-zero entries in u_s
                for li in range(2):
                    for lj in range(2):
                        cu = u_s[li*2+lj]
                        if cu == 0: continue
                        for li_p in range(2):
                            for lj_p in range(2):
                                cv = v_s[li_p*2+lj_p]
                                if cv == 0: continue
                                
                                # Iterate over L_blocks and R_blocks
                                for br_L, bc_L, sgn_L in L_blocks:
                                    idx_A = (2*br_L+li)*4 + (2*bc_L+lj)
                                    coeff_A = sgn_L * cu
                                    
                                    for br_R, bc_R, sgn_R in R_blocks:
                                        idx_B = (2*br_R+li_p)*4 + (2*bc_R+lj_p)
                                        coeff_B = sgn_R * cv
                                        
                                        # Iterate over output entries of the block
                                        for ci in range(2):
                                            for cj in range(2):
                                                w_block = w_s[cj*2+ci]
                                                if w_block == 0: continue
                                                
                                                # Iterate over reconstruction
                                                for (br, bc), pairs in recon.items():
                                                    for (tt, sign) in pairs:
                                                        if tt == t:
                                                            i_out = 2*br + ci
                                                            k_out = 2*bc + cj
                                                            w_idx = k_out * 4 + i_out
                                                            coeff_W = sign * w_block
                                                            
                                                            U4[idx_A][r_idx] += coeff_A
                                                            V4[idx_B][r_idx] += coeff_B
                                                            W4[w_idx][r_idx] += coeff_W
        
        # Convert to integers if possible
        U_int = [[int(round(x)) for x in row] for row in U4]
        V_int = [[int(round(x)) for x in row] for row in V4]
        W_int = [[int(round(x)) for x in row] for row in W4]
        
        if _verify_decomposition(U_int, V_int, W_int):
            return {"U": U_int, "V": V_int, "W": W_int}
        
        # If integer rounding failed, try half-integers?
        # Or just return the float version if it verifies?
        # The task says "integer/half-integer entries".
        # Let's check if they are half-integers.
        U_half = [[int(round(x*2))/2 for x in row] for row in U4]
        V_half = [[int(round(x*2))/2 for x in row] for row in V4]
        W_half = [[int(round(x*2))/2 for x in row] for row in W4]
        
        if _verify_decomposition(U_half, V_half, W_half):
            return {"U": U_half, "V": V_half, "W": W_half}
            
        # Fallback to schoolbook
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    result = _build_4x4_strassen()
    return result
# EVOLVE-BLOCK-END
