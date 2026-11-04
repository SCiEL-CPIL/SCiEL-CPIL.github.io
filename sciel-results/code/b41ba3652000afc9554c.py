"""Tensor-train / hierarchical compression for 4x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Strategy: Split k-index (0..6) into two groups:
    #   Group A: k in {0,1,2,3} (4 values) -> use Strassen for 4x4x4
    #   Group B: k in {4,5,6} (3 values)   -> use schoolbook for 4x4x3
    #
    # For 4x4x4 via Strassen at block level:
    # View 4x4x4 as 2x2 blocks of 2x2 matrices.
    # Use 7 Strassen multiplications, each a 2x2x2 product.
    # Each 2x2x2 product uses schoolbook (rank 4).
    # Total rank for 4x4x4 part = 7 * 4 = 28.
    #
    # For 4x4x3 via schoolbook: rank = 4*4*3 = 48.
    #
    # Total rank = 28 + 48 = 76.
    
    # Strassen decomposition for 2x2x2 (rank 7):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For the 4x4x4 part, we have 2x2 blocks.
    # A[i,j] where i = 2*I + ip, j = 2*J + jp
    # B[j,k] where j = 2*J + jp, k = 2*K + kp
    # C[i,k] where i = 2*I + ip, k = 2*K + kp
    #
    # Strassen term t has:
    # - a_coeffs: dict (I,J) -> coeff for A block
    # - b_coeffs: dict (J,K) -> coeff for B block
    # - c_coeffs: dict (I,K) -> coeff for C block
    #
    # The Strassen term computes:
    # M_t = (sum_{I,J} a_IJ * A[I,J]) * (sum_{J,K} b_JK * B[J,K])
    #
    # Where A[I,J] is the 2x2 submatrix with rows 2I,2I+1 and cols 2J,2J+1
    # B[J,K] is the 2x2 submatrix with rows 2J,2J+1 and cols 2K,2K+1
    #
    # M_t is a 2x2 matrix. Entry (ip,kp) of M_t:
    # M_t[ip,kp] = sum_{J} (sum_{I} a_IJ * A[2I+ip, 2J+jp]) * (sum_{K} b_JK * B[2J+jp, 2K+kp])
    #
    # Wait, this is still a sum. Let me think more carefully.
    #
    # Actually, the Strassen algorithm works as follows for 2x2x2:
    # We form 7 linear combinations of the 4 A entries and 4 B entries,
    # multiply them (7 scalar multiplications), then combine the results.
    #
    # For block Strassen (4x4x4 as 2x2 blocks of 2x2):
    # We form 7 linear combinations of the 4 A blocks and 4 B blocks.
    # Each "multiplication" is a 2x2 matrix multiplication.
    #
    # So for Strassen term t:
    # A_comb = sum_{(I,J) in a_coeffs} a_IJ * A_block[I,J]  (a 2x2 matrix)
    # B_comb = sum_{(J,K) in b_coeffs} b_JK * B_block[J,K]  (a 2x2 matrix)
    # M_t = A_comb * B_comb  (a 2x2 matrix)
    #
    # Then C_block[I,K] += c_IK * M_t for each (I,K) in c_coeffs.
    #
    # Now, M_t = A_comb * B_comb. Using schoolbook for this 2x2 multiplication:
    # M_t[ip,kp] = sum_{jp in {0,1}} A_comb[ip,jp] * B_comb[jp,kp]
    #
    # A_comb[ip,jp] = sum_{(I,J): 2I+ip = ip, 2J+jp = jp} a_IJ * A[2I+ip, 2J+jp]
    #                = sum_{I,J: I=ip/2... wait, ip is 0 or 1, so I is determined by ip?
    # No. A_block[I,J][ip,jp] = A[2I+ip, 2J+jp].
    # A_comb[ip,jp] = sum_{I,J} a_IJ * A[2I+ip, 2J+jp]
    #
    # Similarly B_comb[jp,kp] = sum_{J,K} b_JK * B[2J+jp, 2K+kp]
    #
    # So M_t[ip,kp] = sum_{jp} [sum_{I,J} a_IJ * A[2I+ip, 2J+jp]] * [sum_{J,K} b_JK * B[2J+jp, 2K+kp]]
    #
    # This is a bilinear form. To express it as a sum of rank-1 terms in the
    # full 4x4x4 tensor, I need to expand the products.
    #
    # M_t[ip,kp] = sum_{jp} sum_{I,J} sum_{J',K} a_IJ * b_J'K * A[2I+ip, 2J+jp] * B[2J+jp, 2J'+kp]
    #
    # Wait, the jp index in A_comb and B_comb must match. Let me redo:
    # M_t[ip,kp] = sum_{jp in {0,1}} A_comb[ip,jp] * B_comb[jp,kp]
    # A_comb[ip,jp] = sum_{I,J} a_IJ * A[2I+ip, 2J+jp]
    # B_comb[jp,kp] = sum_{J,K} b_JK * B[2J+jp, 2K+kp]
    #
    # So M_t[ip,kp] = sum_{jp} (sum_{I,J} a_IJ * A[2I+ip, 2J+jp]) * (sum_{J,K} b_JK * B[2J+jp, 2K+kp])
    #
    # This is a product of two sums, which expands to:
    # = sum_{jp} sum_{I,J} sum_{J',K} a_IJ * b_J'K * A[2I+ip, 2J+jp] * B[2J+jp, 2J'+kp]
    #
    # For this to be a valid decomposition, the jp indices in A and B must match:
    # A[2I+ip, 2J+jp] and B[2J+jp, 2J'+kp] share the index 2J+jp.
    # So J and J' must be the same! Let me relabel:
    # = sum_{jp} sum_{I,J,K} a_IJ * b_JK * A[2I+ip, 2J+jp] * B[2J+jp, 2K+kp]
    #
    # Now, the contribution to C from Strassen term t:
    # C[2I+ip, 2K+kp] += c_IK * M_t[ip,kp]
    # = c_IK * sum_{jp} sum_{I',J,K'} a_I'J * b_JK' * A[2I'+ip, 2J+jp] * B[2J+jp, 2K'+kp]
    #
    # This is getting messy with index collisions. Let me use different index names.
    #
    # Let me use: for Strassen term t, the contribution to C[i,k] where i=2I+ip, k=2K+kp:
    # C[2I+ip, 2K+kp] += c_IK * sum_{jp} (sum_{I_a,J_a} a_IaJa * A[2Ia+ip, 2Ja+jp]) * (sum_{J_b,K_b} b_JbKb * B[2Jb+jp, 2Kb+kp])
    #
    # For this to be a proper bilinear form matching the matrix product:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The term A[2Ia+ip, 2Ja+jp] * B[2Jb+jp, 2Kb+kp] only contributes to C[2Ia+ip, 2Kb+kp]
    # if 2Ja+jp = 2Jb+jp, i.e., Ja = Jb.
    #
    # So the effective contribution is:
    # C[2I+ip, 2K+kp] += c_IK * sum_{jp} sum_{Ia,J,Kb} a_IaJ * b_JKb * A[2Ia+ip, 2J+jp] * B[2J+jp, 2Kb+kp]
    #
    # But we need this to equal:
    # C[2I+ip, 2K+kp] = sum_J sum_jp A[2I+ip, 2J+jp] * B[2J+jp, 2K+kp]
    #
    # So the Strassen decomposition must satisfy:
    # For all I, K, ip, kp:
    # sum_t c_t_IK * sum_{jp} sum_{Ia,J,Kb} [a_t_IaJ * b_t_JKb * delta(Ia,I) * delta(Kb,K)] = 1
    #
    # Wait, that's not right either. Let me think about this differently.
    #
    # The key insight: each "rank-1 term" in the full tensor decomposition
    # corresponds to a single (i,j,k) triple with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    #
    # Strassen doesn't directly give rank-1 terms of the full tensor. Instead,
    # it gives a different factorization structure.
    #
    # Let me think about it this way. The multiplication tensor T for 4x4x4 is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    #
    # A rank-r decomposition is: T = sum_{t=1}^{r} u_t * v_t * w_t
    # where u_t is a vector in R^{16}, v_t in R^{16}, w_t in R^{16}.
    #
    # Strassen's algorithm for 2x2x2 gives rank 7. For 4x4x4, we can use
    # block Strassen: treat 4x4x4 as a 2x2x2 problem where each "scalar" is
    # a 2x2 matrix. The 7 matrix multiplications each have rank 4 (schoolbook),
    # so total rank = 7*4 = 28.
    #
    # To construct the explicit rank-1 terms:
    # For each Strassen term t (0..6), and for each sub-multiplication (ip, jp, kp) in {0,1}^3:
    # The term contributes to C[2I+ip, 2K+kp] the value:
    #   c_t_IK * a_t_I'J * b_t_JK' * A[2I'+ip, 2J+jp] * B[2J+jp, 2K'+kp]
    #
    # But this is a product of 4 things, not 2. So it's not a single rank-1 term.
    #
    # I think the issue is that block Strassen doesn't directly give a rank
    # decomposition in the usual sense. Let me reconsider.
    #
    # Actually, I think I'm overcomplicating this. Let me just verify:
    # Does Strassen for 2x2x2 give rank 7? Yes.
    # Does block Strassen for 4x4x4 give rank 28? 
    #
    # The block Strassen algorithm computes:
    # C = A * B where A, B are 4x4.
    # Split into 2x2 blocks: A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Compute 7 products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # Then:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # Each M_t is a 2x2 matrix product. If we compute each using schoolbook (4 multiplications),
    # total scalar multiplications = 7 * 4 = 28.
    #
    # Each scalar multiplication corresponds to one rank-1 term in the tensor decomposition.
    # So the rank is 28.
    #
    # Now let me construct the explicit terms.
    #
    # For Strassen term t, the product M_t = A_comb * B_comb where:
    # A_comb = sum_{(I,J)} a_t[I,J] * A_block[I,J]
    # B_comb = sum_{(J,K)} b_t[J,K] * B_block[J,K]
    #
    # M_t[ip,kp] = sum_{jp} A_comb[ip,jp] * B_comb[jp,kp]
    #
    # A_comb[ip,jp] = sum_{I,J} a_t[I,J] * A[2I+ip, 2J+jp]
    # B_comb[jp,kp] = sum_{J,K} b_t[J,K] * B[2J+jp, 2K+kp]
    #
    # So M_t[ip,kp] = sum_{jp} sum_{I,J} sum_{J',K} a_t[I,J] * b_t[J',K] * A[2I+ip, 2J+jp] * B[2J+jp, 2J'+kp]
    #
    # For the indices to match (2J+jp = 2J'+jp), we need J = J'.
    # So: M_t[ip,kp] = sum_{jp} sum_{I,J,K} a_t[I,J] * b_t[J,K] * A[2I+ip, 2J+jp] * B[2J+jp, 2K+kp]
    #
    # The contribution to C[2I_c+ip, 2K_c+kp] from term t is:
    # sum_{(I_c,K_c): c_t[I_c,K_c] != 0} c_t[I_c,K_c] * M_t[ip,kp]
    # = sum_{I_c,K_c} c_t[I_c,K_c] * sum_{jp} sum_{I,J,K} a_t[I,J] * b_t[J,K] * A[2I+ip, 2J+jp] * B[2J+jp, 2K+kp]
    #
    # We need this to equal: sum_{J,jp} A[2I_c+ip, 2J+jp] * B[2J+jp, 2K_c+kp]
    #
    # So for each (I_c, K_c, ip, kp):
    # sum_t sum_{jp} sum_{I,J,K} c_t[I_c,K_c] * a_t[I,J] * b_t[J,K] * [I=I_c] * [K=K_c] = 1
    #
    # Wait, that's not quite right. The indices I and K in the sum are the same as I_c and K_c
    # for the term to contribute. So:
    # sum_t sum_{jp} c_t[I_c,K_c] * a_t[I_c,J] * b_t[J,K_c] = 1 for all J,jp
    #
    # Hmm, but this must hold for all J and jp, which means the sum must be independent of J and jp.
    # That seems too strong. Let me reconsider.
    #
    # Actually, I think the correct way to think about it is:
    # The full 4x4x4 multiplication tensor has entries T[i,j,k] = 1 (for all i,j,k in 0..3).
    # A rank-r decomposition is T = sum_t u_t * v_t * w_t where:
    # u_t[i*m+j] = alpha_t(i,j)
    # v_t[j*p+k] = beta_t(j,k)  
    # w_t[k*n+i] = gamma_t(k,i)
    #
    # And for all (i,j,k): sum_t alpha_t(i,j) * beta_t(j,k) * gamma_t(k,i) = 1
    #
    # For schoolbook: alpha_t(i,j) = 1 if t corresponds to (i,j,k), else 0.
    #
    # For Strassen: we need to find such alpha, beta, gamma with r=28.
    #
    # Let me try a different approach. Instead of trying to derive the Strassen
    # decomposition analytically, let me just use schoolbook for the whole thing
    # but see if I can find any structure.
    #
    # Actually, let me just implement the schoolbook decomposition correctly.
    # Rank = 112. That's the same as the seed, so no improvement.
    #
    # Let me try yet another approach: use the fact that for 4x4x7, we can
    # potentially use a better algorithm than pure schoolbook.
    #
    # For now, let me just return the schoolbook decomposition to ensure validity.
    
    rank = n * m * p  # 112
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
