"""Tensor decomposition for 3x5x5 matrix multiplication using DFT over Z_5."""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    q = 5  # order of cyclic group
    
    # Roots of unity over Z_5
    # We use a primitive 5th root of unity. In GF(21) or similar, but for real-valued,
    # we'll use complex roots and then find a real/rational representation.
    # Actually, let's use the fact that the DFT matrix for size 5 has entries that are
    # 5th roots of unity. We need exact rational/integer coefficients.
    
    # The DFT-based approach for vector-matrix product (treating each of the 3 rows independently):
    # For a single 1x5 x 5x5 product, the rank is at most 5 using circulant structure.
    # 
    # The idea: If B is circulant with first column b = [b_0, b_1, b_2, b_3, b_4],
    # then A*B where A is 1x5 can be done via:
    # 1. DFT of a -> alpha (5 values)
    # 2. Pointwise multiply by DFT of b -> alpha * beta
    # 3. Inverse DFT -> result
    #
    # For general B, we can't use this directly. But the tensor for 1x5x5 multiplication
    # has a structure that can be decomposed.
    #
    # Actually, let me think more carefully. The bilinear algorithm for a single 1x5 * 5x5:
    # We have 5 inputs from A (a_0..a_4) and 25 inputs from B (b_{j,k}).
    # Output is c_k = sum_j a_j * b_{j,k} for k=0..4.
    #
    # The rank of the 1x5x5 tensor: this is the multiplication of a 1x5 row vector by a 5x5 matrix.
    # This is equivalent to 5 independent inner products of length 5.
    # Each inner product can be done with rank 5 (the standard dot product).
    # But we can do better using shared structure.
    #
    # Let's use the approach from the directive: treat the 5x5 matrix as having a circulant-like
    # structure via DFT. The key insight is that we can diagonalize the multiplication using DFT.
    #
    # For the full 3x5x5, we'll do 3 independent 1x5x5 multiplications.
    # If each 1x5x5 has rank 5, total rank is 15.
    
    # Let's construct a rank-5 bilinear algorithm for 1x5 * 5x5.
    # 
    # Using DFT over Z_5 with a primitive root. Let omega = e^(2*pi*i/5).
    # The DFT matrix F has entries F[j,k] = omega^(jk) for j,k in {0,1,2,3,4}.
    # 
    # The approach:
    # - Define intermediate values using the DFT structure
    # 
    # Actually, for exact rational/integer entries, we need to be careful.
    # Let's use a different approach: the rank of the 1x5x5 tensor is at most 5.
    # 
    # One construction: 
    # Let the input vector be a = [a_0, a_1, a_2, a_3, a_4]
    # Let the matrix B have entries b_{j,k} for j,k in {0,...,4}
    # Output c_k = sum_j a_j b_{j,k}
    #
    # We can write this as: for each "frequency" r in {0,1,2,3,4},
    # define u_r = sum_j a_j * omega^(jr)  (DFT of a)
    # define v_r = something involving B
    # Then c = IDFT of (u * v)
    #
    # But this requires complex numbers. For exact rational/integer, we need a different approach.
    #
    # Let me try a simpler approach: just use the fact that each output c_k depends on all a_j
    # through the row b_{0,k}, b_{1,k}, ..., b_{4,k} of B.
    #
    # c_k = a_0*b_{0,k} + a_1*b_{1,k} + a_2*b_{2,k} + a_3*b_{3,k} + a_4*b_{4,k}
    #
    # This is a sum of 5 products. If we do this for each k, we have 5 such sums.
    # The rank needed is at most 5*5 = 25 for the naive approach, but we can share.
    #
    # Actually, the rank of the 1x5x5 tensor is 5. Here's why:
    # The tensor T_{j,k,l} = delta_{k,l} (where j indexes the row of A, k indexes the column of B's row,
    # and l indexes the output). Wait, let me be more careful.
    #
    # The tensor for multiplying A (1x5) by B (5x5) to get C (1x5):
    # T_{j, k, l} where j in {0..4} is the input from A, k in {0..4} is... hmm.
    #
    # Let me use the standard tensor notation. For matrix multiplication (n x m) * (m x p):
    # T_{i,j,k} where i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # T_{i,j,k} = 1 if the product involves a_{i,j} * b_{j,k} to form c_{i,k}
    # For 1x5x5: T_{j,k} = delta_{j,k} (since it's just inner product for each output)
    #
    # Wait, for 1x5 * 5x5 -> 1x5:
    # c_k = sum_j a_j b_{j,k}
    # The tensor has dimensions (5, 5, 5) where the first 5 is j (index in A), second 5 is (j,k) pair... 
    # 
    # Let me use the U,V,W formulation directly.
    # For rank r, we need U (5 x r), V (25 x r), W (5 x r) such that:
    # c_k = sum_c U[j,c] * V[j*5+k,c] * W[k,c]  ... no, that's not right either.
    #
    # The standard formulation: C_{i,k} = sum_c U_{i,j, c} * V_{j, k, c} * W_{i, k, c}
    # But U, V, W are 2D matrices reshaped.
    #
    # From the seed: U has shape (n*m, rank), indexed by [i*m + j, c]
    # V has shape (m*p, rank), indexed by [j*p + k, c]
    # W has shape (p*n, rank), indexed by [k*n + i, c]
    #
    # For 3x5x5: U is (15, rank), V is (25, rank), W is (15, rank)
    # c_{i,k} = sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*3+i, c]  ... this isn't quite right.
    #
    # Actually, the decomposition is:
    # M(A, B)_{i,k} = sum_c U_{i,m+j, c} * V_{j,p+k, c} * W_{k,n+i, c}
    # where the sum over c gives the bilinear form, and we sum over j implicitly? No...
    #
    # Let me re-read the seed code. In the schoolbook:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1  
    # W[k*n+i][column] = 1
    # And the product is: result[i, k] = sum over all columns c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    #
    # No wait, the standard tensor decomposition for matrix multiplication:
    # The multiplication tensor T is in (F^n x F^m)^* ⊗ (F^m x F^p)^* ⊗ (F^n x F^p)
    # T = sum_{i,j,k} e_i^* ⊗ e_j ⊗ e_{j,k} ⊗ ... 
    #
    # In the bilinear algorithm formulation:
    # C_{i,k} = sum_c sum_j U[i,j,c] A_{i,j} V[c] B_{j,k} W[c,k] ... 
    #
    # I think the interface is:
    # Given A (n x m) and B (m x p), compute C = A * B (n x p)
    # C_{i,k} = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    #
    # Yes! That's it. For each intermediate column c:
    # alpha_{i,c} = sum_j U[i*m+j, c] * A[i,j]
    # beta_{k,c} = sum_j V[j*p+k, c] * B[j,k]
    # C_{i,k} = sum_c W[k*n+i, c] * alpha_{i,c} * beta_{k,c}
    #
    # So for 3x5x5:
    # U: (15, rank) where row index is i*5 + j, i in [0,3), j in [0,5)
    # V: (25, rank) where row index is j*5 + k, j in [0,5), k in [0,5)
    # W: (15, rank) where row index is k*3 + i, k in [0,5), i in [0,3)
    #
    # For the schoolbook, each (i,j,k) triple gets its own column c.
    #
    # Now, for the DFT-based approach with rank 15:
    # We process each row i of A independently. For a fixed i, we need to compute
    # c_{i,k} = sum_j A_{i,j} B_{j,k} for k=0..4.
    #
    # This is a 1x5 * 5x5 multiplication for each i. If we can do each with rank 5,
    # total rank is 3*5 = 15.
    #
    # For the 1x5 * 5x5 case, using the formulation:
    # For fixed i, define:
    # alpha_j = A_{i,j}  (but U can mix over j)
    # beta_{j,k} = B_{j,k}  (but V can mix over j,k)
    # 
    # We want: c_k = sum_j A_{i,j} B_{j,k}
    #
    # Using DFT: let omega = e^(2*pi*i/5). Define:
    # For r in {0,1,2,3,4}:
    #   u_r = sum_j A_{i,j} * omega^(jr)  (DFT of A's row)
    #   For each output k, we need to combine with B.
    #
    # The trick: c_k = (1/5) sum_r u_r * v_r(k) where v_r(k) depends on B and k.
    #
    # v_r(k) = sum_j B_{j,k} * omega^(-jr)  ... this is the DFT of the k-th column of B.
    #
    # So: c_k = (1/5) sum_{r=0}^{4} (sum_j A_{i,j} omega^{jr}) * (sum_j B_{j,k} omega^{-jr})
    #
    # This gives rank 5 for the 1x5x5 case! But it uses complex numbers.
    #
    # For exact rational/integer coefficients, we need to work in an extension field.
    # However, the problem says "integer/half-integer entries". Complex numbers are not allowed.
    #
    # We need to find a real/rational representation. The 5th roots of unity satisfy:
    # omega^0 + omega^1 + omega^2 + omega^3 + omega^4 = 0
    # omega^5 = 1
    # The minimal polynomial is x^4 + x^3 + x^2 + x + 1 = 0
    #
    # We can represent elements of Q(omega) as a_0 + a_1*omega + a_2*omega^2 + a_3*omega^3
    # (since omega^4 = -1 - omega - omega^2 - omega^3)
    #
    # This gives a 4-dimensional vector space over Q. Each "frequency" component r corresponds to
    # a vector in this space. The DFT-based algorithm would then have rank 5 (one for each r),
    # but each entry in U, V, W would be in Q(omega), which we can represent as 4 rational numbers.
    #
    # But the problem requires scalar entries (integer/half-integer), not vectors.
    #
    # Alternative: Use the fact that we only need the final result in Q, and the complex parts cancel.
    # We can pair r and 5-r (complex conjugates) and combine them.
    #
    # r=0 is real. r=1 and r=4 are conjugates. r=2 and r=3 are conjugates.
    # So we need: r=0 (1 component), r=1,4 (2 real components each: real and imaginary), r=2,3 (2 real components each).
    # Total: 1 + 2 + 2 = 5 real-valued intermediate computations per 1x5x5 product.
    #
    # This matches! Each intermediate is a real number, and we get rank 5 per 1x5x5.
    #
    # Let's construct this explicitly.
    
    # For r=0:
    # u_0 = sum_j A_{i,j}
    # v_0(k) = sum_j B_{j,k}
    # Contribution to c_k: (1/5) * u_0 * v_0(k)
    
    # For r=1 and r=4 (conjugate pair):
    # u_1 = sum_j A_{i,j} omega^j  (complex)
    # u_4 = conjugate of u_1 = sum_j A_{i,j} omega^{-j} = sum_j A_{i,j} omega^{4j}
    # 
    # Let's define:
    # Re(u_1) = sum_j A_{i,j} cos(2*pi*j/5)
    # Im(u_1) = sum_j A_{i,j} sin(2*pi*j/5)
    # 
    # Similarly for v_1(k) and v_4(k).
    # 
    # The contribution from r=1 and r=4 to c_k is:
    # (1/5) * (u_1 * v_4(k) + u_4 * v_1(k))
    # = (1/5) * 2 * Re(u_1 * conjugate(v_1(k)))
    # 
    # This requires 2 intermediate values for u (Re and Im) and 2 for v, giving 2 "columns" 
    # in the bilinear sense? No, each column c corresponds to one bilinear product.
    
    # I think I'm overcomplicating this. Let me just use the standard rank-5 algorithm 
    # for 1x5 dot product with 5x5, which is essentially the DFT approach but written 
    # with explicit rational coefficients using the cyclotomic field.
    
    # Actually, let me just use numpy to verify and construct the exact coefficients.
    
    omega = np.exp(2j * np.pi / 5)
    
    # Precompute DFT-related matrices
    # For 1x5 * 5x5 with rank 5 using DFT:
    # U_part[i, c] where i in {0..4}, c in {0..4}
    # V_part[j, k, c] where j,k in {0..4}, c in {0..4}  -> flatten to (25, 5)
    # W_part[k, c] where k in {0..4}, c in {0..4}
    
    # The algorithm:
    # For c = 0..4 (representing frequency r = c):
    #   u_c = sum_j U_part[j, c] * A[j]   where U_part[j, c] = omega^(j*c)
    #   For each k:
    #     v_c_k = sum_j V_part[j, k, c] * B[j, k]  where V_part[j, k, c] = omega^(-j*c)
    #   c_k += (1/5) * W_part[k, c] * u_c * v_c_k  where W_part[k, c] = 1
    #
    # But this uses complex numbers. To get rational entries, we need to work in the cyclotomic field.
    
    # Let's use a different, cleaner approach. The rank of the (1,5,5) tensor is 5.
    # We can find an explicit rank-5 decomposition over the rationals using the following construction:
    #
    # Based on the fact that the tensor is essentially the "dot product" tensor extended.
    
    # For now, let me just construct the rank-15 solution by using the schoolbook but 
    # sharing columns across the 3 rows using DFT-like structure.
    
    # Actually, the simplest valid rank-15 construction:
    # For each of the 3 rows i, use a separate rank-5 decomposition of the 1x5x5 tensor.
    
    # Let me construct an explicit rank-5 decomposition for 1x5 * 5x5.
    
    # Using the identity: for any vectors a, b, and matrix B:
    # (a^T B)_k = sum_j a_j B_{j,k}
    
    # A rank-5 decomposition using the "Fourier" basis over Z_5:
    # Define for r = 0,1,2,3,4:
    #   U_j = exp(2*pi*i*j*r/5)
    #   V_{j,k} = exp(-2*pi*i*j*r/5)  
    #   W_k = 1/5
    #
    # Then: sum_r U_j * V_{j,k} * W_k * a_j * B_{j,k} ... no, this doesn't work directly.
    
    # Let me use a concrete, verified construction. The rank of the (1,n,n) tensor is n.
    # One explicit decomposition for n=5:
    
    # I'll use the following rank-5 algorithm for computing c = A * B where A is 1x5, B is 5x5:
    # 
    # Let a = [a0, a1, a2, a3, a4], B has rows b0, b1, b2, b3, b4 (each 1x5)
    # c_k = sum_j a_j * B[j,k]
    #
    # Define intermediates:
    # t0 = a0 + a1 + a2 + a3 + a4
    # t1 = a0 + a1*ω + a2*ω² + a3*ω³ + a4*ω⁴
    # t2 = a0 + a1*ω² + a2*ω⁴ + a3*ω + a4*ω³
    # t3 = a0 + a1*ω³ + a2*ω + a3*ω⁴ + a4*ω²
    # t4 = a0 + a1*ω⁴ + a2*ω³ + a3*ω² + a4*ω
    #
    # For each output k:
    # s_k_0 = B[0,k] + B[1,k] + B[2,k] + B[3,k] + B[4,k]
    # s_k_1 = B[0,k] + B[1,k]*ω⁻¹ + B[2,k]*ω⁻² + B[3,k]*ω⁻³ + B[4,k]*ω⁻⁴
    # etc.
    #
    # c_k = (1/5)(t0*s_k_0 + t1*s_k_1 + ... + t4*s_k_4)
    #
    # This is bilinear: each term t_r * s_k_r is a product of a linear form in a and a linear form in B.
    # So for fixed r and k, we have a bilinear term. But we need to organize into U, V, W form.
    
    # For the full 3x5x5, we'll have 3x5 = 15 columns (5 per row i).
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 15 x 15
    V = [[0] * rank for _ in range(m * p)]  # 25 x 15
    W = [[0] * rank for _ in range(p * n)]  # 15 x 15
    
    # We need exact rational/half-integer entries. The omega values are not rational.
    # We need to represent the cyclotomic field elements.
    
    # This is getting complex. Let me use a simpler approach that definitely works with rationals.
    
    # Alternative: Just use the schoolbook decomposition. It's guaranteed to work.
    # The rank will be 75, which is the same as the seed.
    
    # But the directive asks for rank 15. Let me think if there's a way to do it with rationals.
    
    # Actually, I realize I should just verify if the DFT approach can be expressed with rational coefficients
    # by working in the cyclotomic field and then eliminating the complex basis.
    
    # The field Q(ω) where ω = e^(2πi/5) has degree 4 over Q.
    # Elements can be written as a + bω + cω² + dω³ with a,b,c,d ∈ Q.
    # 
    # The multiplication in this field is defined by ω⁴ = -1-ω-ω²-ω³.
    
    # For the rank-5 algorithm, we have 5 "frequencies" but ω⁰, ω¹, ω², ω³, ω⁴ where ω⁴ is 
    # expressible in terms of ω, ω², ω³. So the 5 frequencies span a 4-dimensional Q-space.
    
    # This means we might need rank > 5 per 1x5x5 if we stick to rational scalars.
    
    # Given the complexity and the risk of errors, let me just output the schoolbook decomposition.
    # It's correct and matches the seed score.
    
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
