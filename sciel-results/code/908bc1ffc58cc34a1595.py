import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    # Strategy: Hierarchical block composition via Kronecker product.
    #
    # Key insight: The 2x4 * 4x8 multiplication can be decomposed as:
    #   C = A * B  where A is 2x4, B is 4x8
    #
    # Reshape B (4x8) into B' (4, 2, 4) by splitting the 8 columns into
    # 4 groups of 2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8].
    #
    # Then C[i, g, k] = sum_{j=0}^{3} A[i, j] * B'[j, g, k]
    # where g in 0..3, k in 0..1.
    #
    # Further, split A (2x4) into A' (2, 2, 2): A[i, h, j] with h in 0..1, j in 0..1.
    # So A[i, 2h+j] = A'[i, h, j].
    #
    # Then C[i, g, k] = sum_{h=0}^{1} sum_{j=0}^{1} A'[i, h, j] * B'[2h+j, g, k]
    #
    # Define M_h[g, k] = B'[2h, g, k] and N_h[g, k] = B'[2h+1, g, k] for h=0,1.
    # Wait, that doesn't quite work because j indexes both A and B.
    #
    # Let me re-approach. The standard tensor for 2x4*4x8:
    # T[i, j, k] where i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # T[i,j,k] = 1 (the structure tensor has 1s everywhere in the valid index range,
    # but the "meaning" is that A[i,j]*B[j,k] contributes to C[i,k]).
    #
    # Actually, the correct framing: the multiplication tensor M_{n,m,p} has
    # entries T[i,j,k] = 1 for all valid i,j,k, and the bilinear form is:
    #   (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is in U* x V* x W* (dual spaces), with U = R^{n*m}, V = R^{m*p}, W = R^{n*p}.
    #
    # For rank decomposition: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t in R^{n*m}, v_t in R^{m*p}, w_t in R^{n*p}.
    #
    # Schoolbook: r = n*m*p = 64, with u_{ijk} = e_{im+j}, v_{ijk} = e_{jp+k}, w_{ijk} = e_{kn+i}.
    #
    # Now, for the Kronecker structure:
    # 2x4*4x8 = (2x2*2x2) ⊗ (2x2*2x2) in a specific sense?
    #
    # Let's think of it as: A is 2x4 = 2x(2*2), B is 4x8 = (2*2)x(2*4).
    # Hmm, the inner dimension 4 = 2*2.
    #
    # Alternative: Think of the problem as computing 4 independent 2x2*2x2 multiplications
    # but with shared A rows.
    #
    # Actually, let me use a cleaner decomposition:
    # 2x4 * 4x8: The inner dimension is 4.
    # Split the inner dimension into two halves of size 2:
    #   C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    #
    # Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2).
    # Let B1 = B[0:2, :] (2x8), B2 = B[2:4, :] (2x8).
    # Then C = A1 * B1 + A2 * B2.
    #
    # Each term A_h * B_h is a 2x2 * 2x8 multiplication.
    # 2x2 * 2x8: inner dim 2, output 2x8.
    # This can be seen as 4 independent 2x2*2x2 multiplications? No.
    # 2x2 * 2x8: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k].
    # This is like 8 independent 2x1*1x1? No, it's 2*2*2 = 8 scalar mults schoolbook.
    #
    # Can we use Strassen-like tricks on 2x2*2x8?
    # 2x2*2x8 = 2x2 * (2x2 + 2x2 + 2x2 + 2x2) = 4 copies of 2x2*2x2.
    # Using Strassen for each: 4*7 = 28 per A_h*B_h.
    # Total: 2*28 = 56. Same as before.
    #
    # BUT: Can we share the Strassen factors between A1*B1 and A2*B2?
    # The Strassen decomposition of 2x2*2x2 uses 7 scalar mults.
    # For A1*B1: the "A" part is A1 (2x2), "B" part is B1 (2x8) = 4 blocks of (2x2).
    # For A2*B2: the "A" part is A2 (2x2), "B" part is B2 (2x8) = 4 blocks of (2x2).
    #
    # If we use the SAME Strassen U-factors for A1 and A2... no, they're different matrices.
    #
    # Let me try a different structural decomposition.
    #
    # KEY INSIGHT: Use the fact that 2x4*4x8 is isomorphic to 2x8*8x2 in some sense?
    # No, that's not right either.
    #
    # Let me try the most direct Kronecker product approach:
    # The tensor for 2x4*4x8 can be written as M_{2,2,2} ⊗ M_{2,2,4}?
    # No, the dimensions don't match up cleanly.
    #
    # Actually: 2x4*4x8. Let's write 4 = 2*2 and 8 = 2*4.
    # So it's like (2x(2*2)) * ((2*2)x(2*4)).
    # The inner dimension is 4 = 2*2.
    #
    # If we view A as a 2x2x2 tensor A[i, a, b] where j = 2a+b,
    # and B as a 2x2x4 tensor B[a, b, k] where the row index is 2a+b,
    # then C[i, k] = sum_{a,b} A[i, a, b] * B[a, b, k].
    #
    # This is a "tensor contraction" over a 2x2 index.
    # We can use Strassen for the 2x2 contraction!
    #
    # Strassen for 2x2*2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #
    # In our case, the "2x2 matrices" being multiplied are:
    #   "A-block" = A[i, :, :] which is 2x2 (indices a,b)
    #   "B-block" = B[:, :, k] which is 2x2 (indices a,b)
    #
    # Wait, B is 4x8. If we index B by (row, col) = (2a+b, k),
    # then B[a, b, k] = B[2a+b, k].
    #
    # So for fixed i and k, we have a 2x2 matrix "A_ik" = A[i, :, :] (same for all k!)
    # and a 2x2 matrix "B_ik" = B[:, :, k] (depends on k).
    #
    # C[i, k] = sum_{a,b} A[i, a, b] * B[a, b, k] = trace(A[i] * B_k^T)? 
    # Actually it's the Frobenius inner product: <A[i, :, :], B[:, :, k]>.
    #
    # So for each k, we compute the Frobenius inner product of A[i] (2x2) and B_k (2x2).
    # This is equivalent to a 2x2 matrix multiplication followed by trace?
    # No, it's just element-wise multiply and sum.
    #
    # The Frobenius inner product of two 2x2 matrices can be computed with Strassen-like tricks?
    # <X, Y> = X00*Y00 + X01*Y01 + X10*Y10 + X11*Y11.
    # This is just 4 scalar multiplications + 3 additions.
    # Can we do it in fewer? For a fixed pair, no (it's just a dot product of 4 vectors).
    # But if we're doing this for multiple k values, we might share factors.
    #
    # Let's reconsider. The full problem:
    # C[i, k] = sum_{a=0}^{1} sum_{b=0}^{1} A[i, 2a+b] * B[2a+b, k]
    #
    # Let's define for each (a,b) pair, the "column" of B: B[2a+b, :] is a vector of length 8.
    # And A[i, 2a+b] is a scalar.
    #
    # C[i, :] = sum_{a,b} A[i, 2a+b] * B[2a+b, :]
    #
    # This is a linear combination of 4 rows of B, with coefficients from row i of A.
    # For each i, we compute a 1x8 vector as a linear combination of 4 rows of B.
    #
    # This is just matrix-vector multiplication: C[i, :] = A[i, :] * B.
    # Which is what we already knew.
    #
    # OK let me try yet another approach. What about using the known rank bounds?
    #
    # The rank of M_{2,4,8} satisfies:
    #   r(M_{2,4,8}) <= r(M_{2,2,2}) * r(M_{2,2,4})? No, that's not a valid inequality.
    #   r(M_{2,4,8}) <= r(M_{2,2,2}) + r(M_{2,2,6})? No.
    #
    # Valid submultiplicativity: r(M_{n,m,p}) <= r(M_{n1,m1,p1}) * r(M_{n2,m2,p2})
    # if (n,m,p) = (n1*n2, m1*m2, p1*p2). But 2x4x8 doesn't factor that way.
    #
    # However: r(M_{2,4,8}) <= r(M_{2,2,2}) * r(M_{1,2,4}) = 7 * 8 = 56.
    # Because 2x4x8 = (2x1) x (2x2) x (2x4)? No, that's not right either.
    #
    # The correct Kronecker: M_{n,m,p} ⊗ M_{n',m',p'} = M_{nn', mm', pp'}.
    # So M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4}? Check: (2*1, 2*2, 2*4) = (2,4,8). Yes!
    # So r(M_{2,4,8}) <= r(M_{2,2,2}) * r(M_{1,2,4}) = 7 * r(M_{1,2,4}).
    #
    # What is r(M_{1,2,4})? This is 1x2 * 2x4.
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0..3.
    # This is 8 scalar multiplications schoolbook.
    # Can we do better? r(M_{1,2,4}) <= r(M_{1,1,1}) * r(M_{1,2,4})? Trivial.
    # r(M_{1,2,4}) = 8? Probably, since it's just 2x4 = 8 independent products.
    # Actually, M_{1,2,4} is the tensor for computing y = a*x1 + b*x2 where x is 4-dim.
    # This is 2*4 = 8 scalar mults. No savings possible (each output depends on both inputs
    # and each input pair is independent).
    #
    # So r(M_{2,4,8}) <= 7 * 8 = 56.
    #
    # Can we do better? What about M_{2,4,8} = M_{2,2,4} ⊗ M_{1,2,2}?
    # Check: (2*1, 2*2, 4*2) = (2,4,8). Yes!
    # r(M_{2,4,8}) <= r(M_{2,2,4}) * r(M_{1,2,2}).
    #
    # r(M_{1,2,2}): 1x2 * 2x2. C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]. 4 scalar mults.
    # r(M_{2,2,4}): 2x2 * 2x4. C[i,k] = sum_j A[i,j]*B[j,k]. 8 scalar mults schoolbook.
    # Can we do better? M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2}. r <= 7*2 = 14.
    # Or M_{2,2,4}: is it 2 copies of M_{2,2,2}? No, the inner dim is 2, not 4.
    # M_{2,2,4}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0..3.
    # This is 2*2*2 = 8 scalar mults.
    # Can we use Strassen? Strassen works for 2x2*2x2 (inner dim 2, output 2x2).
    # Here inner dim is 2, but output is 2x4.
    # We can split the 4 columns into 2 pairs: 2 copies of 2x2*2x2.
    # So r(M_{2,2,4}) <= 2 * r(M_{2,2,2}) = 2*7 = 14.
    #
    # So r(M_{2,4,8}) <= r(M_{2,2,4}) * r(M_{1,2,2}) <= 14 * 4 = 56.
    #
    # Another factorization: M_{2,4,8} = M_{2,4,2} ⊗ M_{1,1,4}?
    # (2*1, 4*1, 2*4) = (2,4,8). Yes!
    # r(M_{2,4,8}) <= r(M_{2,4,2}) * r(M_{1,1,4}) = r(M_{2,4,2}) * 4.
    #
    # r(M_{2,4,2}): 2x4 * 4x2. C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] for i=0,1, k=0,1.
    # 2*4*2 = 16 scalar mults schoolbook.
    # Can we do better? M_{2,4,2} = M_{2,2,2} ⊗ M_{1,2,1}? (2*1, 2*2, 2*1) = (2,4,2). Yes!
    # r(M_{2,4,2}) <= r(M_{2,2,2}) * r(M_{1,2,1}) = 7 * 2 = 14.
    #
    # So r(M_{2,4,8}) <= 14 * 4 = 56.
    #
    # It seems 56 is a solid upper bound from these factorizations.
    #
    # Now, can we beat 56? The question is whether the Kronecker bound is tight.
    # For small matrices, the rank is often equal to the Kronecker product bound.
    # But sometimes it's lower.
    #
    # Let me try to construct an explicit rank-56 decomposition using the Kronecker product
    # of Strassen (rank 7) for M_{2,2,2} and schoolbook (rank 8) for M_{1,2,4}.
    #
    # M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4}
    #
    # Strassen for M_{2,2,2}:
    # U (2x2 -> 7): 
    #   u1 = [1, 0, 0, 1]  (A00+A11)
    #   u2 = [0, 0, 1, 1]  (A10+A11)  
    #   u3 = [1, 0, 0, 0]  (A00)
    #   u4 = [0, 0, 0, 1]  (A11)
    #   u5 = [1, 1, 0, 0]  (A00+A10)
    #   u6 = [0, 0, 1, -1] (A10-A00)
    #   u7 = [0, 1, 0, -1] (A01-A11)
    #
    # Wait, I need to be careful with the index ordering.
    # For 2x2 matrix A = [[A00, A01], [A10, A11]], flattened as [A00, A01, A10, A11].
    #
    # Strassen factors for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # U (7x4): each row is the coefficients for A.
    # U[0] = [1, 0, 0, 1]   # A00 + A11
    # U[1] = [0, 0, 1, 1]   # A10 + A11
    # U[2] = [1, 0, 0, 0]   # A00
    # U[3] = [0, 0, 0, 1]   # A11
    # U[4] = [1, 1, 0, 0]   # A00 + A10
    # U[5] = [0, 0, 1, -1]  # A10 - A00
    # U[6] = [0, 1, 0, -1]  # A01 - A11
    #
    # V (7x4): each row is the coefficients for B.
    # V[0] = [1, 0, 0, 1]   # B00 + B11
    # V[1] = [1, 0, 0, 0]   # B00
    # V[2] = [0, 1, 0, -1]  # B01 - B11
    # V[3] = [0, 0, 1, -1]  # B10 - B00
    # V[4] = [0, 0, 0, 1]   # B11
    # V[5] = [1, 0, 1, 0]   # B00 + B10
    # V[6] = [0, 1, 0, 1]   # B01 + B11
    #
    # W (7x4): each row is the coefficients for C.
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    #
    # W[0] = [1, 0, 0, 1]   # M1 - M4 - M6 + M7 -> C00: 1, C01: 0, C10: 0, C11: 1
    # W[1] = [0, 1, 0, 0]   # M3 + M5 -> C01
    # W[2] = [0, 0, 1, 0]   # M2 + M4 -> C10
    # W[3] = [0, 0, 0, 1]   # M1 - M3 + M6 - M7 -> C11
    #
    # Wait, I need to be more careful. The W matrix has shape (n*p, rank) = (4, 7).
    # W[k*n+i, t] where k is col of C, i is row of C.
    # For 2x2: C is 2x2, so indices are (i,k) with i in {0,1}, k in {0,1}.
    # Flattened: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3.
    #
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    #
    # So W (4x7):
    # W[0] = [1, 0, 0, -1, 0, -1, 1]   # C00
    # W[1] = [0, 0, 1, 0, 1, 0, 0]     # C01
    # W[2] = [0, 1, 0, 1, 0, 0, 0]     # C10
    # W[3] = [1, 0, -1, 0, 0, 1, -1]   # C11
    #
    # Now for M_{1,2,4} (1x2 * 2x4):
    # A is 1x2: [A0, A1]
    # B is 2x4: [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    # C is 1x4: C[k] = A0*B[0,k] + A1*B[1,k]
    #
    # Schoolbook rank 8:
    # U (2x8): U[0] = [1,0,0,0,0,0,0,0], U[1] = [0,0,0,0,1,0,0,0]
    # V (8x8): identity-like
    # W (4x8): W[0] = [1,0,0,0,0,0,0,0], etc.
    #
    # Actually for M_{1,2,4}, the schoolbook decomposition is:
    # For t = 0..7: t = 2*j + k where j in {0,1}, k in {0,1,2,3}
    # U[j, t] = 1 if t = 2*j+k... no.
    #
    # U has shape (1*2, 8) = (2, 8).
    # V has shape (2*4, 8) = (8, 8).
    # W has shape (4*1, 8) = (4, 8).
    #
    # For t = 2*j + k:
    # U[j, t] = 1
    # V[2*j + k, t] = 1  (B[j, k])
    # W[k, t] = 1  (C[k])
    #
    # So:
    # U = [[1,0,0,0,1,0,0,0],
    #      [0,0,0,0,0,0,0,0]]  # Wait, that's wrong.
    #
    # Let me redo. t ranges 0..7. t = 2*j + k.
    # j=0, k=0: t=0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # j=0, k=1: t=1. U[0,1]=1, V[1,1]=1, W[1,1]=1.
    # j=0, k=2: t=2. U[0,2]=1, V[2,2]=1, W[2,2]=1.
    # j=0, k=3: t=3. U[0,3]=1, V[3,3]=1, W[3,3]=1.
    # j=1, k=0: t=4. U[1,4]=1, V[4,4]=1, W[0,4]=1.
    # j=1, k=1: t=5. U[1,5]=1, V[5,5]=1, W[1,5]=1.
    # j=1, k=2: t=6. U[1,6]=1, V[6,6]=1, W[2,6]=1.
    # j=1, k=3: t=7. U[1,7]=1, V[7,7]=1, W[3,7]=1.
    #
    # So:
    # U = [[1,1,1,1,0,0,0,0],
    #      [0,0,0,0,1,1,1,1]]
    #
    # V = I_8 (identity)
    #
    # W = [[1,0,0,0,1,0,0,0],
    #      [0,1,0,0,0,1,0,0],
    #      [0,0,1,0,0,0,1,0],
    #      [0,0,0,1,0,0,0,1]]
    #
    # Now, the Kronecker product M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4}:
    # The total rank is 7 * 8 = 56.
    #
    # The Kronecker product of tensors:
    # (U1 ⊗ U2) has shape (n1*m1 * n2*m2, r1*r2) = (4*2, 7*8) = (8, 56).
    # (V1 ⊗ V2) has shape (m1*p1 * m2*p2, r1*r2) = (2*2 * 2*4, 56) = (32, 56).
    # (W1 ⊗ W2) has shape (p1*n1 * p2*n2, r1*r2) = (2*2 * 4*1, 56) = (16, 56).
    #
    # For the Kronecker product of decompositions:
    # U_total[t1*r2 + t2, i1*m1*j2... ] 
    #
    # Actually, the Kronecker product of two tensor decompositions:
    # If T1 = sum_t1 u1_t1 ⊗ v1_t1 ⊗ w1_t1 and T2 = sum_t2 u2_t2 ⊗ v2_t2 ⊗ w2_t2,
    # then T1 ⊗ T2 = sum_{t1,t2} (u1_t1 ⊗ u2_t2) ⊗ (v1_t1 ⊗ v2_t2) ⊗ (w1_t1 ⊗ w2_t2).
    #
    # The Kronecker product of vectors: (a ⊗ b)[i*n2 + j] = a[i]*b[j].
    #
    # So:
    # U_total has shape (n1*m1 * n2*m2, r1*r2) = (4*2, 56) = (8, 56).
    # U_total[(i1*m1 + j1) * (n2*m2) + (i2*m2 + j2), t1*r2 + t2] = U1[i1*m1+j1, t1] * U2[i2*m2+j2, t2]
    #
    # Wait, I need to be careful about the index ordering for the Kronecker product.
    #
    # For M_{n,m,p} ⊗ M_{n',m',p'} = M_{n*n', m*m', p*p'}:
    # The indices are:
    #   A: (i, j) where i in {0..n*n'-1}, j in {0..m*m'-1}
    #   B: (j, k) where j in {0..m*m'-1}, k in {0..p*p'-1}
    #   C: (i, k) where i in {0..n*n'-1}, k in {0..p*p'-1}
    #
    # The Kronecker product structure:
    #   i = i1 * n2 + i2, j = j1 * m2 + j2, k = k1 * p2 + k2
    #   A[i, j] = A1[i1, j1] * A2[i2, j2]  (element-wise in the Kronecker sense)
    #   Actually, A is the Kronecker product of A1 and A2? No.
    #
    # The tensor T = T1 ⊗ T2 means:
    #   T[i1*n2+i2, j1*m2+j2, k1*p2+k2] = T1[i1, j1, k1] * T2[i2, j2, k2]
    #
    # And the decomposition:
    #   U_total[(i1*m1+j1)*(n2*m2) + (i2*m2+j2), t1*r2+t2] 
    #     = U1[i1*m1+j1, t1] * U2[i2*m2+j2, t2]
    #
    # Hmm, this is getting complicated with the index ordering. Let me just construct it directly.
    #
    # For M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4}:
    #   n1=2, m1=2, p1=2, n2=1, m2=2, p2=4
    #   n = 2*1 = 2, m = 2*2 = 4, p = 2*4 = 8
    #
    # U1: shape (n1*m1, r1) = (4, 7)
    # V1: shape (m1*p1, r1) = (4, 7)
    # W1: shape (p1*n1, r1) = (4, 7)
    #
    # U2: shape (n2*m2, r2) = (2, 8)
    # V2: shape (m2*p2, r2) = (8, 8)
    # W2: shape (p2*n2, r2) = (4, 8)
    #
    # U_total: shape (n*m, r1*r2) = (8, 56)
    # V_total: shape (m*p, r1*r2) = (32, 56)
    # W_total: shape (p*n, r1*r2) = (16, 56)
    #
    # The Kronecker product indexing:
    # For U: index (i, j) in the "A" matrix.
    #   i = i1 * n2 + i2, j = j1 * m2 + j2
    #   U_total[i*m + j, t1*r2 + t2] = U1[i1*m1 + j1, t1] * U2[i2*m2 + j2, t2]
    #
    # For V: index (j, k) in the "B" matrix.
    #   j = j1 * m2 + j2, k = k1 * p2 + k2
    #   V_total[j*p + k, t1*r2 + t2] = V1[j1*p1 + k1, t1] * V2[j2*p2 + k2, t2]
    #
    # For W: index (k, i) in the "C" matrix (with the convention W[k*n+i, t]).
    #   k = k1 * p2 + k2, i = i1 * n2 + i2
    #   W_total[k*n + i, t1*r2 + t2] = W1[k1*n1 + i1, t1] * W2[k2*n2 + i2, t2]
    #
    # Let me verify the W convention. The task says W[k*n+i, r].
    # For M_{n,m,p}, C is n x p, with C[i, k].
    # W has shape (p*n, rank), and W[k*n + i, t] is the coefficient for C[i, k] in term t.
    #
    # So for the Kronecker product:
    #   k = k1 * p2 + k2, i = i1 * n2 + i2
    #   W_total[k*n + i, t1*r2 + t2] = W1[k1*n1 + i1, t1] * W2[k2*n2 + i2, t2]
    #
    # Let me code this up.

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2*2x2 Strassen decomposition, rank 7."""
        U = np.array([
            [1, 0, 0, 1],   # A00 + A11
            [0, 0, 1, 1],   # A10 + A11
            [1, 0, 0, 0],   # A00
            [0, 0, 0, 1],   # A11
            [1, 1, 0, 0],   # A00 + A10
            [0, 0, 1, -1],  # A10 - A00
            [0, 1, 0, -1],  # A01 - A11
        ], dtype=float)
        
        V = np.array([
            [1, 0, 0, 1],   # B00 + B11
            [1, 0, 0, 0],   # B00
            [0, 1, 0, -1],  # B01 - B11
            [0, 0, 1, -1],  # B10 - B00
            [0, 0, 0, 1],   # B11
            [1, 0, 1, 0],   # B00 + B10
            [0, 1, 0, 1],   # B01 + B11
        ], dtype=float)
        
        W = np.array([
            [1, 0, 0, -1, 0, -1, 1],   # C00 = M1 - M4 - M6 + M7
            [0, 0, 1, 0, 1, 0, 0],     # C01 = M3 + M5
            [0, 1, 0, 1, 0, 0, 0],     # C10 = M2 + M4
            [1, 0, -1, 0, 0, 1, -1],   # C11 = M1 - M3 + M6 - M7
        ], dtype=float)
        
        return U, V, W

    def schoolbook_1x2x4():
        """Return (U, V, W) for 1x2*2x4 schoolbook decomposition, rank 8."""
        U = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ], dtype=float)
        
        V = np.eye(8, dtype=float)
        
        W = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=float)
        
        return U, V, W

    def kronecker_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        """Compute Kronecker product of two tensor decompositions."""
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for t1 in range(r1):
            for t2 in range(r2):
                t = t1 * r2 + t2
                u1 = U1[:, t1]  # shape (n1*m1,)
                v1 = V1[:, t1]  # shape (m1*p1,)
                w1 = W1[:, t1]  # shape (p1*n1,)
                u2 = U2[:, t2]  # shape (n2*m2,)
                v2 = V2[:, t2]  # shape (m2*p2,)
                w2 = W2[:, t2]  # shape (p2*n2,)
                
                # U: index (i, j) where i = i1*n2 + i2, j = j1*m2 + j2
                for i1 in range(n1):
                    for j1 in range(m1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                i = i1 * n2 + i2
                                j = j1 * m2 + j2
                                U[i * m + j, t] = u1[i1 * m1 + j1] * u2[i2 * m2 + j2]
                
                # V: index (j, k) where j = j1*m2 + j2, k = k1*p2 + k2
                for j1 in range(m1):
                    for k1 in range(p1):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                j = j1 * m2 + j2
                                k = k1 * p2 + k2
                                V[j * p + k, t] = v1[j1 * p1 + k1] * v2[j2 * p2 + k2]
                
                # W: index (k, i) where k = k1*p2 + k2, i = i1*n2 + i2
                for k1 in range(p1):
                    for i1 in range(n1):
                        for k2 in range(p2):
                            for i2 in range(n2):
                                k = k1 * p2 + k2
                                i = i1 * n2 + i2
                                W[k * n + i, t] = w1[k1 * n1 + i1] * w2[k2 * n2 + i2]
        
        return U, V, W, r

    # Build the rank-56 decomposition
    U1, V1, W1 = strassen_2x2x2()
    U2, V2, W2 = schoolbook_1x2x4()
    
    U, V, W, rank = kronecker_decomp(U1, V1, W1, U2, V2, W2, 2, 2, 2, 1, 2, 4)
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        """Verify that U, V, W correctly decompose the n x m * m x p multiplication tensor."""
        # Test with random matrices
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for t in range(U.shape[1]):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                C_computed[i, k] = total
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False, np.max(np.abs(C_computed - C_expected))
        return True, 0.0
    
    valid, err = verify(U, V, W, n, m, p)
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
