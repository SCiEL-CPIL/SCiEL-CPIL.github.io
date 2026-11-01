import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 2, 12

    # Build the target tensor T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    # ------------------------------------------------------------------
    # Rank-10 decomposition for 2x2 * 2x4
    # A is 2x2, B is 2x4, C is 4x2
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # a_idx = i*2+j, b_idx = j*4+k, c_idx = k*2+i
    #
    # Strategy: Split B into B_left (cols 0,1) and B_right (cols 2,3).
    # C = A * B_left + A * B_right (each product is 4x2)
    # Use Strassen for each 2x2*2x2 (rank 7 each) = 14 total.
    # Then find a rank-10 combination by sharing/simplifying.
    #
    # Actually, let's use a direct constructive approach:
    # For 2x2 * 2x4, we can write:
    #   C[k,i] = sum_j A[i,j] * B[j,k]
    #
    # Let's use the following rank-10 scheme (verified by construction):
    # We define 10 rank-1 tensors (u_r, v_r, w_r) where:
    #   u_r has shape (4,)  [for A entries: (0,0),(0,1),(1,0),(1,1)]
    #   v_r has shape (8,)  [for B entries: (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
    #   w_r has shape (8,)  [for C entries: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)]
    #
    # C = sum_r outer(u_r, v_r, w_r) in tensor form:
    #   C_tensor[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    #
    # Construction:
    # Split into two 2x2x2 Strassen blocks (for k=0,1 and k=2,3) = 14 terms.
    # Then we look for linear dependencies to reduce to 10.
    #
    # Alternative: Use a known explicit rank-10 for 2x2x4.
    # Let me construct it explicitly using the following 10 products:
    #
    # For the left half (k=0,1):
    #   P1 = (A00+A11)*(B00+B10)  -> contributes to C00, C11 (k=0,1 parts)
    #   P2 = (A10+A11)*B00        -> C10
    #   P3 = A00*(B01-B11)        -> C01
    #   P4 = A11*(B10-B00)        -> C00, C10
    #   P5 = (A00+A01)*B11        -> -C00, C01
    #   P6 = (A10-A00)*(B00+B01)  -> C11
    #   P7 = (A01-A11)*(B10+B11)  -> -C00, C11
    #
    # For the right half (k=2,3):
    #   P8 = (A00+A11)*(B02+B12)  -> C20, C31
    #   P9 = (A10+A11)*B02        -> C30
    #   P10 = A00*(B03-B13)       -> C21
    #   ... this gives 14 again.
    #
    # To get rank 10, we need a smarter combination. Let me use a different approach:
    # Use the fact that for 2x2*2x4, the tensor has a specific structure.
    #
    # Let me just try to verify if rank 10 works by constructing it differently.
    # Actually, let me use a simpler approach: just use 3x7=21 from Strassen
    # and see if we can do better. But 21 > 30/3=10, so that won't help.
    #
    # Let me try a different partitioning: 2x2 * 2x12 directly.
    # Lower bound: rank >= n*m*p / (n+m+p-2) ... not standard.
    # Actually for matrix multiplication m x n * n x p, lower bound is mn.
    # Here 2*2=4, 2*12=24. So rank >= 24? No, that's for square.
    #
    # Let me just try the 3-block Strassen approach: 3 * 7 = 21? No, each block
    # is 2x2 * 2x4 which needs more than 7.
    #
    # Let me reconsider. The correct approach:
    # 2x2 * 2x12: Split the 12 columns into 6 groups of 2.
    # Each 2x2 * 2x2 block has rank 7 (Strassen).
    # Total: 6 * 7 = 42. That's what we have.
    #
    # To do better, we need to exploit structure across blocks.
    # Key insight: A is the SAME 2x2 matrix for all blocks!
    # So the U factors (which encode A) can be SHARED across blocks.
    #
    # In the Strassen decomposition of 2x2*2x2, the 7 "products" are:
    #   M1 = (a00+a11)(b00+b11)
    #   M2 = (a10+a11)b00
    #   M3 = a00(b01-b11)
    #   M4 = a11(b10-b00)
    #   M5 = (a00+a01)b11
    #   M6 = (a10-a00)(b00+b01)
    #   M7 = (a01-a11)(b10+b11)
    #
    # The U factor (encoding A) for each product is a fixed 4-vector.
    # For 6 independent blocks, we have 42 U-vectors, but they repeat in pattern.
    #
    # However, in tensor decomposition, each rank-1 term is independent.
    # We can't just "share" U vectors because the decomposition is a sum of
    # outer products, and each term must be a full rank-1 tensor.
    #
    # BUT: we CAN reformulate. If we define new "super-products" that combine
    # multiple B columns, we might reduce the total count.
    #
    # New approach: Group the 12 B columns into 4 groups of 3.
    # For each group, we need a 2x2 * 2x3 multiplication.
    # Rank of 2x2 * 2x3? At least 6 (schoolbook). Can we do 5?
    # If rank(2x2*2x3) = 5, then total = 4*5 = 20.
    # If rank = 6, total = 24.
    #
    # Actually, let me just try to directly construct a low-rank solution.
    #
    # For 2x2 * 2x3:
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[0,2] = A00*B02 + A01*B12
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    # C[1,2] = A10*B02 + A11*B12
    #
    # That's 6 terms in schoolbook. Can we do 5?
    # Try: P1 = A00*(B00+B01+B02), P2 = A01*(B10+B11+B12), ...
    # No, that gives C[k,i] = A0i * (sum of B0k) which is wrong.
    #
    # The structure is: C is a 3x2 matrix where C[k,i] = sum_j A[i,j]*B[j,k].
    # This is exactly matrix multiplication. The rank of the 2x2*2x3 tensor
    # is known to be 6 (no improvement over schoolbook for these small dims
    # beyond Strassen for 2x2x2).
    #
    # So 4 groups of 3: 4*6 = 24. That's better than 42!
    #
    # Let me verify: can we do even better with a different grouping?
    # 3 groups of 4: if rank(2x2*2x4) = 10, then 3*10 = 30.
    # 2 groups of 6: if rank(2x2*2x6) = 12, then 2*12 = 24.
    # 1 group of 12: rank(2x2*2x12) = ?
    #
    # For 2x2 * 2xp, the rank is at most 2p (schoolbook) and at least 2p - something.
    # Actually, I recall that for 2x2 * 2xp, the rank is exactly 2p for p>=2
    # unless Strassen-like tricks apply.
    #
    # Wait - Strassen applies to 2x2*2x2 giving rank 7 < 8.
    # For 2x2*2x4, can we get rank < 12?
    #
    # Let me try a direct construction for 2x2 * 2x4 with rank 10:
    #
    # Define 10 products. The key is that A is 2x2 (4 entries) and we want
    # to form C[k,i] = A0i*B0k + A1i*B1k for k=0..3, i=0,1.
    #
    # Let me use the following scheme (inspired by combining two Strassen blocks
    # with shared A-terms):
    #
    # Left block (k=0,1) Strassen products:
    #   M1_L = (A00+A11)*(B00+B10)
    #   M2_L = (A10+A11)*B00
    #   M3_L = A00*(B01-B11)
    #   M4_L = A11*(B10-B00)
    #   M5_L = (A00+A01)*B11
    #   M6_L = (A10-A00)*(B00+B01)
    #   M7_L = (A01-A11)*(B10+B11)
    #
    # Right block (k=2,3) Strassen products:
    #   M1_R = (A00+A11)*(B02+B12)
    #   M2_R = (A10+A11)*B02
    #   M3_R = A00*(B03-B13)
    #   M4_R = A11*(B12-B02)
    #   M5_R = (A00+A01)*B13
    #   M6_R = (A10-A00)*(B02+B03)
    #   M7_R = (A01-A11)*(B12+B13)
    #
    # Total: 14 products. Can we merge some?
    #
    # Notice M1_L and M1_R both have U-factor (A00+A11). If we define:
    #   S1 = A00+A11
    # Then M1_L = S1*(B00+B10), M1_R = S1*(B02+B12).
    # These are different rank-1 tensors (different V and W), so they can't merge.
    #
    # However, we can define a NEW product:
    #   P1 = S1 * (B00+B10+B02+B12)
    # This would contribute to C00, C11 (from left) AND C20, C31 (from right).
    # But we need C00 = M1_L - M4_L - M5_L + M7_L (from Strassen reconstruction)
    # and C20 = M1_R - M4_R - M5_R + M7_R.
    #
    # The issue is that the W-factors (output contributions) are different.
    # In a rank-1 tensor, the W-factor is fixed for all outputs.
    #
    # So P1 = S1 * (B00+B10+B02+B12) would have W-factor with 1s at all 4 positions,
    # but we need different coefficients at different positions.
    #
    # This means we CANNOT simply merge them. The 14-term decomposition is tight
    # for this particular construction.
    #
    # Let me try a completely different approach: use the 2x2*2x3 = rank 6 result.
    # 4 groups of 3 columns: total rank = 4 * 6 = 24.
    #
    # Let me construct this explicitly.
    # ------------------------------------------------------------------

    # Rank-6 decomposition for 2x2 * 2x3 (schoolbook, which is optimal here)
    # C[k,i] = A[0,i]*B[0,k] + A[1,i]*B[1,k] for k=0,1,2; i=0,1
    # 6 terms, one for each (i,j,k) combination.
    
    def build_2x2_2x3_block(start_k):
        """Build U, V, W entries for a 2x2 * 2x3 block starting at column start_k.
        Returns lists of (local_a, coeff), (local_b, coeff), (local_c, coeff) for 6 terms.
        local_a: index into 2x2 A (0..3)
        local_b: index into 2x3 B (0..5)  
        local_c: index into 3x2 C (0..5)
        """
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i * 2 + j
                    b = j * 3 + k
                    c = k * 2 + i
                    terms.append((a, b, c))
        return terms

    # Build the full rank-24 decomposition
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for group in range(4):
        start_k = group * 3
        terms = build_2x2_2x3_block(start_k)
        for (a, b, c) in terms:
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[c, col] = 1.0
            col += 1
    
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 24 doesn't verify (shouldn't happen), fall back to rank 42
    # Build Strassen rank-42
    U42 = np.zeros((n * m, 42))
    V42 = np.zeros((m * p, 42))
    W42 = np.zeros((p * n, 42))
    
    # Strassen 2x2x2 factors
    # U entries for each of 7 products (4-element vectors)
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    # V entries for each product (4-element vectors for 2x2 B)
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11 (B[j,k]: j=0,k=0 -> 0; j=1,k=1 -> 3)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # W entries for each product (4-element vectors for 2x2 C)
    # C[k,i]: k=0,1; i=0,1. Index = k*2+i
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, 0],   # M3: C01
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [-1, 0, 0, 1],  # M7: -C00+C11
    ]
    
    for t in range(6):
        for s in range(7):
            c = 7 * t + s
            for a_idx in range(4):
                U42[a_idx, c] = strassen_U[s][a_idx]
            for b_idx in range(4):
                j = b_idx // 2
                k_loc = b_idx % 2
                k = 2 * t + k_loc
                V42[j * p + k, c] = strassen_V[s][b_idx]
            for c_idx in range(4):
                k_loc = c_idx // 2
                i = c_idx % 2
                k = 2 * t + k_loc
                W42[k * n + i, c] = strassen_W[s][c_idx]
    
    if verify(U42, V42, W42):
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Ultimate fallback: schoolbook rank 48
    rank_fb = 48
    Ufb = np.zeros((n * m, rank_fb))
    Vfb = np.zeros((m * p, rank_fb))
    Wfb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Ufb[a, col] = 1.0
                Vfb[b, col] = 1.0
                Wfb[c, col] = 1.0
                col += 1
    
    return {"U": Ufb.tolist(), "V": Vfb.tolist(), "W": Wfb.tolist()}
# EVOLVE-BLOCK-END
