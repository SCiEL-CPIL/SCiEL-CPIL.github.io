import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 5 blocks of 2 columns each
    num_blocks = p // 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for b in range(num_blocks):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # Part 1: Strassen for 2x2 x 2x2
        # A is 2x2 (cols 0,1), B is 2x2 (rows 0,1; cols k1,k2)
        # C is 2x2 (rows 0,1; cols k1,k2)
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00 (i=0, j=0), A11 (i=1, j=1)
        # V: B00 (j=0, k=k1), B11 (j=1, k=k2)
        # W: C00 (k=k1, i=0), C11 (k=k2, i=1) -- Wait, W index is k*n+i
        # W for C00 is k1*n + 0. W for C11 is k2*n + 1.
        # But Strassen mixes them.
        
        # Let's define the 11 terms explicitly.
        # Indices for U: i*m + j. i in {0,1}, j in {0,1,2}
        # Indices for V: j*p + k. j in {0,1,2}, k in {0..9}
        # Indices for W: k*n + i. k in {0..9}, i in {0,1}
        
        # Term 1: (A00 + A11)(B00 + B11)
        # U: 1 at (0,0) and (1,1)
        # V: 1 at (0,k1) and (1,k2)
        # W: 1 at (k1,0) and (k2,1)
        # This is not a rank 1 tensor. It's a product of sums.
        # We need to expand it into rank 1 components?
        # No, Strassen's algorithm defines 7 intermediate products.
        # Each intermediate product is a scalar.
        # The decomposition of the tensor T_2,2,2 into 7 rank-1 tensors is:
        # T = sum_{s=1}^7 u_s v_s w_s
        
        # Let's retrieve the standard Strassen decomposition for 2x2x2.
        # u_s are vectors in R^4 (flattened A)
        # v_s are vectors in R^4 (flattened B)
        # w_s are vectors in R^4 (flattened C)
        
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        
        # m1 = (a00 + a11)(b00 + b11)
        # m2 = (a10 + a11)b00
        # m3 = a00(b01 - b11)
        # m4 = a11(b10 - b00)
        # m5 = (a00 + a10)b11
        # m6 = (a11 - a00)(b00 + b01)
        # m7 = (a01 - a11)(b10 + b11)
        
        # c00 = m1 - m4 + m5
        # c01 = m3 + m7
        # c10 = m2 + m4
        # c11 = m1 - m3 + m6
        
        # We need to map these to U, V, W.
        # For each block b, we have 7 terms from Strassen and 4 from Outer Product.
        
        # Strassen Terms:
        # 1. m1
        # U: a00 (0,0), a11 (1,1) -> U[0, c]=1, U[3, c]=1
        # V: b00 (0,k1), b11 (1,k2) -> V[k1, c]=1, V[p+k2, c]=1
        # W: c00 (k1,0), c11 (k2,1) -> W[k1*n+0, c]=1, W[k2*n+1, c]=1
        # Wait, W is added/subtracted.
        # c00 gets +m1. c11 gets +m1.
        # So W[k1*n+0, c] += 1, W[k2*n+1, c] += 1.
        
        # 2. m2
        # U: a10 (1,0), a11 (1,1) -> U[3, c]=1, U[4, c]=1
        # V: b00 (0,k1) -> V[k1, c]=1
        # W: c10 (k1,1) -> W[k1*n+1, c] += 1
        
        # 3. m3
        # U: a00 (0,0) -> U[0, c]=1
        # V: b01 (0,k2), b11 (1,k2) -> V[k2, c]=1, V[p+k2, c]=-1
        # W: c01 (k2,0) -> W[k2*n+0, c] += 1
        
        # 4. m4
        # U: a11 (1,1) -> U[3, c]=1
        # V: b10 (1,k1), b00 (0,k1) -> V[p+k1, c]=1, V[k1, c]=-1
        # W: c00 (k1,0) -> W[k1*n+0, c] -= 1
        #    c10 (k1,1) -> W[k1*n+1, c] += 1
        
        # 5. m5
        # U: a00 (0,0), a10 (1,0) -> U[0, c]=1, U[3, c]=1
        # V: b11 (1,k2) -> V[p+k2, c]=1
        # W: c00 (k1,0) -> W[k1*n+0, c] += 1
        
        # 6. m6
        # U: a11 (1,1), a00 (0,0) -> U[3, c]=1, U[0, c]=-1
        # V: b00 (0,k1), b01 (0,k2) -> V[k1, c]=1, V[k2, c]=1
        # W: c11 (k2,1) -> W[k2*n+1, c] += 1
        
        # 7. m7
        # U: a01 (0,1), a11 (1,1) -> U[1, c]=1, U[3, c]=-1
        # V: b10 (1,k1), b11 (1,k2) -> V[p+k1, c]=1, V[p+k2, c]=1
        # W: c01 (k2,0) -> W[k2*n+0, c] += 1
        
        # Outer Product Terms (A[:,2] * B[2,:]):
        # 8. a02 b2k1
        # U: a02 (0,2) -> U[2, c]=1
        # V: b2k1 (2,k1) -> V[2*p+k1, c]=1
        # W: c0k1 (k1,0) -> W[k1*n+0, c] += 1
        
        # 9. a02 b2k2
        # U: a02 (0,2) -> U[2, c]=1
        # V: b2k2 (2,k2) -> V[2*p+k2, c]=1
        # W: c0k2 (k2,0) -> W[k2*n+0, c] += 1
        
        # 10. a12 b2k1
        # U: a12 (1,2) -> U[5, c]=1
        # V: b2k1 (2,k1) -> V[2*p+k1, c]=1
        # W: c1k1 (k1,1) -> W[k1*n+1, c] += 1
        
        # 11. a12 b2k2
        # U: a12 (1,2) -> U[5, c]=1
        # V: b2k2 (2,k2) -> V[2*p+k2, c]=1
        # W: c1k2 (k2,1) -> W[k2*n+1, c] += 1
        
        # We will add these to the global U, V, W matrices.
        
        # Helper to add a term
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
            U[u_idx][col_offset] += u_val
            V[v_idx][col_offset] += v_val
            W[w_idx][col_offset] += w_val

        # Term 1
        add_term(0, 1, k1, 1, k1*n+0, 1) # a00 b00 c00
        add_term(3, 1, k1, 1, k2*n+1, 1) # a11 b00 c11 -- WAIT
        # m1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
        # This is not a single rank 1 tensor.
        # Strassen's algorithm works by computing 7 products, then combining them.
        # The tensor decomposition of T_2,2,2 into 7 rank 1 tensors is:
        # T = sum_{s=1}^7 U_s V_s W_s
        # Where U_s, V_s, W_s are vectors.
        # My previous mapping was incorrect. I was trying to map the *products* to rank 1 tensors, but each product is a scalar, not a tensor.
        # The rank 1 tensors correspond to the 7 intermediate values.
        
        # Correct Strassen Decomposition:
        # U1 = [1, 0, 0, 1] (a00, a01, a10, a11) -> a00 + a11
        # V1 = [1, 0, 0, 1] (b00, b01, b10, b11) -> b00 + b11
        # W1 = [1, 0, 0, 1] (c00, c01, c10, c11) -> c00 + c11
        # Wait, c00 + c11 is not the output.
        # The outputs are c00, c01, c10, c11.
        # The decomposition is:
        # c00 = m1 - m4 + m5
        # c01 = m3 + m7
        # c10 = m2 + m4
        # c11 = m1 - m3 + m6
        
        # So W vectors are:
        # W1 = [1, 0, 0, -1] ? No.
        # m1 contributes to c00 (+1) and c11 (+1).
        # So W1 = [1, 0, 0, 1] in basis (c00, c01, c10, c11).
        # m2 contributes to c10 (+1).
        # W2 = [0, 0, 1, 0].
        # m3 contributes to c01 (+1) and c11 (-1).
        # W3 = [0, 1, 0, -1].
        # m4 contributes to c00 (-1) and c10 (+1).
        # W4 = [-1, 0, 1, 0].
        # m5 contributes to c00 (+1).
        # W5 = [1, 0, 0, 0].
        # m6 contributes to c11 (+1).
        # W6 = [0, 0, 0, 1].
        # m7 contributes to c01 (+1).
        # W7 = [0, 1, 0, 0].
        
        # U vectors:
        # U1 = [1, 0, 0, 1]
        # U2 = [0, 0, 1, 1]
        # U3 = [1, 0, 0, 0]
        # U4 = [0, 0, 0, 1]
        # U5 = [1, 0, 1, 0]
        # U6 = [1, 0, 0, -1]
        # U7 = [0, 1, 0, -1]
        
        # V vectors:
        # V1 = [1, 0, 0, 1]
        # V2 = [1, 0, 0, 0]
        # V3 = [0, 1, -1, 0]
        # V4 = [0, 0, 1, -1]
        # V5 = [0, 0, 0, 1]
        # V6 = [1, 1, 0, 0]
        # V7 = [0, 0, 1, 1]
        
        # We map these to the global indices.
        # For block b, k1, k2.
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in local 2x2.
        # Global U indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1).
        # B indices: 0:(0,k1), 1:(0,k2), 2:(1,k1), 3:(1,k2) in local 2x2.
        # Global V indices: k1, k2, p+k1, p+k2.
        # C indices: 0:(k1,0), 1:(k2,0), 2:(k1,1), 3:(k2,1) in local 2x2.
        # Global W indices: k1*n+0, k2*n+0, k1*n+1, k2*n+1.
        
        # We will add these 7 terms.
        
        # Term 1
        add_term(0, 1, k1, 1, k1*n+0, 1) # U1[0], V1[0], W1[0]
        add_term(4, 1, p+k2, 1, k2*n+1, 1) # U1[3], V1[3], W1[3]
        
        # Term 2
        add_term(3, 1, k1, 1, k1*n+1, 1) # U2[2], V2[0], W2[2]
        add_term(4, 1, k1, 1, k1*n+1, 1) # U2[3], V2[0], W2[2]
        
        # Term 3
        add_term(0, 1, k2, 1, k2*n+0, 1) # U3[0], V3[1], W3[1]
        add_term(0, 1, p+k2, -1, k2*n+1, -1) # U3[0], V3[2], W3[3]
        
        # Term 4
        add_term(4, 1, p+k1, 1, k1*n+0, -1) # U4[3], V4[2], W4[0]
        add_term(4, 1, k1, -1, k1*n+1, 1) # U4[3], V4[0], W4[2]
        
        # Term 5
        add_term(0, 1, p+k2, 1, k1*n+0, 1) # U5[0], V5[3], W5[0]
        add_term(3, 1, p+k2, 1, k1*n+0, 1) # U5[2], V5[3], W5[0]
        
        # Term 6
        add_term(4, 1, k1, 1, k2*n+1, 1) # U6[3], V6[0], W6[3]
        add_term(0, -1, k2, 1, k2*n+1, 1) # U6[0], V6[1], W6[3]
        
        # Term 7
        add_term(1, 1, p+k1, 1, k2*n+0, 1) # U7[1], V7[2], W7[1]
        add_term(4, -1, p+k2, 1, k2*n+0, 1) # U7[3], V7[3], W7[1]
        
        # Outer Product Terms
        # 8. a02 b2k1
        add_term(2, 1, 2*p+k1, 1, k1*n+0, 1)
        
        # 9. a02 b2k2
        add_term(2, 1, 2*p+k2, 1, k2*n+0, 1)
        
        # 10. a12 b2k1
        add_term(5, 1, 2*p+k1, 1, k1*n+1, 1)
        
        # 11. a12 b2k2
        add_term(5, 1, 2*p+k2, 1, k2*n+1, 1)
        
        col_offset += 11
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
