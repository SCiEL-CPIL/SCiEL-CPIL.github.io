import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    # Indices
    # U[i, k] corresponds to A[i // 4, i % 4] ? No.
    # A is 2x4. U is indexed by A's entries.
    # U[row * 4 + col, k]
    # V is indexed by B's entries.
    # V[row * 4 + col, k]
    # W is indexed by C's entries.
    # W[col * 2 + row, k] (Trace convention W[k*n+i, r] -> W[j*n+i, k] for C[i,j])
    
    def u_idx(r, c): return r * 4 + c
    def v_idx(r, c): return r * 4 + c
    def w_idx(r, c): return c * 2 + r
    
    current_rank = 0
    
    # We have 4 blocks of B (2x2 each)
    # B is 4x4.
    # B[0:2, 0:2] is B00
    # B[0:2, 2:4] is B01
    # B[2:4, 0:2] is B10
    # B[2:4, 2:4] is B11
    
    # A is 2x4.
    # A[0:2, 0:2] is A0
    # A[0:2, 2:4] is A1
    
    # C is 2x4.
    # C[0:2, 0:2] is C0
    # C[0:2, 2:4] is C1
    
    # C0 = A0 * B00 + A1 * B10
    # C1 = A0 * B01 + A1 * B11
    
    # We can treat this as two independent 2x4x2 multiplications?
    # C0 = [A0, A1] * [B00; B10]
    # C1 = [A0, A1] * [B01; B11]
    
    # Since A is shared, we can potentially save ranks.
    # But for now, we use the block Strassen approach for each 2x2x2 block.
    # Wait, the current code uses 7 ranks per 2x2 block of C?
    # There are 4 2x2 blocks in C? No, C is 2x4, so two 2x2 blocks.
    # Each 2x2 block C_ij is sum of two 2x2 products.
    # C0 = A0 B00 + A1 B10.
    # This is not a single 2x2x2 multiplication.
    # It is a sum of two.
    # Rank of sum of two 2x2x2 multiplications is 14 if independent.
    # But A0, A1 are shared.
    
    # The current program uses 7 ranks per "block" in a loop of 4 blocks.
    # Blocks are (0,0,0,0,0), (0,2,2,0,0), (0,0,0,2,2), (0,2,2,2,2).
    # This corresponds to:
    # 1. A rows 0-1, cols 0-1 (A0) with B rows 0-1, cols 0-1 (B00) -> C rows 0-1, cols 0-1 (C0)
    # This is just A0 * B00.
    # But C0 also needs A1 * B10.
    # So the current program is NOT computing the full multiplication?
    # Let's check the current program's logic.
    # It defines 4 blocks.
    # Block 1: ar=0, ac=0, br=0, bc=0, cc=0.
    # u11 = u(0,0), u22 = u(1,1).
    # v11 = v(0,0), v22 = v(1,1).
    # w11 = w(0,0), w22 = w(1,1).
    # This computes part of C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2].
    # It ignores A[0:2, 2:4] and B[2:4, 0:2].
    # So the current program is WRONG for the full 2x4x4 multiplication?
    # Or does it sum them up?
    # The `set_u` functions ADD to the flat array.
    # So if multiple blocks contribute to the same U, V, W entries, they sum up.
    # Block 1 contributes to C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2].
    # Block 2: ar=0, ac=2, br=2, bc=0, cc=0.
    # u11 = u(0,2), u22 = u(1,3).
    # v11 = v(2,0), v22 = v(3,1).
    # w11 = w(0,0), w22 = w(1,1).
    # This contributes to C[0:2, 0:2] using A[0:2, 2:4] and B[2:4, 0:2].
    # So C[0:2, 0:2] is sum of Block 1 and Block 2 contributions.
    # This is correct!
    # So the current program computes the full multiplication by summing 4 block contributions.
    # Each block is a 2x2x2 multiplication (rank 7).
    # Total rank 28.
    
    # So the current program is correct and has rank 28.
    
    # Can we reduce it?
    # The blocks are independent in terms of B and C, but share A?
    # Block 1 uses A[0:2, 0:2].
    # Block 2 uses A[0:2, 2:4].
    # Block 3 uses A[0:2, 0:2] (ar=0, ac=0) and B[0:2, 2:4] (br=0, bc=2) -> C[0:2, 2:4] (cc=2).
    # Block 4 uses A[0:2, 2:4] (ar=0, ac=2) and B[2:4, 2:4] (br=2, bc=2) -> C[0:2, 2:4] (cc=2).
    
    # So A[0:2, 0:2] is used in Block 1 and Block 3.
    # A[0:2, 2:4] is used in Block 2 and Block 4.
    # This sharing allows for rank reduction.
    # The rank 24 algorithm exploits this.
    
    # Since I cannot derive the exact rank 24 coefficients, I will stick to rank 28.
    
    blocks = [
        (0, 0, 0, 0, 0), # A0, B00 -> C0
        (0, 2, 2, 0, 0), # A1, B10 -> C0
        (0, 0, 0, 2, 2), # A0, B01 -> C1
        (0, 2, 2, 2, 2)  # A1, B11 -> C1
    ]
    
    for (ar, ac, br, bc, cc) in blocks:
        u11 = u_idx(ar, ac)
        u12 = u_idx(ar, ac + 1)
        u21 = u_idx(ar + 1, ac)
        u22 = u_idx(ar + 1, ac + 1)
        
        v11 = v_idx(br, bc)
        v12 = v_idx(br, bc + 1)
        v21 = v_idx(br + 1, bc)
        v22 = v_idx(br + 1, bc + 1)
        
        w11 = w_idx(0, cc)
        w12 = w_idx(0, cc + 1)
        w21 = w_idx(1, cc)
        w22 = w_idx(1, cc + 1)
        
        # Strassen 7 multiplications for 2x2x2
        # M1 = (U11+U22)(V11+V22) -> W11, W22
        # M2 = (U21+U22)V11 -> W21, W22
        # M3 = U11(V12-V22) -> W12, W22
        # M4 = U22(V21-V11) -> W11, W21
        # M5 = (U11+U12)V22 -> W11, W12
        # M6 = (U21-U11)(V11+V12) -> W22
        # M7 = (U12-U22)(V21+V22) -> W11
        
        # M1
        U[u11, current_rank] += 1
        U[u22, current_rank] += 1
        V[v11, current_rank] += 1
        V[v22, current_rank] += 1
        W[w11, current_rank] += 1
        W[w22, current_rank] += 1
        current_rank += 1
        
        # M2
        U[u21, current_rank] += 1
        U[u22, current_rank] += 1
        V[v11, current_rank] += 1
        W[w21, current_rank] += 1
        W[w22, current_rank] -= 1
        current_rank += 1
        
        # M3
        U[u11, current_rank] += 1
        V[v12, current_rank] += 1
        V[v22, current_rank] -= 1
        W[w12, current_rank] += 1
        W[w22, current_rank] += 1
        current_rank += 1
        
        # M4
        U[u22, current_rank] += 1
        V[v21, current_rank] += 1
        V[v11, current_rank] -= 1
        W[w11, current_rank] += 1
        W[w21, current_rank] += 1
        current_rank += 1
        
        # M5
        U[u11, current_rank] += 1
        U[u12, current_rank] += 1
        V[v22, current_rank] += 1
        W[w11, current_rank] -= 1
        W[w12, current_rank] += 1
        current_rank += 1
        
        # M6
        U[u21, current_rank] += 1
        U[u11, current_rank] -= 1
        V[v11, current_rank] += 1
        V[v12, current_rank] += 1
        W[w22, current_rank] += 1
        current_rank += 1
        
        # M7
        U[u12, current_rank] += 1
        U[u22, current_rank] -= 1
        V[v21, current_rank] += 1
        V[v22, current_rank] += 1
        W[w11, current_rank] += 1
        current_rank += 1
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
