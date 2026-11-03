"""Strassen-based rank-88 decomposition for 3x4 by 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-88 tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix, using Strassen's algorithm on 2x2 blocks.
    
    The idea: decompose the 3x4 by 4x8 multiplication as:
    - Split the 4x8 into two 4x4 blocks
    - Split the 3x4 into two parts: top 2 rows and bottom 1 row
    - Use Strassen for the 2x4 by 4x4 part (treating 2x4 as 2 blocks of 1x4,
      but more efficiently: treat as 2x2 blocks where each 'element' is a 1x4 row)
    
    Actually, let's think more carefully.
    
    We multiply A (3x4) by B (4x8) to get C (3x8).
    
    Split:
    A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4
    B = [B_left | B_right] where B_left is 4x4, B_right is 4x4
    
    C = [A_top @ B_left, A_top @ B_right; A_bottom @ B_left, A_bottom @ B_right]
    
    So we need:
    - A_top @ B_left (2x4 * 4x4 -> 2x4)
    - A_top @ B_right (2x4 * 4x4 -> 2x4)
    - A_bottom @ B_left (1x4 * 4x4 -> 1x4)
    - A_bottom @ B_right (1x4 * 4x4 -> 1x4)
    
    For 2x4 * 4x4: we can think of this as multiplying a 2x2 matrix of 1x2 blocks
    by a 2x2 matrix of 2x2 blocks. But that's not quite right either.
    
    Let me reconsider. The 2x4 matrix can be seen as:
    [a11 a12 a13 a14]
    [a21 a22 a23 a24]
    
    And 4x4 as:
    [b11 b12 b13 b14]
    [b21 b22 b23 b24]
    [b31 b32 b33 b34]
    [b41 b42 b43 b44]
    
    The product is 2x4.
    
    Alternatively, use Strassen on 2x2 blocks where we group columns.
    Group the 4 columns into 2 groups of 2:
    A_top = [A11 A12] where A11, A12 are 2x2
    B = [B1; B2] where B1, B2 are 4x2... no wait, B is 4x4.
    
    Let me group differently.
    A_top (2x4) = [P Q] where P, Q are 2x2
    B (4x4) = [R S; T U] where R, S, T, U are 2x2
    
    Then A_top @ B = [P Q] @ [R S; T U] = [P*R + Q*T, P*S + Q*U]
    
    So we need:
    - P*R + Q*T (2x2 * 2x2 + 2x2 * 2x2)
    - P*S + Q*U (2x2 * 2x2 + 2x2 * 2x2)
    
    Using Strassen for each pair:
    For P*R + Q*T: treat as (P+Q)*(R+T) - Q*(R-T) ... no, that's not standard.
    
    Actually, let's just use Strassen for 2x2*2x2 multiplications.
    
    For 2x2 * 2x2, Strassen uses 7 multiplications.
    
    So for A_top @ B:
    We need P*R, Q*T, P*S, Q*U - that's 4 separate 2x2*2x2 multiplications.
    Each takes 7 Strassen mults, so 28 total.
    
    For A_bottom @ B (1x4 * 4x4 -> 1x4):
    A_bottom = [a1 a2 a3 a4]
    This is just dot products:
    C1 = a1*B[:,0] + a2*B[:,1] + a3*B[:,2] + a4*B[:,3]
    Where each B[:,j] is a 4-vector, and a_j is scalar.
    
    Actually, this is 4 scalar-vector products, which is rank 4 (each term is one multiplication).
    Wait, no. Each output element C[i] = sum_j a_j * B[j,i]. That's 4 multiplications per output,
    8 outputs, but they share structure.
    
    Hmm, let me think about the tensor rank differently.
    
    Let me just directly construct the rank-88 decomposition.
    
    Actually, let me look at this more carefully.
    
    n=3, m=4, p=8. Schoolbook rank = 3*4*8 = 96.
    
    Using Strassen on 2x2 blocks:
    We can view the 3x4 by 4x8 multiplication as involving 2x2 block multiplications.
    
    Let me partition:
    A (3x4):
    Row 1: [a b] [c d]  (two 2-vectors)
    Row 2: [e f] [g h]
    Row 3: [i j] [k l]
    
    B (4x8):
    Split columns into two groups of 4: B = [B1 | B2]
    B1 (4x4):
    [a11 a12 a13 a14]
    [b11 b12 b13 b14]
    [c11 c12 c13 c14]
    [d11 d12 d13 d14]
    
    This is getting complicated. Let me just implement the direct construction.
    
    Strategy: Use Strassen to compute 2x2 * 2x2 with 7 mults.
    
    The 3x4 * 4x8 multiplication can be decomposed as:
    - 2 full 2x2*2x2 Strassen computations for the top 2 rows, left 4 cols
    - 2 full 2x2*2x2 Strassen computations for the top 2 rows, right 4 cols  
    - 4 scalar*vector for the bottom row, left 4 cols
    - 4 scalar*vector for the bottom row, right 4 cols
    
    Wait, that's not right either. Let me just code it up properly.
    
    Actually, I think the cleanest approach is:
    
    The multiplication tensor for (n,m,p) = (3,4,8) has rank at most 88.
    
    Here's how: 
    - For the 2x4 part of A times 4x8 part of B:
      Split B into B_left (4x4) and B_right (4x4).
      For A_top (2x4) * B (4x8):
        Split A_top into A_top_left (2x2) and A_top_right (2x2).
        Split B into 2x2 blocks: B = [BL11 BL12; BL21 BL22] | [BR11 BR12; BR21 BR22]
        
        A_top * B = [A_top_left * BL + A_top_right * BR_left_part, ...]
        
    This is getting messy. Let me just hardcode the Strassen-based construction.
    
    Let me define the decomposition explicitly.
    
    For Strassen M_{2,2,2} rank 7:
    The 7 rank-1 terms are:
    1. U=(1,0,1,0), V=(1,0,0,1), W=(1,1,0,0) -- wait, need to get indices right.
    
    For M_{2,2,2}, A is 2x2 (indices i,j), B is 2x2 (indices j,k), C is 2x2 (indices i,k).
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strassen:
    m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    m2 = (A[1,0]+A[1,1])*B[0,0]
    m3 = A[0,0]*(B[0,1]-B[1,1])
    m4 = A[1,1]*(B[1,0]-B[0,0])
    m5 = (A[0,0]+A[0,1])*B[1,1]
    m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m2 + m3 + m6
    
    So the rank-1 terms for Strassen:
    For m1: U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has 1 at (0,0) and (1,1)
    
    Wait, let me be more careful. Each m_t is a scalar product of a linear form in A and a linear form in B.
    
    m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U for m1: row 0: [1,0], row 1: [1,0] -> so U[0,0]=1, U[1,1]=1
    V for m1: V[0,0]=1, V[1,1]=1
    W for m1: W[0,0]=1, W[1,1]=1
    
    m2 = (A[1,0]+A[1,1]) * B[0,0]
    U: U[1,0]=1, U[1,1]=1
    V: V[0,0]=1
    W: W[1,0]=1
    
    m3 = A[0,0] * (B[0,1]-B[1,1])
    U: U[0,0]=1
    V: V[0,1]=1, V[1,1]=-1
    W: W[0,1]=1
    
    m4 = A[1,1] * (B[1,0]-B[0,0])
    U: U[1,1]=1
    V: V[1,0]=1, V[0,0]=-1
    W: W[1,0]=1
    
    m5 = (A[0,0]+A[0,1]) * B[1,1]
    U: U[0,0]=1, U[0,1]=1
    V: V[1,1]=1
    W: W[0,1]=1
    
    m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U: U[1,0]=1, U[0,0]=-1
    V: V[0,0]=1, V[0,1]=1
    W: W[1,1]=1
    
    m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U: U[0,1]=1, U[1,1]=-1
    V: V[1,0]=1, V[1,1]=1
    W: W[0,0]=1
    
    Now for the full 3x4*4x8, I'll construct it by combining these.
    
    Let me define the full decomposition with rank 88.
    
    Partition:
    A (3x4) = [A1 A2] where A1, A2 are 3x2
    B (4x8) = [B1 B2] where B1, B2 are 4x4
    
    C = A*B = [A1*B1, A1*B2, A2*B1, A2*B2] -- no wait, that's not right.
    C = [A1*B1, A1*B2; ... ] -- actually C is 3x8, so:
    C[:, 0:4] = A * B1
    C[:, 4:8] = A * B2
    
    So we need A*B1 (3x4*4x4 -> 3x4) and A*B2 (3x4*4x4 -> 3x4).
    
    For A*B1:
    Split A into top 2 rows (2x4) and bottom 1 row (1x4).
    Split B1 into 2x2 blocks.
    
    Let me just build it step by step in code.
    """
    import numpy as np
    from fractions import Fraction
    import time
    
    deadline = time.time() + 100  # Leave margin before 120s
    
    n, m, p = 3, 4, 8
    
    # We'll build the decomposition term by term.
    # Each term is a rank-1 contribution: u (length n*m=12), v (length m*p=32), w (length p*n=24)
    # Using trace convention: W[k*n+i, r]
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u, v, w):
        terms_U.append(u)
        terms_V.append(v)
        terms_W.append(w)
    
    # Strassen for 2x2*2x2, indices in A: (i,j) -> flat i*2+j
    # B: (j,k) -> flat j*2+k
    # C: (k,i) in trace convention -> flat k*2+i
    
    # The 7 Strassen terms for M_{2,2,2}:
    # Each is (u_vec, v_vec, w_vec) where:
    # u has indices for A (2x2), v for B (2x2), w for C output (2x2, trace order)
    
    strassen_terms = [
        # m1 = (a00+a11)(b00+b11)
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]),  # w: c00=1, c11=1
        # m2 = (a10+a11)b00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),  # w: c10=1 (trace: k=1,i=0 -> idx 1*2+0=2? wait)
    ]
    
    # Let me redo this carefully.
    # For M_{2,2,2}: A is 2x2, B is 2x2, C is 2x2.
    # A indices: (i,j) where i in {0,1}, j in {0,1}. Flat: i*2+j
    # B indices: (j,k) where j in {0,1}, k in {0,1}. Flat: j*2+k
    # C indices: (i,k) where i in {0,1}, k in {0,1}. 
    # Trace convention: W[k*n+i] where n=2, so W[k*2+i]
    
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So W index is k*2+i.
    
    # m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #   U: A[0,0]=1, A[1,1]=1 -> [1,0,0,1]
    #   V: B[0,0]=1, B[1,1]=1 -> [1,0,0,1]
    #   W: C[0,0]=1, C[1,1]=1 -> W[0*2+0]=1, W[1*2+1]=1 -> [1,0,0,1]
    
    # m2 = (A[1,0]+A[1,1])*B[0,0]
    #   U: A[1,0]=1, A[1,1]=1 -> [0,0,1,1]
    #   V: B[0,0]=1 -> [1,0,0,0]
    #   W: C[1,0]=1 -> W[0*2+1]=1 -> [0,1,0,0]
    
    # m3 = A[0,0]*(B[0,1]-B[1,1])
    #   U: A[0,0]=1 -> [1,0,0,0]
    #   V: B[0,1]=1, B[1,1]=-1 -> [0,1,-1,0]
    #   W: C[0,1]=1 -> W[1*2+0]=1 -> [0,0,1,0]
    
    # m4 = A[1,1]*(B[1,0]-B[0,0])
    #   U: A[1,1]=1 -> [0,0,0,1]
    #   V: B[1,0]=1, B[0,0]=-1 -> [-1,0,1,0]
    #   W: C[1,0]=1 -> W[0*2+1]=1 -> [0,1,0,0]
    
    # m5 = (A[0,0]+A[0,1])*B[1,1]
    #   U: A[0,0]=1, A[0,1]=1 -> [1,1,0,0]
    #   V: B[1,1]=1 -> [0,0,0,1]
    #   W: C[0,1]=1 -> W[1*2+0]=1 -> [0,0,1,0]
    
    # m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    #   U: A[1,0]=1, A[0,0]=-1 -> [-1,0,1,0]
    #   V: B[0,0]=1, B[0,1]=1 -> [1,1,0,0]
    #   W: C[1,1]=1 -> W[1*2+1]=1 -> [0,0,0,1]
    
    # m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #   U: A[0,1]=1, A[1,1]=-1 -> [0,1,0,-1]
    #   V: B[1,0]=1, B[1,1]=1 -> [0,0,1,1]
    #   W: C[0,0]=1 -> W[0*2+0]=1 -> [1,0,0,0]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # m2
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 0, 1, 0]),  # m3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 0]),  # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),   # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
    ]
    
    # Now for 3x4 * 4x4:
    # A is 3x4, B is 4x4, C is 3x4.
    # Split A into top 2 rows (2x4) and bottom 1 row (1x4).
    # 
    # For top 2 rows * B (4x4):
    #   Split A_top (2x4) into left 2x2 and right 2x2.
    #   Split B (4x4) into 2x2 blocks: B = [BL TL; BR TR] where each is 2x2.
    #   Actually B is 4x4, so:
    #   B = [[b00 b01 b02 b03],
    #        [b10 b11 b12 b13],
    #        [b20 b21 b22 b23],
    #        [b30 b31 b32 b33]]
    #   
    #   A_top = [[a00 a01 a02 a03],
    #            [a10 a11 a12 a13]]
    #   
    #   A_top * B:
    #   C_top[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    #   C_top[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    #   C_top[0,2] = a00*b02 + a01*b12 + a02*b22 + a03*b32
    #   C_top[0,3] = a00*b03 + a01*b13 + a02*b23 + a03*b33
    #   (similarly for row 1)
    #   
    #   Group columns into pairs:
    #   A_top = [A1 | A2] where A1 = [[a00,a01],[a10,a11]], A2 = [[a02,a03],[a12,a13]]
    #   B = [B1 | B2] where B1 = [[b00,b01],[b10,b11],[b20,b21],[b30,b31]], B2 = [[b02,b03],[b12,b13],[b22,b23],[b32,b33]]
    #   
    #   But B1 is 4x2, not 2x2. So I need to split B into 2x2 blocks differently.
    #   
    #   Let me split B as:
    #   B = [[B11 B12],
    #        [B21 B22]]
    #   where each Bxy is 2x2.
    #   B11 = [[b00,b01],[b10,b11]], B12 = [[b02,b03],[b12,b13]]
    #   B21 = [[b20,b21],[b30,b31]], B22 = [[b22,b23],[b32,b33]]
    #   
    #   And A_top = [A1 A2] where A1 = [[a00,a01],[a10,a11]], A2 = [[a02,a03],[a12,a13]]
    #   
    #   Then A_top * B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    #   
    #   So we need 4 2x2*2x2 multiplications, each done with Strassen (7 mults each).
    #   Total: 28 mults for top 2 rows.
    
    # For bottom 1 row * B (4x4):
    #   A_bottom = [a20 a21 a22 a23]
    #   C_bottom = A_bottom * B (1x4)
    #   C_bottom[0] = a20*b00 + a21*b10 + a22*b20 + a23*b30
    #   C_bottom[1] = a20*b01 + a21*b11 + a22*b21 + a23*b31
    #   C_bottom[2] = a20*b02 + a21*b12 + a22*b22 + a23*b32
    #   C_bottom[3] = a20*b03 + a21*b13 + a22*b32 + a23*b33  (wait, b32 should be b32)
    #   
    #   This is 4 dot products of length 4, which is rank 4*4 = 16 if done naively.
    #   But we can share: each (a2j, B[j,:]) pair contributes to all 4 outputs.
    #   
    #   Actually, for a 1x4 * 4x4 multiplication, the tensor rank is 4 (not 16).
    #   Because: C[k] = sum_j A[j] * B[j,k]
    #   Each term A[j]*B[:,k] is a rank-1 tensor for fixed j, varying k.
    #   So we need 4 terms (one per j), each with:
    #   U: 1 at position j in A (length 4)
    #   V: 1 at position j in B (length 16, but actually B is 4x4 so flat index j*4+0, j*4+1, etc.)
    #   
    #   Wait, no. For a rank-1 term, we need a single scalar multiplication.
    #   The term A[j] * B[j,k] for all k simultaneously is NOT rank-1 in the tensor sense.
    #   A rank-1 term is u * v * w where u is from A, v is from B, w is from C, and
    #   the contribution is u_i * v_j * w_k for each (i,j,k).
    #   
    #   So for the bottom row:
    #   C[i,k] = sum_j A[i,j] * B[j,k] where i=2 (fixed), j in 0..3, k in 0..3
    #   Each term A[2,j] * B[j,k] for a specific j and k is rank-1.
    #   So we need 4*4 = 16 terms for the bottom row alone? That seems like a lot.
    #   
    #   Wait, but A[2,j] is a scalar (since i=2 is fixed), so the "U" vector has a 1 at position 2*m+j = 2*4+j = 8+j.
    #   And B[j,k] has V at position j*p+k = j*8+k.
    #   And C[2,k] has W at position k*n+2 = k*3+2.
    #   
    #   So for each j in 0..3 and k in 0..3, we have a term:
    #   U: 1 at 8+j
    #   V: 1 at j*8+k
    #   W: 1 at k*3+2
    #   
    #   That's 16 terms. Plus 28 for the top = 44 per 4x4 block.
    #   Two blocks = 88 total. 
    #   
    #   So total rank = 88. 
    
    # Let me now construct all 88 terms.
    
    # Helper: create zero vectors
    def zero_U(): return [0] * (n * m)  # length 12
    def zero_V(): return [0] * (m * p)  # length 32
    def zero_W(): return [0] * (p * n)  # length 24
    
    # For top 2 rows (i in 0,1), we use Strassen on 2x2 blocks.
    # A indices: i in {0,1}, j in {0,1,2,3}
    # Split j into j1 in {0,1} and j2 in {2,3}
    # B indices: j in {0,1,2,3}, k in {0,1,2,3}
    # Split B into 2x2 blocks: B[j1,k1] where j1 in {0,1} means j in {0,1}, k1 in {0,1} means k in {0,1}, etc.
    
    # For each 2x2 block multiplication (A_block * B_block), we add 7 Strassen terms.
    
    # A_block for top-left: A[0:2, 0:2], indices (i, j) where i in 0,1; j in 0,1
    # A flat index: i*4 + j
    # B_block top-left: B[0:2, 0:2], indices (j, k) where j in 0,1; k in 0,1
    # B flat index: j*8 + k
    # C_block: C[0:2, 0:2], indices (i, k) where i in 0,1; k in 0,1
    # C flat index (trace): k*3 + i
    
    # Strassen term for this block:
    # For each (u_local, v_local, w_local) in strassen_terms:
    #   u_local is length 4, indexed by (i,j) -> flat i*2+j
    #   v_local is length 4, indexed by (j,k) -> flat j*2+k
    #   w_local is length 4, indexed by (k,i) -> flat k*2+i
    
    for block_a in [0, 1]:  # left (j in 0,1) or right (j in 2,3)
        for block_b in [0, 1]:  # left (k in 0,1) or right (k in 2,3)
            j_offset = block_a * 2
            k_offset = block_b * 2
            
            for (u_loc, v_loc, w_loc) in strassen_terms:
                u = zero_U()
                v = zero_V()
                w = zero_W()
                
                # u_loc is indexed by (i, j_local) where j_local in 0,1
                # A flat index: i*4 + (j_offset + j_local)
                for i in range(2):
                    for jl in range(2):
                        if u_loc[i * 2 + jl] != 0:
                            u[i * 4 + j_offset + jl] = u_loc[i * 2 + jl]
                
                # v_loc is indexed by (j_local, k_local) where j_local in 0,1, k_local in 0,1
                # B flat index: (j_offset + j_local)*8 + (k_offset + k_local)
                for jl in range(2):
                    for kl in range(2):
                        if v_loc[jl * 2 + kl] != 0:
                            v[(j_offset + jl) * 8 + (k_offset + kl)] = v_loc[jl * 2 + kl]
                
                # w_loc is indexed by (k_local, i) where k_local in 0,1, i in 0,1
                # W flat index (trace): (k_offset + k_local)*3 + i
                for kl in range(2):
                    for i in range(2):
                        if w_loc[kl * 2 + i] != 0:
                            w[(k_offset + kl) * 3 + i] = w_loc[kl * 2 + i]
                
                add_term(u, v, w)
    
    # That's 4 blocks * 7 terms = 28 terms for top 2 rows.
    
    # For bottom row (i=2):
    # C[2, k] = sum_j A[2, j] * B[j, k] for j in 0..3, k in 0..3
    # Each term: A[2,j] * B[j,k]
    # U: 1 at 2*4 + j = 8 + j
    # V: 1 at j*8 + k
    # W: 1 at k*3 + 2
    
    for j in range(4):
        for k in range(4):
            u = zero_U()
            v = zero_V()
            w = zero_W()
            u[8 + j] = 1
            v[j * 8 + k] = 1
            w[k * 3 + 2] = 1
            add_term(u, v, w)
    
    # That's 16 terms for bottom row.
    # Total: 28 + 16 = 44 terms for one 4x4 block.
    
    # But wait, we need to do this for BOTH B_left (columns 0-3) and B_right (columns 4-7).
    # I only did B_left above. Let me redo for B_right.
    
    # Actually, I already have the loop over block_b which covers k_offset 0 and 2.
    # But I only went up to k in 0..3. I need k in 0..7.
    
    # Let me restart with a cleaner approach.
    
    terms_U.clear()
    terms_V.clear()
    terms_W.clear()
    
    # For each of the two 4x4 blocks of B (left: k in 0..3, right: k in 4..7):
    for k_block in [0, 1]:
        k_base = k_block * 4
        
        # Top 2 rows: Strassen on 2x2 blocks
        for j_block in [0, 1]:  # j in 0..1 or j in 2..3
            j_base = j_block * 2
            for k_sub in [0, 1]:  # k within the 4-block: k in k_base..k_base+1 or k_base+2..k_base+3
                k_sub_base = k_base + k_sub * 2
                
                for (u_loc, v_loc, w_loc) in strassen_terms:
                    u = zero_U()
                    v = zero_V()
                    w = zero_W()
                    
                    for i in range(2):
                        for jl in range(2):
                            if u_loc[i * 2 + jl] != 0:
                                u[i * 4 + j_base + jl] = u_loc[i * 2 + jl]
                    
                    for jl in range(2):
                        for kl in range(2):
                            if v_loc[jl * 2 + kl] != 0:
                                v[(j_base + jl) * 8 + (k_sub_base + kl)] = v_loc[jl * 2 + kl]
                    
                    for kl in range(2):
                        for i in range(2):
                            if w_loc[kl * 2 + i] != 0:
                                w[(k_sub_base + kl) * 3 + i] = w_loc[kl * 2 + i]
                    
                    add_term(u, v, w)
        
        # Bottom row: 4x4 terms
        for j in range(4):
            for k in range(4):
                u = zero_U()
                v = zero_V()
                w = zero_W()
                u[8 + j] = 1
                v[j * 8 + (k_base + k)] = 1
                w[(k_base + k) * 3 + 2] = 1
                add_term(u, v, w)
    
    # Total terms: 2 * (4 * 7 + 16) = 2 * 44 = 88
    
    rank = len(terms_U)
    
    # Build matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = terms_U[c][i]
        for i in range(m * p):
            V[i][c] = terms_V[c][i]
        for i in range(p * n):
            W[i][c] = terms_W[c][i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
