"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Uses a structured approach:
    1. Start with the schoolbook decomposition (rank 60).
    2. Attempt to reduce rank by identifying and applying known bilinear identities (like Strassen-style)
       on sub-blocks where possible.
    3. Since exact minimal rank for 2x5x6 is not trivially known to be less than 60 in simple structured ways
       without complex search, we implement a robust search for linear dependencies among the tensor slices
       to see if we can drop terms.
       
    However, for 2x5x6, the schoolbook rank is 60. 
    Known lower bounds: The rank of matrix multiplication tensor M_{n,m,p} is at least nm, mp, and np.
    Here: 10, 30, 12. So rank >= 30.
    Upper bound: 60.
    
    A common improvement is using Strassen's algorithm for 2x2 blocks.
    We can partition the 2x5 and 5x6 matrices into blocks.
    Let A be 2x5, B be 5x6.
    We can view A as [A1 A2] where A1 is 2x2, A2 is 2x3? Or 2x1, 2x4?
    Let's try to use the structure.
    
    Actually, a more reliable "improvement" strategy in these coding tasks when the optimal isn't immediately obvious
    is to perform a local search over the space of decompositions or to use a known specific decomposition if available.
    
    For 2x5x6, one can decompose the 5x5 part? No.
    Let's look at the dimensions: n=2, m=5, p=6.
    
    If we split the middle dimension m=5 into 2+3.
    A = [A_2 | A_3] (2x2 and 2x3)
    B = [B_2; B_3] (2x6 and 3x6)
    C = A_2 B_2 + A_3 B_3
    
    A_2 B_2 is 2x2x2 multiplication. Rank 7 (Strassen) instead of 8.
    A_3 B_3 is 2x3x6 multiplication.
    Rank of 2x3x6?
    Schoolbook: 2*3*6 = 36.
    Can we do better?
    2x3x6 can be seen as two 2x3x3 multiplications?
    B_3 is 3x6 = [B_3a | B_3b] where each is 3x3.
    C_right = A_3 B_3a + A_3 B_3b.
    This is 2x3x3 + 2x3x3.
    Rank of 2x3x3?
    Schoolbook: 18.
    Is there a better rank for 2x3x3?
    Lower bound: max(6, 9, 6) = 9.
    Known rank of 2x3x3 is 14? Or 15?
    Actually, let's look up or derive.
    
    Alternatively, just stick to the schoolbook if no simple improvement is found, but the prompt asks for a stronger algorithm.
    The "stronger algorithm" here implies searching for a lower rank.
    
    Let's implement a basic rank-reduction search.
    We start with the schoolbook basis.
    We try to find linear combinations of the tensor terms that allow us to eliminate some terms.
    
    Specifically, we represent the tensor T_{i,j,k} = delta_{i,j,k} (standard multiplication tensor).
    We look for a decomposition T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    
    If we use Strassen for the 2x2x2 part:
    A_2 B_2 uses 7 terms.
    A_3 B_3:
    Let's split A_3 (2x3) and B_3 (3x6).
    We can further split B_3 into three 3x2 blocks?
    B_3 = [B_31, B_32, B_33] each 3x2.
    Then C = A_2 B_2 + A_3 B_31 + A_3 B_32 + A_3 B_33.
    
    This is getting complex. Let's try a simpler heuristic:
    Use the schoolbook decomposition.
    Then, check if any of the rank-1 tensors are linearly dependent with others in a way that allows reduction.
    Actually, the standard way to find lower rank is to solve a system of equations.
    
    Given the time constraint and the difficulty of finding a guaranteed lower rank for 2x5x6 from scratch,
    I will implement a search that tries to merge terms using Strassen identities on 2x2 sub-blocks.
    
    Decomposition:
    1. Split the 5x6 matrix B into two parts: B_top (2x6) and B_bottom (3x6).
    2. Split the 2x5 matrix A into two parts: A_left (2x2) and A_right (2x3).
    3. C = A_left * B_top + A_right * B_bottom.
    
    Term 1: A_left (2x2) * B_top (2x6).
    This is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.
    We can view this as two separate 2x2x2 multiplications?
    B_top = [B_t1, B_t2, B_t3] where each is 2x2.
    So C1 = A_left * B_t1 + A_left * B_t2 + A_left * B_t3.
    Each A_left * B_ti is 2x2x2.
    Using Strassen for 2x2x2, each takes rank 7.
    Total for this part: 7 * 3 = 21.
    
    Term 2: A_right (2x3) * B_bottom (3x6).
    B_bottom = [B_b1, B_b2, B_b3] where each is 3x2.
    C2 = A_right * B_b1 + A_right * B_b2 + A_right * B_b3.
    Each A_right * B_bi is 2x3x2.
    What is the rank of 2x3x2?
    Schoolbook: 2*3*2 = 12.
    Can we do better?
    2x3x2 is isomorphic to 2x2x3 (by transposing B).
    Rank of 2x2x3?
    Schoolbook: 12.
    Lower bound: max(4, 6, 6) = 6.
    Known rank of 2x2x3 is 10? Or 11?
    Actually, for 2x2xn, the rank is known to be 2n for n>=2? No.
    For 2x2x2, rank is 7.
    For 2x2x3, one can use 2x2x2 (7) + 2x2x1 (2) = 9?
    Let's check:
    A (2x2) * B (2x3).
    B = [b1, b2, b3] where bi are 2x1.
    C = A*b1 + A*b2 + A*b3.
    Each A*bi is 2x2x1. Rank 4.
    Total 12.
    But we can combine.
    Let's use the known rank of 2x2xn.
    It is known that R(2,2,n) <= 2n + 1 for n>=2?
    For n=2, R=7 (2*2+3? no).
    For n=3, R(2,2,3) = 10?
    Let's assume we can do 2x3x2 with rank 10.
    Then total for this part: 10 * 3 = 30.
    Total rank: 21 + 30 = 51.
    This is better than 60.
    
    However, implementing the specific Strassen-like decomposition for 2x2x3 requires the actual coefficients.
    If I don't have the exact coefficients, I might produce an invalid decomposition.
    
    Let's stick to a safe, known decomposition:
    The schoolbook decomposition is rank 60.
    If I can find *any* valid decomposition with rank < 60, I win.
    
    Let's try to implement the Strassen 2x2x2 decomposition for the 2x2x2 blocks.
    And for the 2x3x2 blocks, use schoolbook (rank 12) unless I can find a simple reduction.
    
    Wait, 2x3x2 schoolbook is 12.
    If I use schoolbook for the 2x3x2 parts, the total rank is:
    3 * 7 (for 2x2x2) + 3 * 12 (for 2x3x2) = 21 + 36 = 57.
    This is < 60.
    
    So the plan:
    1. Decompose A (2x5) into A1 (2x2) and A2 (2x3).
    2. Decompose B (5x6) into B1 (2x6) and B2 (3x6).
    3. C = A1*B1 + A2*B2.
    4. Decompose B1 into three 2x2 blocks: B1_1, B1_2, B1_3.
    5. Decompose B2 into three 3x2 blocks: B2_1, B2_2, B2_3.
    6. C = sum_{k=1..3} (A1*B1_k) + sum_{k=1..3} (A2*B2_k).
    7. For each A1*B1_k (2x2x2), use Strassen (rank 7).
    8. For each A2*B2_k (2x3x2), use Schoolbook (rank 12).
    9. Total Rank = 3*7 + 3*12 = 21 + 36 = 57.
    
    This is a valid, constructible improvement.
    
    Let's implement the Strassen decomposition for 2x2x2.
    Standard Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need to map this to the tensor decomposition format U, V, W.
    The tensor T is defined by T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]B[j,k].
    In the decomposition T = sum_r U_r \otimes V_r \otimes W_r,
    U_r is a vector of size n*m (here 2*2=4 for the sub-block),
    V_r is a vector of size m*p (here 2*2=4 for the sub-block),
    W_r is a vector of size p*n (here 2*2=4 for the sub-block).
    
    For the sub-block 2x2x2:
    Indices for U: (i, j) with i,j in {0,1}.
    Indices for V: (j, k) with j,k in {0,1}.
    Indices for W: (k, i) with k,i in {0,1}. (Note: W[k*n+i] in the global problem, but locally it's W[k*2+i]).
    
    Let's define the local U, V, W for Strassen.
    Local U size 4, Local V size 4, Local W size 4.
    
    For each of the 7 terms:
    Term 1: (A00+A11) * (B00+B11)
    U: A00=1, A11=1 -> [1, 0, 0, 1] (indices 00, 01, 10, 11)
    V: B00=1, B11=1 -> [1, 0, 0, 1] (indices 00, 01, 10, 11)
    W: Contributes to C00 (+1), C11 (+1).
    C00 is W[0*2+0] = W[0]. C11 is W[1*2+1] = W[3].
    So W: [1, 0, 0, 1].
    
    Term 2: (A10+A11) * B00
    U: A10=1, A11=1 -> [0, 0, 1, 1]
    V: B00=1 -> [1, 0, 0, 0]
    W: Contributes to C10 (+1), C11 (-1).
    C10 is W[0*2+1] = W[2]. C11 is W[3].
    So W: [0, 0, 1, -1].
    
    Term 3: A00 * (B01-B11)
    U: A00=1 -> [1, 0, 0, 0]
    V: B01=1, B11=-1 -> [0, 1, 0, -1]
    W: Contributes to C00 (-1), C01 (+1).
    C00 is W[0]. C01 is W[1].
    So W: [-1, 1, 0, 0].
    
    Term 4: A11 * (B10-B00)
    U: A11=1 -> [0, 0, 0, 1]
    V: B10=1, B00=-1 -> [-1, 0, 1, 0]
    W: Contributes to C00 (+1), C10 (+1).
    C00 is W[0]. C10 is W[2].
    So W: [1, 0, 1, 0].
    
    Term 5: (A00+A01) * B11
    U: A00=1, A01=1 -> [1, 1, 0, 0]
    V: B11=1 -> [0, 0, 0, 1]
    W: Contributes to C00 (-1), C01 (+1).
    C00 is W[0]. C01 is W[1].
    So W: [-1, 1, 0, 0].
    
    Term 6: (A10-A00) * (B00+B01)
    U: A10=1, A00=-1 -> [-1, 0, 1, 0]
    V: B00=1, B01=1 -> [1, 1, 0, 0]
    W: Contributes to C11 (+1).
    C11 is W[3].
    So W: [0, 0, 0, 1].
    
    Term 7: (A01-A11) * (B10+B11)
    U: A01=1, A11=-1 -> [0, 1, 0, -1]
    V: B10=1, B11=1 -> [0, 0, 1, 1]
    W: Contributes to C00 (+1).
    C00 is W[0].
    So W: [1, 0, 0, 0].
    
    Now, we need to embed these into the global U, V, W matrices.
    
    Global Dimensions: n=2, m=5, p=6.
    U size: 2*5 = 10.
    V size: 5*6 = 30.
    W size: 6*2 = 12.
    
    Mapping:
    A is 2x5. Indices (i, j) with i in {0,1}, j in {0..4}.
    U index: i*5 + j.
    
    B is 5x6. Indices (j, k) with j in {0..4}, k in {0..5}.
    V index: j*6 + k.
    
    C is 2x6. Indices (i, k) with i in {0,1}, k in {0..5}.
    W index: k*2 + i. (Task says W[k*n+i]).
    
    Sub-block 1: A1 (2x2) uses A[0:2, 0:2].
    A1[i,j] corresponds to A[i, j].
    U indices for A1: i*5 + j for i in {0,1}, j in {0,1}.
    i=0, j=0 -> 0
    i=0, j=1 -> 1
    i=1, j=0 -> 5
    i=1, j=1 -> 6
    
    Sub-block 1: B1_k (2x2) uses B[0:2, 2k:2k+2].
    B1_k[j,k'] corresponds to B[j, 2k+k'].
    V indices for B1_k: j*6 + (2k+k') for j in {0,1}, k' in {0,1}.
    For k=0:
    j=0, k'=0 -> 0
    j=0, k'=1 -> 1
    j=1, k'=0 -> 6
    j=1, k'=1 -> 7
    
    Sub-block 1: C1_k (2x2) uses C[0:2, 2k:2k+2].
    C1_k[i,k'] corresponds to C[i, 2k+k'].
    W indices for C1_k: (2k+k')*2 + i for i in {0,1}, k' in {0,1}.
    For k=0:
    i=0, k'=0 -> 0*2+0 = 0
    i=1, k'=0 -> 0*2+1 = 1
    i=0, k'=1 -> 1*2+0 = 2
    i=1, k'=1 -> 1*2+1 = 3
    
    Wait, the local W indices in Strassen were:
    W[0] -> C00
    W[1] -> C01
    W[2] -> C10
    W[3] -> C11
    
    Mapping local (i, k') to global:
    Local C00 -> i=0, k'=0 -> Global C[0, 2k+0] -> W index (2k+0)*2 + 0.
    Local C01 -> i=0, k'=1 -> Global C[0, 2k+1] -> W index (2k+1)*2 + 0.
    Local C10 -> i=1, k'=0 -> Global C[1, 2k+0] -> W index (2k+0)*2 + 1.
    Local C11 -> i=1, k'=1 -> Global C[1, 2k+1] -> W index (2k+1)*2 + 1.
    
    So for each k in 0..2:
    Local W[0] maps to Global W[(2k)*2 + 0]
    Local W[1] maps to Global W[(2k+1)*2 + 0]
    Local W[2] maps to Global W[(2k)*2 + 1]
    Local W[3] maps to Global W[(2k+1)*2 + 1]
    
    Similarly for U and V.
    
    Sub-block 2: A2 (2x3) uses A[0:2, 2:5].
    A2[i,j'] corresponds to A[i, 2+j'].
    U indices for A2: i*5 + (2+j') for i in {0,1}, j' in {0,1,2}.
    i=0, j'=0 -> 2
    i=0, j'=1 -> 3
    i=0, j'=2 -> 4
    i=1, j'=0 -> 7
    i=1, j'=1 -> 8
    i=1, j'=2 -> 9
    
    Sub-block 2: B2_k (3x2) uses B[2:5, 2k:2k+2].
    B2_k[j', k'] corresponds to B[2+j', 2k+k'].
    V indices for B2_k: (2+j')*6 + (2k+k') for j' in {0,1,2}, k' in {0,1}.
    
    Sub-block 2: C2_k (2x2) uses C[0:2, 2k:2k+2].
    Same W mapping as before.
    
    For the 2x3x2 part, we use schoolbook.
    A2 (2x3) * B2_k (3x2).
    Terms: for each i in {0,1}, j' in {0,1,2}, k' in {0,1}.
    U: A2[i, j'] = 1.
    V: B2_k[j', k'] = 1.
    W: C2_k[i, k'] = 1.
    
    Let's code this.
    
    """
    n, m, p = 2, 5, 6
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        rank += 1
    
    # Strassen 2x2x2 local vectors
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (Local C00, C01, C10, C11)
    
    strassen_terms = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [-1, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 uses A[0:2, 0:2]
    # B1 uses B[0:2, 0:6]
    # Split B1 into 3 blocks of 2x2: B1_0, B1_1, B1_2
    # B1_k uses B[0:2, 2k:2k+2]
    
    for k in range(3):
        # For each Strassen term
        for u_loc, v_loc, w_loc in strassen_terms:
            u_glob = [0] * (n * m)
            v_glob = [0] * (m * p)
            w_glob = [0] * (p * n)
            
            # Fill U
            # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global U indices for A1:
            # (0,0) -> 0*5+0 = 0
            # (0,1) -> 0*5+1 = 1
            # (1,0) -> 1*5+0 = 5
            # (1,1) -> 1*5+1 = 6
            u_map = [0, 1, 5, 6]
            for l_idx, val in enumerate(u_loc):
                if val != 0:
                    u_glob[u_map[l_idx]] = val
            
            # Fill V
            # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global V indices for B1_k:
            # (0,0) -> 0*6 + (2k+0) = 2k
            # (0,1) -> 0*6 + (2k+1) = 2k+1
            # (1,0) -> 1*6 + (2k+0) = 6+2k
            # (1,1) -> 1*6 + (2k+1) = 7+2k
            v_map = [2*k, 2*k+1, 6+2*k, 7+2*k]
            for l_idx, val in enumerate(v_loc):
                if val != 0:
                    v_glob[v_map[l_idx]] = val
            
            # Fill W
            # Local W indices: 0:C00, 1:C01, 2:C10, 3:C11
            # Global W indices for C1_k:
            # C00 -> i=0, k'=0 -> C[0, 2k+0] -> W[(2k+0)*2 + 0] = 4k
            # C01 -> i=0, k'=1 -> C[0, 2k+1] -> W[(2k+1)*2 + 0] = 4k+2
            # C10 -> i=1, k'=0 -> C[1, 2k+0] -> W[(2k+0)*2 + 1] = 4k+1
            # C11 -> i=1, k'=1 -> C[1, 2k+1] -> W[(2k+1)*2 + 1] = 4k+3
            w_map = [4*k, 4*k+2, 4*k+1, 4*k+3]
            for l_idx, val in enumerate(w_loc):
                if val != 0:
                    w_glob[w_map[l_idx]] = val
            
            add_term(u_glob, v_glob, w_glob)
    
    # Part 2: A2 (2x3) * B2 (3x6)
    # A2 uses A[0:2, 2:5]
    # B2 uses B[2:5, 0:6]
    # Split B2 into 3 blocks of 3x2: B2_0, B2_1, B2_2
    # B2_k uses B[2:5, 2k:2k+2]
    
    for k in range(3):
        # Schoolbook for 2x3x2
        # i in 0..1, j' in 0..2, k' in 0..1
        for i in range(2):
            for jp in range(3):
                for kp in range(2):
                    u_glob = [0] * (n * m)
                    v_glob = [0] * (m * p)
                    w_glob = [0] * (p * n)
                    
                    # U: A2[i, jp] -> A[i, 2+jp] -> U[i*5 + 2+jp]
                    u_idx = i * 5 + (2 + jp)
                    u_glob[u_idx] = 1
                    
                    # V: B2_k[jp, kp] -> B[2+jp, 2k+kp] -> V[(2+jp)*6 + 2k+kp]
                    v_idx = (2 + jp) * 6 + (2 * k + kp)
                    v_glob[v_idx] = 1
                    
                    # W: C2_k[i, kp] -> C[i, 2k+kp] -> W[(2k+kp)*2 + i]
                    w_idx = (2 * k + kp) * 2 + i
                    w_glob[w_idx] = 1
                    
                    add_term(u_glob, v_glob, w_glob)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
