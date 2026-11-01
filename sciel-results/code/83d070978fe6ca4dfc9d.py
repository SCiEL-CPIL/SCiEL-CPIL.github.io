import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return U, V, W

def _tensor_to_vector(n, m, p):
    """Flatten the multiplication tensor into a vector."""
    vec = np.zeros(n * m * m * p * p * n)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                vec[idx] = 1.0
                idx += 1
    return vec

def _verify_decomposition(U, V, W, n, m, p):
    """Check if the decomposition exactly matches the tensor."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_comp, T_target, atol=1e-9)

def _block_decomposition():
    """
    Construct a rank-50 decomposition using block structure.
    We partition the 2x4x7 multiplication into blocks:
    A (2x4) is split into A1 (2x2) and A2 (2x2) along columns.
    B (4x7) is split into B1 (2x7) and B2 (2x7) along rows.
    C = A1 B1 + A2 B2.
    
    For the first term A1 B1 (2x2 times 2x7):
    We can use Strassen's algorithm for the 2x2 part of the product.
    However, the second dimension is 7.
    Let's refine the block strategy.
    
    A 2x4 matrix A can be viewed as [A_left, A_right] where A_left, A_right are 2x2.
    A 4x7 matrix B can be viewed as [B_top; B_bottom] where B_top, B_bottom are 2x7.
    C = A_left B_top + A_right B_bottom.
    
    Term 1: C1 = A_left * B_top (2x2 * 2x7 -> 2x7)
    Term 2: C2 = A_right * B_bottom (2x2 * 2x7 -> 2x7)
    
    For a 2x2 * 2xk multiplication, the rank is known to be 3 (if k=2, rank 7? No).
    Wait, 2x2 * 2x2 is rank 7.
    2x2 * 2x1 is rank 3 (standard result: rank of 2x2x1 tensor is 3).
    
    So, if we split the 7 columns of B into 7 columns of width 1?
    B_top is 2x7. We can write B_top = [b1, b2, ..., b7] where bi is 2x1.
    Then A_left * B_top = A_left * b1 + ... + A_left * b7.
    Each term A_left * bi is a 2x2 * 2x1 multiplication.
    The rank of 2x2x1 matrix multiplication is 3.
    So A_left * B_top can be done with 7 * 3 = 21 rank terms.
    Similarly, A_right * B_bottom can be done with 7 * 3 = 21 rank terms.
    Total rank = 21 + 21 = 42.
    
    Is rank 42 possible?
    Let's verify the rank of 2x2x1 multiplication.
    Tensor T_{2x2x1} has indices i in {0,1}, j in {0,1}, k in {0}.
    T_{ij0} = A_{ij} B_{j0} ... wait.
    Matrix multiplication C = A B.
    If A is 2x2 and B is 2x1, C is 2x1.
    C_i = sum_j A_{ij} B_j.
    Tensor T_{i, j, k} where i is row of C (2), j is "input" index?
    The standard tensor is T_{i, j, k} = delta_{i, something}?
    No, the tensor for matrix multiplication C = A B is defined by C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor is T_{i, j, k} which is 1 if the indices correspond to the term A_{i,j} B_{j,k}.
    Specifically, T_{i, j, k} = 1 if the j-th element of the j-th column of B is multiplied by the j-th element of the i-th row of A to contribute to C_{i,k}.
    Actually, T_{i, j, k} is 1 if the pair (i, j) in A and (j, k) in B contribute to (i, k) in C.
    For 2x2x1:
    A is 2x2, B is 2x1, C is 2x1.
    C_0 = A_00 B_00 + A_01 B_10
    C_1 = A_10 B_00 + A_11 B_10
    
    The tensor T has dimensions (2, 2, 2) for (A_idx, B_idx, C_idx)?
    No, the factors are U (n*m), V (m*p), W (p*n).
    Here n=2, m=2, p=1.
    U is 2x2=4. V is 2x1=2. W is 1x2=2.
    T_{a, b, c} = 1 if a=(i,j), b=(j,k), c=(k,i).
    a=0 (0,0), b=0 (0,0), c=0 (0,0) -> 1
    a=1 (0,1), b=1 (1,0), c=0 (0,0) -> 1
    a=2 (1,0), b=0 (0,0), c=1 (0,1) -> 1
    a=3 (1,1), b=1 (1,0), c=1 (0,1) -> 1
    
    Can we decompose this with rank 3?
    Yes, it is a known result that rank(2x2x1) = 3.
    The decomposition is:
    r=0: U[0]=1, V[0]=1, W[0]=1
    r=1: U[1]=1, V[1]=1, W[0]=1
    r=2: U[2]=1, U[3]=1, V[0]=1, V[1]=-1, W[1]=1 ... something like that.
    
    Let's find the rank 3 decomposition for 2x2x1.
    We need U (4x3), V (2x3), W (2x3).
    T = sum_r u_r v_r w_r^T.
    T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r].
    
    Let's try to solve for rank 3.
    Actually, let's just use the known Strassen-like or explicit decomposition.
    For 2x2x1, the minimal rank is 3.
    One such decomposition:
    r=0: U=[1,0,0,0], V=[1,0], W=[1,0]  -> T[0,0,0]=1
    r=1: U=[0,1,0,0], V=[0,1], W=[1,0]  -> T[1,1,0]=1
    r=2: U=[0,0,1,1], V=[1,-1], W=[0,1] -> T[2,0,1]=1, T[3,1,1]=1.
    Check:
    r=0: T[0,0,0] += 1*1*1 = 1. Correct.
    r=1: T[1,1,0] += 1*1*1 = 1. Correct.
    r=2: T[2,0,1] += 1*1*1 = 1. T[3,1,1] += 1*(-1)*1 = -1. 
    Wait, T[3,1,1] should be 1.
    Let's adjust signs.
    r=2: U=[0,0,1,1], V=[1,1], W=[0,1].
    T[2,0,1] += 1*1*1 = 1.
    T[3,1,1] += 1*1*1 = 1.
    Are there any other non-zero entries?
    r=2 contributes to T[a,b,c] if U[a,2]!=0, V[b,2]!=0, W[c,2]!=0.
    U[2,2]=1, U[3,2]=1. V[0,2]=1, V[1,2]=1. W[1,2]=1.
    So T[2,0,1] += 1. T[2,1,1] += 1. T[3,0,1] += 1. T[3,1,1] += 1.
    But T[2,1,1] and T[3,0,1] should be 0.
    So this simple sum doesn't work directly without cancellation.
    
    Let's use a linear solver to find the rank 3 decomposition for 2x2x1.
    Or better, use the known formula.
    The rank of 2x2x1 is indeed 3.
    
    Let's construct the full 2x4x7 decomposition using 14 blocks of 2x2x1.
    Total rank = 14 * 3 = 42.
    
    We need to map the indices correctly.
    
    Block (l, k) where l in {0,1} (left/right of A) and k in {0..6} (column of B).
    A_block is A[:, l*2 : l*2+2]. (2x2)
    B_block is B[l*2 : l*2+2, k]. (2x1)
    C_block is C[:, k]. (2x1)
    
    For each block, we have a 2x2x1 multiplication.
    We use the rank 3 decomposition for each.
    
    Let's find the rank 3 decomposition for 2x2x1 programmatically or hardcode it.
    Hardcoding is safer if we verify it.
    
    Let's solve for U, V, W for 2x2x1 with rank 3.
    n=2, m=2, p=1.
    U: 4x3, V: 2x3, W: 2x3.
    T: 4x2x2.
    T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1.
    
    Let's try to find it.
    We can use the fact that the tensor is symmetric in some sense? No.
    
    Actually, let's just use the trivial rank 4 for 2x2x1? No, we want minimal.
    If we use rank 4 for each block, total rank = 14 * 4 = 56 (trivial).
    If we use rank 3, total rank = 42.
    
    Let's verify if rank 3 is possible for 2x2x1.
    Yes, it is a standard result.
    
    Let's construct the rank 3 decomposition for 2x2x1.
    We can use a simple script in our head or derive it.
    T_{a,b,c} = sum_{r=0}^2 U_{a,r} V_{b,r} W_{c,r}.
    
    Let's guess a structure.
    r=0: U[0]=1, V[0]=1, W[0]=1. (Contributes to T[0,0,0])
    r=1: U[1]=1, V[1]=1, W[0]=1. (Contributes to T[1,1,0])
    r=2: Must contribute to T[2,0,1] and T[3,1,1].
    And cancel any unwanted contributions.
    
    Let's use a solver in the code to find the rank 3 decomposition for 2x2x1 once, then reuse it.
    Since the problem is small, we can just hardcode the solution if we find it, or solve it at runtime (very fast).
    
    Let's solve it at runtime.
    """
    n, m, p = 2, 2, 1
    R = 3
    # Initialize with random values? No, let's use a structured approach or just solve the linear system for fixed V, W?
    # Actually, for such a small tensor, we can just use the known decomposition.
    # Or we can use the fact that 2x2x1 is equivalent to 2x1x2 (transpose).
    
    # Let's try to find a solution.
    # T = [
    #   [ [1, 0],
    #     [0, 0] ],
    #   [ [0, 0],
    #     [1, 0] ],
    #   [ [0, 1],
    #     [0, 0] ],
    #   [ [0, 0],
    #     [1, 0] ]
    # ]
    # Wait, indices:
    # a: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # b: 0(0,0), 1(1,0)
    # c: 0(0,0), 1(0,1)
    
    # T[0,0,0] = 1
    # T[1,1,0] = 1
    # T[2,0,1] = 1
    # T[3,1,1] = 1
    
    # Let's try:
    # r=0: U=[1,0,0,0], V=[1,0], W=[1,0] -> T[0,0,0]+=1
    # r=1: U=[0,1,0,0], V=[0,1], W=[1,0] -> T[1,1,0]+=1
    # r=2: U=[0,0,1,1], V=[1,1], W=[0,1] -> T[2,0,1]+=1, T[2,1,1]+=1, T[3,0,1]+=1, T[3,1,1]+=1
    # We have extra T[2,1,1]=1 and T[3,0,1]=1.
    # We need to cancel them.
    # We can add more terms? No, rank is 3.
    # So we must adjust the values.
    
    # Let's use variables.
    # This is a non-convex problem.
    # However, we can use the known result that rank is 3.
    # Let's just use the trivial rank 4 for 2x2x1 to be safe?
    # Rank 4 for 2x2x1 gives total rank 14 * 4 = 56.
    # Rank 3 for 2x2x1 gives total rank 14 * 3 = 42.
    
    # Is there a simpler block decomposition?
    # What if we use Strassen for 2x2x2?
    # Split 7 into 2+2+2+1?
    # A (2x4) = [A1, A2] (2x2 each)
    # B (4x7) = [B1; B2] (2x7 each)
    # C = A1 B1 + A2 B2.
    
    # B1 is 2x7. Split into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1).
    # A1 B1 = A1 B1a + A1 B1b + A1 B1c + A1 B1d.
    # A1 B1a is 2x2 * 2x2. Rank 7 (Strassen).
    # A1 B1b is 2x2 * 2x2. Rank 7.
    # A1 B1c is 2x2 * 2x2. Rank 7.
    # A1 B1d is 2x2 * 2x1. Rank 3.
    # Total for A1 B1: 7+7+7+3 = 24.
    # Total for A2 B2: 24.
    # Total rank: 48.
    
    # This is better than 56, but worse than 42.
    # Can we do better with 2x2x1?
    # If we can confirm rank 3 for 2x2x1, then 42 is achievable.
    
    # Let's try to find the rank 3 decomposition for 2x2x1.
    # I will write a small solver in the code to find it if possible, or use a known one.
    # Actually, I can just use the 2x2x2 Strassen and pad with zeros for the 1 column?
    # No, that would be rank 7.
    
    # Let's assume rank 3 is possible and try to find it.
    # If I can't find it easily, I will fall back to rank 48 (Strassen blocks).
    # Or rank 50 (Strassen 2x2x2 + Schoolbook 2x2x1).
    # 2x2x2 Strassen is rank 7.
    # 2x2x1 Schoolbook is rank 4? No, 2x2x1 is rank 3.
    # If I use rank 4 for 2x2x1, total is 48.
    # If I use rank 3 for 2x2x1, total is 42.
    
    # Let's implement the 2x2x2 Strassen and 2x2x1 Rank-3 (if found) or Rank-4 (trivial).
    
    # For now, let's implement the 2x2x2 Strassen + 2x2x1 Trivial (Rank 4) strategy.
    # Total rank = 3 * 7 + 1 * 4 = 25 per A-block.
    # Two A-blocks -> 50.
    # This is a safe, known-correct decomposition with rank 50.
    
    # Let's implement this.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We need to map this to U, V, W.
    # U is for A (2x2), V is for B (2x2), W is for C (2x2).
    # Indices:
    # A: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # B: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # C: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    
    def add_strassen_block(base_u, base_v, base_w, u_offset, v_offset, w_offset):
        # u_offset, v_offset, w_offset are the starting indices in the global U, V, W arrays for this block.
        # But we are building lists, so we just append.
        
        # m1
        U1 = np.zeros(4); U1[0]=1; U1[3]=1
        V1 = np.zeros(4); V1[0]=1; V1[3]=1
        W1 = np.zeros(4); W1[0]=1; W1[3]=1
        # Wait, m1 = (a+d)(e+h) = ae + ah + de + dh
        # T[0,0,0] (a*e) -> U[0], V[0], W[0]
        # T[0,3,1] (a*h) -> U[0], V[3], W[1]
        # T[3,0,2] (d*e) -> U[3], V[0], W[2]
        # T[3,3,3] (d*h) -> U[3], V[3], W[3]
        # This is not a single rank-1 term.
        # Strassen's algorithm computes products of linear combinations.
        # The tensor decomposition of Strassen is rank 7.
        # Each term is a rank-1 tensor in the space of (linear combo of A, linear combo of B, linear combo of C).
        
        # Term 1: m1 = (a+d)(e+h)
        # U1: [1, 0, 0, 1] (a+d)
        # V1: [1, 0, 0, 1] (e+h)
        # W1: [1, 1, 1, 1] (contributes to all 4 C elements? No.)
        # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
        # So m1 contributes to C[0,0] and C[1,1].
        # W1: [1, 0, 0, 1]
        
        U_list.append(np.array([1, 0, 0, 1]))
        V_list.append(np.array([1, 0, 0, 1]))
        W_list.append(np.array([1, 0, 0, 1]))
        
        # Term 2: m2 = (c+d)e
        # U2: [0, 0, 1, 1]
        # V2: [1, 0, 0, 0]
        # W2: [0, 0, 1, 0] (contributes to C[1,0])
        
        U_list.append(np.array([0, 0, 1, 1]))
        V_list.append(np.array([1, 0, 0, 0]))
        W_list.append(np.array([0, 0, 1, 0]))
        
        # Term 3: m3 = a(f-h)
        # U3: [1, 0, 0, 0]
        # V3: [0, 1, 0, -1]
        # W3: [0, 1, 0, 0] (contributes to C[0,1])
        
        U_list.append(np.array([1, 0, 0, 0]))
        V_list.append(np.array([0, 1, 0, -1]))
        W_list.append(np.array([0, 1, 0, 0]))
        
        # Term 4: m4 = d(g-f)
        # U4: [0, 0, 0, 1]
        # V4: [0, -1, 1, 0]
        # W4: [1, 0, 1, 0] (contributes to C[0,0] and C[1,0])
        
        U_list.append(np.array([0, 0, 0, 1]))
        V_list.append(np.array([0, -1, 1, 0]))
        W_list.append(np.array([1, 0, 1, 0]))
        
        # Term 5: m5 = (a+b)h
        # U5: [1, 1, 0, 0]
        # V5: [0, 0, 0, 1]
        # W5: [0, 1, -1, 0] (contributes to C[0,1] and -C[1,0])
        
        U_list.append(np.array([1, 1, 0, 0]))
        V_list.append(np.array([0, 0, 0, 1]))
        W_list.append(np.array([0, 1, -1, 0]))
        
        # Term 6: m6 = (c-a)(e+f)
        # U6: [-1, 0, 1, 0]
        # V6: [1, 1, 0, 0]
        # W6: [0, 0, 0, 1] (contributes to C[1,1])
        
        U_list.append(np.array([-1, 0, 1, 0]))
        V_list.append(np.array([1, 1, 0, 0]))
        W_list.append(np.array([0, 0, 0, 1]))
        
        # Term 7: m7 = (b-d)(g+h)
        # U7: [0, 1, 0, -1]
        # V7: [0, 0, 1, 1]
        # W7: [1, 0, 0, 0] (contributes to C[0,0])
        
        U_list.append(np.array([0, 1, 0, -1]))
        V_list.append(np.array([0, 0, 1, 1]))
        W_list.append(np.array([1, 0, 0, 0]))
        
    # Schoolbook 2x2x1 decomposition (Rank 4)
    # A (2x2) * B (2x1) -> C (2x1)
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    def add_schoolbook_221_block():
        # Term 1: A[0,0]B[0,0]
        U_list.append(np.array([1, 0, 0, 0]))
        V_list.append(np.array([1, 0]))
        W_list.append(np.array([1, 0]))
        
        # Term 2: A[0,1]B[1,0]
        U_list.append(np.array([0, 1, 0, 0]))
        V_list.append(np.array([0, 1]))
        W_list.append(np.array([1, 0]))
        
        # Term 3: A[1,0]B[0,0]
        U_list.append(np.array([0, 0, 1, 0]))
        V_list.append(np.array([1, 0]))
        W_list.append(np.array([0, 1]))
        
        # Term 4: A[1,1]B[1,0]
        U_list.append(np.array([0, 0, 0, 1]))
        V_list.append(np.array([0, 1]))
        W_list.append(np.array([0, 1]))
        
    # Now assemble the full 2x4x7 decomposition.
    # A (2x4) is split into A_left (cols 0-1) and A_right (cols 2-3).
    # B (4x7) is split into B_top (rows 0-1) and B_bottom (rows 2-3).
    # B_top is 2x7. Split into 3 blocks of 2x2 and 1 block of 2x1.
    # B_bottom is 2x7. Split into 3 blocks of 2x2 and 1 block of 2x1.
    
    # We need to map the local indices to global indices.
    # Global U: 2x4 = 8.
    # Global V: 4x7 = 28.
    # Global W: 7x2 = 14.
    
    # Local U (2x2) indices 0,1,2,3 map to global U indices.
    # For A_left: local 0->global 0, local 1->global 1, local 2->global 2, local 3->global 3.
    # For A_right: local 0->global 4, local 1->global 5, local 2->global 6, local 3->global 7.
    
    # Local V (2x2) indices 0,1,2,3 map to global V indices.
    # B_top, block 0 (cols 0-1): local 0->global 0, local 1->global 1, local 2->global 2, local 3->global 3.
    # B_top, block 1 (cols 2-3): local 0->global 4, local 1->global 5, local 2->global 6, local 3->global 7.
    # B_top, block 2 (cols 4-5): local 0->global 8, local 1->global 9, local 2->global 10, local 3->global 11.
    # B_top, block 3 (col 6): local 0->global 12, local 1->global 13.
    
    # B_bottom, block 0 (cols 0-1): local 0->global 14, local 1->global 15, local 2->global 16, local 3->global 17.
    # B_bottom, block 1 (cols 2-3): local 0->global 18, local 1->global 19, local 2->global 20, local 3->global 21.
    # B_bottom, block 2 (cols 4-5): local 0->global 22, local 1->global 23, local 2->global 24, local 3->global 25.
    # B_bottom, block 3 (col 6): local 0->global 26, local 1->global 27.
    
    # Local W (2x2) indices 0,1,2,3 map to global W indices.
    # Block 0 (col 0): local 0->global 0, local 1->global 1, local 2->global 2, local 3->global 3.
    # Block 1 (col 1): local 0->global 4, local 1->global 5, local 2->global 6, local 3->global 7.
    # Block 2 (col 2): local 0->global 8, local 1->global 9, local 2->global 10, local 3->global 11.
    # Block 3 (col 3): local 0->global 12, local 1->global 13, local 2->global 14, local 3->global 15.
    # Block 4 (col 4): local 0->global 16, local 1->global 17, local 2->global 18, local 3->global 19.
    # Block 5 (col 5): local 0->global 20, local 1->global 21, local 2->global 22, local 3->global 23.
    # Block 6 (col 6): local 0->global 24, local 1->global 25. (For 2x1, W is 2x1)
    
    # Wait, W is p*n = 7*2 = 14.
    # W indices: k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # ...
    # k=6, i=0 -> 12
    # k=6, i=1 -> 13
    
    # My mapping above for W was wrong.
    # Let's re-map W.
    # W global index = k * 2 + i.
    # Block 0 (k=0): local 0(i=0)->0, local 1(i=1)->1.
    # Block 1 (k=1): local 0(i=0)->2, local 1(i=1)->3.
    # Block 2 (k=2): local 0(i=0)->4, local 1(i=1)->5.
    # Block 3 (k=3): local 0(i=0)->6, local 1(i=1)->7.
    # Block 4 (k=4): local 0(i=0)->8, local 1(i=1)->9.
    # Block 5 (k=5): local 0(i=0)->10, local 1(i=1)->11.
    # Block 6 (k=6): local 0(i=0)->12, local 1(i=1)->13.
    
    # For 2x2x2 Strassen, W is 2x2.
    # Local W indices: 0(i=0,k=0), 1(i=0,k=1), 2(i=1,k=0), 3(i=1,k=1)?
    # No, standard matrix C is 2x2.
    # C[0,0], C[0,1], C[1,0], C[1,1].
    # Global W index for C[i,k] is k*2 + i.
    # So C[0,0] -> 0. C[0,1] -> 2. C[1,0] -> 1. C[1,1] -> 3.
    # Local W indices in Strassen:
    # 0: C[0,0]
    # 1: C[0,1]
    # 2: C[1,0]
    # 3: C[1,1]
    
    # So for Block 0 (k=0,1):
    # Local 0 (C[0,0]) -> Global 0
    # Local 1 (C[0,1]) -> Global 2
    # Local 2 (C[1,0]) -> Global 1
    # Local 3 (C[1,1]) -> Global 3
    
    # For Block 1 (k=2,3):
    # Local 0 (C[0,0]) -> Global 4
    # Local 1 (C[0,1]) -> Global 6
    # Local 2 (C[1,0]) -> Global 5
    # Local 3 (C[1,1]) -> Global 7
    
    # For Block 2 (k=4,5):
    # Local 0 (C[0,0]) -> Global 8
    # Local 1 (C[0,1]) -> Global 10
    # Local 2 (C[1,0]) -> Global 9
    # Local 3 (C[1,1]) -> Global 11
    
    # For Block 3 (k=6, 2x1):
    # Local 0 (C[0,0]) -> Global 12
    # Local 1 (C[1,0]) -> Global 13
    
    # Now we can implement the mapping.
    
    U_global = []
    V_global = []
    W_global = []
    
    def add_block(u_local, v_local, w_local, u_map, v_map, w_map):
        U_new = np.zeros(8)
        V_new = np.zeros(28)
        W_new = np.zeros(14)
        
        for i, val in enumerate(u_local):
            if val != 0:
                U_new[u_map[i]] = val
        for i, val in enumerate(v_local):
            if val != 0:
                V_new[v_map[i]] = val
        for i, val in enumerate(w_local):
            if val != 0:
                W_new[w_map[i]] = val
                
        U_global.append(U_new)
        V_global.append(V_new)
        W_global.append(W_new)
        
    # A_left maps: 0->0, 1->1, 2->2, 3->3
    # A_right maps: 0->4, 1->5, 2->6, 3->7
    
    u_map_left = [0, 1, 2, 3]
    u_map_right = [4, 5, 6, 7]
    
    # B_top block 0 (k=0,1) maps: 0->0, 1->1, 2->2, 3->3
    # B_top block 1 (k=2,3) maps: 0->4, 1->5, 2->6, 3->7
    # B_top block 2 (k=4,5) maps: 0->8, 1->9, 2->10, 3->11
    # B_top block 3 (k=6) maps: 0->12, 1->13
    
    v_map_top0 = [0, 1, 2, 3]
    v_map_top1 = [4, 5, 6, 7]
    v_map_top2 = [8, 9, 10, 11]
    v_map_top3 = [12, 13]
    
    # B_bottom block 0 (k=0,1) maps: 0->14, 1->15, 2->16, 3->17
    # B_bottom block 1 (k=2,3) maps: 0->18, 1->19, 2->20, 3->21
    # B_bottom block 2 (k=4,5) maps: 0->22, 1->23, 2->24, 3->25
    # B_bottom block 3 (k=6) maps: 0->26, 1->27
    
    v_map_bot0 = [14, 15, 16, 17]
    v_map_bot1 = [18, 19, 20, 21]
    v_map_bot2 = [22, 23, 24, 25]
    v_map_bot3 = [26, 27]
    
    # W block 0 (k=0,1) maps: 0->0, 1->2, 2->1, 3->3
    # W block 1 (k=2,3) maps: 0->4, 1->6, 2->5, 3->7
    # W block 2 (k=4,5) maps: 0->8, 1->10, 2->9, 3->11
    # W block 3 (k=6) maps: 0->12, 1->13
    
    w_map0 = [0, 2, 1, 3]
    w_map1 = [4, 6, 5, 7]
    w_map2 = [8, 10, 9, 11]
    w_map3 = [12, 13]
    
    # A_left * B_top
    # Block 0 (2x2x2 Strassen)
    for r in range(7):
        # We need to re-add the Strassen terms with the correct maps.
        # Instead of re-adding, let's just call add_block with the specific terms.
        pass
        
    # Let's just generate the terms again inside the loop.
    
    def generate_strassen_terms():
        terms = []
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        terms.append(([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]))
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]))
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return terms
        
    def generate_schoolbook_221_terms():
        terms = []
        terms.append(([1, 0, 0, 0], [1, 0], [1, 0]))
        terms.append(([0, 1, 0, 0], [0, 1], [1, 0]))
        terms.append(([0, 0, 1, 0], [1, 0], [0, 1]))
        terms.append(([0, 0, 0, 1], [0, 1], [0, 1]))
        return terms
        
    strassen_terms = generate_strassen_terms()
    schoolbook_terms = generate_schoolbook_221_terms()
    
    # A_left * B_top
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_block(u, v, w, u_map_left, v_map_top0, w_map0)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_block(u, v, w, u_map_left, v_map_top1, w_map1)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_block(u, v, w, u_map_left, v_map_top2, w_map2)
    for r in range(4):
        u, v, w = schoolbook_terms[r]
        add_block(u, v, w, u_map_left, v_map_top3, w_map3)
        
    # A_right * B_bottom
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_block(u, v, w, u_map_right, v_map_bot0, w_map0)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_block(u, v, w, u_map_right, v_map_bot1, w_map1)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_block(u, v, w, u_map_right, v_map_bot2, w_map2)
    for r in range(4):
        u, v, w = schoolbook_terms[r]
        add_block(u, v, w, u_map_right, v_map_bot3, w_map3)
        
    U = np.array(U_global).T
    V = np.array(V_global).T
    W = np.array(W_global).T
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    
    # Get trivial solution as fallback
    U_triv, V_triv, W_triv = _build_trivial(n, m, p)
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Try block decomposition
    if time.time() - start_time < deadline - 10:
        U_block, V_block, W_block = _block_decomposition()
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            best_U, best_V, best_W = U_block, V_block, W_block
            # Rank is 50.
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
