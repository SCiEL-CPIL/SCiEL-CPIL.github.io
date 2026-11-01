"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x14 matrix.
    
    Strategy:
    The bilinear complexity of M_{2,2,14} is at most 2 * M_{2,7} (since 2x14 = 2x(2*7) can be split into two 2x7 blocks).
    Actually, we can view the 2x14 matrix as a single row of 14 elements, or two rows of 7.
    
    Let's analyze the structure.
    A is 2x2. B is 2x14. C is 2x14.
    C_{i,k} = sum_{j=1}^2 A_{i,j} B_{j,k}
    
    This is equivalent to two independent multiplications:
    Row 1 of C: C_{1,k} = A_{1,1} B_{1,k} + A_{1,2} B_{2,k} for k=1..14
    Row 2 of C: C_{2,k} = A_{2,1} B_{1,k} + A_{2,2} B_{2,k} for k=1..14
    
    Each row calculation is a vector of length 14, computed from two input vectors of length 14 and four scalars.
    Specifically, for a fixed i, C_i = A_{i,1} B_1 + A_{i,2} B_2.
    This is a linear combination of two vectors B_1 and B_2 with coefficients from A.
    The rank of the tensor for computing C_i (length 14) from A_{i,1}, A_{i,2} and B_1, B_2 (length 14 each) is not simply additive in the standard sense because the B vectors are shared between the two rows? No, B is the same for both rows.
    
    Wait, the tensor is T_{i,k,j,l} where A_{i,j} and B_{l,k} produce C_{i,k}.
    Indices:
    A: i (1..2), j (1..2)
    B: l (1..2), k (1..14)
    C: i (1..2), k (1..14)
    
    T_{i,k, j,l} = delta_{j,l} ? No.
    C_{i,k} = sum_{j=1}^2 A_{i,j} B_{j,k}
    So the tensor T has indices (i, k) for output, and (j, l) for inputs where A_{i,j} and B_{l,k}.
    Actually, the standard tensor T_{ijk} for M_{n,m,p} has C_{ik} = sum_j A_{ij} B_{jk}.
    Here n=2, m=2, p=14.
    T_{i, k, j} is not quite right because B has two dimensions (row l and col k).
    Let's flatten indices.
    Input A: size 4. Indices (i, j) with i in {0,1}, j in {0,1}.
    Input B: size 28. Indices (l, k) with l in {0,1}, k in {0..13}.
    Output C: size 28. Indices (i, k) with i in {0,1}, k in {0..13}.
    
    The bilinear form is:
    C_{i,k} = sum_{j=0}^1 A_{i,j} B_{j,k}
    
    This tensor can be seen as a block diagonal structure?
    For each k (0..13), the output C_{i,k} depends on A_{i,0}, A_{i,1} and B_{0,k}, B_{1,k}.
    Specifically, the pair (C_{0,k}, C_{1,k}) is determined by (A_{0,0}, A_{0,1}, A_{1,0}, A_{1,1}) and (B_{0,k}, B_{1,k}).
    Let A_col_k be the vector of A entries? No, A is fixed for all k.
    
    Let's look at the dependence on k.
    For a fixed k, we compute a 2x1 vector C_k = [C_{0,k}, C_{1,k}]^T.
    C_k = A * B_k, where B_k is the k-th column of B (size 2x1).
    So we are computing 14 independent 2x1 matrix multiplications?
    No, A is shared.
    
    This is equivalent to multiplying a 2x2 matrix A by a 2x14 matrix B.
    This is the same as computing 14 independent 2x1 multiplications using the same A.
    The rank of the tensor T is the minimum rank of the decomposition.
    
    We can use the fact that M_{2,2,14} is isomorphic to a sub-tensor of M_{2,2,14}.
    Actually, we can embed this into a larger problem or use known bounds.
    
    Known result: The bilinear complexity of M_{n,m,p} is at most M_{n,m,p-1} + M_{1,m,1} ?
    Or M_{n,m,p} <= M_{n,m,1} * p ? No.
    
    Let's look at the structure again.
    C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}
    
    This is a linear map from A (4 dims) and B (28 dims) to C (28 dims).
    For each k, the output C_k (2 dims) is a linear combination of B_{0,k} and B_{1,k} with coefficients A_{i,0} and A_{i,1}.
    
    We can define the tensor T_{i,k, j, l} where i is row of C/A, k is col of C/B, j is col of A, l is row of B.
    T_{i,k,j,l} = 1 if j=l, else 0?
    Wait, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    So the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    Thus, the tensor entry for (i,k) output, (i,j) input A, (j,k) input B is 1.
    Indices:
    Output: (i, k)
    Input A: (i, j)
    Input B: (j, k)
    
    This is exactly the matrix multiplication tensor M_{2,2,14} where the "inner" dimension is j (size 2), and the "outer" dimensions are i (size 2) and k (size 14).
    
    The rank of M_{n,m,p} is known to be <= n*m*p.
    For M_{2,2,14}, the schoolbook rank is 2*2*14 = 56.
    
    Can we do better?
    We can use the fact that M_{2,2,14} is a sub-tensor of M_{2,2,14}.
    Also, M_{2,2,14} is isomorphic to M_{2,14,2} (by symmetry of matrix multiplication? No, M_{n,m,p} is not symmetric in all indices, but M_{n,m,p} is isomorphic to M_{p,m,n} and M_{m,n,p} with appropriate permutations).
    Actually, M_{n,m,p} is isomorphic to M_{p,m,n}.
    So M_{2,2,14} is isomorphic to M_{14,2,2}.
    
    The rank of M_{14,2,2} is the same as M_{2,2,14}.
    
    Let's consider the block structure.
    We can view the 2x14 matrix B as a 2x2 block matrix with 1x7 blocks?
    B = [ B_{0,0} B_{0,1} B_{0,2} ... B_{0,6} ]  (each block is 2x1? No, 1x7? No, B is 2x14. So each block is 2x1 or 1x7 depending on layout).
    
    Let's try to use a known decomposition for small matrices.
    For M_{2,2,2}, the rank is 7 (Strassen).
    For M_{2,2,p}, the rank is at most 7 * ceil(p/2)? Or something similar.
    
    Actually, there is a result by Pan (1978) and others on the rank of rectangular matrix multiplication.
    However, we can construct a decomposition explicitly.
    
    Let's try to reduce the rank by combining terms.
    The schoolbook decomposition has 56 terms:
    For each i in {0,1}, j in {0,1}, k in {0..13}:
    U_{i,j, c} = 1
    V_{j,k, c} = 1
    W_{i,k, c} = 1
    
    We want to find a set of rank-1 tensors that sum to the multiplication tensor.
    
    Consider the case p=2. M_{2,2,2} has rank 7.
    The 7 terms of Strassen for M_{2,2,2} can be adapted.
    For p=14, we have 7 pairs of columns (k, k+1).
    We can apply the Strassen-like decomposition for each pair of columns.
    Since the columns are independent (no coupling between k and k+1 in the output), we can process each pair independently.
    For each pair of columns (2t, 2t+1), we compute C_{:, 2t} and C_{:, 2t+1}.
    This is equivalent to multiplying A (2x2) by B_pair (2x2).
    The rank of M_{2,2,2} is 7.
    So for each pair, we need 7 terms.
    Total rank = 7 * 7 = 49.
    
    This is better than 56.
    
    Let's verify the Strassen decomposition for M_{2,2,2}.
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    
    Strassen's 7 products:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    We need to map this to the tensor decomposition format.
    The tensor T_{i,k,j,l} for M_{2,2,2} has 8 non-zero entries (if we consider the standard basis).
    Wait, the tensor is defined by the bilinear map.
    The rank is the minimum number of rank-1 tensors that sum to T.
    
    For each pair of columns (k, k+1), we have a sub-tensor T_pair.
    T_pair is isomorphic to M_{2,2,2}.
    The rank of T_pair is 7.
    Since the 7 pairs are disjoint in their output indices (k) and input B indices (k), but share the input A indices, we can sum the decompositions.
    
    For pair t (columns 2t, 2t+1):
    We have 7 terms.
    Each term is a rank-1 tensor:
    u_t, s * v_t, s * w_t, s
    
    The u vector has support on A indices (i, j).
    The v vector has support on B indices (l, k) for k in {2t, 2t+1}.
    The w vector has support on C indices (i, k) for k in {2t, 2t+1}.
    
    Since the supports of v and w for different t are disjoint, we can simply concatenate the decompositions.
    Total rank = 7 * 7 = 49.
    
    Let's construct the 49 terms.
    
    For each t in 0..6:
    k0 = 2t
    k1 = 2t + 1
    
    B indices:
    l=0, k=k0 -> index 0*14 + k0 = k0
    l=0, k=k1 -> index k1
    l=1, k=k0 -> index 14 + k0
    l=1, k=k1 -> index 14 + k1
    
    A indices:
    i=0, j=0 -> 0
    i=0, j=1 -> 1
    i=1, j=0 -> 2
    i=1, j=1 -> 3
    
    C indices:
    i=0, k=k0 -> 0*14 + k0 = k0
    i=0, k=k1 -> k1
    i=1, k=k0 -> 14 + k0
    i=1, k=k1 -> 14 + k1
    
    Strassen terms for pair t:
    
    1. P1 = (A00 + A11) * (B0,k0 + B1,k1)  -- Wait, Strassen uses (e+h) which is B00 + B11 in the 2x2 block.
       Here, the 2x2 block B_pair is:
       [[B0,k0, B0,k1],
        [B1,k0, B1,k1]]
       
       So e=B0,k0, f=B0,k1, g=B1,k0, h=B1,k1.
       
       P1 = (A00 + A11) * (B0,k0 + B1,k1)
       U: A00=1, A11=1
       V: B0,k0=1, B1,k1=1
       W: C00 += 1, C11 += 1, C01 += 1, C10 += 1 ?
       No, W is the output coefficient.
       The rank-1 tensor is u * v * w.
       u is vector over A, v over B, w over C.
       The contribution to C is u * v * w.
       
       Let's check the sum:
       sum_s u_s * v_s * w_s
       
       For P1:
       u = [1, 0, 0, 1] (A00, A01, A10, A11)
       v = [1, 0, 0, 1] (B0,k0, B0,k1, B1,k0, B1,k1)
       w = [1, 1, 1, 1] (C0,k0, C0,k1, C1,k0, C1,k1)
       
       Product:
       (A00 + A11) * (B0,k0 + B1,k1) * (C0,k0 + C0,k1 + C1,k0 + C1,k1)
       
       This expands to:
       A00 B0,k0 C0,k0 + A00 B0,k0 C0,k1 + A00 B0,k0 C1,k0 + A00 B0,k0 C1,k1
       + A00 B1,k1 C0,k0 + ...
       
       This is not correct. The rank-1 tensor u v^T w^T contributes to the tensor T.
       T_{i,k,j,l} = sum_s u_{i,j,s} v_{l,k,s} w_{i,k,s}
       
       We want T to be the multiplication tensor.
       T_{i,k,j,l} = 1 if j=l, else 0.
       
       So for P1:
       u_{0,0} = 1, u_{1,1} = 1.
       v_{0,k0} = 1, v_{1,k1} = 1.
       w_{0,k0} = 1, w_{0,k1} = 1, w_{1,k0} = 1, w_{1,k1} = 1.
       
       Sum over s:
       For (i=0, k=k0, j=0, l=0):
       u_{0,0} v_{0,k0} w_{0,k0} = 1*1*1 = 1. Correct (A00 B0,k0).
       
       For (i=0, k=k0, j=0, l=1):
       u_{0,0} v_{1,k0} w_{0,k0} = 1*0*1 = 0. Correct.
       
       For (i=0, k=k0, j=1, l=0):
       u_{0,1} v_{0,k0} w_{0,k0} = 0*1*1 = 0. Correct.
       
       For (i=0, k=k0, j=1, l=1):
       u_{0,1} v_{1,k0} w_{0,k0} = 0*0*1 = 0. Correct.
       
       But wait, P1 also contributes to other terms.
       For (i=0, k=k1, j=0, l=0):
       u_{0,0} v_{0,k1} w_{0,k1} = 1*0*1 = 0.
       
       For (i=0, k=k1, j=0, l=1):
       u_{0,0} v_{1,k1} w_{0,k1} = 1*1*1 = 1.
       This corresponds to A00 B1,k1 C0,k1.
       But in the multiplication, C0,k1 = A00 B0,k1 + A01 B1,k1.
       So A00 B1,k1 is not a term in the expansion of C0,k1.
       The term is A01 B1,k1.
       
       So P1 as defined above is NOT a valid rank-1 component of the multiplication tensor.
       The multiplication tensor T has T_{i,k,j,l} = delta_{j,l}.
       Any rank-1 component u v w must satisfy that u_{i,j} v_{l,k} w_{i,k} is non-zero only when j=l?
       No, the sum of rank-1 components must equal T.
       The individual components can have non-zero entries where T is zero, as long as they cancel out.
       
       However, Strassen's algorithm works because the combination of the 7 terms yields the correct result.
       The 7 terms are not individually "valid" in the sense of matching T's support, but their sum does.
       
       So we can use the 7 terms of Strassen for each pair of columns.
       
       Let's list the 7 terms for a pair (k0, k1).
       Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
       Indices for B: 0:(0,k0), 1:(0,k1), 2:(1,k0), 3:(1,k1)
       Indices for C: 0:(0,k0), 1:(0,k1), 2:(1,k0), 3:(1,k1)
       
       P1 = (A0+A3)(B0+B3)
       u1 = [1, 0, 0, 1]
       v1 = [1, 0, 0, 1]
       w1 = [1, 1, 1, 1]  -- Wait, w1 must be such that the sum works.
       
       Let's check the Strassen combination again.
       C00 = P1 + P4 - P5 + P7
       C01 = P3 + P5
       C10 = P2 + P4
       C11 = P1 - P2 + P3 + P6
       
       This implies that the tensor T is the sum of 7 rank-1 tensors, where the w vectors are:
       w1 = [1, 0, 0, 1]  (contributes to C00 and C11)
       w2 = [0, 0, 1, 0]  (contributes to C10) -> No, P2 contributes to C10 and C11.
       
       Let's derive the w vectors from the linear combinations.
       T = sum_{s=1}^7 u_s v_s w_s
       
       P1 = (A0+A3)(B0+B3)
       P2 = (A2+A3)B0
       P3 = A1(B1-B3)
       P4 = A3(B2-B0)
       P5 = (A0+A1)B3
       P6 = (A2-A0)(B0+B1)
       P7 = (A1-A3)(B2+B3)
       
       C0 = P1 + P4 - P5 + P7
       C1 = P3 + P5
       C2 = P2 + P4
       C3 = P1 - P2 + P3 + P6
       
       So:
       w1 = [1, 0, 0, 1]
       w2 = [0, 0, 1, -1]
       w3 = [0, 1, 0, 1]
       w4 = [1, 0, 1, 0]
       w5 = [-1, 1, 0, 0]
       w6 = [0, 0, 0, 1]
       w7 = [1, 0, 0, 0]
       
       Let's check:
       C0 = 1*P1 + 0*P2 + 0*P3 + 1*P4 - 1*P5 + 0*P6 + 1*P7 = P1 + P4 - P5 + P7. Correct.
       C1 = 0*P1 + 0*P2 + 1*P3 + 0*P4 + 1*P5 + 0*P6 + 0*P7 = P3 + P5. Correct.
       C2 = 0*P1 + 1*P2 + 0*P3 + 1*P4 + 0*P5 + 0*P6 + 0*P7 = P2 + P4. Correct.
       C3 = 1*P1 - 1*P2 + 1*P3 + 0*P4 + 0*P5 + 1*P6 + 0*P7 = P1 - P2 + P3 + P6. Correct.
       
       Now we need the u and v vectors.
       
       P1 = (A0+A3)(B0+B3)
       u1 = [1, 0, 0, 1]
       v1 = [1, 0, 0, 1]
       
       P2 = (A2+A3)B0
       u2 = [0, 0, 1, 1]
       v2 = [1, 0, 0, 0]
       
       P3 = A1(B1-B3)
       u3 = [0, 1, 0, 0]
       v3 = [0, 1, 0, -1]
       
       P4 = A3(B2-B0)
       u4 = [0, 0, 0, 1]
       v4 = [0, 0, 1, -1]  -- Wait, B2 is (1,k0), B0 is (0,k0).
       v4 = [-1, 0, 1, 0]
       
       P5 = (A0+A1)B3
       u5 = [1, 1, 0, 0]
       v5 = [0, 0, 0, 1]
       
       P6 = (A2-A0)(B0+B1)
       u6 = [-1, 0, 1, 0]
       v6 = [1, 1, 0, 0]
       
       P7 = (A1-A3)(B2+B3)
       u7 = [0, 1, 0, -1]
       v7 = [0, 0, 1, 1]
       
       Now we can construct the full decomposition.
       Rank = 49.
       
       We need to map the local indices to global indices.
       
       Global A indices: 0..3
       Global B indices: 0..27
       Global C indices: 0..27
       
       For pair t:
       k0 = 2t
       k1 = 2t + 1
       
       Local B index 0 -> Global B index k0
       Local B index 1 -> Global B index k1
       Local B index 2 -> Global B index 14 + k0
       Local B index 3 -> Global B index 14 + k1
       
       Local C index 0 -> Global C index k0
       Local C index 1 -> Global C index k1
       Local C index 2 -> Global C index 14 + k0
       Local C index 3 -> Global C index 14 + k1
       
       Local A index 0 -> Global A index 0
       Local A index 1 -> Global A index 1
       Local A index 2 -> Global A index 2
       Local A index 3 -> Global A index 3
       
       We will create U, V, W matrices of size (4, 49), (28, 49), (28, 49).
       
       Let's implement this.
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank = 7 * (p/2) = 7 * 7 = 49
    rank = 7 * (p // 2)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen templates
    # u, v, w are lists of 4 elements each (local indices)
    templates = [
        # P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # P3
        ([0, 1, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # P4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for t in range(p // 2):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Local to Global B index mapping
        b_map = [k0, k1, 14 + k0, 14 + k1]
        # Local to Global C index mapping
        c_map = [k0, k1, 14 + k0, 14 + k1]
        # Local to Global A index mapping
        a_map = [0, 1, 2, 3]
        
        for u_loc, v_loc, w_loc in templates:
            for i_loc, val in enumerate(u_loc):
                if val != 0:
                    i_glob = a_map[i_loc]
                    U[i_glob][col] = val
            
            for i_loc, val in enumerate(v_loc):
                if val != 0:
                    i_glob = b_map[i_loc]
                    V[i_glob][col] = val
            
            for i_loc, val in enumerate(w_loc):
                if val != 0:
                    i_glob = c_map[i_loc]
                    W[i_glob][col] = val
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
