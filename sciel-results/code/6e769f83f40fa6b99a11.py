import numpy as np
import time
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The tensor is T_{i,j,k} = delta_{i,j,k} for C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U has shape (n*m, rank), indexed by (i, j) -> i*m + j
    # V has shape (m*p, rank), indexed by (j, k) -> j*p + k
    # W has shape (p*n, rank), indexed by (k, i) -> k*n + i  (trace convention W[k*n+i, r])
    
    # Target: U_{ij,r} * V_{jk,r} * W_{ki,r} = sum_r ... = delta_{i,j,k} * 1? 
    # No, the tensor is T_{ij,k} = 1 if we consider A_{ij} B_{jk} -> C_{ik}.
    # The decomposition is: C_{ik} = sum_j A_{ij} B_{jk} = sum_r U_{ij,r} V_{jr,r} W_{kr,r}? 
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is represented by a tensor T_{i,j,k} where T_{i,j,k} = 1.
    # A rank-1 term contributes u_{i,j} v_{j,k} w_{k,i} to T_{i,j,k}?
    # No, the factorization is usually written as:
    # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) * (sum_k V_{jk,r} B_{jk})? No.
    
    # Let's look at the interface:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # The product is sum_r U_{ij,r} V_{jk,r} W_{ki,r} ?
    # If we fix i,j,k, the contribution to the tensor T_{i,j,k} is sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}.
    # We want this sum to equal 1 for all i,j,k (since T_{i,j,k} = 1 for the matrix multiplication tensor).
    # Wait, is T_{i,j,k} = 1?
    # A_{ij} B_{jk} sums over j. The "tensor" of matrix multiplication is the set of coefficients 1 for each (i,j,k).
    # So yes, we want sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify the seed's logic with this understanding.
    # Seed: n=2, m=3, p=7.
    # U indices: i*3 + j. i in {0,1}, j in {0,1,2}.
    # V indices: j*7 + k. j in {0,1,2}, k in {0..6}.
    # W indices: k*2 + i. k in {0..6}, i in {0,1}.
    
    # The seed uses Strassen blocks for j in {0,1} and k pairs.
    # This implies the tensor is separable by j? No, Strassen mixes i and k.
    # But j is the summation index.
    # Actually, the tensor T_{i,j,k} = 1.
    # If we fix j, we have a sub-tensor T_{i,k}^{(j)} = 1 for all i,k.
    # The rank of the full tensor is the sum of ranks of slices? No, not necessarily.
    # However, if the support of U, V, W for different j's are disjoint, then it is a direct sum.
    # In the seed:
    # j=0 and j=1 are handled together by Strassen blocks (which mix i and k).
    # j=2 is handled separately (schoolbook).
    # So the seed treats j=0,1 as one block and j=2 as another.
    # Within the j=0,1 block, it uses Strassen for 2x2 matrix multiplication (i in {0,1}, k in pairs).
    # This works because for a fixed j, the operation is C_{ik} += A_{ij} B_{jk}.
    # If we group j=0,1, we are effectively doing a 2x2 matrix multiplication for each pair of k's?
    # No, A is 2x3, B is 3x7.
    # C = A B.
    # C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k} + A_{i,2}B_{2,k}.
    # The seed decomposes this into:
    # 1. Terms involving A_{i,0} and A_{i,1} (j=0,1).
    # 2. Terms involving A_{i,2} (j=2).
    
    # For the j=0,1 part:
    # C'_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k}.
    # This is the product of a 2x2 matrix A' (rows i, cols 0,1) and a 2x7 matrix B' (rows 0,1, cols k).
    # The seed splits k into pairs (0,1), (2,3), (4,5) and handles k=6 separately.
    # For a pair of k's, say k=0,1, we have a 2x2 sub-problem:
    # C'_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}
    # C'_{i,1} = A_{i,0}B_{0,1} + A_{i,1}B_{1,1}
    # This is exactly 2x2x2 matrix multiplication, which has rank 7 (Strassen).
    # So for each pair of k's, we use 7 terms.
    # There are 3 pairs: (0,1), (2,3), (4,5). So 3 * 7 = 21 terms.
    # For k=6, we have a 2x2x1 problem (A is 2x2, B is 2x1, C is 2x1).
    # C'_{i,6} = A_{i,0}B_{0,6} + A_{i,1}B_{1,6}.
    # This has rank 4 (schoolbook).
    # So total for j=0,1 is 21 + 4 = 25 terms.
    
    # For j=2:
    # C''_{i,k} = A_{i,2}B_{2,k}.
    # This is an outer product of a 2x1 vector and a 1x7 vector.
    # It decomposes into 2 * 7 = 14 rank-1 terms.
    # Total rank = 25 + 14 = 39.
    
    # Can we do better?
    # The rank of 2x3x2 is 10 (known).
    # If we could group j=0,1,2 differently?
    # The tensor is T_{i,j,k} = 1.
    # This tensor is the direct sum of m copies of the 2x1xp tensor? No.
    # T = sum_j T^{(j)} where T^{(j)}_{i,k} = 1.
    # But the indices are (i,j,k).
    # The support of T^{(j)} is {i in [0,2), k in [0,7)} at fixed j.
    # Since the j-index is distinct for each slice, and the tensor is a direct sum along the j-axis (in the sense that slices don't overlap in j), the rank is the sum of the ranks of the slices IF the decomposition can be chosen to have disjoint supports in the j-dimension of U and V.
    # U_{ij,r}, V_{jk,r}.
    # If we choose U and V to be zero for j not in the slice, then the terms for different j are orthogonal in the r-space? No, they share r.
    # But if we use distinct r's for each j-slice, then the rank is additive.
    # Can we mix j's?
    # If we allow U_{ij,r} to be non-zero for multiple j's, we might reduce rank.
    # However, the tensor T_{i,j,k} = 1 is "separable" in j in the sense that T_{i,j,k} = T_{i,k} * 1_j.
    # Actually, T_{i,j,k} = 1 for all i,j,k.
    # This is the tensor of the bilinear map (A,B) -> C.
    # The rank of the 2x3x7 tensor is the minimum number of rank-1 tensors summing to it.
    
    # Known ranks:
    # 2x2x2: 7
    # 2x2x3: 11? (or 10?)
    # 2x3x3: 13?
    # 2x3x7: ?
    
    # Let's check if we can improve the 2x2x1 part (k=6, j=0,1).
    # Rank is 4. Optimal.
    # Let's check the 2x1x7 part (j=2).
    # Rank is 14. Optimal.
    # Let's check the 2x2x2 parts.
    # Rank is 7. Optimal.
    
    # So the only place to improve is if we can mix the j=0,1 and j=2 slices, or mix the k-pairs.
    # Mixing k-pairs:
    # Instead of (0,1), (2,3), (4,5), (6), what if we use a 2x2x4 decomposition?
    # The rank of 2x2x4 is 13? (2x2x3 is 11, 2x2x4 is 13?).
    # If rank(2x2x4) < 7+7=14, we save.
    # 13 < 14, so saving 1.
    # Then we have k=4,5,6 left. 2x2x3. Rank 11?
    # 7+7+4 = 18.
    # 13+11 = 24.
    # Original: 21+4=25.
    # So 24 < 25. We save 1.
    # Total would be 24 + 14 = 38.
    
    # Is rank(2x2x3) 11?
    # Yes, it is known that r(2x2x3) = 11.
    # Is rank(2x2x4) 13?
    # Yes, r(2x2x4) = 13.
    
    # So we can potentially get rank 38.
    # Can we do better?
    # What about 2x2x7?
    # r(2x2x7) = ?
    # If we use one big block for j=0,1 and all k=0..6.
    # r(2x2x7) = ?
    # Lower bound: max(4, 14, 14) = 14.
    # Upper bound: 7*4 + 4 = 32? No.
    # Using 2x2x2 blocks: 3*7 + 4 = 25.
    # Using 2x2x4 + 2x2x3: 13 + 11 = 24.
    # Is there a better decomposition for 2x2x7?
    # r(2x2xn) is known for small n.
    # r(2x2x1) = 4
    # r(2x2x2) = 7
    # r(2x2x3) = 11
    # r(2x2x4) = 13
    # r(2x2x5) = 15?
    # r(2x2x6) = 17?
    # r(2x2x7) = 19?
    
    # If r(2x2x7) = 19, then total rank = 19 + 14 = 33.
    # This would be a huge improvement.
    
    # Let's check the sequence for r(2x2xn).
    # It is not linear.
    # However, 2x2xn matrix multiplication is equivalent to multiplying a 2x2 matrix by a 2xn matrix.
    # This is a bilinear map.
    # The rank is the tensor rank of the 2x2xn tensor.
    
    # According to literature (e.g., Landsberg, "Tensor Geometry"):
    # r(2x2xn) for n=1..7:
    # n=1: 4
    # n=2: 7
    # n=3: 11
    # n=4: 13
    # n=5: 15
    # n=6: 17
    # n=7: 19
    
    # If this is correct, then r(2x2x7) = 19.
    # Then total rank = 19 + 14 = 33.
    
    # Let's try to construct a rank 33 decomposition.
    # We need a rank 19 decomposition for 2x2x7.
    # And a rank 14 decomposition for 2x1x7 (which is just schoolbook).
    
    # How to construct rank 19 for 2x2x7?
    # We can use the known decompositions for 2x2x4 (rank 13) and 2x2x3 (rank 11).
    # 13 + 11 = 24.
    # But we want 19.
    # So we need a more efficient decomposition.
    # Is there a direct construction for 2x2x7 with rank 19?
    
    # Alternatively, maybe we can mix j=0,1,2 together?
    # The full tensor is 2x3x7.
    # If we view it as a 2x(3x7) tensor? No.
    # If we view it as a (2x3) x 7 tensor?
    # The rank of the 2x3x7 tensor is what we want.
    
    # Let's assume the rank is 33 and try to find a decomposition.
    # If we can't find one, we fall back to 38 or 39.
    
    # Since constructing a rank 19 2x2x7 tensor is non-trivial, let's first try to verify if 38 is achievable.
    # We know 2x2x4 (13) + 2x2x3 (11) = 24 for j=0,1.
    # Plus 14 for j=2.
    # Total 38.
    
    # Let's implement the 2x2x4 and 2x2x3 decompositions.
    # We need the explicit rank-1 terms for 2x2x4 and 2x2x3.
    
    # For 2x2x2, we have Strassen (7 terms).
    # For 2x2x3, we can use a known rank 11 decomposition.
    # For 2x2x4, we can use a known rank 13 decomposition.
    
    # However, finding these explicit decompositions is complex.
    # A simpler approach:
    # Use Strassen for 2x2x2 (7 terms) for k=0,1.
    # Use Strassen for 2x2x2 (7 terms) for k=2,3.
    # Use Strassen for 2x2x2 (7 terms) for k=4,5.
    # Use Schoolbook for 2x2x1 (4 terms) for k=6.
    # Total 25.
    
    # To get 24, we need to save 1 term by combining k=4,5,6 into a 2x2x3 block.
    # Rank(2x2x3) = 11.
    # 7+7+11 = 25. No saving.
    # Wait, 7 (for 0,1) + 7 (for 2,3) + 11 (for 4,5,6) = 25.
    # Original: 7+7+7+4 = 25.
    # So no saving there.
    
    # What if we combine k=0,1,2,3 into 2x2x4?
    # Rank(2x2x4) = 13.
    # 13 + 7 (for 4,5) + 4 (for 6) = 24.
    # Yes! 13 < 7+7=14.
    # So we save 1 term.
    # Total for j=0,1 is 24.
    # Total rank = 24 + 14 = 38.
    
    # So we need a rank 13 decomposition for 2x2x4.
    # And a rank 7 decomposition for 2x2x2 (Strassen).
    # And a rank 4 decomposition for 2x2x1 (Schoolbook).
    
    # Let's find the rank 13 decomposition for 2x2x4.
    # This is a known problem.
    # One way to construct it is to use the fact that 2x2x4 is related to 2x2x2.
    # Or use a known table.
    
    # Since I don't have the explicit 13-term decomposition memorized, I will try to generate it or use a fallback.
    # If I can't find it, I'll stick with 39.
    # But 38 is a clear improvement if I can find the 2x2x4 decomposition.
    
    # Let's try to search for a 2x2x4 decomposition of rank 13.
    # This is a small search problem.
    # Variables: U (4x13), V (8x13), W (8x13).
    # Equations: sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.
    # i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # 16 equations.
    # We can fix some variables to reduce the search space.
    # For example, set U and V to have sparse structures.
    
    # Given the time constraint, I will implement a simple search for the 2x2x4 decomposition.
    # If it fails, I return the rank 39 solution.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # U: 2x2 (i,j), V: 2x2 (j,k), W: 2x2 (k,i)
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Strassen:
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # M2: (A10+A11)B00 -> C10, -C11
        # M3: A00(B01-B11) -> C01, C11
        # M4: A11(B10-B00) -> C00, C10
        # M5: (A00+A01)B11 -> -C00, C01
        # M6: (A10-A00)(B00+B01) -> C11
        # M7: (A01-A11)(B10+B11) -> C00
        
        U = [[0]*7 for _ in range(4)] # i*2+j
        V = [[0]*7 for _ in range(4)] # j*2+k
        W = [[0]*7 for _ in range(4)] # k*2+i
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    # We need a 2x2x4 decomposition of rank 13.
    # If we can't find it quickly, we use 2x2x2 + 2x2x2 (rank 14).
    
    # For now, let's just return the rank 39 solution to ensure validity.
    # The search for 38 or 33 is too risky without pre-computed decompositions.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        nonlocal col
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1
        add_term([0, 4], [b00, b11], [c00, c11])
        # M2
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        # M3
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        # M4
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        # M5
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        # M6
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        # M7
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # k=6 block
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # j=2 block
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
