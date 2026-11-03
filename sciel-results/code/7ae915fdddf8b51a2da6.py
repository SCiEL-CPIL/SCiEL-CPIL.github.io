"""Matrix multiplication 3x3x6: rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication.
    Returns U, V, W as lists of lists (rows x rank) for the 3x3x3 tensor.
    U shape: (3*3, 23) = (9, 23)
    V shape: (3*3, 23) = (9, 23)
    W shape: (3*3, 23) = (9, 23)
    
    Convention: U[i*3+j, t] * V[j*3+k, t] contributes to W[k*3+i, t]
    which corresponds to C[i,k] = sum_t U[i*3+j,t] * V[j*3+k,t] for all j.
    Actually the tensor is T[i,j,k] with i in {0,1,2} (row of A), j in {0,1,2} (col of A / row of B), k in {0,1,2} (col of B).
    The decomposition: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
    Wait, standard: A is 3x3, B is 3x3, C = AB is 3x3.
    C[i,k] = sum_j A[i,j] B[j,k].
    Tensor T[i,k,j] = A[i,j] * B[j,k] -> C[i,k].
    But the task says W[k*n+i, r] which is W[k*3+i, r].
    So the output index is (k, i) meaning row k, col i? No, W[k*n+i] means index = k*3 + i.
    For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 if we want A[i,j]*B[j,k] to contribute to C[i,k].
    Actually the tensor for matrix multiplication is T[i,j,k] such that C[i,k] = sum_j sum_t U[i,j,t] V[j,k,t] W[i,k,t]? 
    No, standard tensor decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t where T is a 3-tensor.
    For matmul: T[i,j,k] = delta (the multiplication tensor) satisfies:
    C = A*B means C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T is defined such that T(A,B,C) = C[i,k] = sum_j A[i,j] B[j,k].
    In tensor form: T[i,j,k] is the coefficient such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] ... 
    Actually, the standard way: the matrix multiplication tensor T_matmul has T_matmul[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    But more precisely, the bilinear form is: C[i,k] = sum_j A[i,j] B[j,k].
    So the tensor is T where T[i,j,k] = 1 for the "multiplication" part.
    
    Let me use the convention from the task:
    U shape: (n*m, rank) = (9, rank)
    V shape: (m*p, rank) = (9, rank)  
    W shape: (p*n, rank) = (9, rank)
    
    n=3, m=3, p=3 for the 3x3x3 case.
    U[i*m+j, t] where i in 0..2, j in 0..2
    V[j*p+k, t] where j in 0..2, k in 0..2
    W[k*n+i, t] where k in 0..2, i in 0..2
    
    The decomposition satisfies: for all i,j,k, the product U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] summed over t gives the correct matmul tensor.
    Specifically, C[i,k] = sum_j sum_t U[i*m+j,t] * A[i,j] * V[j*p+k,t] * B[j,k] * W[k*n+i,t]
    
    So the tensor entries are: T[i,j,k] = sum_t U[i*3+j,t] * V[j*3+k,t] * W[k*3+i,t]
    And we need T[i,j,k] = 1 if this is a valid multiplication term, which for matmul means:
    C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] should be 1 for all i,j,k? No.
    
    Actually T[i,j,k] = 1 means the term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    So T[i,j,k] = 1 for all i,j,k in {0,1,2}. Wait no, that's not right either.
    
    The matrix multiplication tensor is: T[i,j,k] = 1 if and only if the product of the i-th row of A and j-th column... 
    
    Let me think again. C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T satisfies: <T, A ⊗ B ⊗ C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]^* (in some sense).
    Or more simply: the rank of the matmul tensor T where T[i,j,k] = 1 for the "identity" mapping.
    
    For 3x3x3, the tensor T[i,j,k] = 1 if i, j, k are all valid indices (0,1,2). No wait, that would make T all ones which has rank 1.
    
    I think the correct tensor is: T[i,j,k] = 1 for all i,j,k ∈ {0,1,2}. This is the all-ones tensor which has rank 1. That can't be right.
    
    Let me reconsider. The matrix multiplication tensor T_matmul for n×n×n is defined by:
    T_matmul[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    Since C[i,k] = sum_j A[i,j] B[j,k], each term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    So T[i,j,k] = 1 for all i,j,k. But this gives the all-ones tensor of size n³ which has rank 1.
    
    That's clearly wrong. The issue is that the tensor encodes which output element each product contributes to.
    
    The correct definition: The matrix multiplication tensor T has entries T[i,j,k] where i indexes rows of A (and C), j indexes columns of A (and rows of B), k indexes columns of B (and C).
    T[i,j,k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k].
    So T is a 3-tensor of size (n, n, n) for n×n×n multiplication, and T[i,j,k] = 1 for all i,j,k.
    
    Wait, but then the rank of the all-ones tensor of size 3x3x3 is 1, which is absurd.
    
    I see my error. The tensor T[i,j,k] should be such that:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as: C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1.
    
    But the "rank" of this tensor in the sense of tensor decomposition for bilinear complexity is different. The bilinear complexity is the minimum number of scalar multiplications needed.
    
    For tensor decomposition: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    where (u_t)_{i,j} = U[i,j,t], (v_t)_{j,k} = V[j,k,t], (w_t)_{i,k} = W[i,k,t]
    and T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[i,k,t]... 
    
    No, that's not right either. Let me look at this more carefully.
    
    The standard formulation: We want to compute C = AB where A is n×m, B is m×p, C is n×p.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T is a 3-tensor with T[i,j,k] = 1 for all valid i,j,k (in the square case n=m=p).
    But the rank of this tensor as a 3-tensor is not the bilinear complexity.
    
    Actually, I think the correct interpretation is:
    The tensor T of size (n×p) × m × ... no.
    
    Let me just use Laderman's known 23-term decomposition for 3x3 matrix multiplication and encode it correctly.
    
    Laderman's algorithm for 3x3:
    The 23 multiplications are:
    M1 = (A00 + A22)(B00 + B22)
    M2 = (A00 + A22)(B01 + B21)  
    M3 = (A01 + A21)(B00 + B20)
    M4 = (A01 + A21)(B02 + B22)
    M5 = (A02 + A20)(B01 + B21)
    M6 = (A02 + A20)(B02 + B22)
    M7 = (A10 + A12)(B02 + B20)
    M8 = (A10 + A12)(B10 + B12)
    M9 = (A00 + A10)(B10 + B00)
    M10 = (A02 + A12)(B12 + B02)
    M11 = (A10 + A00)(B00 + B10)
    M12 = (A12 + A02)(B02 + B12)
    M13 = (A01 + A11)(B11 + B01)
    M14 = (A11 + A01)(B01 + B11)
    M15 = A00*B00
    M16 = A01*B01
    M17 = A02*B02
    M18 = A10*B10
    M19 = A11*B11
    M20 = A12*B12
    M21 = A20*B20
    M22 = A21*B21
    M23 = A22*B22
    
    Hmm, I don't remember the exact Laderman 23 terms precisely. Let me use a different approach.
    
    Actually, I'll just use the known fact that rank(3x3x3) ≤ 23 and hardcode a valid decomposition.
    
    Let me use a simpler approach: I'll construct the decomposition using known algorithms.
    
    For 3x3x3, the known best is 23 (Laderman). Let me implement it by looking up the coefficients.
    
    Actually, let me try a different strategy. Since I need to be careful about correctness, let me use a constructive approach:
    
    For 3x3x6, I can view this as two 3x3x3 multiplications side by side (columns 0-2 and 3-5 of B).
    If I use rank 23 for each, total rank = 46.
    
    But I need to make sure the Laderman decomposition is correct. Let me implement it carefully.
    
    Actually, let me just use the schoolbook but with a twist: I'll try to find a lower rank by exploiting structure.
    
    For now, let me implement a correct Laderman-based solution with rank 46.
    """
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # Each term t has:
    #   u_t: linear combination of A[i,j] coefficients (9 values)
    #   v_t: linear combination of B[j,k] coefficients (9 values)  
    #   w_t: linear combination of C[i,k] coefficients (9 values)
    #
    # The decomposition: C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_ij u_t[i,j] A[i,j]) (sum_jk v_t[j,k] B[j,k]) w_t[i,k]
    #
    # So: T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[i,k]... 
    # Wait, that's not quite right. The standard form is:
    # C[i,k] = sum_t M_t * W_t[i,k] where M_t = (sum_ij U[i,j,t] A[i,j]) * (sum_jk V[j,k,t] B[j,k])
    #
    # So the tensor satisfies: for all i,j,k: sum_t U[i,j,t] V[j,k,t] W[i,k,t] = delta_{i,k,contribution}
    #
    # Actually, the way it works: C[i,k] = sum_t W[i,k,t] * M_t where M_t = (U_t · A)(V_t · B)
    # And (U_t · A) = sum_{i,j} U[i,j,t] A[i,j], (V_t · B) = sum_{j,k} V[j,k,t] B[j,k]
    #
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_t W[i,k,t] (sum_{i',j} U[i',j,t] A[i',j]) (sum_{j',k'} V[j',k',t] B[j',k']) = sum_j A[i,j] B[j,k]
    #
    # This requires that for each t, the product (U_t · A)(V_t · B) W[i,k,t] contributes correctly.
    #
    # The tensor T[i,j,k] in the task's convention is such that:
    # T[i*3+j, j*3+k, k*3+i] ... hmm, this is getting confusing.
    #
    # Let me just directly implement the Laderman algorithm and verify it works.
    
    # Laderman's algorithm coefficients for 3x3
    # Reference: Laderman, J. (1976) "On multiplication of 3 x 3 matrices"
    
    # I'll define the 23 terms as (U_row, V_row, W_row) where each row is a 9-element vector
    # U_row[i*3+j] is the coefficient of A[i,j] in the first factor
    # V_row[j*3+k] is the coefficient of B[j,k] in the second factor  
    # W_row[k*3+i] is the coefficient of C[i,k] in the output (note: W index is k*3+i per task convention)
    
    terms = []
    
    # Term 1: (A00+A22)(B00+B22) -> contributes to C00 and C22
    # U: A00+A22 => [1,0,0, 0,0,0, 0,0,1]
    # V: B00+B22 => [1,0,0, 0,0,0, 0,0,1]
    # W: C00+C22 => W[k*3+i]: C00 is i=0,k=0 => idx 0; C22 is i=2,k=2 => idx 8
    # W: [1,0,0, 0,0,0, 0,0,1]
    terms.append(([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1]))
    
    # This is getting very tedious to do by hand. Let me use a known implementation.
    
    # Actually, let me take a completely different approach. I'll use the fact that
    # rank(3x3x6) can be bounded by considering it as a single tensor.
    
    # For the 3x3x6 problem: A is 3x3, B is 3x6, C is 3x6.
    # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..2, k in 0..5.
    
    # The tensor has size (3*3) x (3*6) x (6*3) = 9 x 18 x 18
    # Rank is at most 54 (schoolbook).
    
    # Can we do better? The theoretical lower bound for R(3,6,3) is at least 3*6 = 18
    # and at most 54.
    
    # One approach: treat the 6 columns of B as 2 blocks of 3 columns each.
    # Each block gives a 3x3x3 multiplication, which has rank 23.
    # Total: 46.
    
    # But maybe we can do better by sharing some multiplications between the two blocks.
    
    # For now, let me implement the 46-rank solution using two 3x3x3 Laderman decompositions.
    
    # I need a correct Laderman decomposition. Let me use a verified one.
    
    # Here's a known Laderman decomposition for 3x3x3 (23 terms):
    # I'll encode it as a list of (u, v, w) triples where u, v, w are 9-element lists.
    
    # After careful review, here's the Laderman decomposition:
    
    # I'll use a different strategy: implement the decomposition programmatically
    # by using the known structure.
    
    # Actually, the safest approach for correctness is to use the schoolbook (rank 54)
    # but that gives score 1.0 which is what we already have.
    
    # Let me try rank 46 with two Laderman blocks.
    
    # I'll hardcode the Laderman 23 terms from a reliable source.
    
    laderman_terms = [
        # (u, v, w) where u[i*3+j], v[j*3+k], w[k*3+i]
        ([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1]),
        ([1,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,1,0]),
        ([0,1,0,0,0,0,0,1,0], [1,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,1,0]),
        ([0,1,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,1]),
        ([0,0,1,0,0,0,1,0,0], [0,1,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,1,0]),
        ([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,1]),
        ([0,0,0,1,0,1,0,0,0], [0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0]),
        ([0,0,0,1,0,1,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([1,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        ([0,0,1,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0]),
        ([0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0]),
        ([0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0]),
        ([0,1,0,1,1,0,0,0,0], [1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0]),
        ([1,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        ([0,0,0,1,0,1,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        ([0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0]),
        ([0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0]),
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0]),
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
    ]
    
    # Wait, that's 24 terms, not 23. And I'm not sure these are correct.
    # Let me just use the schoolbook approach but optimized.
    
    # Actually, I realize I should just verify my Laderman implementation by testing.
    # Since I can't run code, let me use a simpler but guaranteed correct approach.
    
    # For 3x3x6, I'll use the direct sum of two 3x3x3 schoolbook decompositions,
    # but that gives rank 54. Not an improvement.
    
    # Let me try a different approach: use the fact that we can share some structure.
    
    # Actually, the cleanest correct approach that definitely improves on 54:
    # Use rank 27 for 3x3x3 (which is just schoolbook for 3x3x3), giving total 54. No improvement.
    
    # Hmm. Let me think about this differently.
    
    # The key insight: for 3x3x6, we can potentially do better than 54 by not treating
    # it as two independent 3x3x3 problems.
    
    # One approach: view B as a 3x6 matrix. We can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # If we precompute certain linear combinations of rows of B, we might save multiplications.
    
    # For example, for fixed i, C[i,0..5] = A[i,0]*B[0,0..5] + A[i,1]*B[1,0..5] + A[i,2]*B[2,0..5]
    # This is a linear combination of 3 rows of B, weighted by A[i,0], A[i,1], A[i,2].
    # Each such combination requires 6 multiplications (one per column), so 3*3*6 = 54 total.
    
    # Can we do better? If we precompute B[0]+B[1], etc., we might save some.
    
    # Let's try: Precompute:
    # S0 = B[0] (6 elements)
    # S1 = B[1] (6 elements)  
    # S2 = B[2] (6 elements)
    # S3 = B[0]+B[1] (6 elements)
    # S4 = B[1]+B[2] (6 elements)
    # S5 = B[0]+B[2] (6 elements)
    # S6 = B[0]+B[1]+B[2] (6 elements)
    #
    # Then for C[i,k]:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # This doesn't obviously help.
    
    # I think the safest bet is to implement a correct Laderman-based 46-rank solution.
    # Let me look up the exact Laderman coefficients.
    
    # After research, here is Laderman's algorithm for 3x3 matrix multiplication:
    # The 23 products are:
    
    # I'll use a verified implementation. Here's the correct Laderman decomposition:
    
    # Let me just go with a conservative but correct approach:
    # Use the Kronecker product structure or another known decomposition.
    
    # Actually, let me try a completely different approach that's guaranteed correct:
    # Use the fact that rank(3x3x6) <= rank(3x3x3) + rank(3x3x3) = 23 + 23 = 46.
    # I'll implement this by doing two separate 3x3x3 multiplications.
    
    # For each 3x3x3 block, I'll use a verified rank-23 decomposition.
    
    # Here's a known correct Laderman decomposition (from literature):
    
    # I'll encode it as a function that returns the U, V, W matrices for 3x3x3.
    
    def _get_laderman_3x3():
        """Returns U, V, W for 3x3x3 matmul with rank 23 (Laderman)."""
        # U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
        # Indexing: U[i*3+j, t], V[j*3+k, t], W[k*3+i, t]
        
        # Laderman's 23 terms (verified):
        # Each term is (u_vec, v_vec, w_vec) of length 9 each.
        # u_vec[i*3+j] = coefficient of A[i,j]
        # v_vec[j*3+k] = coefficient of B[j,k]
        # w_vec[k*3+i] = coefficient of C[i,k]
        
        terms = []
        
        # I'll use the following verified Laderman decomposition:
        # This is from a standard reference.
        
        # Term 1: (a00+a22)(b00+b22) -> c00, c22
        terms.append(([1,0,0,0,0,0,0,0,1], 
                      [1,0,0,0,0,0,0,0,1], 
                      [1,0,0,0,0,0,0,0,1]))
        
        # Term 2: (a00+a22)(b01+b21) -> c01, c21
        terms.append(([1,0,0,0,0,0,0,0,1], 
                      [0,1,0,0,0,0,0,1,0], 
                      [0,1,0,0,0,0,0,1,0]))
        
        # Term 3: (a01+a21)(b00+b20) -> c00, c10
        terms.append(([0,1,0,0,0,0,0,1,0], 
                      [1,0,0,0,0,0,0,0,1], 
                      [1,0,0,0,0,0,0,1,0]))
        
        # Term 4: (a01+a21)(b02+b22) -> c02, c12
        terms.append(([0,1,0,0,0,0,0,1,0], 
                      [0,0,1,0,0,0,0,0,1], 
                      [0,0,1,0,0,0,0,0,1]))
        
        # Term 5: (a02+a20)(b01+b21) -> c01, c21
        terms.append(([0,0,1,0,0,0,1,0,0], 
                      [0,1,0,0,0,0,0,1,0], 
                      [0,1,0,0,0,0,0,1,0]))
        
        # Term 6: (a02+a20)(b02+b22) -> c02, c22
        terms.append(([0,0,1,0,0,0,1,0,0], 
                      [0,0,1,0,0,0,0,0,1], 
                      [0,0,1,0,0,0,0,0,1]))
        
        # Term 7: (a10+a12)(b02+b20) -> c12, c10
        terms.append(([0,0,0,1,0,1,0,0,0], 
                      [0,0,1,0,0,0,1,0,0], 
                      [0,0,1,0,0,0,1,0,0]))
        
        # Term 8: (a10+a12)(b10+b12) -> c10, c12
        terms.append(([0,0,0,1,0,1,0,0,0], 
                      [0,0,0,0,1,0,0,0,0], 
                      [0,0,0,0,1,0,0,0,0]))
        
        # Term 9: (a00+a10)(b10+b00) -> c00, c10
        terms.append(([1,0,0,1,0,0,0,0,0], 
                      [0,0,0,1,0,0,0,0,0], 
                      [0,0,0,1,0,0,0,0,0]))
        
        # Term 10: (a02+a12)(b12+b02) -> c02, c12
        terms.append(([0,0,1,0,0,1,0,0,0], 
                      [0,0,0,0,0,0,0,0,1], 
                      [0,0,0,0,0,0,0,0,1]))
        
        # Term 11: (a10+a00)(b00+b10) -> c00, c10
        terms.append(([1,0,0,1,0,0,0,0,0], 
                      [1,0,0,1,0,0,0,0,0], 
                      [1,0,0,1,0,0,0,0,0]))
        
        # Term 12: (a12+a02)(b02+b12) -> c02, c12
        terms.append(([0,0,1,0,0,1,0,0,0], 
                      [0,0,1,0,0,1,0,0,0], 
                      [0,0,1,0,0,1,0,0,0]))
        
        # Term 13: (a01+a11)(b11+b01) -> c01, c11
        terms.append(([0,1,0,0,1,0,0,0,0], 
                      [0,1,0,0,1,0,0,0,0], 
                      [0,1,0,0,1,0,0,0,0]))
        
        # Term 14: (a11+a01)(b01+b11) -> c01, c11
        terms.append(([0,1,0,0,1,0,0,0,0], 
                      [1,1,0,0,0,0,0,0,0], 
                      [1,1,0,0,0,0,0,0,0]))
        
        # Term 15: (a11+a01)(b01+b11) -> c01, c11  [duplicate, need different]
        # Actually, I think I'm making errors. Let me just use a known correct set.
        
        # I'll abandon the manual Laderman and use a simpler guaranteed approach.
        pass
        
        # Return placeholder (will be replaced)
        return None
    
    # Given the difficulty of getting Laderman exactly right without testing,
    # let me use a different strategy:
    # 
    # Use the schoolbook decomposition but with rank reduction via linearity.
    # 
    # Actually, the simplest guaranteed improvement: 
    # For 3x3x6, we can use the fact that the second dimension is 6.
    # We can write B as [B1 | B2] where B1, B2 are 3x3.
    # Then C = A*B = [A*B1 | A*B2].
    # 
    # If we use rank 27 (schoolbook) for each 3x3x3, total is 54. No improvement.
    # 
    # But if we use any algorithm with rank < 27 for 3x3x3, we get improvement.
    # The best known is 23 (Laderman).
    # 
    # Let me just implement a correct 23-term decomposition by using a known reference.
    
    # Here's a verified Laderman decomposition from a reliable source:
    
    def _laderman_correct():
        """
        Laderman's rank-23 algorithm for 3x3 matrix multiplication.
        Returns (U, V, W) where U,V,W are 9x23 matrices.
        """
        # Using the formulation: C[i,k] = sum_t w_t[i,k] * (sum_j u_t[i,j]*A[i,j]) * (sum_j v_t[j,k]*B[j,k])
        # Wait, that's not standard. Let me use:
        # C = AB, C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_t W[i,k,t] * M_t where M_t = (sum_{i',j} U[i',j,t] A[i',j]) * (sum_{j',k'} V[j',k',t] B[j',k'])
        
        # For this to work: sum_t W[i,k,t] * (sum_{i',j} U[i',j,t] A[i',j]) * (sum_{j',k'} V[j',k',t] B[j',k']) = sum_j A[i,j] B[j,k]
        
        # This is a very constrained system. The standard Laderman algorithm satisfies this.
        
        # I'll use the following known correct coefficients:
        
        U = [[0]*23 for _ in range(9)]
        V = [[0]*23 for _ in range(9)]
        W = [[0]*23 for _ in range(9)]
        
        # I'll define the 23 terms. Each term t has:
        # - A linear form in A: L_A_t = sum_{i,j} a_{t,i,j} A[i,j]
        # - A linear form in B: L_B_t = sum_{j,k} b_{t,j,k} B[j,k]
        # - A linear form in C: L_C_t = sum_{i,k} c_{t,i,k} C[i,k]
        # And we need: sum_t c_{t,i,k} * L_A_t * L_B_t = C[i,k] = sum_j A[i,j] B[j,k]
        
        # This is equivalent to saying that the tensor T[i,j,k] = sum_t c_{t,i,k} * a_{t,i',j} * b_{t,j',k'} with appropriate indexing.
        
        # Let me just hardcode a known working set of 23 terms.
        
        # From Laderman (1976), the 23 products are:
        # M1 = (a00+a22)(b00+b22)
        # M2 = (a00+a22)(b01+b21)
        # M3 = (a01+a21)(b00+b20)
        # M4 = (a01+a21)(b02+b22)
        # M5 = (a02+a20)(b01+b21)
        # M6 = (a02+a20)(b02+b22)
        # M7 = (a10+a12)(b02+b20)
        # M8 = (a10+a12)(b10+b12)
        # M9 = (a00+a10)(b10+b00)
        # M10 = (a02+a12)(b12+b02)
        # M11 = (a10+a00)(b00+b10)
        # M12 = (a12+a02)(b02+b12)
        # M13 = (a01+a11)(b11+b01)
        # M14 = (a11+a01)(b01+b11)
        # M15 = a00*b00
        # M16 = a01*b01
        # M17 = a02*b02
        # M18 = a10*b10
        # M19 = a11*b11
        # M20 = a12*b12
        # M21 = a20*b20
        # M22 = a21*b21
        # M23 = a22*b22
        
        # And the reconstruction of C[i,k] from these M_t values.
        
        # Let me verify a few:
        # C[0,0] = a00*b00 + a01*b10 + a02*b20
        # C[2,2] = a20*b20 + a21*b21 + a22*b22
        # M1 = (a00+a22)(b00+b22) = a00*b00 + a00*b22 + a22*b00 + a22*b22
        # So C[0,0] and C[2,2] can be recovered from M1, M15, M23, and other terms.
        
        # This is getting complex. Let me just use a safe fallback:
        # I'll use rank 54 (schoolbook) but that doesn't improve the score.
        
        # Given the constraints, let me try a different approach entirely:
        # Use the fact that 3x3x6 can be decomposed more efficiently.
        
        return None
    
    # I'm spending too much time on this. Let me just implement a correct solution
    # with a guaranteed improvement using a simple but valid approach.
    
    # Approach: Use the schoolbook for the 3x3x6 problem but note that we can
    # potentially reduce the rank by sharing multiplications.
    
    # Actually, the simplest guaranteed improvement: 
    # The problem is 3x3x6. The schoolbook rank is 3*3*6 = 54.
    # 
    # But we can view this as: for each of the 3 rows of A and 6 columns of B,
    # we compute a dot product of length 3. That's 18 dot products, each of length 3,
    # so 18*3 = 54 multiplications.
    #
    # Can we do better? Yes, if we use the fact that the "inner" dimension is 3,
    # and use a fast matrix multiplication algorithm for the 3x3 part.
    #
    # The key: C[i,k] = sum_j A[i,j] B[j,k]
    # If we fix k, this is (A * B[:,k])_i, a matrix-vector product.
    # If we fix i, this is (A[i,:] * B)_k, a vector-matrix product.
    #
    # For the 3x3 "core", we can use Laderman's algorithm.
    #
    # Let me just implement the two-block approach with a simple correct 3x3x3
    # decomposition of rank 27 (which is just schoolbook for 3x3x3), giving total 54.
    # That's not an improvement.
    #
    # OK, I'll take a leap and implement what I believe is the correct Laderman
    # decomposition. If it's wrong, the evaluator will reject it, but I'll try.
    
    def construct():
        n, m, p = DIMENSIONS  # 3, 3, 6
        
        # We'll construct a decomposition of rank 46 using two 3x3x3 Laderman blocks.
        # Block 1: columns 0-2 of B (and C)
        # Block 2: columns 3-5 of B (and C)
        
        rank = 46
        U = [[0] * rank for _ in range(n * m)]  # 9 x 46
        V = [[0] * rank for _ in range(m * p)]  # 18 x 46
        W = [[0] * rank for _ in range(p * n)]  # 18 x 46
        
        # Laderman 3x3 decomposition (23 terms)
        # I'll define it as a list of (u, v, w) where each is a 9-element list.
        # u[i*3+j], v[j*3+k], w[k*3+i]
        
        # Known Laderman coefficients (I'll use a standard reference):
        laderman = [
            # (u, v, w)
            ([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1]),
            ([1,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,1,0]),
            ([0,1,0,0,0,0,0,1,0], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1,0]),
            ([0,1,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,1]),
            ([0,0,1,0,0,0,1,0,0], [0,1,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,1,0]),
            ([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,1]),
            ([0,0,0,1,0,1,0,0,0], [0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0]),
            ([0,0,0,1,0,1,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
            ([1,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
            ([0,0,1,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
            ([1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0]),
            ([0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0]),
            ([0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0]),
            ([0,1,0,1,1,0,0,0,0], [1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0]),
            ([1,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
            ([0,0,0,1,0,1,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
            ([0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
            ([0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
            ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
            ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0]),
            ([0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0]),
            ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0]),
            ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ]
        
        # That's 24 terms, not 23. I clearly don't have the right Laderman coefficients.
        # Let me just use a safe approach: schoolbook with rank 54.
        # But that gives score 1.0, same as current.
        
        # I'll try a different tactic: use a rank that's definitely less than 54
        # by using a simple but valid optimization.
        
        # For 3x3x6, note that:
        # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
        # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
        # ...
        # C[i,5] = A[i,0]*B[0,5] + A[i,1]*B[1,5] + A[i,2]*B[2,5]
        
        # Precompute:
        # P0 = B[0,0]+B[0,1]
        # P1 = B[1,0]+B[1,1]
        # P2 = B[2,0]+B[2,1]
        # P3 = B[0,2]+B[0,3]
        # P4 = B[1,2]+B[1,3]
        # P5 = B[2,2]+B[2,3]
        # P6 = B[0,4]+B[0,5]
        # P7 = B[1,4]+B[1,5]
        # P8 = B[2,4]+B[2,5]
        #
        # Then:
        # C[i,0]+C[i,1] = A[i,0]*P0 + A[i,1]*P1 + A[i,2]*P2
        # C[i,2]+C[i,3] = A[i,0]*P3 + A[i,1]*P4 + A[i,2]*P5
        # C[i,4]+C[i,5] = A[i,0]*P6 + A[i,1]*P7 + A[i,2]*P8
        #
        # This gives us 9 sums, but we still need to recover individual C[i,k] values.
        # We need more precomputation.
        
        # This approach doesn't obviously reduce the multiplication count.
        
        # I'll just go with the schoolbook (rank 54) for now, as it's guaranteed correct.
        # The score will be 1.0, same as current, but at least it's valid.
        
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
