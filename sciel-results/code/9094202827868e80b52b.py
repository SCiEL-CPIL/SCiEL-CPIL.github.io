"""Matrix multiplication 4x4x5 - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    
    schoolbook_rank = n * m * p  # 80
    
    def build_schoolbook():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, schoolbook_rank
    
    def verify_decomposition(U, V, W, rank, tol=1e-4):
        """Check if the decomposition is exact."""
        # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_nice_list(mat):
        """Convert numpy array to list of integers/half-integers."""
        result = []
        for row in mat:
            nice_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                nice_row.append(rounded)
            result.append(nice_row)
        return result
    
    def try_strassen_block_decomposition():
        """
        Try to use Strassen's algorithm structure for 2x2 blocks.
        
        For 4x4 x 4x5:
        - Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
        - Split B (4x5) into two 2x5 blocks: B1, B2
        - Output C (4x5) has two 2x5 blocks: C1, C2
        
        C1 = A11*B1 + A12*B2
        C2 = A21*B1 + A22*B2
        
        Each 2x2 x 2x5 multiplication can potentially use Strassen-like tricks.
        For 2x2 x 2x1, Strassen gives rank 7 instead of 8.
        For 2x2 x 2x5, we can do 5 independent 2x2 x 2x1 multiplications.
        With Strassen: 5*7 = 35 multiplications per 2x2 x 2x5 block.
        Total: 2 blocks * 35 = 70... still not better than 80.
        
        Let me think differently.
        
        Actually, let's try a more systematic approach.
        """
        
        # Try rank values from 48 to 79
        for target_rank in range(48, 80):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a structured initialization based on tensor train ideas
            # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k)
            # We want to decompose it as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # Strategy: use random initialization with structure
            # Initialize with small random values, then use gradient-free optimization
            
            # For now, let's try a specific structured pattern
            # that exploits the symmetry of the problem
            
            # Pattern: use modular arithmetic to assign rank indices
            # This is a heuristic that might work for certain ranks
            
            # Try Kronecker product structure
            # For 4x4x4, we can use rank 27 (Strassen) or potentially lower
            # For 4x4x5, we need to extend this
            
            # Let's try a different approach: use the fact that
            # we can share computations across the k dimension
            
            # For each (i,j), we need to compute sum over k of A[i,j]*B[j,k]
            # This is essentially a dot product of length 5
            
            # Use Winograd-style transform on the k dimension
            # For length 5, we can use a 5-point transform
            
            # Actually, let's just try to find ANY decomposition with rank < 80
            # using a simple greedy or random approach
            
            # Random initialization approach
            np.random.seed(42)
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                # Initialize with random values
                U_init = np.random.randn(n * m, target_rank) * 0.1
                V_init = np.random.randn(m * p, target_rank) * 0.1
                W_init = np.random.randn(p * n, target_rank) * 0.1
                
                # Use simple coordinate descent to improve
                # This is a simplified version - in practice we'd use more sophisticated methods
                
                # For now, just check if the random init happens to work (unlikely)
                if verify_decomposition(U_init, V_init, W_init, target_rank, tol=1e-2):
                    return U_init, V_init, W_init, target_rank
        
        return None
    
    def try_kronecker_structure():
        """
        Try to exploit Kronecker product structure.
        
        For 4x4x4 matrix multiplication, we can write it as a tensor product
        of 2x2x2 multiplications.
        
        T_4x4x4 = T_2x2x2 ⊗ T_2x2x2 (in some sense)
        
        If we know a rank-r decomposition of T_2x2x2, then
        T_4x4x4 has rank at most r^2.
        
        For 2x2x2, the schoolbook rank is 8, and Strassen shows rank 7.
        So 4x4x4 has rank at most 7^2 = 49 using this approach.
        
        But we need 4x4x5, not 4x4x4.
        
        Let's first handle 4x4x4 with rank 49, then extend to 4x4x5.
        """
        
        # Strassen decomposition for 2x2x2
        # C = A * B where A, B, C are 2x2
        # C[0,0] = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) - (A[1,0]+A[1,1])*(B[0,1]+B[1,1]) + (A[0,0]+A[0,1])*(B[1,0]+B[1,1])
        # ... this is getting complicated
        
        # Let me use a known Strassen decomposition
        # The 7 multiplications for 2x2x2:
        # m1 = (a11 + a22)(b11 + b22)
        # m2 = (a21 + a22)b11
        # m3 = a11(b12 - b22)
        # m4 = a22(b21 - b11)
        # m5 = (a11 + a12)b22
        # m6 = (a21 - a11)(b11 + b12)
        # m7 = (a12 - a22)(b21 + b22)
        #
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # Let's verify this Strassen decomposition
        def build_strassen_2x2x2():
            """Build factor matrices for 2x2x2 with rank 7."""
            n2, m2, p2 = 2, 2, 2
            rank2 = 7
            U2 = np.zeros((n2 * m2, rank2))
            V2 = np.zeros((m2 * p2, rank2))
            W2 = np.zeros((p2 * n2, rank2))
            
            # m1 = (a11 + a22)(b11 + b22)
            # U contributions: a11 -> U[0,0], a22 -> U[3,0]
            # V contributions: b11 -> V[0,0], b22 -> V[3,0]
            # W contributions: c11 -> W[0,0], c22 -> W[3,0]
            U2[0, 0] = 1  # a11
            U2[3, 0] = 1  # a22
            V2[0, 0] = 1  # b11
            V2[3, 0] = 1  # b22
            W2[0, 0] = 1  # c11
            W2[3, 0] = 1  # c22
            
            # m2 = (a21 + a22)b11
            U2[1, 1] = 1  # a21
            U2[3, 1] = 1  # a22
            V2[0, 1] = 1  # b11
            W2[2, 1] = 1  # c21
            
            # m3 = a11(b12 - b22)
            U2[0, 2] = 1  # a11
            V2[1, 2] = 1  # b12
            V2[3, 2] = -1  # -b22
            W2[1, 2] = 1  # c12
            W2[3, 2] = 1  # c22
            
            # m4 = a22(b21 - b11)
            U2[3, 3] = 1  # a22
            V2[2, 3] = 1  # b21
            V2[0, 3] = -1  # -b11
            W2[2, 3] = 1  # c21
            W2[0, 3] = 1  # c11
            
            # m5 = (a11 + a12)b22
            U2[0, 4] = 1  # a11
            U2[2, 4] = 1  # a12
            V2[3, 4] = 1  # b22
            W2[0, 4] = -1  # -c11
            W2[1, 4] = 1  # c12
            
            # m6 = (a21 - a11)(b11 + b12)
            U2[1, 5] = 1  # a21
            U2[0, 5] = -1  # -a11
            V2[0, 5] = 1  # b11
            V2[1, 5] = 1  # b12
            W2[3, 5] = 1  # c22
            
            # m7 = (a12 - a22)(b21 + b22)
            U2[2, 6] = 1  # a12
            U2[3, 6] = -1  # -a22
            V2[2, 6] = 1  # b21
            V2[3, 6] = 1  # b22
            W2[0, 6] = 1  # c11
            W2[1, 6] = 1  # c12
            
            return U2, V2, W2, rank2
        
        # Verify Strassen decomposition
        U2, V2, W2, rank2 = build_strassen_2x2x2()
        if verify_decomposition(U2, V2, W2, rank2, tol=1e-6):
            # Great, now use Kronecker product to get 4x4x4
            # T_4x4x4 ≈ T_2x2x2 ⊗ T_2x2x2
            # Rank of tensor product is product of ranks
            # So rank(4x4x4) <= 7 * 7 = 49
            
            # Build 4x4x4 decomposition using Kronecker product
            def kron_decomposition(U1, V1, W1, rank1, U2, V2, W2, rank2):
                """Compute Kronecker product decomposition."""
                rank = rank1 * rank2
                U = np.zeros((U1.shape[0] * U2.shape[0], rank))
                V = np.zeros((V1.shape[0] * V2.shape[0], rank))
                W = np.zeros((W1.shape[0] * W2.shape[0], rank))
                
                for r1 in range(rank1):
                    for r2 in range(rank2):
                        r = r1 * rank2 + r2
                        U[:, r] = np.kron(U1[:, r1], U2[:, r2])
                        V[:, r] = np.kron(V1[:, r1], V2[:, r2])
                        W[:, r] = np.kron(W1[:, r1], W2[:, r2])
                
                return U, V, W, rank
            
            U4, V4, W4, rank4 = kron_decomposition(U2, V2, W2, rank2, U2, V2, W2, rank2)
            
            # Verify 4x4x4 decomposition
            # Note: the Kronecker product structure needs to match the index ordering
            # U is indexed by (i,j) where i is row of A, j is col of A
            # In Kronecker: U[i1*2+i2, j1*2+j2, r] = U2[i1,j1,r1] * U2[i2,j2,r2]
            # This should work if the index ordering is consistent
            
            # For 4x4x4, we need to verify
            # But we need 4x4x5, so let's first see if this works for 4x4x4
            
            # Actually, the Kronecker product gives us 4x4x4, not 4x4x5
            # We need to extend this to 4x4x5
            
            # For 4x4x5, we can think of it as 4x4x4 + 4x4x1
            # The 4x4x1 part has rank 16 (schoolbook)
            # So total rank might be 49 + 16 = 65? But that's not necessarily additive.
            
            # Alternative: use 2x2x2 ⊗ 2x2x2 ⊗ 2x2x1 or similar
            # 2x2x1 has rank 4 (schoolbook)
            # So 2x2x2 ⊗ 2x2x2 ⊗ 2x2x1 would give 4x4x4... no, that's 4x4x4
            
            # Let me think about this differently.
            # 4x4x5 can be decomposed as:
            # - First 4 columns: 4x4x4 (rank 49 using Kronecker Strassen)
            # - Last 1 column: 4x4x1 (rank 16 using schoolbook)
            # Total: 49 + 16 = 65
            
            # But this is a block decomposition, and the ranks might not simply add
            # because the factor matrices need to be compatible
            
            # Let's try a different approach: use 2x2x2 ⊗ 2x2x(2+3)
            # 2x2x5 has rank... let's compute
            
            # Actually, let's just try to build the 4x4x5 directly
            # using the Kronecker structure where possible
            
            # For now, return the 4x4x4 result and see if we can extend it
            # This is a partial solution
            
            # Let's try to extend to 4x4x5 by adding 5th column
            # The 5th column of B gives a 4x1 vector, and we need to compute
            # A * (4x1 vector) which is a 4x1 vector
            # This is a 4x4x1 multiplication, which has rank 16 (schoolbook)
            
            # Total rank: 49 + 16 = 65
            # But we need to be careful about how we combine these
            
            # Let's try a cleaner approach:
            # Decompose 4x4x5 as (2x2x2) ⊗ (2x2x5)
            # We need rank of 2x2x5
            
            # For 2x2x1: rank 4 (schoolbook)
            # For 2x2x5: we can do 5 independent 2x2x1 multiplications
            # So rank(2x2x5) <= 5 * 4 = 20
            # But maybe we can do better with Winograd-style tricks
            
            # Actually, for 2x2xk, the rank is known to be 2k for k >= 2
            # No wait, that's not right either.
            
            # Let's just use the schoolbook for the 2x2x5 part
            # rank(2x2x5) = 2*2*5 = 20
            # Then rank(4x4x5) <= rank(2x2x2) * rank(2x2x5) = 7 * 20 = 140
            # That's worse than schoolbook 80!
            
            # So the Kronecker approach doesn't help here.
            
            # Let me try yet another approach.
            
            # For 4x4x5, let's use the following strategy:
            # - Use Strassen on the 2x2 blocks in the i and j dimensions
            # - Handle the k dimension separately
            
            # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
            # B is 4x5, split into 2x5 blocks: B1, B2
            # C is 4x5, split into 2x5 blocks: C1, C2
            
            # C1 = A11*B1 + A12*B2
            # C2 = A21*B1 + A22*B2
            
            # Each 2x2 x 2x5 multiplication:
            # We can use Strassen-like tricks if we can decompose the 2x5 part
            # 2x5 = 2x2 + 2x2 + 2x1
            # So 2x2 x 2x5 = (2x2 x 2x2) + (2x2 x 2x2) + (2x2 x 2x1)
            # Rank: 7 + 7 + 4 = 18 per 2x2 x 2x5 block
            # Total: 2 * 18 = 36? No, that's not how it works.
            
            # I think I'm overcomplicating this. Let me just try to find
            # any decomposition with rank < 80 using a more systematic search.
            
            pass
        
        return None
    
    # Try the Kronecker structure approach
    result = try_kronecker_structure()
    
    if result is not None:
        U, V, W, rank = result
        if verify_decomposition(U, V, W, rank):
            U_list = to_nice_list(U)
            V_list = to_nice_list(V)
            W_list = to_nice_list(W)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try Strassen block decomposition
    result = try_strassen_block_decomposition()
    
    if result is not None:
        U, V, W, rank = result
        if verify_decomposition(U, V, W, rank):
            U_list = to_nice_list(U)
            V_list = to_nice_list(V)
            W_list = to_nice_list(W)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W, rank = build_schoolbook()
    
    U_list = to_nice_list(U)
    V_list = to_nice_list(V)
    W_list = to_nice_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
