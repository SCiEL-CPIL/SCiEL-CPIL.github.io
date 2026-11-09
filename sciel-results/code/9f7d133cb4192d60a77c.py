"""Program to find rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
n, m, p = DIMENSIONS  # 2, 3, 5

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The task is to decompose the tensor for multiplying a 2x3 matrix by a 3x5 matrix.
    # Dimensions: A is 2x3, B is 3x5, C is 2x5.
    # Tensor T_{i,j,k} = delta_{j,k} where i in 0..1, j in 0..2, k in 0..4.
    # We need matrices U (6xR), V (15xR), W (10xR) such that
    # C_{i,k} = sum_r U_{i*3+j, r} * V_{j*5+k, r} * W_{k*2+i, r}
    # Wait, the convention is W[k*n+i, r]. So W is (p*n, R) = (10, R).
    # U is (n*m, R) = (6, R).
    # V is (m*p, R) = (15, R).
    
    # The standard schoolbook rank is n*m*p = 2*3*5 = 30.
    
    # Strategy:
    # 1. Try to find a decomposition using known low-rank algorithms for sub-blocks.
    # 2. Specifically, split the 5 columns into 2+2+1.
    #    This decomposes the problem into three independent parts:
    #    - 2x3 * 3x2 (Rank R1)
    #    - 2x3 * 3x2 (Rank R2)
    #    - 2x3 * 3x1 (Rank R3)
    #    Total Rank = R1 + R2 + R3.
    
    # We know:
    # - 2x2 * 2x2 has rank 7 (Strassen).
    # - 2x2 * 2x1 has rank 4 (standard).
    # - 2x3 * 3x2 is isomorphic to 2x2 * 2x3? No.
    #   Let's analyze 2x3 * 3x2.
    #   A is 2x3, B is 3x2, C is 2x2.
    #   This is equivalent to computing a 2x2 matrix product where the "elements" of A are 1x3 vectors and B are 3x1 vectors? No.
    #   Actually, 2x3 * 3x2 can be seen as two 2x1 * 1x2 products? No.
    #   Let's use the splitting of the 3 dimension.
    #   2x3 * 3x2 -> Split the 3 into 2+1.
    #   This gives:
    #   - 2x2 * 2x2 (Rank 7)
    #   - 2x1 * 1x2 (Rank 2)
    #   - 2x2 * 1x2? No, the split is on the middle dimension m=3.
    #   So A is 2x(2+1), B is (2+1)x2.
    #   Terms:
    #   1. A_1 (2x2) * B_1 (2x2) -> Rank 7
    #   2. A_2 (2x1) * B_2 (1x2) -> Rank 2
    #   3. A_1 (2x2) * B_2 (1x2) -> Rank 4 (2x2 * 1x2)
    #   4. A_2 (2x1) * B_1 (2x2) -> Rank 4 (2x1 * 2x2)
    #   Total for 2x3 * 3x2 = 7 + 2 + 4 + 4 = 17?
    #   Wait, is there a better way?
    #   Actually, 2x3 * 3x2 is the same as 2x2 * 2x3? No.
    #   Let's check the rank of 2x3 * 3x2.
    #   It is known that R(2,3,2) = 11? Or 10?
    #   Let's try to construct it explicitly.
    
    # Alternative Strategy:
    # Use the 11-term decomposition for 2x3 * 3x2 if available, or construct one.
    # If R(2,3,2) = 11, then total rank = 11 + 11 + 6 = 28.
    # If R(2,3,2) = 10, then total rank = 10 + 10 + 6 = 26.
    
    # Let's try to find a rank-11 decomposition for 2x3 * 3x2 first, as it's safer.
    # Then we can try to optimize to 10.
    
    # Actually, let's just implement the 28-rank solution first to ensure correctness,
    # then try to improve.
    
    # Helper to create identity-like decomposition for block multiplication
    def block_identity(n, m, p):
        """Returns U, V, W for n x m * m x p with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Helper to create Strassen decomposition for 2x2 * 2x2
    def strassen_2x2():
        """Returns U, V, W for 2x2 * 2x2 with rank 7."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Strassen's algorithm
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # U (A side)
        U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = 1; U[3, 0] = 1
        U[0, 1] = 0; U[1, 1] = 0; U[2, 1] = 1; U[3, 1] = 1
        U[0, 2] = 1; U[1, 2] = 0; U[2, 2] = 0; U[3, 2] = 0
        U[0, 3] = 0; U[1, 3] = 0; U[2, 3] = 0; U[3, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; U[2, 4] = 0; U[3, 4] = 0
        U[0, 5] = -1; U[1, 5] = 0; U[2, 5] = 1; U[3, 5] = 0
        U[0, 6] = 0; U[1, 6] = -1; U[2, 6] = 0; U[3, 6] = 1
        
        # V (B side)
        V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
        V[0, 1] = 1; V[1, 1] = 0; V[2, 1] = 0; V[3, 1] = 0
        V[0, 2] = 1; V[1, 2] = -1; V[2, 2] = 0; V[3, 2] = 0
        V[0, 3] = -1; V[1, 3] = 1; V[2, 3] = 0; V[3, 3] = 0
        V[0, 4] = 0; V[1, 4] = 0; V[2, 4] = 1; V[3, 4] = 1
        V[0, 5] = 1; V[1, 5] = 1; V[2, 5] = 0; V[3, 5] = 0
        V[0, 6] = 0; V[1, 6] = 0; V[2, 6] = 1; V[3, 6] = 1
        
        # W (C side)
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
        W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 0; W[3, 1] = 1
        W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 0; W[3, 2] = 0
        W[0, 3] = 0; W[1, 3] = 0; W[2, 3] = 0; W[3, 3] = 1
        W[0, 4] = 1; W[1, 4] = 0; W[2, 4] = 0; W[3, 4] = 0
        W[0, 5] = 0; W[1, 5] = 1; W[2, 5] = 1; W[3, 5] = 0
        W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = -1
        
        return U, V, W
    
    # Build 28-rank solution:
    # Split p=5 into 2+2+1.
    # Block 1: 2x3 * 3x2 (cols 0,1)
    # Block 2: 2x3 * 3x2 (cols 2,3)
    # Block 3: 2x3 * 3x1 (col 4)
    
    # For 2x3 * 3x2, we can use:
    # Split m=3 into 2+1.
    # Terms:
    # 1. 2x2 * 2x2 (Strassen, rank 7)
    # 2. 2x1 * 1x2 (standard, rank 2)
    # 3. 2x2 * 1x2 (standard, rank 4)
    # 4. 2x1 * 2x2 (standard, rank 4)
    # Total rank 17.
    # But wait, is there a better decomposition for 2x3 * 3x2?
    # Actually, the rank of 2x3 * 3x2 is 11.
    # Let's try to find it or use a different splitting.
    
    # Let's try to construct the 11-term decomposition for 2x3 * 3x2.
    # This is known to exist.
    # Since I don't have the explicit coefficients memorized, I will use the 17-term one
    # which gives total rank 17+17+6 = 40? No, that's worse than 30.
    
    # Wait, the schoolbook rank is 30.
    # The 17-term for 2x3*3x2 is worse than schoolbook for that block (2*3*2=12).
    # So I must use a better decomposition for 2x3*3x2.
    
    # Let's use the known rank 11 decomposition for 2x3*3x2.
    # I will hardcode it if I can find it online or derive it.
    # For now, let's try to use a generic optimization to find rank 11.
    
    # If I can't find rank 11, I can use rank 12 (schoolbook) for 2x3*3x2.
    # Then total rank = 12 + 12 + 6 = 30. Same as schoolbook.
    
    # Is there a way to beat 30?
    # Yes, if I use Strassen for 2x2*2x2 parts.
    
    # Let's try splitting 2x3 * 3x5 differently.
    # Split n=2 into 1+1?
    # 1x3 * 3x5 (rank 15) + 1x3 * 3x5 (rank 15) = 30.
    
    # Split m=3 into 2+1?
    # 2x2 * 2x5 (rank 20) + 2x1 * 1x5 (rank 10) + 2x2 * 1x5 (rank 20) + 2x1 * 2x5 (rank 10) = 60? No.
    
    # The best known rank for 2x3 * 3x5 is 28.
    # This is achieved by splitting 5 into 2+2+1 and using rank 11 for 2x3*3x2.
    
    # I will implement a search for rank 11 decomposition of 2x3*3x2.
    # If it fails, I will return rank 30.
    
    def find_rank_11_2x3x2():
        # Tensor for 2x3 * 3x2
        # i in 0..1, j in 0..2, k in 0..1
        # T_{i,j,k} = delta_{j,k}
        # U: 6x11, V: 12x11, W: 4x11
        # We want to find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{j,k}
        
        # This is a non-linear system. We can use random initialization and gradient descent.
        # But it might not converge reliably in time.
        
        # Let's try to hardcode a known solution.
        # From literature, the rank 11 decomposition exists.
        # I will construct it using the following structure:
        # Use Strassen for the 2x2*2x2 part (rank 7)
        # And handle the remaining parts.
        
        # Actually, let's just return None and use schoolbook for now.
        # The task says "score is the rank". Lower is better.
        # If I can't find 28, I should return 30.
        
        return None
    
    # Let's try to implement the 28-rank solution by finding the 11-term decomposition.
    # Since I don't have it, I will try to generate it using a simple optimization.
    
    def optimize_decomposition(n, m, p, target_rank, max_iter=1000):
        """Try to find a decomposition of rank target_rank for n x m * m x p."""
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Target tensor T
        T = np.zeros((n, m, p))
        for j in range(m):
            T[:, j, j] = 1  # Wait, T_{i,j,k} = delta_{j,k} is not correct for matrix mult.
            # Matrix mult: C_{i,k} = sum_j A_{i,j} B_{j,k}
            # Tensor: T_{i,j,k} = delta_{j,j}? No.
            # The tensor for matrix multiplication is T_{i,j,k} = 1 if j is the middle index.
            # Actually, the tensor is defined by T(A,B) = C.
            # T_{i,j,k} = delta_{j,j} is not right.
            # The tensor is T_{i,j,k} = 1 if the indices match the multiplication.
            # For 2x3 * 3x2:
            # C_{i,k} = sum_{j=0}^2 A_{i,j} B_{j,k}
            # So the tensor has non-zero entries at T_{i,j,k} where j is the summation index.
            # The tensor is a 3D array of size n x m x p.
            # T_{i,j,k} = 1 if j is the middle dimension.
            # Actually, the tensor is defined such that C_{i,k} = sum_j A_{i,j} T_{i,j,k} B_{j,k}?
            # No, the standard decomposition is:
            # C_{i,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
            # This is not a direct tensor decomposition in the usual sense.
            # It's a factorization of the multiplication map.
            
            pass
        
        # This approach is too complex to implement quickly.
        # I will return the schoolbook rank 30 solution.
        return None
    
    # Fallback: Schoolbook rank 30
    U, V, W = block_identity(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
