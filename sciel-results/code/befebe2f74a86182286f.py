import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n, m, p = 2, 3, 6
    
    # Dimensions for the tensor T_{a,b,c,d,i,k}
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Indices:
    # a in 0..n-1, b in 0..m-1 -> U index u = a*m + b
    # c in 0..m-1, d in 0..p-1 -> V index v = c*p + d
    # k in 0..p-1, i in 0..n-1 -> W index w = k*n + i

    # Strategy:
    # Decompose M_{2,3,6} using the structure of M_{2,2,2} (Strassen) and M_{1,3,3} (Identity).
    # M_{2,3,6} can be seen as a block matrix multiplication where the 3x6 matrix is viewed as three 3x2 blocks.
    # However, a simpler and robust approach is to use the Kronecker product of decompositions.
    # Let's use the known rank 7 decomposition for M_{2,2,2} (Strassen).
    # And the rank 3 decomposition for M_{1,3,3} (which is just identity, rank 3).
    # Wait, M_{2,3,6} is not a direct Kronecker product of M_{2,2,2} and M_{1,3,3} because the dimensions don't align perfectly in a simple Kronecker sense (2x3x6 vs 2x2x2 and 1x3x3).
    
    # Let's look at the dimensions:
    # Input A is 2x3. Input B is 3x6. Output C is 2x6.
    # We can split B into three 3x2 matrices: B1, B2, B3.
    # Then C = A * [B1, B2, B3] = [A*B1, A*B2, A*B3].
    # Each A*Bj is a 2x2 matrix multiplication.
    # So we need 3 independent 2x2 multiplications.
    # If we use Strassen for each 2x2, the rank is 3 * 7 = 21.
    
    # Let's construct this rank 21 decomposition.
    # Strassen decomposition for M_{2,2,2}:
    # U_s (4, 7), V_s (4, 7), W_s (4, 7)
    # The standard Strassen indices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We will define U_s, V_s, W_s for the 2x2 case.
    # U index u = a*2 + b (a,b in {0,1})
    # V index v = c*2 + d (c,d in {0,1})
    # W index w = k*2 + i (k,i in {0,1})
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    W_s[1, 0] = 1; W_s[2, 0] = 1
    
    # M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # M3
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[0, 2] = 1; W_s[1, 2] = 1
    
    # M4
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[2, 3] = 1; W_s[3, 3] = 1
    
    # M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[1, 4] = 1; W_s[3, 4] = 1
    
    # M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = 1
    
    # M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[1, 6] = 1

    # Now construct U, V, W for M_{2,3,6}
    # U: (6, 21)
    # V: (18, 21)
    # W: (12, 21)
    
    # The 3 blocks of B correspond to columns (0,1), (2,3), (4,5) of B.
    # For block j (j=0,1,2), we use the Strassen decomposition.
    # U for block j: The input A is the same for all blocks.
    # So U[:, j*7:(j+1)*7] should be U_s?
    # Wait, the input to the j-th 2x2 multiplication is A (2x2) and B_j (2x2).
    # But A is 2x3. We only use the first 2 columns of A for B1? No.
    # A is 2x3. B is 3x6.
    # C = A B.
    # C[:, 0:2] = A[:, 0:3] B[0:3, 0:2]
    # C[:, 2:4] = A[:, 0:3] B[0:3, 2:4]
    # C[:, 4:6] = A[:, 0:3] B[0:3, 4:6]
    
    # The Strassen decomposition works for 2x2 matrices.
    # Here we are multiplying a 2x3 matrix by a 3x2 matrix.
    # This is NOT a 2x2 multiplication.
    # So we cannot simply use Strassen on the blocks.
    
    # We need a decomposition for M_{2,3,2}.
    # Let's find the rank of M_{2,3,2}.
    # Schoolbook: 2*3*2 = 12.
    # Can we do better?
    # M_{2,3,2} = M_{2,2,2} \oplus M_{2,1,2}? No.
    # M_{2,3,2} can be viewed as multiplying a 2x3 matrix by a 3x2 matrix.
    # This is equivalent to two 2x3 matrix multiplications? No.
    
    # Let's stick to the safe rank 34 solution if we can't find a better one quickly.
    # But wait, the current program's fallback is rank 34.
    # The score is 1.0588 which corresponds to rank 34 (seed 36 / 34 = 1.0588).
    # If we can find a rank 33 solution, score = 36/33 = 1.0909.
    # If we can find a rank 32 solution, score = 36/32 = 1.125.
    
    # Let's try to construct a rank 33 solution by combining a rank 11 M_{2,3,2} block?
    # We established M_{2,3,2} rank is likely > 11.
    
    # Alternative: Use the rank 17 solution for M_{2,3,3} (from the current program's build_rank34).
    # The current program uses two copies of a rank 17 M_{2,3,3} to make rank 34 M_{2,3,6}.
    # Is there a way to share computations between the two M_{2,3,3} blocks?
    # M_{2,3,6} = M_{2,3,3} \oplus M_{2,3,3}.
    # The inputs A are the same. The inputs B are different (B1, B2).
    # The outputs C are different (C1, C2).
    # If we can find a decomposition where the U matrix is shared, we save rank.
    # U is (6, r). V is (18, r). W is (12, r).
    # If we use the same U for both blocks, we need U to be (6, r).
    # But the rank adds up unless there's overlap in the U space.
    
    # Let's try a different structural approach.
    # What if we use a rank 11 decomposition for M_{2,2,2} (impossible, min is 7)?
    # What if we use a rank 10 decomposition for M_{2,3,2}?
    
    # Let's try to verify if a rank 33 solution exists.
    # Lower bound for M_{2,3,6}:
    # Rank(M_{2,3,6}) >= Rank(M_{2,3,3}) + Rank(M_{2,3,3}) - Rank(M_{2,3,3} \cap M_{2,3,3})? No.
    
    # Let's just return the rank 34 solution, but optimized for speed.
    # The current program takes too long due to the failed search for rank 33.
    # We will skip the search and just return the known rank 34 solution.
    # This ensures we get at least 1.0588.
    
    def build_rank34():
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    U, V, W = build_rank34()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
