import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # We want to find a decomposition of rank r such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Strategy: Try to find a low-rank decomposition.
    # The schoolbook rank is n*m*p = 48.
    # Known bounds for 2x4x6:
    # We can try to exploit the structure.
    # Let's attempt to find a rank 40 or lower decomposition.
    # A common technique is to use the "border rank" or specific structured decompositions.
    # However, finding an exact integer/half-integer decomposition of minimal rank is hard.
    
    # Let's try a systematic search for a rank r < 48.
    # We can fix U and V to have a specific sparse pattern and solve for W, or vice versa.
    
    # Let's try rank 40.
    # Actually, let's try to construct a decomposition using a known lower bound or a clever trick.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x4x4, we can use 2 blocks of 2x2x2, rank 14.
    # For 2x4x6, we can use 3 blocks of 2x4x2? No, B is 4x6.
    # Split B into three 4x2 matrices: B1, B2, B3.
    # C = A @ B1 + A @ B2 + A @ B3.
    # Each A @ B_i is 2x4 @ 4x2 = 2x2.
    # Rank of 2x4 @ 4x2 multiplication?
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
    # Let A be 2x4, B be 4x2.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k].
    # We can decompose this.
    # If we use Strassen-like for 2x2x2, can we extend?
    # 2x4 @ 4x2 can be seen as 2x(2+2) @ (2+2)x2.
    # Split A into A1 (2x2), A2 (2x2). B into B1 (2x2), B2 (2x2).
    # C = A1 B1 + A2 B2.
    # Each term is 2x2 @ 2x2.
    # If we use rank 7 for each 2x2 @ 2x2, total rank 14.
    # So 2x4 @ 4x2 has rank at most 14.
    # Then 2x4 @ 4x6 = 3 * (2x4 @ 4x2) has rank at most 3 * 14 = 42.
    
    # Can we do better?
    # Maybe the 2x4 @ 4x2 multiplication has rank less than 14?
    # Let's check the rank of 2x4 @ 4x2.
    # This is the tensor T_{ijk} where i in 0..1, j in 0..3, k in 0..1.
    # We can try to find its minimal rank.
    # If rank is 13, total is 39.
    # If rank is 12, total is 36.
    
    # Let's try to find a decomposition for 2x4 @ 4x2 with rank 13.
    # Or just use the 14 and see if we can optimize the combination.
    
    # Let's implement the 2x2 @ 2x2 Strassen decomposition first.
    # Strassen for 2x2 @ 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # This gives rank 7 for 2x2 @ 2x2.
    
    # Now for 2x4 @ 4x2:
    # A = [[a0, a1, a2, a3], [b0, b1, b2, b3]]
    # B = [[c0, d0], [c1, d1], [c2, d2], [c3, d3]]
    # C = [[sum a_i B_i0, sum a_i B_i1], [sum b_i B_i0, sum b_i B_i1]]
    # Split A into A1 = [[a0, a1], [b0, b1]] and A2 = [[a2, a3], [b2, b3]].
    # Split B into B1 = [[c0, d0], [c1, d1]] and B2 = [[c2, d2], [c3, d3]].
    # C = A1 B1 + A2 B2.
    # Use Strassen for A1 B1 and A2 B2.
    # This uses 7 + 7 = 14 terms.
    
    # Let's try to find a better decomposition for 2x4 @ 4x2.
    # Maybe rank 13?
    # We can try to solve for a rank 13 decomposition of 2x4 @ 4x2.
    
    # Let's write a function to find a decomposition of a given rank for small tensors.
    
    def find_decomp(n, m, p, rank, max_iter=1000):
        # T[i,j,k] = 1
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # We fix U and V to be random or structured, then solve for W.
        # Or we can use a more robust method.
        
        # Let's try a simple approach: fix U and V to be identity-like, solve for W.
        # This might not work for low rank.
        
        # Let's use the fact that we can set U and V to have specific values.
        # For 2x4 @ 4x2, n=2, m=4, p=2.
        # U is 8 x rank.
        # V is 8 x rank.
        # W is 4 x rank.
        
        # Let's try to find a rank 13 decomposition for 2x4 @ 4x2.
        # We can use a grid search or random search.
        # Given the time limit, let's try a few random initializations and solve for W.
        
        import random
        random.seed(42)
        
        for _ in range(max_iter):
            # Initialize U and V randomly
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Set some entries to 1, -1, 0 randomly
            for c in range(rank):
                for idx in range(n*m):
                    if random.random() < 0.5:
                        U[idx, c] = random.choice([1, -1])
                for idx in range(m*p):
                    if random.random() < 0.5:
                        V[idx, c] = random.choice([1, -1])
            
            # Now solve for W
            # For each (i,k), sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
            # This is a linear system for W.
            # Let's form the matrix M where M[(i,k), c] = sum_j U[i*m+j, c] V[j*p+k, c]
            # Then M @ W_col = 1 for each (i,k).
            
            M = np.zeros((n*p, rank))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            M[i*p+k, c] += U[i*m+j, c] * V[j*p+k, c]
            
            # Now we have M @ W = 1
            # W is (p*n, rank).
            # We need to solve for W.
            # This is an underdetermined system if rank > n*p.
            # We can use least squares or find a particular solution.
            
            try:
                W, residuals, rank_w, s = np.linalg.lstsq(M.T, np.ones(n*p), rcond=None)
                # W is (rank, n*p)
                W = W.T # (n*p, rank)
                
                # Check if the solution is exact
                # Reconstruct T
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            T[i,j,k] = val
                
                # Check if T is all 1s
                if np.allclose(T, 1.0, atol=1e-9):
                    # We found a decomposition
                    # Convert to integer/half-integer if possible
                    # For now, return as is
                    return U, V, W
            except:
                continue
        
        return None
    
    # Let's try to find a rank 13 decomposition for 2x4 @ 4x2
    U13, V13, W13 = find_decomp(2, 4, 2, 13, max_iter=50)
    
    if U13 is not None:
        # We have a rank 13 decomposition for 2x4 @ 4x2
        # Now we need to combine three such decompositions for 2x4 @ 4x6
        # B is split into B1, B2, B3 (each 4x2)
        # A is the same for all.
        # So we can reuse the same U for all three?
        # No, the U in the tensor decomposition corresponds to the entries of A.
        # In the 2x4 @ 4x2 decomposition, U[i*m+j, c] corresponds to A[i,j].
        # In the 2x4 @ 4x6 decomposition, U[i*m+j, c] corresponds to A[i,j].
        # So the U part is the same!
        # The V part corresponds to B[j,k].
        # In 2x4 @ 4x2, V[j*p+k, c] corresponds to B[j,k] where k in 0..1.
        # In 2x4 @ 4x6, V[j*p+k, c] corresponds to B[j,k] where k in 0..5.
        # So we can construct the global V by placing the V13 blocks.
        # Similarly, W corresponds to C[i,k].
        # In 2x4 @ 4x2, W[k*n+i, c] corresponds to C[i,k] where k in 0..1.
        # In 2x4 @ 4x6, W[k*n+i, c] corresponds to C[i,k] where k in 0..5.
        
        # Let's construct the global decomposition.
        # Global rank = 3 * 13 = 39.
        rank_global = 3 * 13
        U_global = np.zeros((n*m, rank_global))
        V_global = np.zeros((m*p, rank_global))
        W_global = np.zeros((p*n, rank_global))
        
        # For each block b in 0..2:
        # U_global[:, b*13:(b+1)*13] = U13
        # V_global: for j in 0..3, k in b*2..b*2+1, V_global[j*6+k, b*13+c] = V13[j*2+k-b*2, c]
        # W_global: for k in b*2..b*2+1, i in 0..1, W_global[k*2+i, b*13+c] = W13[(k-b*2)*2+i, c]
        
        for b in range(3):
            # U
            U_global[:, b*13:(b+1)*13] = U13
            
            # V
            for j in range(4):
                for k in range(2):
                    V_global[j*6 + b*2 + k, b*13:(b+1)*13] = V13[j*2 + k, :]
            
            # W
            for k in range(2):
                for i in range(2):
                    W_global[(b*2 + k)*2 + i, b*13:(b+1)*13] = W13[k*2 + i, :]
        
        # Verify
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank_global):
                        val += U_global[i*m+j, c] * V_global[j*p+k, c] * W_global[k*n+i, c]
                    T[i,j,k] = val
        
        if np.allclose(T, 1.0, atol=1e-9):
            # Convert to list of lists
            U_list = U_global.tolist()
            V_list = V_global.tolist()
            W_list = W_global.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook rank 48
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
