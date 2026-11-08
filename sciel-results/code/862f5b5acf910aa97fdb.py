"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 3, 6
    # Rank 18 construction:
    # We exploit the structure: C[i,k] = sum_j A[i,j] B[j,k]
    # We can group terms. Let's use a block-based approach.
    #
    # Actually, let's use a known lower bound construction or a systematic one.
    # For 2x3x3, rank is 7 (Strassen-like).
    # For 2x3x6, we can view it as two 2x3x3 multiplications side by side.
    # If we compute C1 = A * B1 (2x3) and C2 = A * B2 (2x3) where B = [B1 B2] (3x6)
    # Rank(2x3x3) = 7? No, that's not right. Let me check.
    # Actually, the border rank of matrix multiplication 2x3x3 is not 7.
    # Let's just do a direct optimization or a known construction.
    #
    # Let's try a simple construction:
    # We know that multiplying a 2xm matrix by an mxp matrix has rank at most 2*m*p.
    # But we can do better.
    #
    # Let's use the fact that we can share some of the multiplications.
    # A is 2x3. B is 3x6.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # This is 12 dot products of length 3.
    #
    # Let's try to find a rank-18 decomposition by hand or by small search.
    # Or better, let's use a programmatic search for a valid decomposition.
    # Since the space is small, we can search for a sparse decomposition.
    #
    # Actually, let's just return the schoolbook rank 36 if we can't find better quickly.
    # But we can do better. Let's try to find a rank-27 or lower.
    #
    # Let's use a simple iterative solver to find a low-rank decomposition.
    # We'll use the "power iteration" style or random restarts.
    #
    # For now, let's implement a simple search that tries to find a valid decomposition
    # with rank < 36.
    
    def verify(U, V, W, n, m, p, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                                if abs(s - expected) > 1e-9:
                                    return False
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Let's try a rank 18 construction.
    # Idea: Use the fact that 2x3x6 can be decomposed.
    # Let's just do a quick random search for rank 20-30.
    # If we find one, great. Otherwise, return schoolbook.
    
    start = time.time()
    deadline = start + 100
    
    # First, let's try a known construction for 2x3x3 which has rank 7?
    # Actually, let's just search.
    
    best_result = schoolbook(n, m, p)
    
    # Try to find a better rank
    for rank in range(20, 36):
        if time.time() > deadline:
            break
        for _ in range(50):
            if time.time() > deadline:
                break
            # Random sparse initialization
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for c in range(rank):
                # Each column is sparse
                for i in range(n*m):
                    if np.random.rand() < 0.3:
                        U[i, c] = np.random.choice([-1, 1])
                for i in range(m*p):
                    if np.random.rand() < 0.3:
                        V[i, c] = np.random.choice([-1, 1])
                for i in range(p*n):
                    if np.random.rand() < 0.3:
                        W[i, c] = np.random.choice([-1, 1])
            
            if verify(U, V, W, n, m, p, rank):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        if best_result["U"][0].count(1) < 36:  # Found something better
            break
    
    return best_result
# EVOLVE-BLOCK-END
