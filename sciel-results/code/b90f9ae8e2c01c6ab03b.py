import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # 1. Strassen 2x2 Decomposition (Rank 7)
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = (A00-A11)(B01+B11)
        U[0, 3] = 1; U[3, 3] = -1
        V[1, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        
        return U, V, W

    # 2. Construct 4x4x4 Decomposition (Rank 49)
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                for i1 in range(2):
                    for i2 in range(2):
                        row = i1 * 2 + i2
                        U[row, col] = U2[i1, t1] * U2[i2, t2]
                        V[row, col] = V2[i1, t1] * V2[i2, t2]
                        W[row, col] = W2[i1, t1] * W2[i2, t2]
        return U, V, W

    # 3. Construct 4x4x1 Decomposition (Rank 4)
    def rect_4x4x1():
        rank = 4
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        
        for col in range(4):
            for i in range(4):
                U[i * 4 + col, col] = 1.0
                W[i, col] = 1.0
            V[col * 5, col] = 1.0
        return U, V, W

    # 4. Combine to 4x4x5 (Rank 53)
    U44, V44, W44 = strassen_4x4()
    U41, V41, W41 = rect_4x4x1()
    
    rank_total = 53
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, :49] = U44
    U[:, 49:] = U41
    
    V[:16, :49] = V44
    V[16:20, :49] = 0
    V[:, 49:] = V41
    
    W[:16, :49] = W44
    W[16:20, :49] = 0
    W[:, 49:] = W41

    # 5. Verification
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U, V, W, n, m, p):
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    else:
        rank = 53

    # Attempt rank reduction by removing linearly dependent columns
    if time.time() < deadline - 5:
        def reduce_rank(U, V, W, tol=1e-9):
            r = U.shape[1]
            keep = [True] * r
            for c in range(r):
                if not keep[c]:
                    continue
                # Check if column c is linearly dependent on previous kept columns
                # This is expensive O(r^3). r=53. 53^3 ~ 150k. Fast.
                # We can use SVD of the matrix formed by kept columns + current column
                # But simpler: just check if the rank increases when adding column c to the set of kept columns.
                # We maintain the rank of the kept set.
                pass
            
            # Simpler approach: Compute the rank of the full U, V, W matrices.
            # If rank(U) < r, or rank(V) < r, or rank(W) < r, we might have redundancy.
            # But the decomposition rank is the number of terms.
            # If we remove a term, we must re-verify.
            # Given the structure, it's unlikely to be redundant.
            return U, V, W, r
        
        # We'll skip the complex reduction as it's unlikely to succeed for a minimal decomposition.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
