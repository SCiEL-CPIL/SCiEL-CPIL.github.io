import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    t0 = time.time()
    deadline = t0 + 140
    
    n, m, p = DIMENSIONS
    
    # Helper to find decomposition for small blocks
    def find_small_decomp(n, m, p, rank, max_iter=1000, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        T = np.ones((n, m, p))
        
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            # Update U
            for r in range(rank):
                U[:, r] = np.sum(T * V.T[r].reshape(m, p) * W.T[r].reshape(p, n), axis=(1, 2)).reshape(n*m)
                U[:, r] /= np.linalg.norm(U[:, r])
            # Update V
            for r in range(rank):
                V[:, r] = np.sum(T * U.T[r].reshape(n, m) * W.T[r].reshape(p, n), axis=(0, 2)).reshape(m*p)
                V[:, r] /= np.linalg.norm(V[:, r])
            # Update W
            for r in range(rank):
                W[:, r] = np.sum(T * U.T[r].reshape(n, m) * V.T[r].reshape(m, p), axis=(0, 1)).reshape(p*n)
                W[:, r] /= np.linalg.norm(W[:, r])
                
        # Verify
        def verify(U, V, W, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                        if abs(s - 1.0) > 1e-6:
                            return False
            return True
            
        if verify(U, V, W, rank):
            # Project to half-integers? No, small blocks might need integers.
            # Try rounding to nearest 0.5
            U2 = np.round(U * 2) / 2.0
            V2 = np.round(V * 2) / 2.0
            W2 = np.round(W * 2) / 2.0
            if verify(U2, V2, W2, rank):
                return U2, V2, W2
            else:
                return U, V, W
        return None

    # 1. Solve 2x3x2 (Rank 5)
    # 2. Solve 2x3x3 (Rank 7)
    
    # We need two 2x3x2 blocks for 2x3x4
    # And one 2x3x3 block for 2x3x3
    
    # Try to find rank 5 for 2x3x2
    U_232, V_232, W_232 = find_small_decomp(2, 3, 2, 5, seed=1)
    if U_232 is None:
        # Fallback to schoolbook for this block
        U_232 = np.zeros((6, 6)); V_232 = np.zeros((6, 6)); W_232 = np.zeros((4, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U_232[i*3+j, col] = 1
                    V_232[j*2+k, col] = 1
                    W_232[k*2+i, col] = 1
                    col += 1
        rank_232 = 6
    else:
        rank_232 = 5
        
    # Try to find rank 7 for 2x3x3
    U_233, V_233, W_233 = find_small_decomp(2, 3, 3, 7, seed=2)
    if U_233 is None:
        U_233 = np.zeros((6, 9)); V_233 = np.zeros((9, 9)); W_233 = np.zeros((6, 9))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_233[i*3+j, col] = 1
                    V_233[j*3+k, col] = 1
                    W_233[k*2+i, col] = 1
                    col += 1
        rank_233 = 9
    else:
        rank_233 = 7
        
    # Assemble
    # Total rank = 2 * rank_232 + rank_233
    total_rank = 2 * rank_232 + rank_233
    
    U = np.zeros((6, total_rank))
    V = np.zeros((21, total_rank))
    W = np.zeros((14, total_rank))
    
    col = 0
    # Block 1: k=0,1
    U[:, col:col+rank_232] = U_232
    V[:4, col:col+rank_232] = V_232 # V is 3x2 = 6 rows? No, V is m*p. For 2x3x2, V is 3x2=6.
    # Wait, V_232 shape is (3*2, 5) = (6,5).
    # In full V (3*7=21), indices 0..5 correspond to j=0..2, k=0..1.
    V[0:6, col:col+rank_232] = V_232
    # W_232 shape is (2*2, 5) = (4,5).
    # In full W (7*2=14), indices 0..3 correspond to k=0..1, i=0..1.
    W[0:4, col:col+rank_232] = W_232
    col += rank_232
    
    # Block 2: k=2,3
    U[:, col:col+rank_232] = U_232
    # V indices: j=0..2, k=2..3 -> rows 6..11
    V[6:12, col:col+rank_232] = V_232
    # W indices: k=2..3, i=0..1 -> rows 4..7
    W[4:8, col:col+rank_232] = W_232
    col += rank_232
    
    # Block 3: k=4,5,6
    U[:, col:col+rank_233] = U_233
    # V indices: j=0..2, k=4..6 -> rows 12..20
    V[12:21, col:col+rank_233] = V_233
    # W indices: k=4..6, i=0..1 -> rows 8..13
    W[8:14, col:col+rank_233] = W_233
    col += rank_233
    
    # Verify final
    def verify_full(U, V, W, rank):
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    s = np.sum(U[i*3+j, :rank] * V[j*7+k, :rank] * W[k*2+i, :rank])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
        
    if verify_full(U, V, W, total_rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U0 = np.zeros((6, 42)); V0 = np.zeros((21, 42)); W0 = np.zeros((14, 42))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    U0[i*3+j, col] = 1
                    V0[j*7+k, col] = 1
                    W0[k*2+i, col] = 1
                    col += 1
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
