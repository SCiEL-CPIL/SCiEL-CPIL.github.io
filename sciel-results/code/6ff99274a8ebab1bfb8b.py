import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s
    
    # 1. Define Strassen factors for 2x2x2 using the verified formulation
    # Each factor r has vectors u_r (for A), v_r (for B), w_r (for C), each of length 4
    # Indexing: [i*2+j] for 2x2 matrix entry (i,j)
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1: (A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2: (A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3: A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # M4: A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5: (A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6: (A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7: (A01-A11)(B10+B11)
    ]
    
    # Verify Strassen correctness
    def _verify_strassen():
        U3 = np.zeros((2, 2, 7))
        V3 = np.zeros((2, 2, 7))
        W3 = np.zeros((2, 2, 7))
        for s in range(7):
            u, v, w = S[s]
            for i in range(2):
                for j in range(2):
                    U3[i, j, s] = u[i*2+j]
                    V3[i, j, s] = v[i*2+j]
                    W3[i, j, s] = w[i*2+j]
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if not _verify_strassen():
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 2. Build 4x4x4 factors using recursion (rank 49)
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for s in range(7):
        u_block, v_block, w_block = S[s]
        for r in range(7):
            u_scalar, v_scalar, w_scalar = S[r]
            R = 7 * s + r
            for i in range(4):
                for j in range(4):
                    U4[i, j, R] = u_block[i] * u_scalar[j] if False else 0
                    # Need to handle block structure properly
                    # A is 4x4, split into four 2x2 blocks
                    # Block (bi, bj) contains entries A[2*bi+ei, 2*bj+ej]
                    # u_block[bi*2+bj] is coefficient for block (bi,bj)
                    # u_scalar[ei*2+ej] is coefficient within block
                    # So U4[i,j,R] = sum over bi,bj,ei,ej where i=2*bi+ei, j=2*bj+ej
                    # Since i,j uniquely determine bi,bj,ei,ej:
                    # bi = i//2, ei = i%2, bj = j//2, ej = j%2
                    pass
            
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    ei = i % 2
                    bj = j // 2
                    ej = j % 2
                    U4[i, j, R] = u_block[bi*2+bj] * u_scalar[ei*2+ej]
            
            for j in range(4):
                for k in range(4):
                    bi = j // 2
                    ei = j % 2
                    bj = k // 2
                    ej = k % 2
                    V4[j, k, R] = v_block[bi*2+bj] * v_scalar[ei*2+ej]
            
            for k in range(4):
                for i in range(4):
                    bi = k // 2
                    ei = k % 2
                    bj = i // 2
                    ej = i % 2
                    W4[k, i, R] = w_block[bi*2+bj] * w_scalar[ei*2+ej]
    
    # Verify 4x4x4
    result4 = np.einsum('ijr,jkr,kir->ijk', U4, V4, W4)
    if not np.allclose(result4, 1.0, atol=1e-10):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 3. Build 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half (k < 4): B columns 0-3
    for i in range(4):
        for j in range(4):
            U[i*4 + j, 0:49] = U4[i, j, :]
    
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j, k, :]
    
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k, i, :]
    
    # Second half (k >= 4): B columns 4-7
    for i in range(4):
        for j in range(4):
            U[i*4 + j, 49:98] = U4[i, j, :]
    
    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            V[j*8 + k, 49:98] = V4[j, kp, :]
    
    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            W[k*4 + i, 49:98] = W4[kp, i, :]
    
    # Verify 4x4x8
    def _verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if _verify(U, V, W):
        # Try to reduce rank using linear algebra
        if time.time() < deadline - 30:
            best_rank = 98
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Strategy: try to find linear dependencies among the 98 rank-1 terms
            # For each term r, we have vectors u_r (16-dim), v_r (32-dim), w_r (32-dim)
            # Check if any subset is linearly dependent in a way that allows reduction
            
            # Approach: SVD-based rank reduction
            # Reshape the tensor and try to find a lower-rank decomposition
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # Try removing terms one at a time and solving for the rest
            # This is expensive, so try a limited number of attempts
            
            np.random.seed(42)
            max_attempts = 200
            for attempt in range(max_attempts):
                if time.time() > deadline - 20:
                    break
                
                # Pick a term to remove
                r_remove = np.random.randint(98)
                keep_indices = [r for r in range(98) if r != r_remove]
                
                # Try to find new factors for the remaining 97 terms that still work
                # This is a hard problem; use a simple heuristic
                # Check if the removed term is "redundant" by seeing if its contribution
                # can be absorbed by other terms
                
                # Compute the contribution of term r_remove
                contrib = np.zeros((4, 4, 8))
                for i in range(4):
                    for j in range(4):
                        u_val = U[i*4+j, r_remove]
                        if u_val != 0:
                            for k in range(8):
                                v_val = V[j*8+k, r_remove]
                                w_val = W[k*4+i, r_remove]
                                if v_val != 0 and w_val != 0:
                                    contrib[i,j,k] += u_val * v_val * w_val
                
                # If the contribution is zero, we can remove it
                if np.allclose(contrib, 0, atol=1e-10):
                    # Remove this term
                    new_rank = 97
                    new_U = np.zeros((16, new_rank))
                    new_V = np.zeros((32, new_rank))
                    new_W = np.zeros((32, new_rank))
                    
                    col = 0
                    for r in keep_indices:
                        new_U[:, col] = U[:, r]
                        new_V[:, col] = V[:, r]
                        new_W[:, col] = W[:, r]
                        col += 1
                    
                    if _verify(new_U, new_V, new_W):
                        best_rank = new_rank
                        best_U, best_V, best_W = new_U, new_V, new_W
                        U, V, W = new_U, new_V, new_W
                        continue
                
                # Try removing 2 terms
                if attempt % 10 == 0:
                    r_remove2 = np.random.randint(98)
                    if r_remove2 != r_remove:
                        keep_indices2 = [r for r in range(98) if r not in [r_remove, r_remove2]]
                        
                        # Check combined contribution
                        contrib2 = np.zeros((4, 4, 8))
                        for r in [r_remove, r_remove2]:
                            for i in range(4):
                                for j in range(4):
                                    u_val = U[i*4+j, r]
                                    if u_val != 0:
                                        for k in range(8):
                                            v_val = V[j*8+k, r]
                                            w_val = W[k*4+i, r]
                                            if v_val != 0 and w_val != 0:
                                                contrib2[i,j,k] += u_val * v_val * w_val
                        
                        if np.allclose(contrib2, 0, atol=1e-10):
                            new_rank = 96
                            new_U = np.zeros((16, new_rank))
                            new_V = np.zeros((32, new_rank))
                            new_W = np.zeros((32, new_rank))
                            
                            col = 0
                            for r in keep_indices2:
                                new_U[:, col] = U[:, r]
                                new_V[:, col] = V[:, r]
                                new_W[:, col] = W[:, r]
                                col += 1
                            
                            if _verify(new_U, new_V, new_W):
                                best_rank = new_rank
                                best_U, best_V, best_W = new_U, new_V, new_W
                                U, V, W = new_U, new_V, new_W
                                break
            
            if best_rank < 98:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
