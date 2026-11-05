import numpy as np
from fractions import Fraction
import time

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank.
    
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 3, 4, 5
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W
    
    def build_decomposition():
        U_parts = []
        V_parts = []
        W_parts = []
        
        def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
            for r in range(U_blk.shape[1]):
                U_parts.append(np.zeros(n * m))
                for i, idx in enumerate(u_idx):
                    U_parts[-1][idx] = U_blk[i, r]
                
                V_parts.append(np.zeros(m * p))
                for i, idx in enumerate(v_idx):
                    V_parts[-1][idx] = V_blk[i, r]
                
                W_parts.append(np.zeros(p * n))
                for i, idx in enumerate(w_idx):
                    W_parts[-1][idx] = W_blk[i, r]
        
        # Block 1: A1_top * B_top_left (2x2x2)
        U_blk, V_blk, W_blk = strassen_2x2x2()
        add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_blk, V_blk, W_blk)
        
        # Block 2: A1_top * B_top_mid (2x2x2)
        U_blk, V_blk, W_blk = strassen_2x2x2()
        add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_blk, V_blk, W_blk)
        
        # Block 3: A1_top * B_top_right (2x2x1)
        U_blk, V_blk, W_blk = schoolbook_2x2x1()
        add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
        
        # Block 4: A1_bottom * B_bottom_left (2x2x2)
        U_blk, V_blk, W_blk = strassen_2x2x2()
        add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_blk, V_blk, W_blk)
        
        # Block 5: A1_bottom * B_bottom_mid (2x2x2)
        U_blk, V_blk, W_blk = strassen_2x2x2()
        add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_blk, V_blk, W_blk)
        
        # Block 6: A1_bottom * B_bottom_right (2x2x1)
        U_blk, V_blk, W_blk = schoolbook_2x2x1()
        add_block([2, 3, 6, 7], [14, 19], [12, 13], U_blk, V_blk, W_blk)
        
        # Block 7: A2_top * B_top_left (1x2x2)
        U_blk, V_blk, W_blk = schoolbook_1x2x2()
        add_block([8, 9], [0, 1, 5, 6], [2, 5], U_blk, V_blk, W_blk)
        
        # Block 8: A2_top * B_top_mid (1x2x2)
        U_blk, V_blk, W_blk = schoolbook_1x2x2()
        add_block([8, 9], [2, 3, 7, 8], [8, 11], U_blk, V_blk, W_blk)
        
        # Block 9: A2_top * B_top_right (1x2x1)
        U_blk, V_blk, W_blk = schoolbook_1x2x1()
        add_block([8, 9], [4, 9], [14], U_blk, V_blk, W_blk)
        
        # Block 10: A2_bottom * B_bottom_left (1x2x2)
        U_blk, V_blk, W_blk = schoolbook_1x2x2()
        add_block([10, 11], [10, 11, 15, 16], [2, 5], U_blk, V_blk, W_blk)
        
        # Block 11: A2_bottom * B_bottom_mid (1x2x2)
        U_blk, V_blk, W_blk = schoolbook_1x2x2()
        add_block([10, 11], [12, 13, 17, 18], [8, 11], U_blk, V_blk, W_blk)
        
        # Block 12: A2_bottom * B_bottom_right (1x2x1)
        U_blk, V_blk, W_blk = schoolbook_1x2x1()
        add_block([10, 11], [14, 19], [14], U_blk, V_blk, W_blk)
        
        U = np.array(U_parts).T
        V = np.array(V_parts).T
        W = np.array(W_parts).T
        
        return U, V, W
    
    U, V, W = build_decomposition()
    current_rank = U.shape[1]
    
    # Verify correctness
    def verify(U, V, W):
        # Build the tensor from decomposition
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # where i in [0, n*m), j in [0, m*p), k in [0, p*n)
        # W[k*n+i, r] means W is indexed as W[k*n+i, r]
        # So for output C[i_out, k_out], we need W[k_out*n + i_out, r]
        
        # The multiplication tensor T[i,j,k] should be 1 if i,j,k correspond to
        # C[i_out, k_out] = sum_j A[i_out, j] * B[j, k_out]
        # i = i_out * m + j_in
        # j = j_in * p + k_out  
        # k = k_out * n + i_out
        
        # So T[i, j, k] = 1 if i%m == j//p and k%n == i//m and j%p == k//n
        # Let's verify: i = i_out*m + j_in, j = j_in*p + k_out, k = k_out*n + i_out
        # i%m = j_in, j//p = j_in ✓
        # k%n = i_out, i//m = i_out ✓
        # j%p = k_out, k//n = k_out ✓
        
        n, m, p = 3, 4, 5
        T_target = np.zeros((n*m, m*p, p*n))
        for i_out in range(n):
            for j_in in range(m):
                for k_out in range(p):
                    i = i_out * m + j_in
                    j = j_in * p + k_out
                    k = k_out * n + i_out
                    T_target[i, j, k] = 1.0
        
        # Compute T from decomposition
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.allclose(T_target, T_decomp, atol=1e-10)
    
    if not verify(U, V, W):
        # Fallback: return schoolbook
        U_school = np.zeros((n*m, n*m*p))
        V_school = np.zeros((m*p, n*m*p))
        W_school = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_school[i * m + j, r] = 1
                    V_school[j * p + k, r] = 1
                    W_school[k * n + i, r] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to reduce rank by finding linearly dependent columns
    # If column r of U is in the span of other columns, and similarly for V,
    # we might be able to remove it
    U_float = U.astype(np.float64)
    V_float = V.astype(np.float64)
    W_float = W.astype(np.float64)
    
    # Try greedy removal: for each column, check if removing it still gives correct decomposition
    # This is expensive, so limit the number of attempts
    max_attempts = 200
    attempts = 0
    removed = []
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        found = False
        for r in range(U.shape[1]):
            if r in removed:
                continue
            # Try removing column r
            mask = [i for i in range(U.shape[1]) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if verify(U_test, V_test, W_test):
                # Successfully removed column r
                U = U_test
                V = V_test
                W = W_test
                removed.append(r)
                found = True
                break
        
        if not found:
            break
    
    current_rank = U.shape[1]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
