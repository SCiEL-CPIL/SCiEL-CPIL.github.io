"""Kronecker product decomposition for 4x4 by 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    # 1. Construct Strassen's rank-7 decomposition for 2x2x2
    # U, V, W are 4x7
    # Indices for 2x2x2: u = i*2+j, v = j*2+k, w = k*2+i
    # Strassen's 7 products:
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A10 - A00) * (B00 + B10)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 + A01) * B00
    #
    # We need to express these as linear combinations of A_ij and B_jk.
    # U vectors (coeffs for A):
    # M1: A00 -> [1,0,0,0]
    # M2: A00+A10 -> [1,0,1,0]
    # M3: A10-A00 -> [-1,0,1,0]
    # M4: A11 -> [0,0,0,1]
    # M5: A00+A11 -> [1,0,0,1]
    # M6: A01-A11 -> [0,1,0,-1]
    # M7: A00+A01 -> [1,1,0,0]
    
    U_222 = np.array([
        [1, 1, -1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, -1, 0]
    ])
    
    # V vectors (coeffs for B):
    # M1: B01-B11 -> [0,1,0,-1]
    # M2: B11 -> [0,0,0,1]
    # M3: B00+B10 -> [1,0,1,0]
    # M4: B01-B00 -> [-1,1,0,0]
    # M5: B00+B11 -> [1,0,0,1]
    # M6: B10+B11 -> [0,0,1,1]
    # M7: B00 -> [1,0,0,0]
    
    V_222 = np.array([
        [0, 0, 1, -1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [-1, 1, 0, 0, 1, 1, 0]
    ])
    
    # W vectors (coeffs for C):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # W00: [1, 0, 0, 1, -1, 0, 1]
    # W01: [0, 0, 1, 0, 1, 0, 0]
    # W10: [0, 1, 0, 0, 0, 0, 0]
    # W11: [1, -1, 1, 0, 0, 1, 0]
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Verify 2x2x2
    # T = sum U[:,r] V[:,r] W[:,r]
    # Check a few entries
    # C00 = sum U[0,r] V[0,r] W[0,r] ...
    # This is standard Strassen, assumed correct.
    
    # 2. Construct 4x4x4 decomposition via Kronecker product
    # Rank 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            U_444[:, col] = np.kron(U_222[:, r], U_222[:, s])
            V_444[:, col] = np.kron(V_222[:, r], V_222[:, s])
            W_444[:, col] = np.kron(W_222[:, r], W_222[:, s])
            
    # 3. Construct 4x4x3 decomposition via schoolbook
    # Rank 48
    # Columns 4, 5, 6 of B
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((12, 48)) # m*p = 4*3 = 12
    W_443 = np.zeros((12, 48)) # p*n = 3*4 = 12
    
    col_idx = 0
    for i in range(4):
        for j in range(4):
            for k_offset in range(3):
                k = 4 + k_offset
                u = i * 4 + j
                v = j * 7 + k
                w = k * 4 + i
                
                # Map to local indices for 443 block
                # V local index: j * 3 + k_offset
                # W local index: k_offset * 4 + i
                v_local = j * 3 + k_offset
                w_local = k_offset * 4 + i
                
                U_443[u, col_idx] = 1
                V_443[v_local, col_idx] = 1
                W_443[w_local, col_idx] = 1
                col_idx += 1
                
    # 4. Combine
    U_total = np.hstack([U_444, U_443])
    V_total = np.vstack([V_444, V_443])
    W_total = np.vstack([W_444, W_443])
    
    rank = U_total.shape[1]
    
    # Convert to lists for return
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Verification
    if time.time() < deadline - 5:
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        
        # Check all 112 target entries
        # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        # Target: T[i*4+j, j*7+k, k*4+i] = 1
        # Non-target: 0
        
        # We can check a random subset or all if fast enough
        # 112 entries is small.
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    u = i * 4 + j
                    v = j * 7 + k
                    w = k * 4 + i
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val - 1.0) > 1e-9:
                        raise ValueError(f"Target entry ({u},{v},{w}) = {val}, expected 1")
                        
        # Check a few non-targets
        # u=0, v=0, w=1 (i=0,j=0,k=0 vs w=1 -> k=0,i=1? No w=1 is k=0,i=1. Target is w=0 for i=0,j=0,k=0. So w=1 is non-target for u=0,v=0)
        # Actually, just check all non-targets if time permits
        # Total entries 16*28*28 = 12544.
        # 112 targets. 12432 non-targets.
        # Loop is fast enough.
        target_set = set()
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    target_set.add((i*4+j, j*7+k, k*4+i))
                    
        for u in range(16):
            for v in range(28):
                for w in range(28):
                    if (u, v, w) in target_set:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        raise ValueError(f"Non-target entry ({u},{v},{w}) = {val}, expected 0")
                        
    return result
# EVOLVE-BLOCK-END
