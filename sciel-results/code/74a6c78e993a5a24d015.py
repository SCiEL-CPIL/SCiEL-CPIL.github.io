import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 4, 4, 8
    rank_4x4 = 49
    total_rank = 2 * rank_4x4
    
    # Define 2x2 Strassen factors
    # U2: 4x7, index i*2+j
    U2 = np.array([
        [1, 0, 0, 0],  # M3
        [0, 1, 0, 0],  # M7 part? No, let's use the list from above
        [0, 0, 1, 0],  # M6 part
        [0, 0, 0, 1]   # M4
    ])
    # Wait, the previous list was:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [-1, 0, 1, 0]
    # M7: [0, 1, 0, -1]
    
    U2_list = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]
    ]
    # U2 is 4x7. Row index u = i*2+j.
    # U2[u, r] = U2_list[r][u]
    U2 = np.array(U2_list).T # Shape 4x7
    
    V2_list = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [0, 0, 1, -1], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]
    ]
    V2 = np.array(V2_list).T # Shape 4x7
    
    W2_list = [
        [1, 0, 0, 1], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 1], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, -1]
    ]
    W2 = np.array(W2_list).T # Shape 4x7
    
    # Construct 4x4 factors
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for rh in range(7):
        for rl in range(7):
            rt = rh * 7 + rl
            for ih in range(2):
                for jh in range(2):
                    for il in range(2):
                        for jl in range(2):
                            i = ih * 2 + il
                            j = jh * 2 + jl
                            u = i * 4 + j
                            U4[u, rt] = U2[ih * 2 + jh, rh] * U2[il * 2 + jl, rl]
                            
                            # V index
                            # v = j*4 + k
                            # j = jh*2 + jl
                            # k = kh*2 + kl
                            # We need to iterate k too?
                            # No, V4[v, rt] depends on j and k.
                            # The loop above only fixed i, j.
                            # We need to fill V4 and W4 separately or in a more comprehensive loop.
                            
    # The loop above only fills U4. We need to fill V4 and W4.
    # V4[v, rt] = V2[jh*2+kh, rh] * V2[jl*2+kl, rl]
    # v = j*4 + k = (jh*2+jl)*4 + (kh*2+kl)
    
    # Reset and fill all
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for rh in range(7):
        for rl in range(7):
            rt = rh * 7 + rl
            for ih in range(2):
                for jh in range(2):
                    for kh in range(2):
                        for il in range(2):
                            for jl in range(2):
                                for kl in range(2):
                                    i = ih * 2 + il
                                    j = jh * 2 + jl
                                    k = kh * 2 + kl
                                    
                                    u = i * 4 + j
                                    v = j * 4 + k
                                    w = k * 4 + i
                                    
                                    U4[u, rt] += U2[ih * 2 + jh, rh] * U2[il * 2 + jl, rl]
                                    V4[v, rt] += V2[jh * 2 + kh, rh] * V2[jl * 2 + kl, rl]
                                    W4[w, rt] += W2[kh * 2 + ih, rh] * W2[kl * 2 + il, rl]
                                    
                                    if time.time() > deadline:
                                        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
                                        
    # Construct 4x4x8 factors
    U_full = np.zeros((16, 98))
    V_full = np.zeros((32, 98))
    W_full = np.zeros((32, 98))
    
    # First half (k=0..3)
    U_full[:, 0:49] = U4
    
    for j in range(4):
        for k in range(4):
            v_idx_full = j * 8 + k
            v_idx_sub = j * 4 + k
            V_full[v_idx_full, 0:49] = V4[v_idx_sub, 0:49]
            
    for k in range(4):
        for i in range(4):
            w_idx_full = k * 4 + i
            w_idx_sub = k * 4 + i
            W_full[w_idx_full, 0:49] = W4[w_idx_sub, 0:49]
            
    # Second half (k=4..7)
    U_full[:, 49:98] = U4
    
    for j in range(4):
        for k in range(4):
            v_idx_full = j * 8 + (k + 4)
            v_idx_sub = j * 4 + k
            V_full[v_idx_full, 49:98] = V4[v_idx_sub, 49:98]
            
    for k in range(4):
        for i in range(4):
            w_idx_full = (k + 4) * 4 + i
            w_idx_sub = k * 4 + i
            W_full[w_idx_full, 49:98] = W4[w_idx_sub, 49:98]
            
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
