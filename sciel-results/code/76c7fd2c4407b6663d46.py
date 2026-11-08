import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # New Approach: Numerical Optimization for Tensor Decomposition
    # Target: Find a low-rank decomposition of the tensor T[a,b,c]
    # T[i*4+j, j*8+k, k*2+i] = 1
    
    # We will try to find a decomposition of rank R for the sub-problem M_{2,4,4}
    # and use it twice.
    # If we find rank 21 for M_{2,4,4}, total rank is 42.
    # If we fail, we fall back to rank 28 for M_{2,4,4} (total 56).
    
    # Let's define the tensor for M_{2,4,4}
    # A: 2x4 (8 elems), B: 4x4 (16 elems), C: 2x4 (8 elems)
    # T_sub[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    
    def get_sub_tensor():
        T = np.zeros((8, 16, 8))
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 2 + i
                    T[a, b, c] = 1.0
        return T

    T_sub = get_sub_tensor()
    
    # Try to find a rank-21 decomposition
    # We use a simple iterative method:
    # 1. Initialize U, V, W with random values
    # 2. Update U, V, W to minimize || T - sum U_r V_r W_r ||
    # 3. Snap to integers
    
    best_rank = 56
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: 4 blocks of Strassen 2x2x2 for M_{2,4,4} -> Rank 28
    # Then total rank 56
    
    # Let's try to find a better decomposition for M_{2,4,4}
    # We'll try ranks 21, 22, ..., 28
    
    # For simplicity and robustness, I will implement a known rank-21 decomposition
    # if I can recall it, or use a heuristic.
    # Actually, I will use a "Structured" decomposition that is known to be rank 21.
    # The construction is based on the Strassen algorithm but with "folding".
    
    # Since I cannot guarantee the exact coefficients of the rank-21 decomposition
    # from memory, I will use a **Greedy Linear Algebra** approach to see if
    # we can reduce the rank of the 28-term decomposition.
    
    # Construct the 28-term decomposition for M_{2,4,4}
    # 4 blocks of 2x2x2
    U_sub = np.zeros((8, 28))
    V_sub = np.zeros((16, 28))
    W_sub = np.zeros((8, 28))
    
    S = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    def add_strassen_block_sub(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            u_entries, v_entries, w_entries = S[r]
            for idx, val in u_entries:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_off, lj + j_off
                U_sub[gi * 4 + gj][cr] += val
            for idx, val in v_entries:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_off, lk + k_off
                V_sub[gj * 4 + gk][cr] += val
            for idx, val in w_entries:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_off, li + i_off
                W_sub[gk * 2 + gi][cr] += val

    # 4 blocks for 2x4x4: i_off=0, j_off in {0,2}, k_off in {0,2}
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            add_strassen_block_sub(block_idx * 7, 0, j_off, k_off)
            block_idx += 1
            
    # Now we have a rank-28 decomposition for M_{2,4,4}.
    # We can try to reduce the rank by finding linear dependencies.
    # However, for the purpose of this task, if we can't easily find a lower rank,
    # we should stick to a valid decomposition.
    
    # Let's try to use **Rank 21** if we can find it.
    # I will implement a simple check: if we can find a rank-21 decomposition, use it.
    # Otherwise, use rank 28.
    
    # Since finding rank 21 is hard, I will use a **different known decomposition**.
    # Actually, the rank of M_{2,4,4} is 21.
    # I will use a **hardcoded rank-21 decomposition** if possible.
    # If not, I will use the 28-rank one.
    
    # Let's assume we can use the 28-rank one for now, which gives total 56.
    # To improve, I will try to **merge** the two M_{2,4,4} blocks.
    
    # Full decomposition for M_{2,4,8}
    # Use the 28-rank sub-decomposition twice.
    U_full = np.zeros((8, 56))
    V_full = np.zeros((32, 56))
    W_full = np.zeros((16, 56))
    
    # First half: k in 0..3
    U_full[:, 0:28] = U_sub
    V_full[0:16, 0:28] = V_sub
    W_full[0:8, 0:28] = W_sub
    
    # Second half: k in 4..7
    # Note: V indices for second half are j*8+k where k is 4..7.
    # In the sub-problem, k was 0..3. So we add 4 to the k index in V.
    # V_sub[j*4+k] corresponds to V_full[j*8 + (k+4)]
    for j in range(4):
        for k in range(4):
            V_full[j*8 + k + 4, 28:56] = V_sub[j*4 + k, 0:28]
    
    # W_full for second half: c = k*2 + i.
    # In sub-problem, c = k_sub*2 + i.
    # In full problem, k_full = k_sub + 4.
    # c_full = (k_sub+4)*2 + i = k_sub*2 + i + 8.
    # So W_full[c_full, 28:56] = W_sub[c_sub, 0:28]
    for i in range(2):
        for k_sub in range(4):
            c_sub = k_sub * 2 + i
            c_full = (k_sub + 4) * 2 + i
            W_full[c_full, 28:56] = W_sub[c_sub, 0:28]
            
    # Convert to lists of lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
