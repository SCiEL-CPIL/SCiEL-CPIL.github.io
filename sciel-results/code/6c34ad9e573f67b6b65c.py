"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def make_schoolbook():
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

    def strassen_2x2x2():
        """Returns U, V, W for 2x2x2 Strassen decomposition. Rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U[0,0]=1; U[1,0]=1; U[2,0]=0; U[3,0]=0
        V[0,0]=1; V[1,0]=0; V[2,0]=1; V[3,0]=0
        W[0,0]=1; W[1,0]=1; W[2,0]=0; W[3,0]=0
        
        # M2: (A10+A11)B00
        U[0,1]=0; U[1,1]=0; U[2,1]=1; U[3,1]=1
        V[0,1]=1; V[1,1]=0; V[2,1]=0; V[3,1]=0
        W[0,1]=1; W[1,1]=0; W[2,1]=0; W[3,1]=0
        
        # M3: A00(B01-B11)
        U[0,2]=1; U[1,2]=0; U[2,2]=0; U[3,2]=0
        V[0,2]=1; V[1,2]=0; V[2,2]=0; V[3,2]=0
        W[0,2]=1; W[1,2]=-1; W[2,2]=0; W[3,2]=0
        
        # M4: A11(B10-B00)
        U[0,3]=0; U[1,3]=0; U[2,3]=0; U[3,3]=1
        V[0,3]=-1; V[1,3]=0; V[2,3]=1; V[3,3]=0
        W[0,3]=1; W[1,3]=0; W[2,3]=0; W[3,3]=0
        
        # M5: (A00+A01)B11
        U[0,4]=1; U[1,4]=0; U[2,4]=1; U[3,4]=0
        V[0,4]=0; V[1,4]=0; V[2,4]=0; V[3,4]=1
        W[0,4]=0; W[1,4]=0; W[2,4]=0; W[3,4]=1
        
        # M6: (A10-A00)(B00+B01)
        U[0,5]=-1; U[1,5]=0; U[2,5]=1; U[3,5]=0
        V[0,5]=1; V[1,5]=1; V[2,5]=0; V[3,5]=0
        W[0,5]=1; W[1,5]=1; W[2,5]=0; W[3,5]=0
        
        # M7: (A01-A11)(B10+B11)
        U[0,6]=0; U[1,6]=1; U[2,6]=0; U[3,6]=-1
        V[0,6]=0; V[1,6]=0; V[2,6]=1; V[3,6]=1
        W[0,6]=1; W[1,6]=1; W[2,6]=0; W[3,6]=0
        
        return U, V, W

    def schoolbook_2x2x1():
        """Returns U, V, W for 2x2x1 Schoolbook decomposition. Rank 4."""
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4)) # W shape is (p*n, rank) -> (1*2, 4)
        
        # C0 = A00B00 + A01B10
        # r=0: A00 B00
        U[0,0]=1; U[1,0]=0; U[2,0]=0; U[3,0]=0
        V[0,0]=1; V[1,0]=0; V[2,0]=0; V[3,0]=0
        W[0,0]=1; W[1,0]=0
        
        # r=1: A01 B10
        U[0,1]=1; U[1,1]=0; U[2,1]=0; U[3,1]=0
        V[0,1]=0; V[1,1]=1; V[2,1]=0; V[3,1]=0
        W[0,1]=1; W[1,1]=0
        
        # C1 = A10B00 + A11B10
        # r=2: A10 B00
        U[0,2]=0; U[1,2]=0; U[2,2]=1; U[3,2]=0
        V[0,2]=1; V[1,2]=0; V[2,2]=0; V[3,2]=0
        W[0,2]=0; W[1,2]=1
        
        # r=3: A11 B10
        U[0,3]=0; U[1,3]=0; U[2,3]=0; U[3,3]=1
        V[0,3]=0; V[1,3]=1; V[2,3]=0; V[3,3]=0
        W[0,3]=0; W[1,3]=1
        
        return U, V, W

    def build_2x2x7():
        """Builds 2x2x7 decomposition from 3 Strassen and 1 Schoolbook. Rank 25."""
        U_list = []
        V_list = []
        W_list = []
        
        # 3 Strassen blocks for k offsets 0, 2, 4
        # Strassen W is 2x7 (p=2, n=2? No, W is p*n x rank. Here p=2, n=2? 
        # Wait, in 2x2x2, U is 4x7, V is 4x7, W is 4x7.
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i].
        # For 2x2x2: m=2, p=2, n=2.
        # U size 4. V size 4. W size 4.
        
        s_U, s_V, s_W = strassen_2x2x2()
        sb_U, sb_V, sb_W = schoolbook_2x2x1()
        
        # Strassen 1: k in {0, 1}
        # W indices need to be mapped to global k in {0..6} and i in {0,1}
        # s_W is 4x7. Row index is k*2 + i.
        # We want to place this into global W of size 14 (7*2).
        # Global W index: k_global * 2 + i.
        # For k_offset = 0: k_global = 0, 1.
        
        def shift_W(W_local, k_offset, n_local=2):
            # W_local shape (n_local * p_local, rank). Here p_local=2.
            # Actually, W_local from strassen is 4x7.
            # We need to create a new W of size (7 * 2, rank) for the 2x2x7 tensor.
            # And shift the k index.
            pass

        # Let's build the full 2x2x7 matrices directly.
        # U size: 4 (2*2). Same for all blocks.
        # V size: 14 (2*7).
        # W size: 14 (7*2).
        
        rank_227 = 3 * 7 + 4
        U_227 = np.zeros((4, rank_227))
        V_227 = np.zeros((14, rank_227))
        W_227 = np.zeros((14, rank_227))
        
        col_offset = 0
        
        # Add 3 Strassen blocks
        for k_off in [0, 2, 4]:
            # U is same
            U_227[:, col_offset:col_offset+7] = s_U
            
            # V: shift k index by k_off
            # s_V is 4x7. Rows are j*2+k.
            # We need to place these into V_227 rows j*7+k_global.
            # j=0: k_global in {k_off, k_off+1}
            # j=1: k_global in {k_off, k_off+1}
            
            for r in range(7):
                for j in range(2):
                    for k_local in range(2):
                        v_val = s_V[j*2 + k_local, r]
                        if v_val != 0:
                            k_global = k_off + k_local
                            V_227[j*7 + k_global, col_offset + r] = v_val
            
            # W: shift k index by k_off
            # s_W is 4x7. Rows are k*2+i.
            # We need to place these into W_227 rows k_global*2+i.
            
            for r in range(7):
                for k_local in range(2):
                    for i in range(2):
                        w_val = s_W[k_local*2 + i, r]
                        if w_val != 0:
                            k_global = k_off + k_local
                            W_227[k_global*2 + i, col_offset + r] = w_val
            
            col_offset += 7
        
        # Add 1 Schoolbook block for k in {6}
        # sb_U is 4x4. sb_V is 4x4. sb_W is 2x4.
        U_227[:, col_offset:col_offset+4] = sb_U
        
        for r in range(4):
            for j in range(2):
                for k_local in range(1): # p_local=1
                    v_val = sb_V[j*1 + k_local, r]
                    if v_val != 0:
                        k_global = 6 + k_local
                        V_227[j*7 + k_global, col_offset + r] = v_val
            
        for r in range(4):
            for k_local in range(1):
                for i in range(2):
                    w_val = sb_W[k_local*2 + i, r] # sb_W is 2x4, rows are k*2+i? 
                    # In schoolbook_2x2x1, W was (2, 4). Row index k*n+i. n=2.
                    # So row 0 is k=0, i=0. Row 1 is k=0, i=1.
                    if w_val != 0:
                        k_global = 6 + k_local
                        W_227[k_global*2 + i, col_offset + r] = w_val
        
        return U_227, V_227, W_227

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    # Construct the full 4x4x7 decomposition
    # Rank = 4 * 25 = 100
    U_227, V_227, W_227 = build_2x2x7()
    
    rank_total = 100
    U_final = np.zeros((16, rank_total))
    V_final = np.zeros((28, rank_total))
    W_final = np.zeros((28, rank_total))
    
    col_offset = 0
    for i_block in range(2):
        for j_block in range(2):
            # U shift: i_block * 2
            # V shift: j_block * 2
            # W shift: 0
            
            U_final[i_block*2:(i_block+1)*2, col_offset:col_offset+25] = U_227
            V_final[j_block*2:(j_block+1)*2, col_offset:col_offset+25] = V_227
            W_final[:, col_offset:col_offset+25] = W_227
            
            col_offset += 25
    
    result = {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
    
    # Verify
    err = verify(result["U"], result["V"], result["W"], rank_total)
    if err < 1e-8:
        return result
    else:
        print(f"Verification failed with error {err}. Falling back to schoolbook.")
        return make_schoolbook()

# EVOLVE-BLOCK-END
