import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Decomposition (Rank 49)
    # Using recursive Strassen: 7 macro terms * 7 micro terms
    
    # Define Micro Strassen terms for 2x2x2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # Micro U, V, W are lists of (index, coeff)
    
    micro_terms = []
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # U: 1 at 00, 11. V: 1 at 00, 11. W: 1 at 00, 11.
    micro_terms.append([
        [(0, 1), (3, 1)],  # U
        [(0, 1), (3, 1)],  # V
        [(0, 1), (3, 1)]   # W
    ])
    
    # M2: (A10+A11)B00 -> C10
    # U: 1 at 10, 11. V: 1 at 00. W: 1 at 10 (k=0, i=1 -> 1)
    micro_terms.append([
        [(2, 1), (3, 1)],  # U: 10, 11
        [(0, 1)],          # V: 00
        [(1, 1)]           # W: 10
    ])
    
    # M3: A00(B10-B11) -> C01, C10
    # U: 1 at 00. V: 1 at 10, -1 at 11. W: 1 at 01 (k=1, i=0 -> 2), 1 at 10 (k=0, i=1 -> 1)
    micro_terms.append([
        [(0, 1)],          # U: 00
        [(2, 1), (3, -1)], # V: 10, 11
        [(2, 1), (1, 1)]   # W: 01, 10
    ])
    
    # M4: A11(B01-B00) -> C00, C01
    # U: 1 at 11. V: 1 at 01, -1 at 00. W: 1 at 00 (k=0, i=0 -> 0), 1 at 01 (k=1, i=0 -> 2)
    micro_terms.append([
        [(3, 1)],          # U: 11
        [(1, 1), (0, -1)], # V: 01, 00
        [(0, 1), (2, 1)]   # W: 00, 01
    ])
    
    # M5: (A00+A01)B11 -> C00, C11
    # U: 1 at 00, 01. V: 1 at 11. W: 1 at 00 (k=0, i=0 -> 0), 1 at 11 (k=1, i=1 -> 3)
    micro_terms.append([
        [(0, 1), (1, 1)],  # U: 00, 01
        [(3, 1)],          # V: 11
        [(0, 1), (3, 1)]   # W: 00, 11
    ])
    
    # M6: (A10-A00)(B00+B01) -> C10
    # U: 1 at 10, -1 at 00. V: 1 at 00, 01. W: 1 at 10 (k=0, i=1 -> 1)
    micro_terms.append([
        [(2, 1), (0, -1)], # U: 10, 00
        [(0, 1), (1, 1)],  # V: 00, 01
        [(1, 1)]           # W: 10
    ])
    
    # M7: (A01-A11)(B10+B11) -> C01, C11
    # U: 1 at 01, -1 at 11. V: 1 at 10, 11. W: 1 at 01 (k=1, i=0 -> 2), 1 at 11 (k=1, i=1 -> 3)
    micro_terms.append([
        [(1, 1), (3, -1)], # U: 01, 11
        [(2, 1), (3, 1)],  # V: 10, 11
        [(2, 1), (3, 1)]   # W: 01, 11
    ])
    
    # Define Macro Strassen terms for 4x4x4 (blocks)
    # Blocks: 00, 01, 10, 11
    # Macro U, V, W are lists of (block_index, coeff) where block_index is 0..3
    # Block index b corresponds to (br, bc) = (b//2, b%2)
    
    macro_terms = []
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # U: 1 at 00, 11. V: 1 at 00, 11. W: 1 at 00, 11.
    macro_terms.append([
        [(0, 1), (3, 1)],  # U: blocks 00, 11
        [(0, 1), (3, 1)],  # V: blocks 00, 11
        [(0, 1), (3, 1)]   # W: blocks 00, 11
    ])
    
    # M2: (A10+A11)B00 -> C10
    # U: 1 at 10, 11. V: 1 at 00. W: 1 at 10 (block 10)
    macro_terms.append([
        [(2, 1), (3, 1)],  # U: blocks 10, 11
        [(0, 1)],          # V: block 00
        [(2, 1)]           # W: block 10
    ])
    
    # M3: A00(B10-B11) -> C01, C10
    # U: 1 at 00. V: 1 at 10, -1 at 11. W: 1 at 01 (block 01), 1 at 10 (block 10)
    macro_terms.append([
        [(0, 1)],          # U: block 00
        [(2, 1), (3, -1)], # V: blocks 10, 11
        [(1, 1), (2, 1)]   # W: blocks 01, 10
    ])
    
    # M4: A11(B01-B00) -> C00, C01
    # U: 1 at 11. V: 1 at 01, -1 at 00. W: 1 at 00 (block 00), 1 at 01 (block 01)
    macro_terms.append([
        [(3, 1)],          # U: block 11
        [(1, 1), (0, -1)], # V: blocks 01, 00
        [(0, 1), (1, 1)]   # W: blocks 00, 01
    ])
    
    # M5: (A00+A01)B11 -> C00, C11
    # U: 1 at 00, 01. V: 1 at 11. W: 1 at 00 (block 00), 1 at 11 (block 11)
    macro_terms.append([
        [(0, 1), (1, 1)],  # U: blocks 00, 01
        [(3, 1)],          # V: block 11
        [(0, 1), (3, 1)]   # W: blocks 00, 11
    ])
    
    # M6: (A10-A00)(B00+B01) -> C10
    # U: 1 at 10, -1 at 00. V: 1 at 00, 01. W: 1 at 10 (block 10)
    macro_terms.append([
        [(2, 1), (0, -1)], # U: blocks 10, 00
        [(0, 1), (1, 1)],  # V: blocks 00, 01
        [(2, 1)]           # W: block 10
    ])
    
    # M7: (A01-A11)(B10+B11) -> C01, C11
    # U: 1 at 01, -1 at 11. V: 1 at 10, 11. W: 1 at 01 (block 01), 1 at 11 (block 11)
    macro_terms.append([
        [(1, 1), (3, -1)], # U: blocks 01, 11
        [(2, 1), (3, 1)],  # V: blocks 10, 11
        [(1, 1), (3, 1)]   # W: blocks 01, 11
    ])
    
    U_list = []
    V_list = []
    W_list = []
    
    # Generate 49 terms for 4x4x4
    for mk in range(7):
        mu, mv, mw = macro_terms[mk]
        for ml in range(7):
            u_micro, v_micro, w_micro = micro_terms[ml]
            
            u_global = np.zeros(16)
            v_global = np.zeros(16)
            w_global = np.zeros(16)
            
            # Expand U
            for (mb_idx, m_coeff) in mu:
                mb_r = mb_idx // 2
                mb_c = mb_idx % 2
                for (mi_idx, mi_coeff) in u_micro:
                    mi_r = mi_idx // 2
                    mi_c = mi_idx % 2
                    gi = mb_r * 2 + mi_r
                    gj = mb_c * 2 + mi_c
                    u_global[gi * 4 + gj] += m_coeff * mi_coeff
            
            # Expand V
            for (mb_idx, m_coeff) in mv:
                mb_r = mb_idx // 2
                mb_c = mb_idx % 2
                for (mi_idx, mi_coeff) in v_micro:
                    mi_r = mi_idx // 2
                    mi_c = mi_idx % 2
                    gi = mb_r * 2 + mi_r
                    gj = mb_c * 2 + mi_c
                    v_global[gi * 4 + gj] += m_coeff * mi_coeff
            
            # Expand W
            for (mb_idx, m_coeff) in mw:
                mb_r = mb_idx // 2
                mb_c = mb_idx % 2
                for (mi_idx, mi_coeff) in w_micro:
                    mi_r = mi_idx // 2
                    mi_c = mi_idx % 2
                    gi = mb_r * 2 + mi_r
                    gj = mb_c * 2 + mi_c
                    w_global[gi * 4 + gj] += m_coeff * mi_coeff
            
            U_list.append(u_global)
            V_list.append(v_global)
            W_list.append(w_global)
            
    # Part 2: 4x4x1 Decomposition (Rank 16)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # For each i, j:
    # U[i*4+j, r] = 1
    # V[j*4+4, r] = 1  (B index j*4+4? No. B is 4x5. B[j, 4] is index j*5+4? 
    # Wait. V shape is (m*p, rank) = (4*5, rank) = (20, rank).
    # V index is j*p + k. Here k=4. So V[j*5+4, r] = 1.
    # W shape is (p*n, rank) = (5*4, rank) = (20, rank).
    # W index is k*n + i. Here k=4. So W[4*4+i, r] = 1.
    
    for i in range(4):
        for j in range(4):
            u_new = np.zeros(16)
            v_new = np.zeros(20)
            w_new = np.zeros(20)
            
            u_new[i * 4 + j] = 1
            v_new[j * 5 + 4] = 1
            w_new[4 * 4 + i] = 1
            
            U_list.append(u_new)
            V_list.append(v_new)
            W_list.append(w_new)
            
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    rank = U.shape[1]
    
    # Verify
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                u_val = np.dot(U[:, r], A.flatten())
                v_val = np.dot(V[:, r], B.flatten())
                C_computed[i, k] += u_val * v_val * W[k*n + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-5):
        # Fallback to schoolbook if verification fails
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        rank = n*m*p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_dot_a = np.dot(U[:, r], A.flatten())
                    v_dot_b = np.dot(V[:, r], B.flatten())
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
    
    return True

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
