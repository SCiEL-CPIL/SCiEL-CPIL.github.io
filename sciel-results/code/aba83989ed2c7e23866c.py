import numpy as np
import time
from itertools import combinations

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, rank), V shape: (16, rank), W shape: (16, rank)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Use integer arithmetic to avoid floating point issues
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * 4 + j
    
    def v_idx(j, k):
        return j * 4 + k
    
    def w_idx(k, i):
        return k * 4 + i
    
    def a_key(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_key(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_key(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    terms = []
    
    # Term 0: M0 = (A00+A11)(B00+B11)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_key(0,0,r,c): 1, a_key(1,1,r,c): 1}
            R0[r][c] = {b_key(0,0,r,c): 1, b_key(1,1,r,c): 1}
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            contrib0[(r,c)] = [(c_key(0,0,r,c), 1), (c_key(1,1,r,c), 1)]
    terms.append((L0, R0, contrib0))
    
    # Term 1: M1 = (A10+A11)B00
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_key(1,0,r,c): 1, a_key(1,1,r,c): 1}
            R1[r][c] = {b_key(0,0,r,c): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            contrib1[(r,c)] = [(c_key(1,0,r,c), 1), (c_key(1,1,r,c), -1)]
    terms.append((L1, R1, contrib1))
    
    # Term 2: M2 = A00(B01-B11)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_key(0,0,r,c): 1}
            R2[r][c] = {b_key(0,1,r,c): 1, b_key(1,1,r,c): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            contrib2[(r,c)] = [(c_key(0,1,r,c), 1), (c_key(1,1,r,c), 1)]
    terms.append((L2, R2, contrib2))
    
    # Term 3: M3 = A11(B10-B00)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_key(1,1,r,c): 1}
            R3[r][c] = {b_key(1,0,r,c): 1, b_key(0,0,r,c): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            contrib3[(r,c)] = [(c_key(0,0,r,c), 1), (c_key(1,0,r,c), 1)]
    terms.append((L3, R3, contrib3))
    
    # Term 4: M4 = (A00+A01)B11
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_key(0,0,r,c): 1, a_key(0,1,r,c): 1}
            R4[r][c] = {b_key(1,1,r,c): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            contrib4[(r,c)] = [(c_key(0,0,r,c), -1), (c_key(0,1,r,c), 1)]
    terms.append((L4, R4, contrib4))
    
    # Term 5: M5 = (A10-A00)(B00+B01)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_key(1,0,r,c): 1, a_key(0,0,r,c): -1}
            R5[r][c] = {b_key(0,0,r,c): 1, b_key(0,1,r,c): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            contrib5[(r,c)] = [(c_key(1,1,r,c), 1)]
    terms.append((L5, R5, contrib5))
    
    # Term 6: M6 = (A01-A11)(B10+B11)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_key(0,1,r,c): 1, a_key(1,1,r,c): -1}
            R6[r][c] = {b_key(1,0,r,c): 1, b_key(1,1,r,c): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            contrib6[(r,c)] = [(c_key(0,0,r,c), 1)]
    terms.append((L6, R6, contrib6))
    
    sub_contrib = [
        [((0,0),1), ((1,1),1)],
        [((1,0),1), ((1,1),-1)],
        [((0,1),1), ((1,1),1)],
        [((0,0),1), ((1,0),1)],
        [((0,0),-1), ((0,1),1)],
        [((1,1),1)],
        [((0,0),1)]
    ]
    
    for t in range(7):
        L_t, R_t, contrib_t = terms[t]
        a, b, c, d = L_t[0][0], L_t[0][1], L_t[1][0], L_t[1][1]
        e, f, g, h = R_t[0][0], R_t[0][1], R_t[1][0], R_t[1][1]
        
        def add_dicts(x, y, sign=1):
            d = dict(x)
            for k, v in y.items():
                d[k] = d.get(k, 0) + sign * v
            return d
        
        sub_products = [
            (add_dicts(a, d), add_dicts(e, h)),
            (add_dicts(c, d), dict(e)),
            (dict(a), add_dicts(f, h, -1)),
            (dict(d), add_dicts(g, e, -1)),
            (add_dicts(a, b), dict(h)),
            (add_dicts(c, a, -1), add_dicts(e, f)),
            (add_dicts(b, d, -1), add_dicts(g, h))
        ]
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            for (row, col), coeff in x_k.items():
                U[u_idx(row, col)][rank_idx] += coeff
            
            for (row, col), coeff in y_k.items():
                V[v_idx(row, col)][rank_idx] += coeff
            
            for (pr, pc), sub_coeff in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_key, block_coeff) in contrib_t[(pr, pc)]:
                        out_row, out_col = out_key
                        W[w_idx(out_col, out_row)][rank_idx] += sub_coeff * block_coeff
    
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify correctness
    tensor = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    expected = np.zeros((16, 16, 16))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                expected[i*4+k, k*4+j, j*4+i] = 1.0
    
    if np.max(np.abs(tensor - expected)) > 1e-10:
        # Return the integer version if verification fails (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Check for linear dependencies among the 49 rank-1 components
    # Build the component matrix: each column r is the vectorized tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    comp_matrix = np.einsum('ar,br,cr->abc r', U_np, V_np, W_np).reshape(-1, rank)
    
    # Try removing components with smallest singular values - check all single removals
    sv = np.linalg.svd(comp_matrix, compute_uv=False)
    
    # First pass: try removing any single component
    best_rank = rank
    best_mask = np.ones(rank, dtype=bool)
    
    for remove_idx in range(rank):
        mask = np.ones(rank, dtype=bool)
        mask[remove_idx] = False
        recon = np.einsum('ar,br,cr->abc', U_np[:, mask], V_np[:, mask], W_np[:, mask])
        if np.max(np.abs(recon - expected)) < 1e-10:
            best_rank = rank - 1
            best_mask = mask
            break
    
    # If we found a removable component, try to remove more
    if best_rank < rank:
        current_mask = best_mask.copy()
        for _ in range(rank):  # try up to rank more removals
            found = False
            for remove_idx in range(rank):
                if current_mask[remove_idx]:
                    test_mask = current_mask.copy()
                    test_mask[remove_idx] = False
                    recon = np.einsum('ar,br,cr->abc', U_np[:, test_mask], V_np[:, test_mask], W_np[:, test_mask])
                    if np.max(np.abs(recon - expected)) < 1e-10:
                        current_mask = test_mask
                        best_rank -= 1
                        found = True
                        break
            if not found:
                break
    
    if best_rank < rank:
        U_new = U_np[:, best_mask].tolist()
        V_new = V_np[:, best_mask].tolist()
        W_new = W_np[:, best_mask].tolist()
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U, "V": V, "W": W}
