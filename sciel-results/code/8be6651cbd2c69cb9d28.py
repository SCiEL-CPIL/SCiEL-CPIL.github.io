import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # 2x M_2,4,4 (rank 28 each) + 2x M_1,4,4 (rank 16 each)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Convert to numpy for analysis
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute rank-1 tensors
    R = np.zeros((rank, nm * mp * pn), dtype=float)
    for r in range(rank):
        Rr = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        R[r] = Rr.reshape(-1)
    
    # Try to find linearly dependent sets and reduce rank
    keep = [True] * rank
    
    # First pass: check for identical or proportional rank-1 tensors
    for r in range(rank):
        if time.time() > deadline:
            break
        if not keep[r]:
            continue
        
        for s in range(r + 1, rank):
            if not keep[s]:
                continue
            
            # Check if R[r] and R[s] are proportional
            norm_r = np.linalg.norm(R[r])
            norm_s = np.linalg.norm(R[s])
            
            if norm_r < 1e-10 or norm_s < 1e-10:
                continue
            
            # Check if R[r] = alpha * R[s] for some alpha
            # alpha = dot(R[r], R[s]) / dot(R[s], R[s])
            alpha = np.dot(R[r], R[s]) / (norm_s ** 2)
            
            # Check if R[r] - alpha*R[s] is close to zero
            diff = R[r] - alpha * R[s]
            if np.linalg.norm(diff) / norm_r < 1e-8:
                # They are proportional. We can combine them.
                # T = ... + R[r] + R[s] + ... = ... + (1 + alpha) R[s] + ...
                # So we can remove R[r] and scale R[s] by (1 + alpha)
                # But we need to update the coefficient, which means modifying U, V, W
                # This is tricky because scaling a rank-1 tensor means scaling one of U, V, or W
                # R[s] = U_s ⊗ V_s ⊗ W_s
                # (1+alpha) R[s] = (1+alpha) U_s ⊗ V_s ⊗ W_s = U_s ⊗ V_s ⊗ ((1+alpha) W_s)
                # So we can scale W[:, s] by (1+alpha) and remove term r
                W_np[:, s] *= (1 + alpha)
                keep[r] = False
                break
    
    final_indices = [i for i in range(rank) if keep[i]]
    final_rank = len(final_indices)
    
    U_final = U_np[:, final_indices]
    V_final = V_np[:, final_indices]
    W_final = W_np[:, final_indices]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True
