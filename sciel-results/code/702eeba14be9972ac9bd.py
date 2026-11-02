import time
import numpy as np

# Constants for the problem
N = 2
M = 2
P = 16

def _get_strassen_components():
    """
    Returns U_block, V_block, W_block for a single 2x2 * 2x2 multiplication.
    U_block shape: (2*2, 7) = (4, 7)
    V_block shape: (2*2, 7) = (4, 7)
    W_block shape: (2*2, 7) = (4, 7)
    
    The indices for V and W in the block correspond to local indices 0..3.
    U indices: 0:(i0,j0), 1:(i0,j1), 2:(i1,j0), 3:(i1,j1)
    V indices: 0:(j0,k0), 1:(j0,k1), 2:(j1,k0), 3:(j1,k1)
    W indices: 0:(k0,i0), 1:(k1,i0), 2:(k0,i1), 3:(k1,i1)
    
    Wait, the task specifies:
    U: (n*m, rank) -> rows indexed by i*m+j ? Or i + m*j?
    "U, V, W : factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    And the tensor decomposition condition is likely:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] == T[i,j,k]
    
    Let's check the prompt's current program indexing:
    U: (n, m, rank) -> flattened?
    Prompt code:
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    This implies:
    U_flat index: i*m + j?
    If U is (n*m, rank), and reshaped to (n, m, rank), then U[i, j, r] is at row i*m + j.
    V is (m*p, rank), reshaped to (m, p, rank), so V[j, k, r] is at row j*p + k.
    W is (p*n, rank), reshaped to (p, n, rank), so W[k, i, r] is at row k*n + i.
    
    My Strassen block mapping in the previous code:
    U_core rows: 0..3 corresponding to (i,j) pairs.
    V_core rows: 0..3 corresponding to (j,k) pairs.
    W_core rows: 0..3 corresponding to (k,i) pairs.
    
    I need to map these to the flat indices.
    """
    U_b = np.zeros((4, 7), dtype=np.int64)
    V_b = np.zeros((4, 7), dtype=np.int64)
    W_b = np.zeros((4, 7), dtype=np.int64)
    
    # Strassen 7 products
    # M1 = (A00 + A11) * (B00 + B11)
    U_b[0, 0] = 1; U_b[3, 0] = 1
    V_b[0, 0] = 1; V_b[3, 0] = 1
    W_b[0, 0] = 1; W_b[3, 0] = 1  # W(k,i) where k=0,i=0 and k=1,i=1?
    # Wait, W indices in block:
    # Block W is (p_block * n_block) = 2*2 = 4.
    # Reshape to (p_block, n_block, rank) -> (2, 2, 7).
    # W_flat[row, r] = W_block[p_loc, n_loc, r].
    # p_loc is local k, n_loc is local i.
    # Flat index in block: p_loc * n_block + n_loc.
    
    # Let's stick to the local (k,i) mapping:
    # Local k=0, i=0 -> flat 0
    # Local k=0, i=1 -> flat 1
    # Local k=1, i=0 -> flat 2
    # Local k=1, i=1 -> flat 3
    
    # M1 contributes to C00, C11, C01? No.
    # C = A * B.
    # M1 = (a+d)(e+h).
    # C00 = a e + b h. M1 term: (a+d)(e+h) = ae + ah + de + dh.
    # We need to subtract other terms.
    # Standard Strassen:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P2 + P3 + P6
    
    # Let's map the 7 products to U, V, W contributions.
    # U factors A, V factors B, W factors C.
    # Each P_k is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}? No.
    # In the bilinear form, one term is U_i V_j W_k.
    # The "rank" decomposition defines the 7 products.
    # Product t is associated with U[:,t], V[:,t], W[:,t].
    
    # Product 0 (P1):
    # U: a,d -> (0,0), (1,1) in (i,j) ?
    # a is i=0,j=0 (idx 0). d is i=1,j=1 (idx 3).
    # U_b[0,0]=1; U_b[3,0]=1.
    # V: e,h -> (0,0), (1,1) in (j,k)?
    # e is j=0,k=0 (idx 0). h is j=1,k=1 (idx 3).
    # V_b[0,0]=1; V_b[3,0]=1.
    # W: C00, C11 -> (k,i)
    # C00 is k=0,i=0 (idx 0). C11 is k=1,i=1 (idx 3).
    # W_b[0,0]=1; W_b[3,0]=1.
    
    # Product 1 (P2 = (c+d)e):
    # U: c,d -> i=1,j=0 (idx 2), i=1,j=1 (idx 3).
    # U_b[2,1]=1; U_b[3,1]=1.
    # V: e -> j=0,k=0 (idx 0).
    # V_b[0,1]=1.
    # W: C10, C11?
    # C10 = P2+P6. C11 = P1-P2+...
    # So W should contribute to C10 and C11.
    # C10 is k=0,i=1 (idx 1). C11 is k=1,i=1 (idx 3).
    # W_b[1,1]=1; W_b[3,1]=-1.
    
    # Product 2 (P3 = a(f-h)):
    # U: a -> idx 0.
    # U_b[0,2]=1.
    # V: f,h -> j=0,k=1 (idx 1), j=1,k=1 (idx 3).
    # V_b[1,2]=1; V_b[3,2]=-1.
    # W: C01, C11.
    # C01 = P3+P5. C11 = ... + P3.
    # C01 is k=1,i=0 (idx 2). C11 is k=1,i=1 (idx 3).
    # W_b[2,2]=1; W_b[3,2]=1.
    
    # Product 3 (P4 = d(g-e)):
    # U: d -> idx 3.
    # U_b[3,3]=1.
    # V: g,e -> j=1,k=0 (idx 2), j=0,k=0 (idx 0).
    # V_b[2,3]=1; V_b[0,3]=-1.
    # W: C00.
    # C00 = P1+P4-P5+P7.
    # C00 is k=0,i=0 (idx 0).
    # W_b[0,3]=1.
    
    # Product 4 (P5 = (a+b)h):
    # U: a,b -> idx 0, 1.
    # U_b[0,4]=1; U_b[1,4]=1.
    # V: h -> idx 3.
    # V_b[3,4]=1.
    # W: C00, C01.
    # C00 = ... - P5. C01 = P3 + P5.
    # C00 (idx 0), C01 (idx 2).
    # W_b[0,4]=-1; W_b[2,4]=1.
    
    # Product 5 (P6 = (c-a)(e+f)):
    # U: c,a -> idx 2, 0.
    # U_b[2,5]=1; U_b[0,5]=-1.
    # V: e,f -> idx 0, 1.
    # V_b[0,5]=1; V_b[1,5]=1.
    # W: C10, C11.
    # C10 = P2+P6. C11 = ... + P6.
    # C10 (idx 1), C11 (idx 3).
    # W_b[1,5]=1; W_b[3,5]=1.
    
    # Product 6 (P7 = (b-d)(g+h)):
    # U: b,d -> idx 1, 3.
    # U_b[1,6]=1; U_b[3,6]=-1.
    # V: g,h -> idx 2, 3.
    # V_b[2,6]=1; V_b[3,6]=1.
    # W: C00.
    # C00 = ... + P7.
    # C00 (idx 0).
    # W_b[0,6]=1.
    
    return U_b, V_b, W_b

def construct():
    start = time.time()
    
    n, m, p = N, M, P
    
    U_blk, V_blk, W_blk = _get_strassen_components()
    
    num_blocks = p // 2
    total_rank = num_blocks * 7
    
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    for t in range(num_blocks):
        rank_offset = t * 7
        k_start = t * 2
        
        # Fill U (same for all blocks, just repeated)
        U[:, rank_offset : rank_offset + 7] = U_blk
        
        # Fill V
        # Block V has 4 rows (local j,k).
        # Full V row index = j * p + k.
        # Local j in {0,1}, local k in {0,1} (relative to k_start).
        for l_j in range(2):
            for l_k in range(2):
                full_j = l_j
                full_k = k_start + l_k
                full_idx = full_j * p + full_k
                local_idx = l_j * 2 + l_k
                V[full_idx, rank_offset : rank_offset + 7] = V_blk[local_idx]
                
        # Fill W
        # Block W has 4 rows (local k,i).
        # Full W row index = k * n + i.
        # Local k in {0,1} (relative), local i in {0,1}.
        for l_k in range(2):
            for l_i in range(2):
                full_k = k_start + l_k
                full_i = l_i
                full_idx = full_k * n + full_i
                local_idx = l_k * 2 + l_i
                W[full_idx, rank_offset : rank_offset + 7] = W_blk[local_idx]

    # Verify correctness
    U_r = U.reshape(n, m, total_rank)
    V_r = V.reshape(m, p, total_rank)
    W_r = W.reshape(p, n, total_rank)
    
    # Tensor product reconstruction
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    res = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    
    if not np.allclose(res, 1.0):
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    if time.time() - start > 110:
        # Emergency exit, though construction is fast
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
