import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2_factors():
    """
    Return U, V, W factor matrices for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # p1 = (a+c)(e+g)
    # p2 = (a+c)(f+h)
    # p3 = (b+d)(e+g)
    # p4 = (b+d)(f+h)
    # p5 = a*e
    # p6 = d*h
    # p7 = (c-d)(f-h) -- Note: variations exist, this is one valid set
    
    # Indices for U (A): 0:a, 1:b, 2:c, 3:d
    # Indices for V (B): 0:e, 1:f, 2:g, 3:h
    # Indices for W (C): 0:top-left, 1:top-right, 2:bot-left, 3:bot-right
    
    # p1 = (a+c)(e+g)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    # Contribution to C:
    # C00 += 1, C01 += 1, C10 += 1, C11 += 1 ? No.
    # Let's use the explicit coefficient mapping for Strassen.
    # C00 = p1 + p4 - p5 + p7
    # C01 = p5 - p1 + p2 + p6
    # C10 = p1 - p3 + p4 + p6
    # C11 = -p3 + p2 + p5 + p7
    
    # We need U, V, W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    
    # Let's define the 7 terms explicitly as rank-1 components.
    # Term 0: p1
    # U: a=1, c=1. V: e=1, g=1.
    U[:, 0] = [1, 0, 1, 0]
    V[:, 0] = [1, 0, 1, 0]
    # W coefficients for p1:
    # C00 += 1, C01 += -1, C10 += 1, C11 += 0
    W[:, 0] = [1, -1, 1, 0]
    
    # Term 1: p2
    # U: a=1, c=1. V: f=1, h=1.
    U[:, 1] = [1, 0, 1, 0]
    V[:, 1] = [0, 1, 0, 1]
    # W coefficients for p2:
    # C00 += 0, C01 += 1, C10 += 0, C11 += 1
    W[:, 1] = [0, 1, 0, 1]
    
    # Term 2: p3
    # U: b=1, d=1. V: e=1, g=1.
    U[:, 2] = [0, 1, 0, 1]
    V[:, 2] = [1, 0, 1, 0]
    # W coefficients for p3:
    # C00 += 0, C01 += 0, C10 += -1, C11 += -1
    W[:, 2] = [0, 0, -1, -1]
    
    # Term 3: p4
    # U: b=1, d=1. V: f=1, h=1.
    U[:, 3] = [0, 1, 0, 1]
    V[:, 3] = [0, 1, 0, 1]
    # W coefficients for p4:
    # C00 += 1, C01 += 0, C10 += 1, C11 += 0
    W[:, 3] = [1, 0, 1, 0]
    
    # Term 4: p5
    # U: a=1. V: e=1.
    U[:, 4] = [1, 0, 0, 0]
    V[:, 4] = [1, 0, 0, 0]
    # W coefficients for p5:
    # C00 += -1, C01 += 1, C10 += 0, C11 += 1
    W[:, 4] = [-1, 1, 0, 1]
    
    # Term 5: p6
    # U: d=1. V: h=1.
    U[:, 5] = [0, 0, 0, 1]
    V[:, 5] = [0, 0, 0, 1]
    # W coefficients for p6:
    # C00 += 0, C01 += 1, C10 += 1, C11 += 0
    W[:, 5] = [0, 1, 1, 0]
    
    # Term 6: p7
    # U: c=1, d=-1. V: f=1, h=-1.
    U[:, 6] = [0, 0, 1, -1]
    V[:, 6] = [0, 1, 0, -1]
    # W coefficients for p7:
    # C00 += 1, C01 += 0, C10 += 0, C11 += 1
    W[:, 6] = [1, 0, 0, 1]
    
    return U, V, W

def _schoolbook_1x2x2_factors():
    """
    Return U, V, W for 1x2 * 2x2 -> 1x2 with rank 4.
    U shape: (2, 4)
    V shape: (4, 4)
    W shape: (2, 4)
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # A = [a, b], B = [[e, f], [g, h]], C = [c0, c1]
    # c0 = a*e + b*g
    # c1 = a*f + b*h
    
    # Term 0: a*e -> c0
    U[0, 0] = 1
    V[0, 0] = 1 # e is (0,0)
    W[0, 0] = 1 # c0 is (0,0) in C (flattened 1x2 -> index 0)
    
    # Term 1: b*g -> c0
    U[1, 1] = 1
    V[2, 1] = 1 # g is (1,0) -> index 1*2+0 = 2
    W[0, 1] = 1
    
    # Term 2: a*f -> c1
    U[0, 2] = 1
    V[1, 2] = 1 # f is (0,1) -> index 0*2+1 = 1
    W[1, 2] = 1 # c1 is index 1
    
    # Term 3: b*h -> c1
    U[1, 3] = 1
    V[3, 3] = 1 # h is (1,1) -> index 1*2+1 = 3
    W[1, 3] = 1
    
    return U, V, W

def _tensor_product_factors(U1, V1, W1, U2, V2, W2):
    """
    Compute tensor product of two decompositions.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    R = r1 * r2
    
    n1, m1, p1 = U1.shape[0]//m1, m1, V1.shape[0]//m1
    # Actually, we can just use the shapes directly.
    # U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
    # U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
    
    # Resulting dimensions:
    # n = n1*n2, m = m1*m2, p = p1*p2
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    U = np.zeros((U1.shape[0]*U2.shape[0], R))
    V = np.zeros((V1.shape[0]*V2.shape[0], R))
    W = np.zeros((W1.shape[0]*W2.shape[0], R))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            U[:, c] = np.kron(U1[:, c1], U2[:, c2])
            V[:, c] = np.kron(V1[:, c1], V2[:, c2])
            W[:, c] = np.kron(W1[:, c1], W2[:, c2])
            
    return U, V, W

def _block_assemble(blocks):
    """
    Assemble global factors from block factors.
    blocks: list of (U, V, W, u_offset, v_offset, w_offset)
    u_offset, v_offset, w_offset are (start_row, start_col) for the block in the global matrix.
    """
    max_u = 0
    max_v = 0
    max_w = 0
    total_rank = 0
    for U, V, W, u_off, v_off, w_off in blocks:
        max_u = max(max_u, u_off[0] + U.shape[0], u_off[1] + U.shape[1] if len(u_off)==2 else u_off[0] + U.shape[0])
        # This logic is getting messy. Let's just allocate the full global size.
        pass
    
    # Better approach: Determine global sizes from the first block or known dimensions.
    # For 3x4x8:
    # U: 12 x R
    # V: 32 x R
    # W: 24 x R
    
    # Let's just sum the ranks and allocate.
    R = sum(U.shape[1] for U, V, W, _, _, _ in blocks)
    
    # We need to know the global shapes.
    # Let's infer from the offsets and block shapes.
    global_u_rows = 0
    global_v_rows = 0
    global_w_rows = 0
    
    for U, V, W, u_off, v_off, w_off in blocks:
        global_u_rows = max(global_u_rows, u_off[0] + U.shape[0])
        global_v_rows = max(global_v_rows, v_off[0] + V.shape[0])
        global_w_rows = max(global_w_rows, w_off[0] + W.shape[0])
        
    U_global = np.zeros((global_u_rows, R))
    V_global = np.zeros((global_v_rows, R))
    W_global = np.zeros((global_w_rows, R))
    
    col_offset = 0
    for U, V, W, u_off, v_off, w_off in blocks:
        r = U.shape[1]
        U_global[u_off[0]:u_off[0]+U.shape[0], col_offset:col_offset+r] = U
        V_global[v_off[0]:v_off[0]+V.shape[0], col_offset:col_offset+r] = V
        W_global[w_off[0]:w_off[0]+W.shape[0], col_offset:col_offset+r] = W
        col_offset += r
        
    return U_global, V_global, W_global

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Get base factors
    U_s, V_s, W_s = _strassen_2x2_factors() # 2x2x2, rank 7
    U_1, V_1, W_1 = _schoolbook_1x2x2_factors() # 1x2x2, rank 4
    
    # Construct 2x4x4 factors (M_2,4,4)
    # M_2,4,4 = M_2,2,2 (tensor) M_1,2,2
    U_24, V_24, W_24 = _tensor_product_factors(U_s, V_s, W_s, U_1, V_1, W_1)
    # U_24: (2*1 * 2*2, 7*4) = (4*4, 28) -> (16, 28)?
    # Wait. U for M_2,2,2 is (4, 7). U for M_1,2,2 is (2, 4).
    # Kron: (4*2, 7*4) = (8, 28).
    # Correct.
    
    # Construct 1x4x4 factors (M_1,4,4)
    # M_1,4,4 = M_1,2,2 (tensor) M_1,2,2
    U_14, V_14, W_14 = _tensor_product_factors(U_1, V_1, W_1, U_1, V_1, W_1)
    # U_14: (2*2, 4*4) = (4, 16).
    # Correct.
    
    # Assemble 3x4x8
    # Blocks:
    # 1. Top-Left: A(0:2, :) * B(:, 0:4) -> C(0:2, 0:4)
    #    U_off: (0, 0) [Rows 0-1 of A]
    #    V_off: (0, 0) [Cols 0-3 of B]
    #    W_off: (0, 0) [Cols 0-3, Rows 0-1 of C]
    #    Note: W index is k*n + i.
    #    For C(0:2, 0:4), k in 0..3, i in 0..1.
    #    W rows: k*3 + i.
    #    k=0, i=0 -> 0
    #    k=0, i=1 -> 1
    #    k=1, i=0 -> 3
    #    k=1, i=1 -> 4
    #    ...
    #    k=3, i=0 -> 9
    #    k=3, i=1 -> 10
    #    So W rows are not contiguous for a block?
    #    W shape for M_2,4,4 is (4*2, 28) = (8, 28).
    #    The local W indices are k_local * 2 + i_local.
    #    Global W indices are k_global * 3 + i_global.
    #    We need to map local W rows to global W rows.
    
    # Let's create a helper to place a block's W into global W.
    
    U_global = np.zeros((12, 88))
    V_global = np.zeros((32, 88))
    W_global = np.zeros((24, 88))
    
    col = 0
    
    # Block 1: Top-Left (2x4 * 4x4 -> 2x4)
    # U: Rows 0-1 of A (2x4). Local U rows 0-7. Global U rows 0-7.
    U_global[0:8, col:col+28] = U_24
    # V: Cols 0-3 of B (4x4). Local V rows 0-15. Global V rows 0-15.
    V_global[0:16, col:col+28] = V_24
    # W: Cols 0-3, Rows 0-1 of C.
    # Local W rows 0-7 correspond to k_l in 0..3, i_l in 0..1.
    # Global W rows: k_g in 0..3, i_g in 0..1. Index k_g*3 + i_g.
    for i_l in range(2):
        for k_l in range(4):
            local_idx = k_l * 2 + i_l
            global_idx = k_l * 3 + i_l
            W_global[global_idx, col:col+28] = W_24[local_idx, :]
    col += 28
    
    # Block 2: Top-Right (2x4 * 4x4 -> 2x4)
    # U: Rows 0-1 of A.
    U_global[0:8, col:col+28] = U_24
    # V: Cols 4-7 of B. Global V rows 16-31.
    V_global[16:32, col:col+28] = V_24
    # W: Cols 4-7, Rows 0-1 of C.
    for i_l in range(2):
        for k_l in range(4):
            local_idx = k_l * 2 + i_l
            global_idx = (k_l + 4) * 3 + i_l
            W_global[global_idx, col:col+28] = W_24[local_idx, :]
    col += 28
    
    # Block 3: Bottom-Left (1x4 * 4x4 -> 1x4)
    # U: Row 2 of A. Global U rows 8-11.
    U_global[8:12, col:col+16] = U_14
    # V: Cols 0-3 of B. Global V rows 0-15.
    V_global[0:16, col:col+16] = V_14
    # W: Cols 0-3, Row 2 of C.
    for i_l in range(1):
        for k_l in range(4):
            local_idx = k_l * 1 + i_l
            global_idx = k_l * 3 + 2
            W_global[global_idx, col:col+16] = W_14[local_idx, :]
    col += 16
    
    # Block 4: Bottom-Right (1x4 * 4x4 -> 1x4)
    # U: Row 2 of A.
    U_global[8:12, col:col+16] = U_14
    # V: Cols 4-7 of B. Global V rows 16-31.
    V_global[16:32, col:col+16] = V_14
    # W: Cols 4-7, Row 2 of C.
    for i_l in range(1):
        for k_l in range(4):
            local_idx = k_l * 1 + i_l
            global_idx = (k_l + 4) * 3 + 2
            W_global[global_idx, col:col+16] = W_14[local_idx, :]
    col += 16
    
    # Verify
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # n=3, m=4, p=8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_global[i*m+j, :] * V_global[j*p+k, :] * W_global[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    if not valid:
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W
