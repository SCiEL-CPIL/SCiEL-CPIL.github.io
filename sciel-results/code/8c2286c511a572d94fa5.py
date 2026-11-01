"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build target tensor for verification
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strassen-based decomposition for 2x2 * 2x5
    # Strategy: Split B (2x5) into B1 (2x4) and B2 (2x1)
    # Use Strassen for A*B1: 14 terms (7 for each 2x2 block)
    # Use schoolbook for A*B2: 4 terms
    # Total: 18 terms
    
    rank = 18
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Helper to add a term
    term_idx = 0
    
    def add_term(u_vals, v_vals, w_vals):
        """u_vals, v_vals, w_vals are dicts mapping index->value"""
        nonlocal term_idx
        if term_idx >= rank:
            raise ValueError("Exceeded rank")
        for idx, val in u_vals.items():
            U[idx, term_idx] += val
        for idx, val in v_vals.items():
            V[idx, term_idx] += val
        for idx, val in w_vals.items():
            W[idx, term_idx] += val
        term_idx += 1
    
    # Indices:
    # U index: i*m + j where i in {0,1}, j in {0,1}
    #   i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
    # V index: j*p + k where j in {0,1}, k in {0,1,2,3,4}
    #   j=0,k=0 -> 0; j=0,k=1 -> 1; j=0,k=2 -> 2; j=0,k=3 -> 3; j=0,k=4 -> 4
    #   j=1,k=0 -> 5; j=1,k=1 -> 6; j=1,k=2 -> 7; j=1,k=3 -> 8; j=1,k=4 -> 9
    # W index: k*n + i where k in {0,1,2,3,4}, i in {0,1}
    #   k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    #   k=2,i=0 -> 4; k=2,i=1 -> 5; k=3,i=0 -> 6; k=3,i=1 -> 7
    #   k=4,i=0 -> 8; k=4,i=1 -> 9
    
    # Strassen for first 2x2 block (columns 0,1)
    # M1 = (a00+a11)(b00+b11)
    add_term({0:1, 3:1}, {0:1, 6:1}, {0:1, 1:1, 2:1, 3:1})
    
    # M2 = (a10+a11)b00
    add_term({2:1, 3:1}, {0:1}, {0:-1, 1:1})
    
    # M3 = a00(b11-b10)
    add_term({0:1}, {6:1, 5:-1}, {0:1, 1:1})
    
    # M4 = a11(b00-b10)
    add_term({3:1}, {0:1, 5:-1}, {0:-1, 1:1})
    
    # M5 = (a00+a01)b10
    add_term({0:1, 1:1}, {5:1}, {0:1, 1:-1})
    
    # M6 = (a10+a11)(b01-b00)
    add_term({2:1, 3:1}, {1:1, 0:-1}, {0:1, 1:1})
    
    # M7 = (a01-a10)(b00+b10)
    add_term({1:1, 2:-1}, {0:1, 5:1}, {0:1, 1:-1})
    
    # Strassen for second 2x2 block (columns 2,3)
    # M8 = (a00+a11)(b02+b12)
    add_term({0:1, 3:1}, {2:1, 7:1}, {4:1, 5:1, 6:1, 7:1})
    
    # M9 = (a10+a11)b02
    add_term({2:1, 3:1}, {2:1}, {4:-1, 5:1})
    
    # M10 = a00(b12-b11)
    add_term({0:1}, {7:1, 6:-1}, {4:1, 5:1})
    
    # M11 = a11(b02-b11)
    add_term({3:1}, {2:1, 6:-1}, {4:-1, 5:1})
    
    # M12 = (a00+a01)b11
    add_term({0:1, 1:1}, {6:1}, {4:1, 5:-1})
    
    # M13 = (a10+a11)(b03-b02)
    add_term({2:1, 3:1}, {3:1, 2:-1}, {4:1, 5:1})
    
    # M14 = (a01-a10)(b02+b11)
    add_term({1:1, 2:-1}, {2:1, 6:1}, {4:1, 5:-1})
    
    # Schoolbook for last column (column 4)
    # c0,4 = a00*b04 + a01*b14
    add_term({0:1}, {4:1}, {8:1})
    add_term({1:1}, {9:1}, {8:1})
    
    # c1,4 = a10*b04 + a11*b14
    add_term({2:1}, {4:1}, {9:1})
    add_term({3:1}, {9:1}, {9:1})
    
    # Verify the decomposition
    R = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(R, T, atol=1e-10, rtol=0):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = np.zeros((nm, rank_fb))
        V_fb = np.zeros((mp, rank_fb))
        W_fb = np.zeros((pn, rank_fb))
        for c in range(rank_fb):
            i = c // (m*p)
            rem = c % (m*p)
            j = rem // p
            k = rem % p
            U_fb[i*m+j, c] = 1.0
            V_fb[j*p+k, c] = 1.0
            W_fb[k*n+i, c] = 1.0
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Convert to lists of lists with integer values
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
