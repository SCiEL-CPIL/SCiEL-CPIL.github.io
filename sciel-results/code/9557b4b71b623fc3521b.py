import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    total_rank = 32
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Helper to map and add a block
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, rank_offset):
        # U_block: shape (n_sub * m_sub, r_sub)
        # V_block: shape (m_sub * p_sub, r_sub)
        # W_block: shape (p_sub * n_sub, r_sub)
        # i_offset: offset for i in A (row)
        # j_offset: offset for j in A (col) / B (row)
        # k_offset: offset for k in B (col) / C (row)
        # rank_offset: column index in U, V, W to start writing
        
        n_sub = U_block.shape[0] // m_sub if m_sub else 0 # This is tricky, better pass dims
        # Let's pass explicit sub-dimensions
        pass

    # We will explicitly construct the sub-blocks and fill the main matrices
    
    # --- Primitive Factors ---
    
    # M_222 (Rank 7)
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]).T  # Shape (4, 7)
    V_222 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]).T  # Shape (4, 7)
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ]).T  # Shape (4, 7)
    
    # M_221 (Rank 4)
    U_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).T  # Shape (4, 4)
    V_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]).T  # Shape (2, 4)
    W_221 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T  # Shape (2, 4)
    
    # M_213 (Rank 2)
    U_213 = np.array([
        [1, 0],
        [0, 1]
    ]).T  # Shape (2, 2)
    V_213 = np.array([
        [1, 1],
        [1, 1],
        [1, 1]
    ]).T  # Shape (3, 2)
    W_213 = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]
    ]).T  # Shape (6, 2)
    
    # M_211 (Rank 2)
    U_211 = np.array([
        [1, 0],
        [0, 1]
    ]).T  # Shape (2, 2)
    V_211 = np.array([
        [1, 1]
    ]).T  # Shape (1, 2)
    W_211 = np.array([
        [1, 0],
        [0, 1]
    ]).T  # Shape (2, 2)

    rank_col = 0
    
    # Block 1: M_233 for k in {0,1,2}
    # Sub-block 1a: M_222 (j in {0,1}, k in {0,1})
    # i_offset=0, j_offset=0, k_offset=0
    r_sub = 7
    U[:, rank_col:rank_col+r_sub] = U_222
    V[0*7+0:1*7+1, rank_col:rank_col+r_sub] = 0 # Initialize
    V[1*7+0:2*7+1, rank_col:rank_col+r_sub] = 0
    # Map V_222 (4x7) to V global
    # Local rows: 0 (j=0,k=0), 1 (j=0,k=1), 2 (j=1,k=0), 3 (j=1,k=1)
    V[0, rank_col:rank_col+r_sub] = V_222[0, :]
    V[1, rank_col:rank_col+r_sub] = V_222[1, :]
    V[7, rank_col:rank_col+r_sub] = V_222[2, :]
    V[8, rank_col:rank_col+r_sub] = V_222[3, :]
    # Map W_222 (4x7) to W global
    # Local rows: 0 (k=0,i=0), 1 (k=0,i=1), 2 (k=1,i=0), 3 (k=1,i=1)
    W[0, rank_col:rank_col+r_sub] = W_222[0, :]
    W[1, rank_col:rank_col+r_sub] = W_222[1, :]
    W[2, rank_col:rank_col+r_sub] = W_222[2, :]
    W[3, rank_col:rank_col+r_sub] = W_222[3, :]
    rank_col += r_sub
    
    # Sub-block 1b: M_221 (j in {0,1}, k in {2})
    r_sub = 4
    U[:, rank_col:rank_col+r_sub] = U_221
    V[0*7+2, rank_col:rank_col+r_sub] = V_221[0, :]
    V[1*7+2, rank_col:rank_col+r_sub] = V_221[1, :]
    # W_221 rows: 0 (k=0,i=0), 1 (k=0,i=1) -> Global k=2
    W[2*2+0, rank_col:rank_col+r_sub] = W_221[0, :]
    W[2*2+1, rank_col:rank_col+r_sub] = W_221[1, :]
    rank_col += r_sub
    
    # Sub-block 1c: M_213 (j in {2}, k in {0,1,2})
    r_sub = 2
    # U_213 rows: 0 (i=0), 1 (i=1). Global j=2.
    U[0*3+2, rank_col:rank_col+r_sub] = U_213[0, :]
    U[1*3+2, rank_col:rank_col+r_sub] = U_213[1, :]
    # V_213 rows: 0 (k=0), 1 (k=1), 2 (k=2). Global j=2.
    V[2*7+0, rank_col:rank_col+r_sub] = V_213[0, :]
    V[2*7+1, rank_col:rank_col+r_sub] = V_213[1, :]
    V[2*7+2, rank_col:rank_col+r_sub] = V_213[2, :]
    # W_213 rows: 0..5 (k=0..2, i=0..1)
    W[0:6, rank_col:rank_col+r_sub] = W_213
    rank_col += r_sub
    
    # Block 2: M_233 for k in {3,4,5}
    # Sub-block 2a: M_222 (j in {0,1}, k in {3,4})
    r_sub = 7
    U[:, rank_col:rank_col+r_sub] = U_222
    V[0*7+3, rank_col:rank_col+r_sub] = V_222[0, :]
    V[0*7+4, rank_col:rank_col+r_sub] = V_222[1, :]
    V[1*7+3, rank_col:rank_col+r_sub] = V_222[2, :]
    V[1*7+4, rank_col:rank_col+r_sub] = V_222[3, :]
    # W: k in {3,4}
    W[3*2+0, rank_col:rank_col+r_sub] = W_222[0, :]
    W[3*2+1, rank_col:rank_col+r_sub] = W_222[1, :]
    W[4*2+0, rank_col:rank_col+r_sub] = W_222[2, :]
    W[4*2+1, rank_col:rank_col+r_sub] = W_222[3, :]
    rank_col += r_sub
    
    # Sub-block 2b: M_221 (j in {0,1}, k in {5})
    r_sub = 4
    U[:, rank_col:rank_col+r_sub] = U_221
    V[0*7+5, rank_col:rank_col+r_sub] = V_221[0, :]
    V[1*7+5, rank_col:rank_col+r_sub] = V_221[1, :]
    W[5*2+0, rank_col:rank_col+r_sub] = W_221[0, :]
    W[5*2+1, rank_col:rank_col+r_sub] = W_221[1, :]
    rank_col += r_sub
    
    # Sub-block 2c: M_213 (j in {2}, k in {3,4,5})
    r_sub = 2
    U[0*3+2, rank_col:rank_col+r_sub] = U_213[0, :]
    U[1*3+2, rank_col:rank_col+r_sub] = U_213[1, :]
    V[2*7+3, rank_col:rank_col+r_sub] = V_213[0, :]
    V[2*7+4, rank_col:rank_col+r_sub] = V_213[1, :]
    V[2*7+5, rank_col:rank_col+r_sub] = V_213[2, :]
    # W: k in {3,4,5}
    W[3*2+0, rank_col:rank_col+r_sub] = W_213[0, :]
    W[3*2+1, rank_col:rank_col+r_sub] = W_213[1, :]
    W[4*2+0, rank_col:rank_col+r_sub] = W_213[2, :]
    W[4*2+1, rank_col:rank_col+r_sub] = W_213[3, :]
    W[5*2+0, rank_col:rank_col+r_sub] = W_213[4, :]
    W[5*2+1, rank_col:rank_col+r_sub] = W_213[5, :]
    rank_col += r_sub
    
    # Block 3: M_231 for k in {6}
    # Sub-block 3a: M_221 (j in {0,1}, k in {6})
    r_sub = 4
    U[:, rank_col:rank_col+r_sub] = U_221
    V[0*7+6, rank_col:rank_col+r_sub] = V_221[0, :]
    V[1*7+6, rank_col:rank_col+r_sub] = V_221[1, :]
    W[6*2+0, rank_col:rank_col+r_sub] = W_221[0, :]
    W[6*2+1, rank_col:rank_col+r_sub] = W_221[1, :]
    rank_col += r_sub
    
    # Sub-block 3b: M_211 (j in {2}, k in {6})
    r_sub = 2
    U[0*3+2, rank_col:rank_col+r_sub] = U_211[0, :]
    U[1*3+2, rank_col:rank_col+r_sub] = U_211[1, :]
    V[2*7+6, rank_col:rank_col+r_sub] = V_211[0, :]
    W[6*2+0, rank_col:rank_col+r_sub] = W_211[0, :]
    W[6*2+1, rank_col:rank_col+r_sub] = W_211[1, :]
    rank_col += r_sub

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
