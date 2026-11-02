# EVOLVE-BLOCK-START
import numpy as np

DIMENSIONS = [2, 3, 7]

def _strassen_222_np():
    """Return U, V, W for 2x2x2 Strassen decomposition as numpy arrays."""
    # U: (4, 7), V: (4, 7), W: (4, 7)
    terms = [
        ([1,0,0,1], [1,0,1,0], [1,0,0,-1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _schoolbook_223_np():
    """2x2x3 schoolbook, rank 6."""
    rank = 6
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, col] = 1
                V[j*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_217_np():
    """2x1x7 schoolbook, rank 14."""
    rank = 14
    U = np.zeros((2, rank))
    V = np.zeros((7, rank))
    W = np.zeros((14, rank))
    col = 0
    for i in range(2):
        for k in range(7):
            U[i, col] = 1
            V[k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    return U, V, W

def _combine_227_np():
    """2x2x7: split 7 into 2+2+3. Rank 20."""
    sU, sV, sW = _strassen_222_np()
    sU2, sV2, sW2 = _strassen_222_np()
    scU, scV, scW = _schoolbook_223_np()
    
    tr = 20
    U = np.zeros((4, tr))
    V = np.zeros((14, tr))
    W = np.zeros((14, tr))
    
    # Block 1: 2x2x2 (k=0,1)
    U[:, 0:7] = sU
    # V mapping: V[j*7+k, r] = sV[j*2+k, r] for j=0,1; k=0,1
    # V rows 0..1 map to sV rows 0..1
    # V rows 7..8 map to sV rows 2..3
    V[0:2, 0:7] = sV[0:2, :]
    V[7:9, 0:7] = sV[2:4, :]
    # W mapping: W[k*2+i, r] = sW[k*2+i, r] for k=0,1; i=0,1
    # W rows 0..3 map to sW rows 0..3
    W[0:4, 0:7] = sW
    
    # Block 2: 2x2x2 (k=2,3)
    U[:, 7:14] = sU2
    # V mapping: V[j*7+2+k, r] = sV2[j*2+k, r]
    # V rows 2..3 map to sV2 rows 0..1
    # V rows 9..10 map to sV2 rows 2..3
    V[2:4, 7:14] = sV2[0:2, :]
    V[9:11, 7:14] = sV2[2:4, :]
    # W mapping: W[(2+k)*2+i, r] = sW2[k*2+i, r]
    # W rows 4..7 map to sW2 rows 0..3
    W[4:8, 7:14] = sW2
    
    # Block 3: 2x2x3 (k=4,5,6)
    U[:, 14:20] = scU
    # V mapping: V[j*7+4+k, r] = scV[j*3+k, r]
    # V rows 4..6 map to scV rows 0..2
    # V rows 11..13 map to scV rows 3..5
    V[4:7, 14:20] = scV[0:3, :]
    V[11:14, 14:20] = scV[3:6, :]
    # W mapping: W[(4+k)*2+i, r] = scW[k*2+i, r]
    # W rows 8..13 map to scW rows 0..5
    W[8:14, 14:20] = scW
    
    return U, V, W

def _combine_full_np():
    """Full 2x3x7: split m=3 into 2+1. Rank 34."""
    U227, V227, W227 = _combine_227_np()
    U217, V217, W217 = _schoolbook_217_np()
    
    tr = 34
    U = np.zeros((6, tr))
    V = np.zeros((21, tr))
    W = np.zeros((14, tr))
    
    # Part 1: A1 (2x2) * B1 (2x7)
    # U1 rows 0..3 correspond to U227 rows 0..3
    # U index i*3+j. i=0,1; j=0,1.
    # U[0, :20] = U227[0, :]
    # U[1, :20] = U227[1, :]
    # U[3, :20] = U227[2, :]
    # U[4, :20] = U227[3, :]
    U[0, 0:20] = U227[0, :]
    U[1, 0:20] = U227[1, :]
    U[3, 0:20] = U227[2, :]
    U[4, 0:20] = U227[3, :]
    
    # V1 rows 0..13 correspond to V227 rows 0..13
    V[0:14, 0:20] = V227
    
    # W1 rows 0..13 correspond to W227 rows 0..13
    W[0:14, 0:20] = W227
    
    # Part 2: A2 (2x1) * B2 (1x7)
    # U2 rows 2, 5 correspond to U217 rows 0, 1
    U[2, 20:34] = U217[0, :]
    U[5, 20:34] = U217[1, :]
    
    # V2 rows 14..20 correspond to V217 rows 0..6
    V[14:21, 20:34] = V217
    
    # W2 rows 0..13 correspond to W217 rows 0..13
    W[0:14, 20:34] = W217
    
    return U, V, W

def _verify_np(U, V, W, n, m, p):
    rank = U.shape[1]
    # Tensor T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i, j, k] == 1 for all i,j,k
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute contraction
    # T[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
    # Use einsum
    # 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _combine_full_np()
    
    if _verify_np(U, V, W, n, m, p):
        # Convert to lists of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    # Fallback to schoolbook
    rank = n*m*p
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
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
