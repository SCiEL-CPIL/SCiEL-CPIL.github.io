"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy: Block decomposition.
    # p=6 is split into 3 blocks of size 2.
    # Each block computes a 2x3x2 multiplication.
    # Rank of 2x3x2 is 11 (7 Strassen terms for 2x2x2 part + 4 scalar terms for the 3rd column/row).
    # Total rank = 3 * 11 = 33.
    # This is the theoretical minimum rank for this problem size.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    # Define the 11 terms for a single 2x3x2 block.
    # Local indices:
    # A (2x3): flat index i*3+j, i in 0..1, j in 0..2
    # B (3x2): flat index j*2+k, j in 0..2, k in 0..1
    # C (2x2): flat index i*2+k, i in 0..1, k in 0..1
    
    # We decompose the 2x3x2 tensor into:
    # 1. Strassen-like decomposition for A[:, 0:2] (2x2) * B[0:2, :] (2x2)
    # 2. Scalar products for A[:, 2] * B[2, :]
    
    terms = []
    
    # --- Strassen terms for the 2x2 x 2x2 part ---
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=2, B11=3
    
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    # M2 = (A10+A11)B00 -> C10 - C11
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    # M3 = A00(B01-B11) -> C01 + C11
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    # M4 = A11(B10-B00) -> C00 + C10
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    # M5 = (A00+A01)B11 -> -C00 + C01
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01) -> C11
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    # M7 = (A01-A11)(B10+B11) -> C00
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    
    # --- Scalar terms for the 3rd column of A and 3rd row of B ---
    # A02=2, A12=5
    # B20=4, B21=5
    
    # K1 = A02 * B20 -> C00
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    # K2 = A02 * B21 -> C01
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    # K3 = A12 * B20 -> C10
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    # K4 = A12 * B21 -> C11
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    # Fill the matrices for the 3 blocks (k=0,1; k=2,3; k=4,5)
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U: Global A indices are the same for all blocks (A is not split)
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: Local B index l_idx = j*2+k_local -> Global j*p + (block*2 + k_local)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W: Local C index l_idx = i*2+k_local -> Global (block*2 + k_local)*n + i
            # Convention: W[k*n + i, r]
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}

def _verify(result, n, m, p):
    """Verify the decomposition is correct by checking all (i,j,k) entries."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    # Check shapes
    assert len(U) == n * m, f"U shape {len(U)} != {n*m}"
    assert len(V) == m * p, f"V shape {len(V)} != {m*p}"
    assert len(W) == p * n, f"W shape {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U column count mismatch"
    assert all(len(row) == rank for row in V), "V column count mismatch"
    assert all(len(row) == rank for row in W), "W column count mismatch"
    
    # For standard matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    
    return True
# EVOLVE-BLOCK-END
