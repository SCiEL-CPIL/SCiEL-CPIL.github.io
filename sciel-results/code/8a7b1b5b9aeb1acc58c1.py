import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n*m: return False
    if V.shape[0] != m*p: return False
    if W.shape[0] != p*n: return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]: return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try Rank 10 decomposition for 2x3x2 blocks
    # If successful, total rank 30
    # Else fallback to Rank 11 (total 33)
    
    def build_rank10_232():
        # This is a placeholder for a potential rank 10 decomposition.
        # Based on Strassen (7) + 3 extra terms? 
        # Strassen for 2x2x2 core.
        # Tail: A02, A12, B20, B21.
        # C00 += A02 B20
        # C01 += A02 B21
        # C10 += A12 B20
        # C11 += A12 B21
        # Can we share?
        # M8 = A02 B20
        # M9 = A02 B21
        # M10 = A12 B20
        # M11 = A12 B21
        # Total 11.
        # To get 10, we need one less.
        # Try M8 = (A02+A12)(B20+B21)? No.
        # Return None to indicate failure/unknown.
        return None

    terms_232_rank10 = build_rank10_232()
    
    if terms_232_rank10 is not None:
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        # ... construction ...
        # For now, assume it fails and use rank 11
        pass

    # Fallback: Rank 11 per 2x3x2 block
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Seed's rank 11 terms for 2x3x2
    # Core 2x2x2 Strassen (7 terms) + 4 tail terms
    terms = []
    # Strassen 2x2x2 on indices A00,A01,A10,A11 and B00,B01,B10,B11
    # Output C00, C01, C10, C11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Note: The seed's indices for 2x3x2 were:
    # U: i*3+j
    # V: j*2+k (wait, seed used j*2+k in _construct_rank33? 
    # No, seed used j_local*2+k_local? 
    # In _construct_rank33:
    # j_local = l_idx // 2
    # k_local = l_idx % 2
    # So V index is j*2+k. This implies B is 3x2.
    # But B is 3x2. j in 0,1,2. k in 0,1.
    # Correct.
    
    # Let's use the seed's exact terms which are known to work (score 1.0909)
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
