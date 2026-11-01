"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Use a linear change of basis in the output space to share
    # rank-7 tensor structure across both halves of the 2x3x6 problem.
    # 
    # The 2x3x3 multiplication has rank 7 (Strassen). We can split the 2x3x6
    # problem into two 2x3x3 sub-problems and use a clever basis change to
    # share the rank-7 structure.
    #
    # Key insight: If we can find a linear transformation of the output such that
    # both halves use the same rank-7 decomposition pattern, we get rank 14.
    
    # Build the rank-7 Strassen decomposition for 2x3x3
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # The 7 terms for 2x3x3 (using Strassen for the 2x2 core + 4 additional terms)
    def get_2x3x3_terms():
        terms = []
        
        # Strassen terms for A[:,0:2] * B[0:2,:] -> contributes to C[:,0:2]
        # M1 = (A00+A11)(B00+B11)
        terms.append({
            'u': {0: 1, 4: 1},      # A[0,0], A[1,1]
            'v': {0: 1, 4: 1},      # B[0,0], B[1,1]
            'w': {0: 1, 4: 1}       # C[0,0], C[1,1]
        })
        # M2 = (A10+A11)B00
        terms.append({
            'u': {3: 1, 4: 1},      # A[1,0], A[1,1]
            'v': {0: 1},            # B[0,0]
            'w': {3: 1, 4: -1}      # C[1,0], C[1,1]
        })
        # M3 = A00(B01-B11)
        terms.append({
            'u': {0: 1},            # A[0,0]
            'v': {1: 1, 4: -1},     # B[0,1], B[1,1]
            'w': {1: 1, 4: 1}       # C[0,1], C[1,1]
        })
        # M4 = A11(B10-B00)
        terms.append({
            'u': {4: 1},            # A[1,1]
            'v': {3: 1, 0: -1},     # B[1,0], B[0,0]
            'w': {0: 1, 3: 1}       # C[0,0], C[1,0]
        })
        # M5 = (A00+A01)B11
        terms.append({
            'u': {0: 1, 1: 1},      # A[0,0], A[0,1]
            'v': {4: 1},            # B[1,1]
            'w': {0: -1, 1: 1}      # C[0,0], C[0,1]
        })
        # M6 = (A10-A00)(B00+B01)
        terms.append({
            'u': {3: 1, 0: -1},     # A[1,0], A[0,0]
            'v': {0: 1, 1: 1},      # B[0,0], B[0,1]
            'w': {4: 1}             # C[1,1]
        })
        # M7 = (A01-A11)(B10+B11)
        terms.append({
            'u': {1: 1, 4: -1},     # A[0,1], A[1,1]
            'v': {3: 1, 4: 1},      # B[1,0], B[1,1]
            'w': {0: 1}             # C[0,0]
        })
        
        # Additional terms for the third column/row interactions
        # K1 = A02 * B20 -> C[0,0]
        terms.append({
            'u': {2: 1},            # A[0,2]
            'v': {6: 1},            # B[2,0]
            'w': {0: 1}             # C[0,0]
        })
        # K2 = A02 * B21 -> C[0,1]
        terms.append({
            'u': {2: 1},            # A[0,2]
            'v': {7: 1},            # B[2,1]
            'w': {1: 1}             # C[0,1]
        })
        # K3 = A12 * B20 -> C[1,0]
        terms.append({
            'u': {5: 1},            # A[1,2]
            'v': {6: 1},            # B[2,0]
            'w': {3: 1}             # C[1,0]
        })
        # K4 = A12 * B21 -> C[1,1]
        terms.append({
            'u': {5: 1},            # A[1,2]
            'v': {7: 1},            # B[2,1]
            'w': {4: 1}             # C[1,1]
        })
        
        # Terms for C[:,2] = A * B[:,2]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        
        # L1 = A00 * B02 -> C[0,2]
        terms.append({
            'u': {0: 1},
            'v': {2: 1},
            'w': {6: 1}
        })
        # L2 = A01 * B12 -> C[0,2]
        terms.append({
            'u': {1: 1},
            'v': {5: 1},
            'w': {6: 1}
        })
        # L3 = A02 * B22 -> C[0,2]
        terms.append({
            'u': {2: 1},
            'v': {8: 1},
            'w': {6: 1}
        })
        # L4 = A10 * B02 -> C[1,2]
        terms.append({
            'u': {3: 1},
            'v': {2: 1},
            'w': {9: 1}
        })
        # L5 = A11 * B12 -> C[1,2]
        terms.append({
            'u': {4: 1},
            'v': {5: 1},
            'w': {9: 1}
        })
        # L6 = A12 * B22 -> C[1,2]
        terms.append({
            'u': {5: 1},
            'v': {8: 1},
            'w': {9: 1}
        })
        
        return terms
    
    # Try the rank-14 approach: share structure between the two halves
    # by using a linear change of basis in the output space.
    
    # For the 2x3x6 problem, we have:
    # A: 2x3, B: 3x6, C: 2x6
    # Split B into B_left (3x3) and B_right (3x3)
    # C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    
    # The naive approach uses 2 * 11 = 22 rank terms.
    # We want to find a way to share the rank-7 structure.
    
    # Let's try a different approach: use the fact that the 2x3x3 tensor
    # has rank 7, and see if we can combine the two halves cleverly.
    
    # Actually, let me first implement the rank-22 solution as a baseline,
    # then try to optimize.
    
    terms_2x3x3 = get_2x3x3_terms()
    
    # For now, use the 2-block approach with rank 11 per block = rank 22
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for 2 blocks of 2x3x3
    for block in range(2):
        for t_idx, term in enumerate(terms_2x3x3):
            col = block * 11 + t_idx
            
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            for l_idx, val in term['v'].items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            
            for l_idx, val in term['w'].items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    
    # If verification fails, fall back to the rank-33 solution
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
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
