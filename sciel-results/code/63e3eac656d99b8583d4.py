"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And compare with the matrix multiplication tensor
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # True tensor value: 1 (since C[i,k] includes A[i,j]*B[j,k])
                true_val = 1.0
                
                # Decomposed value
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
    
    # Strategy: Use 3 blocks of 2x3x2, each with rank 11
    # Total rank = 3 * 11 = 33
    # Block 0: C[:,0:2] = A * B[:,0:2]
    # Block 1: C[:,2:4] = A * B[:,2:4]
    # Block 2: C[:,4:6] = A * B[:,4:6]
    
    def build_2x3x2_decomposition():
        """Build a rank-11 decomposition for 2x3x2 multiplication.
        
        A is 2x3 (flat index i*3+j), B is 3x2 (flat index j*2+k), C is 2x2.
        W is indexed as W[k*2+i] (n=2).
        Returns list of 11 terms, each with 'u', 'v', 'w' dicts.
        """
        terms = []
        
        # We need: for all i in {0,1}, j in {0,1,2}, k in {0,1}:
        #   sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        
        # W indexing: W[k*2+i]
        # C[0,0]: k=0,i=0 -> W[0]
        # C[0,1]: k=1,i=0 -> W[2]
        # C[1,0]: k=0,i=1 -> W[1]
        # C[1,1]: k=1,i=1 -> W[3]
        
        # Strassen for A[:,0:2] * B[0:2,0:2] contributing to C[:,0:2]:
        # A sub: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx3, A[1,1]=idx4
        # B sub: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx2, B[1,1]=idx3
        
        # M1 = (A00+A11)(B00+B11) -> C[0,0] and C[1,1]
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        # M2 = (A10+A11)*B00 -> C[1,0] and C[1,1]
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {1: 1, 3: -1}})
        # M3 = A00*(B01-B11) -> C[0,1] and C[1,1]
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {2: 1, 3: 1}})
        # M4 = A11*(B10-B00) -> C[0,0] and C[1,0]
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 1: 1}})
        # M5 = (A00+A01)*B11 -> C[0,0] and C[0,1]
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 2: 1}})
        # M6 = (A10-A00)*(B00+B01) -> C[1,1]
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        # M7 = (A01-A11)*(B10+B11) -> C[0,0]
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
        
        # Scalar terms for column 2 of A and row 2 of B
        # K1: A[0,2]*B[2,0] -> C[0,0]
        terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
        # K2: A[0,2]*B[2,1] -> C[0,1]
        terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {2: 1}})
        # K3: A[1,2]*B[2,0] -> C[1,0]
        terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {1: 1}})
        # K4: A[1,2]*B[2,1] -> C[1,1]
        terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
        
        return terms
    
    def verify_2x3x2(terms):
        """Verify a rank-11 decomposition for 2x3x2."""
        rank = len(terms)
        U = np.zeros((6, rank))
        V = np.zeros((6, rank))
        W = np.zeros((4, rank))
        
        for r, term in enumerate(terms):
            for idx, val in term['u'].items():
                U[idx, r] = val
            for idx, val in term['v'].items():
                V[idx, r] = val
            for idx, val in term['w'].items():
                W[idx, r] = val
        
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    dec_val = 0.0
                    for r in range(rank):
                        dec_val += U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if abs(dec_val - 1.0) > 1e-9:
                        return False, (i, j, k, dec_val)
        return True, None
    
    # Build and verify the 2x3x2 decomposition
    terms_232 = build_2x3x2_decomposition()
    valid, err = verify_2x3x2(terms_232)
    
    if not valid:
        # Fall back to rank-36 schoolbook
        return _construct_rank36()
    
    # Build the full 2x3x6 decomposition using three 2x3x2 blocks
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    for block in range(3):
        for t_idx, term in enumerate(terms_232):
            col = block * 11 + t_idx
            
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            for v_idx, val in term['v'].items():
                j_local = v_idx // 2
                k_local = v_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            for w_idx, val in term['w'].items():
                k_local = w_idx // 2
                i_local = w_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        if _verify(U, V, W):
            return result
        else:
            # Fallback to rank-36
            return _construct_rank36()
    
    return result

def _construct_rank36():
    """Fallback rank-36 schoolbook construction."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
