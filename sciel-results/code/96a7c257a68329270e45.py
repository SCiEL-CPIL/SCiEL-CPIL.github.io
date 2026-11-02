import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
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

def _build_2x3x3_terms():
    """
    Build rank-11 decomposition for 2x3 * 3x3 -> 2x3.
    A is 2x3 (rows i=0,1; cols j=0,1,2), B is 3x3 (rows j=0,1,2; cols k=0,1,2),
    C is 2x3 (rows i=0,1; cols k=0,1,2).
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Flat indices:
    A: i*3+j  -> A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    B: j*3+k  -> B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    C: i*3+k  -> C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
    """
    terms = []
    
    # Strassen 7 terms for A[:,0:2] * B[0:2,:] -> C[:,0:2]
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 4: 1}})
    # M2 = (A10+A11)B00 -> C10, C11
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {3: 1, 4: -1}})
    # M3 = A00(B01-B11) -> C01, C11
    terms.append({'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {1: 1, 4: 1}})
    # M4 = A11(B10-B00) -> C00, C10
    terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 3: 1}})
    # M5 = (A00+A01)B11 -> C00, C01
    terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01) -> C11
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {4: 1}})
    # M7 = (A01-A11)(B10+B11) -> C00
    terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}})
    
    # 4 scalar terms for A[:,2] * B[2,:] -> C[:,2]
    # K1 = A02 * B20 -> C02
    terms.append({'u': {2: 1}, 'v': {6: 1}, 'w': {2: 1}})
    # K2 = A02 * B21 -> C02
    terms.append({'u': {2: 1}, 'v': {7: 1}, 'w': {2: 1}})
    # K3 = A12 * B20 -> C12
    terms.append({'u': {5: 1}, 'v': {6: 1}, 'w': {5: 1}})
    # K4 = A12 * B21 -> C12
    terms.append({'u': {5: 1}, 'v': {7: 1}, 'w': {5: 1}})
    
    # Wait, that's not right. Let me reconsider.
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # The Strassen terms already handle A[i,0]*B[0,2] and A[i,1]*B[1,2] via M terms?
    # No, Strassen handles B[0:2, 0:2], not B[0:2, 2].
    # So we need:
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # That's 4 more scalar terms (2 rows x 3 cols of the extra column).
    
    # Let me redo this properly.
    terms = []
    
    # Strassen 7 terms for A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 4: 1}})  # M1
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {3: 1, 4: -1}})  # M2
    terms.append({'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {1: 1, 4: 1}})  # M3
    terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 3: 1}})  # M4
    terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 1: 1}})  # M5
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {4: 1}})  # M6
    terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}})  # M7
    
    # Now we need C[:,2] = A[:,0:3] * B[:,2]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # That's 6 scalar terms, but we can use 2 more Strassen-like tricks.
    # Actually, for a 2x3 * 3x1 vector multiply, the rank is 3 (schoolbook) 
    # but can be done with rank 3. So 6 scalar terms for both rows.
    # Wait, 2x3 * 3x1: C[0] = A[0,0]B[0]+A[0,1]B[1]+A[0,2]B[2], 
    # C[1] = A[1,0]B[0]+A[1,1]B[1]+A[1,2]B[2]. That's 6 products.
    # Can we do better? The rank of 2x3x1 tensor is 3 (min(2,3,1)*... no, 
    # it's the rank of the bilinear form. For 2x3x1, rank = 3.
    # So we need at least 3 terms for the column k=2 part.
    # But we have 6 scalar terms above. Let's use 3 terms:
    # Actually no, each term is a product of one A element, one B element, 
    # contributing to one or more C elements. For C[:,2], we need:
    # 3 terms won't work because each term can only involve one A[i,j] and one B[j,2].
    # We need: A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2], A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2]
    # That's 6 distinct products. Unless we combine them.
    # We can't combine A[0,0]*B[0,2] with A[1,0]*B[0,2] into one term because 
    # they contribute to different C elements (C[0,2] vs C[1,2]).
    # So we need 6 scalar terms for the k=2 column.
    # Total: 7 + 6 = 13. That's worse than 12 (schoolbook for 2x3x3).
    
    # Hmm, let me reconsider. Maybe 2x3x3 has rank 12, not 11.
    # Schoolbook: 2*3*3 = 18. With Strassen on the 2x2x2 core: 7 + 6 = 13.
    # That's not better than 12.
    
    # Let me try a different approach: use the fact that p=6 and try to find
    # a decomposition with rank less than 33.
    
    # Actually, let me just use the 3-block approach with rank 11 per block = 33,
    # and try to reduce it.
    
    # Alternative: use 2 blocks of 2x3x3 with rank 12 each = 24.
    # 2x3x3 schoolbook rank is 18, but we can do better.
    # Actually, the rank of 2x3x3 is at most 12 (using Strassen on 2x2x2 + 4 extra).
    # Let me verify: 2x3x3, C[i,k] = sum_j A[i,j]B[j,k].
    # Use Strassen for j=0,1 part (2x2x3): rank 7 for the 2x2x2 core, 
    # but we have 3 columns in C. 
    # This is getting complicated. Let me just go with rank 33 and try to reduce.
    
    return None  # This approach is getting too complicated, let's use rank 33

def _rank33_solution():
    """
    Build a rank-33 decomposition using 3 blocks of 2x3x2 with rank 11 each.
    """
    n, m, p = DIMENSIONS
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

def _try_rank_reduction(base_result, target_rank, deadline):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    """
    n, m, p = DIMENSIONS
    U = np.array(base_result['U'], dtype=float)
    V = np.array(base_result['V'], dtype=float)
    W = np.array(base_result['W'], dtype=float)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return base_result
    
    # Build tensor vectors for each column
    def tensor_col(r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    vec[idx] = U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        return vec
    
    # Try removing columns one at a time
    for remove_col in range(current_rank):
        if time.time() > deadline:
            break
        cols_to_keep = [c for c in range(current_rank) if c != remove_col]
        if len(cols_to_keep) < target_rank:
            break
        
        target = tensor_col(remove_col)
        M = np.zeros((n * m * p, len(cols_to_keep)))
        for col_idx, c in enumerate(cols_to_keep):
            M[:, col_idx] = tensor_col(c)
        
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            proj = U_svd @ (U_svd.T @ target)
            residual = target - proj
            if np.linalg.norm(residual) < 1e-8:
                U_new = U[:, cols_to_keep]
                V_new = V[:, cols_to_keep]
                W_new = W[:, cols_to_keep]
                result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                if _verify(result["U"], result["V"], result["W"]):
                    return result
        except np.linalg.LinAlgError:
            continue
    
    return base_result

def construct():
    start_time = time.time()
    deadline = 110.0
    
    best_result = _rank33_solution()
    
    if time.time() < deadline:
        reduced = _try_rank_reduction(best_result, 32, deadline)
        if reduced is not None:
            best_result = reduced
    
    if not _verify(best_result["U"], best_result["V"], best_result["W"]):
        best_result = _rank33_solution()
    
    return best_result
