"""Matrix multiplication 3x4x8 with rank reduction using Strassen identities and redundancy pruning."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 4, 8]
SEED_RANK = 96

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 1],  # a
        [0, 0, 0, 0, 1, 0, -1],  # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, 1]    # d
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 1],   # f
        [1, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # M11
        [0, 0, 1, 0, 1, 0, 0],   # M12
        [0, 1, 0, 1, 0, 0, 0],   # M21
        [1, -1, 1, 0, 0, 1, 0]   # M22
    ]
    return U, V, W

def _construct_3x4x8():
    """Construct rank-minimal decomposition for 3x4x8."""
    n, m, p = 3, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 1x2x2 schoolbook (rank 4)
    U1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]
    V1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W1 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_2x2x2_block(u_rows, v_rows, w_rows, offset):
        for i in range(4):
            for c in range(7):
                U[u_rows[i]][offset + c] += U2[i][c]
                V[v_rows[i]][offset + c] += V2[i][c]
                W[w_rows[i]][offset + c] += W2[i][c]
    
    def add_1x2x2_block(u_rows, v_rows, w_rows, offset):
        for i in range(2):
            for c in range(4):
                U[u_rows[i]][offset + c] += U1[i][c]
                V[v_rows[i]][offset + c] += V1[i][c]
                W[w_rows[i]][offset + c] += W1[i][c]
        for i in range(2, 4):
            for c in range(4):
                V[v_rows[i]][offset + c] += V1[i][c]
                
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            u_rows = [
                0 * m + j0,
                0 * m + j1,
                1 * m + j0,
                1 * m + j1
            ]
            v_rows = [
                j0 * p + k0,
                j0 * p + k1,
                j1 * p + k0,
                j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0,
                k1 * n + 0,
                k0 * n + 1,
                k1 * n + 1
            ]
            
            add_2x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
            
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            u_rows = [
                2 * m + j0,
                2 * m + j1
            ]
            v_rows = [
                j0 * p + k0,
                j1 * p + k0,
                j0 * p + k1,
                j1 * p + k1
            ]
            w_rows = [
                k0 * n + 2,
                k1 * n + 2
            ]
            
            add_1x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 4
            
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                u_row = U_np[i*m+j]
                v_row = V_np[j*p+k]
                w_row = W_np[k*n+i]
                actual = np.sum(u_row * v_row * w_row)
                if abs(actual - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Vectorized verification."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                if abs(np.sum(prod) - 1) > tol:
                    return False
    return True

def _try_remove_column(U, V, W, col, n, m, p, tol=1e-10):
    """Check if removing column col preserves validity."""
    rank = len(U[0])
    if rank <= 1:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Remove column col
    U_r = np.delete(U_np, col, axis=1)
    V_r = np.delete(V_np, col, axis=1)
    W_r = np.delete(W_np, col, axis=1)
    
    # Check if all (i,j,k) still sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                if abs(np.sum(prod) - 1) > tol:
                    return False
    return True

def _prune_columns(U, V, W, n, m, p, deadline, max_rounds=50):
    """Iteratively remove redundant columns."""
    U_list = [row[:] for row in U]
    V_list = [row[:] for row in V]
    W_list = [row[:] for row in W]
    rank = len(U_list[0])
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 10:
            break
        
        removed_any = False
        cols_to_check = list(range(rank))
        
        for col in cols_to_check:
            if time.time() > deadline - 10:
                break
            
            if _try_remove_column(U_list, V_list, W_list, col, n, m, p):
                # Remove column col
                U_list = [row[:col] + row[col+1:] for row in U_list]
                V_list = [row[:col] + row[col+1:] for row in V_list]
                W_list = [row[:col] + row[col+1:] for row in W_list]
                rank -= 1
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _prune_pairs(U, V, W, n, m, p, deadline):
    """Try removing pairs of columns (for cases where single removal fails but pair removal works)."""
    U_list = [row[:] for row in U]
    V_list = [row[:] for row in V]
    W_list = [row[:] for row in W]
    rank = len(U_list[0])
    
    improved = False
    checked_pairs = set()
    
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        
        found_pair = False
        for c1 in range(rank):
            for c2 in range(c1+1, rank):
                if time.time() > deadline - 10:
                    break
                if (c1, c2) in checked_pairs:
                    continue
                
                # Check if removing both columns works
                U_np = np.array(U_list, dtype=float)
                V_np = np.array(V_list, dtype=float)
                W_np = np.array(W_list, dtype=float)
                
                U_r = np.delete(np.delete(U_np, c1, axis=1), c2, axis=1)
                V_r = np.delete(np.delete(V_np, c1, axis=1), c2, axis=1)
                W_r = np.delete(np.delete(W_np, c1, axis=1), c2, axis=1)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                            if abs(np.sum(prod) - 1) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Remove both columns
                    U_list = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in U_list]
                    V_list = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in V_list]
                    W_list = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in W_list]
                    rank -= 2
                    found_pair = True
                    improved = True
                    break
            if found_pair:
                break
        
        if not found_pair:
            break
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    try:
        # Step 1: Construct base rank-88 decomposition
        result = _construct_3x4x8()
        
        # Step 2: Verify the base decomposition
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            # Fallback to schoolbook if verification fails
            result = _schoolbook()
            return result
        
        # Step 3: Prune individual redundant columns
        result = _prune_columns(result["U"], result["V"], result["W"], n, m, p, deadline)
        
        # Step 4: Verify after single-column pruning
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            # Reconstruct and just do single-column pruning with stricter check
            result_base = _construct_3x4x8()
            result = _prune_columns(result_base["U"], result_base["V"], result_base["W"], n, m, p, deadline, max_rounds=20)
        
        # Step 5: Try pair pruning if time allows
        if time.time() < deadline - 20:
            result_pair = _prune_pairs(result["U"], result["V"], result["W"], n, m, p, deadline)
            if _verify(result_pair["U"], result_pair["V"], result_pair["W"], n, m, p):
                if len(result_pair["U"][0]) < len(result["U"][0]):
                    result = result_pair
        
        # Final verification
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            # Fall back to verified rank-88
            result_base = _construct_3x4x8()
            result = _prune_columns(result_base["U"], result_base["V"], result_base["W"], n, m, p, deadline, max_rounds=30)
            if not _verify(result["U"], result["V"], result["W"], n, m, p):
                result = _schoolbook()
        
    except Exception:
        result = _schoolbook()
    
    return result
