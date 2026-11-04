import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Schoolbook rank is 64
    # Strassen-based rank is 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # We have 8 blocks of 2x2x2
    # j0 in {0, 2}
    # k0 in {0, 2, 4, 6}
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Check deadline
            if _time_left() < 1.0:
                break
            
            # Local indices:
            # i in {0, 1}
            # j in {j0, j0+1}
            # k in {k0, k0+1}
            
            # Strassen terms for 2x2x2
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # Let's define a helper to add a term
            def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
                # u_coeffs: dict {(i,j): val}
                # v_coeffs: dict {(j,k): val}
                # w_coeffs: dict {(k,i): val}
                for (i, j), val in u_coeffs.items():
                    if val != 0:
                        U[i*m + j][col_idx] += val
                for (j, k), val in v_coeffs.items():
                    if val != 0:
                        V[j*p + k][col_idx] += val
                for (k, i), val in w_coeffs.items():
                    if val != 0:
                        W[k*n + i][col_idx] += val
            
            # Term 1
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1
            
            # Term 2: M2 = (A10 + A11)B00
            # C10: 1
            # C11: -1
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col
            )
            col += 1
            
            # Term 3: M3 = A00(B01 - B11)
            # C01: 1
            # C11: 1
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1
            
            # Term 4: M4 = A11(B10 - B00)
            # C00: 1
            # C10: 1
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col
            )
            col += 1
            
            # Term 5: M5 = (A00 + A01)B11
            # C00: -1
            # C01: 1
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col
            )
            col += 1
            
            # Term 6: M6 = (A10 - A00)(B00 + B01)
            # C11: 1
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col
            )
            col += 1
            
            # Term 7: M7 = (A01 - A11)(B10 + B11)
            # C00: 1
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col
            )
            col += 1
    
    # Now try to reduce rank by zeroing out components
    # Check if decomposition is valid
    def check_validity(U, V, W, n, m, p):
        # The tensor T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 if this contributes to C[i,k], which it always does in the product
        # Actually the tensor is T[i,j,k] = 1 for all valid (i,j,k) since it's the multiplication tensor
        # Wait, let me reconsider. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # No, that's not right either.
        
        # The tensor for matrix multiplication of n×m by m×p matrices is:
        # T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k]
        # In standard matrix multiplication, every term A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid indices
        
        # Let's verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by zeroing out individual components
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # First verify the base decomposition is valid
    base_valid = check_validity(U, V, W, n, m, p)
    
    if not base_valid:
        # Return the base decomposition as-is
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by zeroing out rows/columns
    # Strategy: for each rank position, try zeroing it out and check validity
    # If valid, we can remove that rank position
    
    # Build a mask of which rank positions are "active"
    active = [True] * current_rank
    
    # Try removing each rank position one at a time
    for attempt in range(current_rank):
        if _time_left() < 5.0:
            break
        
        removed_any = False
        for r in range(current_rank):
            if not active[r]:
                continue
            if _time_left() < 2.0:
                break
            
            # Try zeroing out rank position r
            saved_U = U_np[:, r].copy()
            saved_V = V_np[:, r].copy()
            saved_W = W_np[:, r].copy()
            
            U_np[:, r] = 0
            V_np[:, r] = 0
            W_np[:, r] = 0
            
            # Check validity
            valid = check_validity(U_np.tolist(), V_np.tolist(), W_np.tolist(), n, m, p)
            
            if valid:
                # We can remove this rank position
                active[r] = False
                removed_any = True
            else:
                # Restore
                U_np[:, r] = saved_U
                V_np[:, r] = saved_V
                W_np[:, r] = saved_W
        
        if not removed_any:
            break
    
    # Count active rank positions
    active_indices = [i for i in range(current_rank) if active[i]]
    new_rank = len(active_indices)
    
    if new_rank < current_rank:
        # Compact the matrices
        final_U = U_np[:, active_indices].tolist()
        final_V = V_np[:, active_indices].tolist()
        final_W = W_np[:, active_indices].tolist()
        
        # Verify final result
        if check_validity(final_U, final_V, final_W, n, m, p):
            return {"U": final_U, "V": final_V, "W": final_W}
    
    # If no reduction was possible, return the original
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
