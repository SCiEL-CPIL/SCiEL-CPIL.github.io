import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def verify_exact(U, V, W, tol=1e-8):
        r = U.shape[1]
        U_tens = U.reshape(n, m, r)
        V_tens = V.reshape(m, p, r)
        W_tens = W.reshape(p, n, r)
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_tens, V_tens, W_tens)
        T_target = np.ones((n, m, p))
        err = np.max(np.abs(T_recon - T_target))
        return err < tol, T_recon

    # Schoolbook construction as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen-based construction for rank 54
    def strassen_construction():
        # Split columns: 5 = 2+2+1, rows: 2, columns of B: 6 = 2+2+2
        # Block 1: j=0,1 (2x2 A block, 2x2 B blocks for k=0,1; k=2,3; k=4,5)
        # Block 2: j=2,3 (2x2 A block)
        # Block 3: j=4 (1x1, schoolbook)
        
        # Strassen 2x2: 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        rank = 0
        U = np.zeros((n * m, 54))
        V = np.zeros((m * p, 54))
        W = np.zeros((p * n, 54))
        
        def add_strassen_block(a_row_start, b_col_start, col_offset):
            # a_row_start: starting row index in A for this 2x2 block
            # b_col_start: starting column index in B for this 2x2 block
            # Each Strassen multiplication contributes 7 terms
            
            # A block indices: (a_row_start, b_col_start), (a_row_start, b_col_start+1), 
            #                   (a_row_start+1, b_col_start), (a_row_start+1, b_col_start+1)
            # B block indices: (b_col_start, k), (b_col_start+1, k), etc.
            
            # M1 = (A00+A11)(B00+B11)
            U[a_row_start * m + b_col_start, col_offset] = 1
            U[(a_row_start + 1) * m + b_col_start + 1, col_offset] = 1
            V[b_col_start * p + 0, col_offset] = 1
            V[b_col_start * p + 1, col_offset] = 1
            # Contribution to C: M1 goes to C00 and C11
            W[0 * n + 0, col_offset] = 1  # C[0,0]
            W[0 * n + 1, col_offset] = 0  # C[0,1]
            W[1 * n + 0, col_offset] = 0  # C[1,0]
            W[1 * n + 1, col_offset] = 1  # C[1,1]
            # Wait, this is getting complex. Let me use a different approach.
            pass
        
        # Actually, let me construct it more carefully.
        # For a 2x2 A block multiplied by a 2x2 B block (each 2x2), Strassen uses 7 multiplications.
        # But here, the A block is 2x2 (rows 0-1, cols j,j+1) and B block is 2x2 (rows j,j+1, cols k,k+1).
        # The result is 2x2 (rows 0-1, cols k,k+1).
        
        # Let me build it term by term.
        # For each 2x2 A block and 2x2 B block, we need 7 terms.
        
        # Define Strassen coefficients
        # M1 = (A00+A11) * (B00+B11) -> contributes to C00 (+1), C11 (+1)
        # M2 = (A10+A11) * B00 -> contributes to C10 (+1), C11 (-1)
        # M3 = A00 * (B10-B11) -> contributes to C01 (+1), C10 (+1)
        # M4 = A11 * (B01-B00) -> contributes to C00 (+1), C10 (+1)
        # M5 = (A00+A01) * B11 -> contributes to C00 (-1), C01 (+1)
        # M6 = (A10-A00) * (B00+B10) -> contributes to C11 (+1)
        # M7 = (A01-A11) * (B01+B11) -> contributes to C00 (+1), C11 (+1)
        
        # Let me enumerate each multiplication term.
        # Term index, A coefficient, B coefficient, C coefficient
        
        strassen_terms = [
            # (A00, A11, A01, A10, B00, B01, B10, B11, C00, C01, C10, C11)
            (1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1),  # M1
            (0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, -1), # M2
            (1, 0, 0, 0, 0, 0, 1, -1, 0, 1, 1, 0), # M3
            (0, 1, 0, 0, -1, 1, 0, 0, 1, 0, 1, 0), # M4
            (1, 0, 1, 0, 0, 0, 0, 1, -1, 1, 0, 0), # M5
            (0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1),  # M6
            (0, -1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1), # M7
        ]
        
        col = 0
        # 2x2 A blocks at j=0,1 and j=2,3
        # 2x2 B blocks at k=0,1; k=2,3; k=4,5
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                for term in strassen_terms:
                    a00, a11, a01, a10, b00, b01, b10, b11, c00, c01, c10, c11 = term
                    # A indices: row 0/1, col j_start+j_start+1
                    if a00 != 0:
                        U[0 * m + j_start, col] += a00
                    if a11 != 0:
                        U[1 * m + j_start + 1, col] += a11
                    if a01 != 0:
                        U[0 * m + j_start + 1, col] += a01
                    if a10 != 0:
                        U[1 * m + j_start, col] += a10
                    
                    # B indices: row j_start+j_start+1, col k_start+k_start+1
                    if b00 != 0:
                        V[j_start * p + k_start, col] += b00
                    if b01 != 0:
                        V[j_start * p + k_start + 1, col] += b01
                    if b10 != 0:
                        V[(j_start + 1) * p + k_start, col] += b10
                    if b11 != 0:
                        V[(j_start + 1) * p + k_start + 1, col] += b11
                    
                    # C indices: row 0/1, col k_start+k_start+1
                    if c00 != 0:
                        W[k_start * n + 0, col] += c00
                    if c01 != 0:
                        W[(k_start + 1) * n + 0, col] += c01
                    if c10 != 0:
                        W[k_start * n + 1, col] += c10
                    if c11 != 0:
                        W[(k_start + 1) * n + 1, col] += c11
                    
                    col += 1
        
        # Block 3: j=4, schoolbook
        for k in range(6):
            # A[0,4] * B[4,k] -> C[0,k]
            U[0 * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            # A[1,4] * B[4,k] -> C[1,k]
            U[1 * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + 1, col] = 1
            col += 1
        
        # col should be 54
        assert col == 54, f"Expected 54, got {col}"
        
        # Verify
        err, _ = verify_exact(U, V, W)
        if err:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Return schoolbook if Strassen fails
            return schoolbook()
    
    # Try Strassen construction
    result = strassen_construction()
    rank = len(result['U'][0]) if result['U'] else 60
    print(f"Constructed rank: {rank}")
    
    return result

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    print(f"Rank: {rank}")
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    U_tens = U_np.reshape(n, m, rank)
    V_tens = V_np.reshape(m, p, rank)
    W_tens = W_np.reshape(p, n, rank)
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_tens, V_tens, W_tens)
    T_target = np.ones((n, m, p))
    err = np.max(np.abs(T_recon - T_target))
    print(f"Error: {err}")
