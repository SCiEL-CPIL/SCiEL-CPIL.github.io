"""
Matrix multiplication 4x4x6 tensor decomposition.
Combines Strassen-based block decomposition with structured linear dependency elimination.
Target: find a rank < 77 decomposition by exploiting rectangular asymmetry.
"""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    1. Build a rank-77 decomposition using Kronecker product of Strassen (2x2x2, rank 7)
       and a rank-11 decomposition for 2x2x3.
    2. Apply structured linear dependency elimination to try to reduce rank.
    3. If elimination fails, return the best valid decomposition found.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strassen decomposition for 2x2x2:
    # U1, V1, W1 are (4, 7) matrices
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a10+a11)b00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a00(b01-b11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a11(b10-b00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a00+a01)b11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3: rank-11 decomposition
    # Terms 0-6: Strassen for first two columns
    # Terms 7-10: Trivial for third column
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        
        # V1 rows: j*2+k -> V2 rows: j*3+k
        V2[0][c] = V1[0][c]  # j=0,k=0
        V2[1][c] = V1[1][c]  # j=0,k=1
        V2[3][c] = V1[2][c]  # j=1,k=0
        V2[4][c] = V1[3][c]  # j=1,k=1
        V2[2][c] = 0         # j=0,k=2
        V2[5][c] = 0         # j=1,k=2
        
        # W1 rows: k*2+i -> W2 rows: k*2+i
        W2[0][c] = W1[0][c]  # k=0,i=0
        W2[1][c] = W1[1][c]  # k=0,i=1
        W2[2][c] = W1[2][c]  # k=1,i=0
        W2[3][c] = W1[3][c]  # k=1,i=1
        W2[4][c] = 0         # k=2,i=0
        W2[5][c] = 0         # k=2,i=1
    
    # Trivial part (column 2), terms 7-10
    U2[0][7] = 1
    V2[2][7] = 1
    W2[4][7] = 1
    
    U2[1][8] = 1
    V2[5][8] = 1
    W2[4][8] = 1
    
    U2[2][9] = 1
    V2[2][9] = 1
    W2[5][9] = 1
    
    U2[3][10] = 1
    V2[5][10] = 1
    W2[5][10] = 1
    
    # Now Kronecker product:
    # Total rank = 7 * 11 = 77
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    # Verify the decomposition
    def _verify(U_mat, V_mat, W_mat, n, m, p):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try structured linear dependency elimination
    # Look for integer linear combinations of rank-1 terms that sum to zero
    # This is a hard problem, so we'll try a heuristic approach
    
    def try_dependency_elimination(U, V, W, max_iter=50):
        """
        Try to find linear dependencies among the rank-1 terms.
        Returns (U', V', W', new_rank) if a reduction is found, else None.
        """
        current_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W
        best_rank = current_rank
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Randomly select a subset of columns to test for dependency
            subset_size = min(10, current_rank)
            cols = np.random.choice(current_rank, subset_size, replace=False)
            
            # Check if there's a linear dependency among these columns
            # We want to find coefficients c_r such that sum_r c_r * (u_r, v_r, w_r) = 0
            # This is equivalent to finding a null vector of the concatenated matrix
            
            # Build the concatenated matrix
            concat = np.vstack([
                U[:, cols].flatten(),
                V[:, cols].flatten(),
                W[:, cols].flatten()
            ])
            
            # Find null space
            try:
                U_s, s, Vt = np.linalg.svd(concat, full_matrices=False)
                # Null space is in the last row of Vt if s[-1] is close to 0
                if s[-1] < 1e-10:
                    null_vec = Vt[-1]
                    # Check if null_vec has integer or half-integer entries
                    # Scale to make entries integers
                    max_val = np.max(np.abs(null_vec))
                    if max_val > 0:
                        scaled = null_vec / max_val
                        # Check if entries are close to integers or half-integers
                        is_int = np.all(np.abs(scaled - np.round(scaled)) < 1e-6)
                        is_half = np.all(np.abs(2*scaled - np.round(2*scaled)) < 1e-6)
                        
                        if is_int or is_half:
                            # Found a dependency! Remove one column
                            # Determine which column to remove (the one with largest coefficient)
                            remove_idx = np.argmax(np.abs(null_vec))
                            remove_col = cols[remove_idx]
                            
                            # Remove the column
                            keep_cols = [c for c in range(current_rank) if c != remove_col]
                            new_U = U[:, keep_cols]
                            new_V = V[:, keep_cols]
                            new_W = W[:, keep_cols]
                            
                            # Verify
                            if _verify(new_U, new_V, new_W, n, m, p):
                                best_U, best_V, best_W = new_U, new_V, new_W
                                best_rank = current_rank - 1
                                U, V, W = new_U, new_V, new_W
                                current_rank = best_rank
                            else:
                                # Dependency doesn't actually work, continue
                                pass
            except Exception:
                continue
        
        if best_rank < current_rank:
            return best_U, best_V, best_W, best_rank
        return None
    
    # Try dependency elimination
    result = try_dependency_elimination(U, V, W, max_iter=30)
    if result is not None:
        U, V, W, new_rank = result
        print(f"Reduced rank from 77 to {new_rank}")
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to original rank-77
        rank = 7 * 11
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((24, rank), dtype=int)
        W = np.zeros((24, rank), dtype=int)
        
        for c1 in range(7):
            for c2 in range(11):
                c = c1 * 11 + c2
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                u1 = U1[i1*2+j1][c1]
                                u2 = U2[i2*2+j2][c2]
                                if u1 != 0 and u2 != 0:
                                    row = i1*8 + i2*4 + j1*2 + j2
                                    U[row][c] = u1 * u2
                
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(2):
                            for k2 in range(3):
                                v1 = V1[j1*2+k1][c1]
                                v2 = V2[j2*3+k2][c2]
                                if v1 != 0 and v2 != 0:
                                    row = j1*12 + j2*6 + k1*3 + k2
                                    V[row][c] = v1 * v2
                
                for k1 in range(2):
                    for k2 in range(3):
                        for i1 in range(2):
                            for i2 in range(2):
                                w1 = W1[k1*2+i1][c1]
                                w2 = W2[k2*2+i2][c2]
                                if w1 != 0 and w2 != 0:
                                    row = k1*12 + k2*4 + i1*2 + i2
                                    W[row][c] = w1 * w2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
