"""Strassen-based rank decomposition for 2x4x4 matrix multiplication."""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    r = U.shape[1]
    T = np.ones((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T_computed, T, atol=tol)

def _strassen_decomposition():
    """
    Build Strassen-based decomposition for C = A * B where A is 2x4, B is 4x4.
    
    Split A into two 2x2 blocks: A = [A11 A12; A21 A22] (rows 0-1, cols 0-3)
    Actually A is 2x4, so A = [A_top (2x4), A_bottom (2x4)] -- no, A is 2x4.
    
    Let me reconsider. A is 2x4, B is 4x4, C is 2x4.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Split B into two 4x2 blocks: B = [B_left (4x2), B_right (4x2)]
    C = [C_left (2x2), C_right (2x2)]
    
    C_left[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for k=0,1
    C_right[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for k=2,3
    
    Now split A into two 2x2 blocks: A = [A_left (2x2), A_right (2x2)]
    A_left = A[:, 0:2], A_right = A[:, 2:4]
    
    C_left = A_left * B_left + A_right * B_right  (each product is 2x2)
    C_right = A_left * B_left_right + A_right * B_right_right
    
    Wait, B is 4x4. Let me split B rows into two 2x4 blocks:
    B_top = B[0:2, :] (2x4), B_bot = B[2:4, :] (2x4)
    
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    Let B1 = B[0:2, :] (2x4), B2 = B[2:4, :] (2x4)
    
    C = A1 * B1 + A2 * B2  (2x4 = 2x2 * 2x4 + 2x2 * 2x4)
    
    Now for each 2x2 * 2x4 product, split the 2x4 into two 2x2:
    B1 = [B11 B12] where B11 = B[0:2,0:2], B12 = B[0:2,2:4]
    B2 = [B21 B22] where B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    C_left (2x2) = A1*B11 + A2*B21
    C_right (2x2) = A1*B12 + A2*B22
    
    For C_left: use Strassen on (A1, B11) and (A2, B21) -- but that's two separate 2x2x2 products.
    Each 2x2x2 Strassen uses 7 multiplications.
    
    Total: 4 Strassen products (A1*B11, A2*B21, A1*B12, A2*B22) = 4*7 = 28 terms.
    
    But we can do better by sharing. Let's use Strassen on the full structure.
    
    Actually, let's just build the 28-term decomposition directly.
    
    Strassen for 2x2: 
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    n, m, p = 2, 4, 4
    # We'll build the decomposition with 28 terms (4 Strassen 2x2x2 multiplications)
    # 
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Split: j in {0,1,2,3}, i in {0,1}, k in {0,1,2,3}
    # 
    # Group by (j_pair, k_pair):
    # j_pair 0 = {0,1}, j_pair 1 = {2,3}
    # k_pair 0 = {0,1}, k_pair 1 = {2,3}
    # 
    # C[i,k] = sum_{jp, kp} sum_{j in jp, k in kp} A[i,j]*B[j,k]
    # 
    # For each (jp, kp) pair, we have a 2x2x2 subproblem:
    # A_sub is 2x2 (rows i, cols j in jp)
    # B_sub is 2x2 (rows j in jp, cols k in kp)
    # C_sub is 2x2 (rows i, cols k in kp)
    # 
    # Apply Strassen to each of the 4 subproblems.
    
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen 7 terms for 2x2x2:
    # Term t=0..6
    # A indices (i, j_local), B indices (j_local, k_local), C indices (i, k_local)
    # 
    # M0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M1 = (A[1,0]+A[1,1])*B[0,0]
    # M2 = A[0,0]*(B[0,1]-B[1,1])
    # M3 = A[1,1]*(B[1,0]-B[0,0])
    # M4 = (A[0,0]+A[0,1])*B[1,1]
    # M5 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # 
    # C[0,0] = M0+M3-M4+M6
    # C[0,1] = M2+M4
    # C[1,0] = M1+M3
    # C[1,1] = M0-M1+M2+M5
    
    # For each subproblem (jp, kp), the local indices are:
    # i_local = i (0 or 1)
    # j_local = j - jp*2 (0 or 1)  
    # k_local = k - kp*2 (0 or 1)
    # 
    # Global indices:
    # U index: i*m + j = i*4 + (jp*2 + j_local)
    # V index: j*p + k = (jp*2 + j_local)*4 + (kp*2 + k_local)
    # W index: k*n + i = (kp*2 + k_local)*2 + i
    
    # Strassen coefficients for A (U factor):
    # M0: A[0,0]=1, A[1,1]=1
    # M1: A[1,0]=1, A[1,1]=1
    # M2: A[0,0]=1
    # M3: A[1,1]=1
    # M4: A[0,0]=1, A[0,1]=1
    # M5: A[1,0]=1, A[0,0]=-1
    # M6: A[0,1]=1, A[1,1]=-1
    
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    
    # Strassen coefficients for B (V factor):
    # M0: B[0,0]=1, B[1,1]=1
    # M1: B[0,0]=1
    # M2: B[0,1]=1, B[1,1]=-1
    # M3: B[1,0]=1, B[0,0]=-1
    # M4: B[1,1]=1
    # M5: B[0,0]=1, B[0,1]=1
    # M6: B[1,0]=1, B[1,1]=1
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M4... wait M5
        [(1,0,1), (1,1,1)],     # M6
    ]
    
    # Strassen coefficients for C (W factor):
    # C[0,0] = M0+M3-M4+M6
    # C[0,1] = M2+M4
    # C[1,0] = M1+M3
    # C[1,1] = M0-M1+M2+M5
    
    # W_coeffs[term][output_pos] where output_pos is (i_local, k_local)
    W_coeffs = [
        [(0,0,1), (0,1,0), (1,0,0), (1,1,0)],   # M0: C00+=1, C01+=0, C10+=0, C11+=0... 
    ]
    # Let me redo this more carefully.
    # For each Strassen term t (0-6), the contribution to each C[i_local, k_local]:
    # 
    # C[0,0] = M0 + M3 - M4 + M6
    # C[0,1] = M2 + M4
    # C[1,0] = M1 + M3
    # C[1,1] = M0 - M1 + M2 + M5
    
    # So W coefficient for term t at position (i_local, k_local):
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],    # M0: C00+=1, C11+=1
        1: [(1,0,1), (1,1,-1)],   # M1: C10+=1, C11+=-1
        2: [(0,1,1), (1,1,1)],    # M2: C01+=1, C11+=1
        3: [(0,0,1), (1,0,1)],    # M3: C00+=1, C10+=1
        4: [(0,0,-1), (0,1,1)],   # M4: C00+=-1, C01+=1
        5: [(1,1,1)],             # M5: C11+=1
        6: [(0,0,1)],             # M6: C00+=1
    }
    
    col = 0
    for jp in range(2):
        for kp in range(2):
            for t in range(7):
                # U coefficients
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                # V coefficients
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                # W coefficients
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

def _try_prune(U, V, W, n, m, p):
    """Try to remove zero columns or find linear dependencies to reduce rank."""
    # Check for zero columns
    r = U.shape[1]
    nonzero_cols = []
    for c in range(r):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            nonzero_cols.append(c)
    
    if len(nonzero_cols) < r:
        U = U[:, nonzero_cols]
        V = V[:, nonzero_cols]
        W = W[:, nonzero_cols]
    
    # Check for duplicate columns (up to sign)
    # This is complex; skip for now
    return U, V, W

def _to_list_of_lists(mat):
    return [[float(x) for x in row] for row in mat]

def construct():
    n, m, p = 2, 4, 4
    
    if _remaining() < 10.0:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    # Build Strassen decomposition
    U, V, W = _strassen_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Try to prune
        U, V, W = _try_prune(U, V, W, n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
        else:
            # Pruning broke it, use unpruned
            U, V, W = _strassen_decomposition()
            return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    else:
        # Strassen verification failed, fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
