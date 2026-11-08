import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Products P1..P7
    # U coeffs: list of (local_i, local_j, coeff)
    # V coeffs: list of (local_j, local_k, coeff)
    # W coeffs: list of (local_k, local_i, coeff) for each product
    
    # P1: (A00 + A11)(B00 + B11)
    u1 = [(0, 0, 1), (1, 1, 1)]
    v1 = [(0, 0, 1), (1, 1, 1)]
    w1 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)] # C00:1, C11:1
    
    # P2: (A10 + A11)B00
    u2 = [(1, 0, 1), (1, 1, 1)]
    v2 = [(0, 0, 1)]
    w2 = [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, -1)] # C10:1, C11:-1
    
    # P3: A00(B01 - B11)
    u3 = [(0, 0, 1)]
    v3 = [(0, 1, 1), (1, 1, -1)]
    w3 = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)] # C01:1, C11:1
    
    # P4: A11(B10 - B00)
    u4 = [(1, 1, 1)]
    v4 = [(1, 0, 1), (0, 0, -1)]
    w4 = [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)] # C00:1, C10:1
    
    # P5: (A00 + A01)B11
    u5 = [(0, 0, 1), (0, 1, 1)]
    v5 = [(1, 1, 1)]
    w5 = [(0, 0, -1), (0, 1, 1), (1, 0, 0), (1, 1, 0)] # C00:-1, C01:1
    
    # P6: (A10 - A00)(B00 + B01)
    u6 = [(1, 0, 1), (0, 0, -1)]
    v6 = [(0, 0, 1), (0, 1, 1)]
    w6 = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)] # C11:1
    
    # P7: (A01 - A11)(B10 + B11)
    u7 = [(0, 1, 1), (1, 1, -1)]
    v7 = [(1, 0, 1), (1, 1, 1)]
    w7 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)] # C00:1
    
    products = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Blocks: (j_A_start, j_B_start, k_start)
    # Block 1: A cols 0-1, B rows 0-1, C cols 0-1
    # Block 2: A cols 2-3, B rows 2-3, C cols 0-1
    # Block 3: A cols 0-1, B rows 0-1, C cols 2-3
    # Block 4: A cols 2-3, B rows 2-3, C cols 2-3
    
    blocks = [
        (0, 0, 0),
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2)
    ]
    
    col_offset = 0
    for (ja_start, jb_start, k_start) in blocks:
        for r in range(7):
            u_coeffs, v_coeffs, w_coeffs = products[r]
            col = col_offset + r
            
            # Fill U
            for (li, lj, coeff) in u_coeffs:
                global_i = li
                global_j = ja_start + lj
                U[global_i * m + global_j][col] += coeff
                
            # Fill V
            for (lj, lk, coeff) in v_coeffs:
                global_j = jb_start + lj
                global_k = k_start + lk
                V[global_j * p + global_k][col] += coeff
                
            # Fill W
            for (lk, li, coeff) in w_coeffs:
                global_k = k_start + lk
                global_i = li
                W[global_k * n + global_i][col] += coeff
                
    return {"U": U, "V": V, "W": W}

# Verification (optional, for debugging)
def verify():
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = DIMENSIONS
    
    # Random test
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C = sum_r U[:, r] * V[:, r] * W[:, r] ? No.
    # The tensor is sum_r U_r \otimes V_r \otimes W_r
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Decomposition: C = sum_r (U_r \otimes A) (V_r \otimes B) W_r^T ?
    # Actually, the condition is:
    # C_{ik} = sum_r U_{i,m+j?} ... No.
    # U is (n*m) x rank. Index i*m+j.
    # V is (m*p) x rank. Index j*p+k.
    # W is (p*n) x rank. Index k*n+i.
    
    # The decomposition satisfies:
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # Where T_{i,j,k} = 1 if i,j,k are valid indices for the multiplication term A_{ij}B_{jk}.
    # Wait, the tensor T for matrix multiplication is defined such that
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    # This is equivalent to <A \otimes B, T_{i,k}> = C_{ik} ?
    # The standard tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # The rank decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # This implies:
    # 1 = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} for all i,j,k?
    # No, the tensor entries are 1 for all i,j,k.
    # So we must have sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k.
    
    # Let's check this condition.
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_reconstructed[i, j, k] = val
                
    # It should be all 1s.
    if not np.allclose(T_reconstructed, 1):
        print("Verification failed")
        print(T_reconstructed)
    else:
        print("Verification passed")

if __name__ == "__main__":
    verify()
