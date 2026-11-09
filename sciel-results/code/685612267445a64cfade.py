import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U_S rows: 00, 01, 10, 11
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # V_S rows: 00, 01, 10, 11
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # W_S rows: 00, 01, 10, 11
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Schoolbook 3x3 decomposition
    # Rank 9
    # U_K rows: i*3+j
    U_K = np.eye(9, dtype=float)
    # V_K rows: j*3+k
    V_K = np.eye(9, dtype=float)
    # W_K rows: k*3+i
    # W_K[k*3+i, c] = 1 if c == j?
    # Condition: sum_c U_K[i,j,c] V_K[j,k,c] W_K[k,i,c] = 1
    # U_K[i,j,c] = delta_{j,c}
    # V_K[j,k,c] = delta_{j,c}
    # W_K[k,i,c] must be delta_{j,c} to make product 1.
    # So W_K[k,i,c] = 1 if c == j.
    # But j is not in the row index of W_K.
    # Wait. The condition is sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ki}^{(c)} = 1.
    # For schoolbook, c iterates over j.
    # U_{ij}^{(c)} = 1 if j==c else 0.
    # V_{jk}^{(c)} = 1 if j==c else 0.
    # W_{ki}^{(c)} = 1 if j==c else 0.
    # But W depends on k,i. It doesn't know j.
    # Ah, the decomposition is not unique.
    # For schoolbook, we can set:
    # U_{ij}^{(c)} = delta_{j,c}
    # V_{jk}^{(c)} = delta_{j,c}
    # W_{ki}^{(c)} = delta_{j,c} ... wait, W cannot depend on j.
    
    # Let's re-verify the schoolbook decomposition.
    # We want sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ki}^{(c)} = 1 for all i,j,k.
    # This is impossible if W doesn't depend on j, unless the sum is always 1.
    # If U, V are identity-like, U_{ij}^{(c)} V_{jk}^{(c)} is non-zero only if c=j.
    # Then sum_c ... = U_{ij}^{(j)} V_{jk}^{(j)} W_{ki}^{(j)}.
    # We need this to be 1.
    # So U_{ij}^{(j)} = 1, V_{jk}^{(j)} = 1, W_{ki}^{(j)} = 1.
    # This means W_{ki}^{(c)} = 1 for all c? No.
    # If W_{ki}^{(c)} = 1 for all c, then sum is 1.
    # But rank would be 9.
    # Let's check if W_{ki}^{(c)} = 1 works.
    # sum_c U_{ij}^{(c)} V_{jk}^{(c)} * 1.
    # U_{ij}^{(c)} = delta_{j,c}.
    # V_{jk}^{(c)} = delta_{j,c}.
    # Product is 1 if c=j, 0 otherwise.
    # Sum over c is 1.
    # So yes, W_K can be all ones?
    # But W_K shape is (9, 9).
    # W_K[k*3+i, c] = 1 for all k,i,c.
    # Let's verify.
    # sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ki}^{(c)}
    # = sum_c delta_{j,c} delta_{j,c} * 1
    # = delta_{j,j} = 1.
    # Yes.
    
    W_K = np.ones((9, 9), dtype=float)
    
    # Kronecker products
    U_kron = np.kron(U_S, U_K)
    V_kron = np.kron(V_S, V_K)
    W_kron = np.kron(W_S, W_K)
    
    # Permutations
    # U: rows (i_A, j_A, i_B, j_B) -> i_tot * 6 + j_tot
    # kron_idx = (i_A * 2 + j_A) * 9 + (i_B * 3 + j_B)
    # target_idx = (2 * i_A + i_B) * 6 + (2 * j_A + j_B)
    perm_U = np.zeros(36, dtype=int)
    for i_A in range(2):
        for j_A in range(2):
            for i_B in range(3):
                for j_B in range(3):
                    kron_idx = (i_A * 2 + j_A) * 9 + (i_B * 3 + j_B)
                    target_idx = (2 * i_A + i_B) * 6 + (2 * j_A + j_B)
                    perm_U[target_idx] = kron_idx
    
    # V: rows (j_A, k_A, j_B, k_B) -> j_tot * 6 + k_tot
    # kron_idx = (j_A * 2 + k_A) * 9 + (j_B * 3 + k_B)
    # target_idx = (2 * j_A + j_B) * 6 + (2 * k_A + k_B)
    perm_V = np.zeros(36, dtype=int)
    for j_A in range(2):
        for k_A in range(2):
            for j_B in range(3):
                for k_B in range(3):
                    kron_idx = (j_A * 2 + k_A) * 9 + (j_B * 3 + k_B)
                    target_idx = (2 * j_A + j_B) * 6 + (2 * k_A + k_B)
                    perm_V[target_idx] = kron_idx
    
    # W: rows (k_A, i_A, k_B, i_B) -> k_tot * 6 + i_tot
    # kron_idx = (k_A * 2 + i_A) * 9 + (k_B * 3 + i_B)
    # target_idx = (2 * k_A + k_B) * 6 + (2 * i_A + i_B)
    perm_W = np.zeros(36, dtype=int)
    for k_A in range(2):
        for i_A in range(2):
            for k_B in range(3):
                for i_B in range(3):
                    kron_idx = (k_A * 2 + i_A) * 9 + (k_B * 3 + i_B)
                    target_idx = (2 * k_A + k_B) * 6 + (2 * i_A + i_B)
                    perm_W[target_idx] = kron_idx
    
    U_final = U_kron[perm_U, :]
    V_final = V_kron[perm_V, :]
    W_final = W_kron[perm_W, :]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
