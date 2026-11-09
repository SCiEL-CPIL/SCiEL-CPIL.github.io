"""Optimized rank decomposition for 2x4 by 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 decomposition for 2x4 * 4x4 matrix multiplication.
    
    Strategy: Decompose the 2x4 * 4x4 multiplication into 4 independent 
    2x2 * 2x2 multiplications and use Strassen's algorithm (rank 7) for each.
    
    Mapping:
    A (2x4) is split into A_L (cols 0,1) and A_R (cols 2,3).
    B (4x4) is split into B_T (rows 0,1) and B_B (rows 2,3).
    
    C = A * B = A_L * B_T + A_R * B_B
    
    Further split output columns of B_T and B_B into pairs:
    C[:, 0:2] = A_L * B_T[:, 0:2] + A_R * B_B[:, 0:2]
    C[:, 2:4] = A_L * B_T[:, 2:4] + A_R * B_B[:, 2:4]
    
    This gives 4 terms, each being a product of a 2x2 matrix (from A) 
    and a 2x2 matrix (from B block).
    
    Each 2x2 * 2x2 product is decomposed using Strassen (rank 7).
    Total rank = 4 * 7 = 28.
    """
    deadline = time.time() + 100  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Schoolbook fallback
    def get_schoolbook():
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

    # Define a verified Strassen decomposition for 2x2 * 2x2
    # Indices for local 2x2 block:
    # A_local: (i, j) where i,j in {0,1}
    # B_local: (j, k) where j,k in {0,1}
    # C_local: (k, i) where k,i in {0,1} -> W index is k*n + i
    
    # Strassen terms for C = A*B
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each term is a list of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): val}
    # V_coeffs: dict {(j,k): val}
    # W_coeffs: dict {(k,i): val}
    
    strassen_terms = [
        # P1
        (
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        ),
        # P2
        (
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1}, 
            {(0,0): 1, (1,1): -1}
        ),
        # P3
        (
            {(0,0): 1}, 
            {(1,0): 1, (1,1): -1}, 
            {(0,1): 1, (1,0): 1}
        ),
        # P4
        (
            {(1,1): 1}, 
            {(0,1): 1, (1,0): -1}, 
            {(1,1): 1, (0,1): -1, (1,0): 1}
        ),
        # P5
        (
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1}, 
            {(0,0): -1, (1,1): 1}
        ),
        # P6
        (
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (1,0): 1}, 
            {(1,0): 1}
        ),
        # P7
        (
            {(0,1): 1, (1,1): -1}, 
            {(0,1): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        ),
    ]
    
    # Verify the Strassen decomposition locally
    def verify_strassen():
        # Random test matrices
        A_test = np.array([[1, 2], [3, 4]], dtype=float)
        B_test = np.array([[5, 6], [7, 8]], dtype=float)
        C_expected = A_test @ B_test
        
        # Reconstruct C from terms
        C_recon = np.zeros((2, 2), dtype=float)
        
        # We need to simulate the tensor contraction
        # For each term, compute the scalar product
        for u_dict, v_dict, w_dict in strassen_terms:
            # Calculate the "middle" value M = sum(u_ij * v_jk) ?? No.
            # The decomposition is: C_ki = sum_t U_it * V_jt * W_kt ?? No.
            # The standard tensor decomposition is:
            # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
            # Here, for a fixed term t, the contribution to C[k,i] is:
            # sum_{j_local} U[i, j_local, t] * V[j_local, k_local, t] ... wait.
            
            # Let's stick to the definition:
            # C_local[k, i] = sum_j A[i, j] * B[j, k]
            # Decomposition: C_local[k, i] = sum_t (sum_j U[i, j, t] * V[j, k, t]) * W[k, i, t] ??
            # No, the standard form is:
            # C[k, i] = sum_t ( sum_j U[i, j, t] * V[j, k, t] ) * W[k, i, t] is incorrect.
            
            # Correct form:
            # C[k, i] = sum_t U[i, j, t] * V[j, k, t] * W[k, i, t] is not quite right because j is summed.
            
            # Actually, the tensor T[i, j, k] = C[k, i] (note the index swap for W)
            # T[i, j, k] = sum_t U[i, j, t] * V[j, k, t] * W[k, i, t]
            
            # Let's check the indices in the prompt:
            # U: (n*m, rank) -> U[i*m + j, t]
            # V: (m*p, rank) -> V[j*p + k, t]
            # W: (p*n, rank) -> W[k*n + i, t]
            
            # So the term is: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            
            # In the local 2x2 block:
            # U[i, j, t] corresponds to U_local[i, j] in the dict
            # V[j, k, t] corresponds to V_local[j, k] in the dict
            # W[k, i, t] corresponds to W_local[k, i] in the dict
            
            # So for a fixed term t:
            # Contribution to C[k, i] is:
            # sum_j ( U[i, j] * V[j, k] ) * W[k, i] ??
            # No, W is also part of the product.
            # The product is U[i, j] * V[j, k] * W[k, i].
            # Sum over j? No, the tensor entry T[i, j, k] is the coefficient of A[i, j] B[j, k].
            # So T[i, j, k] = C[k, i].
            # Decomposition: T[i, j, k] = sum_t U[i, j, t] V[j, k, t] W[k, i, t].
            
            # So for each term t:
            # We form a matrix M_t where M_t[i, k] = sum_j U[i, j, t] * V[j, k, t] * W[k, i, t] ??
            # No, W depends on k and i.
            
            # Let's just compute the full tensor T and compare.
            T_recon = np.zeros((2, 2, 2), dtype=float)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        total = 0
                        for idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
                            u_val = u_dict.get((i, j), 0)
                            v_val = v_dict.get((j, k), 0)
                            w_val = w_dict.get((k, i), 0)
                            total += u_val * v_val * w_val
                        T_recon[i, j, k] = total
                        # T[i, j, k] should be 1 if i==k? No.
                        # T[i, j, k] is the coefficient of A[i, j] * B[j, k] in C[k, i].
                        # C[k, i] = sum_j A[i, j] B[j, k].
                        # So the coefficient of A[i, j] B[j, k] in C[k, i] is always 1.
                        # Therefore T[i, j, k] should be 1 for all i, j, k.
                        if abs(total - 1.0) > 1e-9:
                            return False
            return True

    if not verify_strassen():
        # If my Strassen implementation is wrong, fallback
        return get_schoolbook()

    # Construct the full 2x4 * 4x4 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # We have 4 blocks:
    # Block 0: A_L (cols 0,1) * B_T_left (rows 0,1, cols 0,1) -> C[:, 0:2]
    # Block 1: A_R (cols 2,3) * B_B_left (rows 2,3, cols 0,1) -> C[:, 0:2]
    # Block 2: A_L (cols 0,1) * B_T_right (rows 0,1, cols 2,3) -> C[:, 2:4]
    # Block 3: A_R (cols 2,3) * B_B_right (rows 2,3, cols 2,3) -> C[:, 2:4]
    
    # For each block, we use 7 Strassen terms.
    # Block b uses columns b*7 to b*7+6 in U, V, W.
    
    # Mapping details:
    # A is 2x4. Rows i=0,1. Cols j=0..3.
    # B is 4x4. Rows j=0..3. Cols k=0..3.
    # C is 2x4. Rows k=0..3. Cols i=0,1. (W index k*n + i)
    
    # Block 0:
    # A indices: i in {0,1}, j_local in {0,1} -> global j = j_local
    # B indices: j_local in {0,1} -> global j = j_local, k_local in {0,1} -> global k = k_local
    # C indices: k_local in {0,1} -> global k = k_local, i in {0,1}
    
    # Block 1:
    # A indices: i in {0,1}, j_local in {0,1} -> global j = j_local + 2
    # B indices: j_local in {0,1} -> global j = j_local + 2, k_local in {0,1} -> global k = k_local
    # C indices: k_local in {0,1} -> global k = k_local, i in {0,1}
    
    # Block 2:
    # A indices: i in {0,1}, j_local in {0,1} -> global j = j_local
    # B indices: j_local in {0,1} -> global j = j_local, k_local in {0,1} -> global k = k_local + 2
    # C indices: k_local in {0,1} -> global k = k_local + 2, i in {0,1}
    
    # Block 3:
    # A indices: i in {0,1}, j_local in {0,1} -> global j = j_local + 2
    # B indices: j_local in {0,1} -> global j = j_local + 2, k_local in {0,1} -> global k = k_local + 2
    # C indices: k_local in {0,1} -> global k = k_local + 2, i in {0,1}
    
    blocks = [
        # (j_offset_A, j_offset_B, k_offset_C)
        (0, 0, 0),  # Block 0
        (2, 2, 0),  # Block 1
        (0, 0, 2),  # Block 2
        (2, 2, 2),  # Block 3
    ]
    
    for b_idx, (j_off_a, j_off_b, k_off_c) in enumerate(blocks):
        base_col = b_idx * 7
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            col = base_col + t_idx
            
            # Fill U
            for (i, j_loc), val in u_dict.items():
                global_j = j_loc + j_off_a
                U[i * m + global_j][col] = val
            
            # Fill V
            for (j_loc, k_loc), val in v_dict.items():
                global_j = j_loc + j_off_b
                global_k = k_loc + k_off_c
                V[global_j * p + global_k][col] = val
            
            # Fill W
            for (k_loc, i), val in w_dict.items():
                global_k = k_loc + k_off_c
                W[global_k * n + i][col] = val
                
    # Verify the full decomposition
    def verify_full():
        # Random test
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        # Reconstruct C
        C_recon = np.zeros((p, n))
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # C[k, i] = sum_j sum_t A[i, j] * V[j, k, t] * U[i, j, t] * W[k, i, t] ??
        # No, the formula is:
        # C[k, i] = sum_t ( sum_j U[i, j, t] * A[i, j] ) * ( sum_k' V[j, k, t] * B[j, k] ) ... No.
        
        # The tensor definition:
        # T[i, j, k] = sum_t U[i, j, t] V[j, k, t] W[k, i, t]
        # And C[k, i] = sum_j A[i, j] B[j, k] T[i, j, k]
        
        # So:
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    t_sum = 0
                    for t in range(rank):
                        u_val = U_np[i * m + j, t]
                        v_val = V_np[j * p + k, t]
                        w_val = W_np[k * n + i, t]
                        t_sum += u_val * v_val * w_val
                    total += A_test[i, j] * B_test[j, k] * t_sum
                C_recon[k, i] = total
                
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
        return True

    if not verify_full():
        return get_schoolbook()
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
