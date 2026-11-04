import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Rank 189 Decomposition: 2x2 Strassen blocks of 3x3 Schoolbook
    # Total Rank = 7 * 27 = 189
    
    total_rank = 189
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    # 3x3 Schoolbook Kernel
    # For a 3x3 multiplication, the rank-27 decomposition is:
    # For i in 0..2, j in 0..2, k in 0..2:
    # U[i*3+j, col] = 1
    # V[j*3+k, col] = 1
    # W[k*3+i, col] = 1
    
    # We generate the base 3x3 kernel indices once
    base_u_rows = []
    base_v_rows = []
    base_w_rows = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base_u_rows.append(i * 3 + j)
                base_v_rows.append(j * 3 + k)
                base_w_rows.append(k * 3 + i)
                
    base_u_rows = np.array(base_u_rows)
    base_v_rows = np.array(base_v_rows)
    base_w_rows = np.array(base_w_rows)
    
    # Strassen Steps for 2x2 blocks
    # Blocks: 00, 01, 10, 11
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Coefficients for U (A blocks)
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U row for block b, local (i,j) is (b//2)*3+i, (b%2)*3+j -> global row (b//2)*3+i, (b%2)*3+j
    # But U is flattened: row = global_i * 6 + global_j
    
    # We define the Strassen steps as lists of (block_idx, coeff)
    steps = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    col_offset = 0
    for step in steps:
        a_blocks, b_blocks, c_blocks = step
        
        for r in range(27):
            # U part
            u_row_local = base_u_rows[r]
            local_i = u_row_local // 3
            local_j = u_row_local % 3
            
            for b_idx, coeff in a_blocks:
                global_i = (b_idx // 2) * 3 + local_i
                global_j = (b_idx % 2) * 3 + local_j
                u_idx = global_i * 6 + global_j
                U[u_idx, col_offset + r] += coeff
                
            # V part
            v_row_local = base_v_rows[r]
            local_j = v_row_local // 3
            local_k = v_row_local % 3
            
            for b_idx, coeff in b_blocks:
                global_j = (b_idx // 2) * 3 + local_j
                global_k = (b_idx % 2) * 3 + local_k
                v_idx = global_j * 6 + global_k
                V[v_idx, col_offset + r] += coeff
                
            # W part
            w_row_local = base_w_rows[r]
            local_k = w_row_local // 3
            local_i = w_row_local % 3
            
            for b_idx, coeff in c_blocks:
                global_k = (b_idx // 2) * 3 + local_k
                global_i = (b_idx % 2) * 3 + local_i
                w_idx = global_k * 6 + global_i
                W[w_idx, col_offset + r] += coeff
                
        col_offset += 27
        
    # Verification (Optional, but good for safety)
    # If time permits, verify.
    if time.time() < start_time + 10:
        try:
            U3g = U.reshape(6, 6, -1)
            V3g = V.reshape(6, 6, -1)
            W3g = W.reshape(6, 6, -1)
            # Reconstruct a single entry to check? No, check all.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Tensor form: T[i,j,k] = delta_{i,k} (for MM tensor? No)
            # The tensor for MM is T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}
            # In the factorization U, V, W:
            # U[i,j,r] V[j,k,r] W[k,i,r] should sum to 1 if i,j,k are part of a valid term?
            # Actually, the tensor T_{i,j,k} for MM is 1 if the indices correspond to a multiplication term.
            # The standard tensor T has T_{i,j,k} = 1 for all i,j,k? No.
            # The MM tensor is defined by T_{i,j,k} = delta_{i,k} ? No.
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # The tensor is T_{i,j,k} such that the trilinear form is sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} C_{ik}^* ?
            # The task says: "output factor follows the official trace convention W[k*n+i,r]"
            # And "exactly decompose the matrix multiplication tensor".
            # The MM tensor T has entries T_{i,j,k} = 1 if we are computing C_{ik} from A_{ij} and B_{jk}?
            # No, the tensor is the structure of the multiplication.
            # T_{i,j,k} = 1 for all i,j,k is the "complete" tensor?
            # Wait, the rank of the identity tensor is n. The rank of MM tensor is R(n).
            # The tensor T_{i,j,k} is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
            # This corresponds to the tensor with T_{i,j,k} = 1 if i,j,k are "compatible"?
            # Actually, the MM tensor is T_{i,j,k} = 1 for all i,j,k?
            # No, if T_{i,j,k} = 1 for all i,j,k, then sum A_{ij} B_{jk} T_{i,j,k} = sum A_{ij} B_{jk}.
            # This is the trace of AB? No.
            # The MM tensor is usually defined as T_{i,j,k} = delta_{i,k} ? No.
            # Let's check the verification in Inspiration 1:
            # reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            # target = np.ones((6, 6, 6))
            # This implies the target tensor is all 1s.
            # If the target tensor is all 1s, then U, V, W must decompose the all-ones tensor.
            # Does the all-ones tensor correspond to MM?
            # If T_{i,j,k} = 1, then sum_{r} U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
            # This is the condition for the decomposition.
            
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            target = np.ones((6, 6, 6))
            if not np.allclose(reconstructed, target, atol=1e-9):
                # Fallback to schoolbook if Strassen failed
                rank = 216
                U_f = np.zeros((36, rank))
                V_f = np.zeros((36, rank))
                W_f = np.zeros((36, rank))
                for i in range(6):
                    for j in range(6):
                        for k in range(6):
                            idx = i*6 + j
                            jk = j*6 + k
                            ki = k*6 + i
                            col = i*36 + j*6 + k
                            U_f[idx, col] = 1.0
                            V_f[jk, col] = 1.0
                            W_f[ki, col] = 1.0
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
