import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 6, 6, 6
    dim = n  # 6
    block_size = 3
    num_blocks = 2  # 6 = 2 * 3

    # Total rank: 7 (Strassen) * 27 (schoolbook 3x3) = 189
    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps for 2x2 block multiplication
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10 + A11) * B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00 * (B01 - B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11 * (B10 - B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00 + A01) * B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = block_size
    col = 0

    for a_list, b_list, c_list in strassen_steps:
        for r in range(bs):
            for s in range(bs):
                for t in range(bs):
                    # U: left factor, index = i*dim + j
                    for (I, J, ca) in a_list:
                        u_idx = (I * bs + r) * dim + (J * bs + s)
                        U[u_idx, col] += ca

                    # V: right factor, index = j*dim + k
                    for (J, K, cb) in b_list:
                        v_idx = (J * bs + s) * dim + (K * bs + t)
                        V[v_idx, col] += cb

                    # W: output factor, trace convention W[k*dim + i, r]
                    for (I, K, cc) in c_list:
                        w_idx = (K * bs + t) * dim + (I * bs + r)
                        W[w_idx, col] += cc

                    col += 1

                    # Periodic time check
                    if col % 27 == 0 and time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Optional: Validation step to ensure correctness
    # If validation fails, fall back to schoolbook 216
    try:
        # Reshape U, V, W to verify tensor contraction
        # U: (36, 189) -> (6, 6, 189)
        # V: (36, 189) -> (6, 6, 189)
        # W: (36, 189) -> (6, 6, 189)
        
        # We need to check if sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == (A@B)[i,k]
        # Wait, the decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition:
        # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k]) ... No.
        
        # The standard tensor decomposition form:
        # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} ?
        # For matrix multiplication:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # Tensor T_{ijk} where T_{ijk} = 1 if j is the middle index?
        # Actually, the factors are usually defined such that:
        # C_{ik} = sum_r (sum_j U_{ij}^{(r)} A_{ij}) (sum_j V_{jk}^{(r)} B_{jk}) ? No.
        
        # Let's look at the indices in the code:
        # U[u_idx, r] where u_idx = i*6 + j. So U[i,j,r].
        # V[v_idx, r] where v_idx = j*6 + k. So V[j,k,r].
        # W[w_idx, r] where w_idx = k*6 + i. So W[k,i,r].
        
        # The product for a specific r is:
        # X_{ij} = sum? No, U is linear in A.
        # The term is: (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k]) ?
        # No, that would be a scalar.
        
        # The correct contraction for matrix multiplication tensor:
        # C_{ik} = sum_{r} ( sum_{j} U_{ij}^{(r)} A_{ij} ) ? No.
        
        # Let's derive from the block construction.
        # For each r (which corresponds to a specific Strassen step and local indices):
        # M = X * Y
        # X is formed from A blocks. Y is formed from B blocks.
        # X_{local_u, local_v} = sum_{I,J} A_{I,J} * coeff_A
        # Y_{local_v, local_w} = sum_{J,K} B_{J,K} * coeff_B
        # M_{local_u, local_w} = sum_{local_v} X_{local_u, local_v} Y_{local_v, local_w}
        # C_{I,K} += M_{local_u, local_w} * coeff_C
        
        # So, for a fixed r:
        # C_{ik} += coeff_C * ( sum_{j} (sum_{I,J} A_{I,J} coeff_A) ... )
        
        # Actually, the tensor decomposition of matrix multiplication is:
        # T_{i,j,k} (where C_{ik} = sum_j A_{ij} B_{jk})
        # Can be written as sum_r U_{ir} V_{jr} W_{kr} ?
        # No, the indices are A_{ij}, B_{jk}, C_{ik}.
        # The factors are:
        # U_{i,j,r} (coefficient of A_{ij} in the r-th term)
        # V_{j,k,r} (coefficient of B_{jk} in the r-th term)
        # W_{k,i,r} (coefficient of C_{ik} in the r-th term)
        
        # The identity is:
        # sum_{i,j,k} A_{ij} B_{jk} C_{ki} = sum_r (sum_{i,j} U_{ijr} A_{ij}) (sum_{j,k} V_{jkr} B_{jk}) (sum_{k,i} W_{kir} C_{ki})
        # This is the trace form.
        # The task uses W[k*n+i, r].
        
        # To validate:
        # Pick random A, B.
        # Compute C = A @ B.
        # Compute S = sum_r (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k]) * (sum_{k,i} W[k,i,r] C[i,k]) ?
        # No, that's not how you recover C.
        
        # The decomposition asserts that the tensor T is a sum of rank-1 tensors.
        # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} is for general tensors.
        # For MatMul, the tensor is T_{ijk} = delta_{jk}? No.
        # T_{ijk} is 1 if the indices match the matmul pattern?
        # MatMul tensor: T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}.
        # This is equivalent to the tensor M_{ijk} where M_{ijk} = 1 if j is the contraction index?
        # Actually, the matmul tensor is defined by T_{i,j,k} = 1 if the operation is A_{ij} B_{jk} -> C_{ik}.
        # In index notation: T_{i,j,k} = \delta_{j,j} ? No.
        # The matmul tensor is rank 1 in each mode? No.
        
        # Let's just check the matrix multiplication result directly using the linear forms.
        # For each r:
        # L_r(A) = sum_{i,j} U[i,j,r] A[i,j]
        # R_r(B) = sum_{j,k} V[j,k,r] B[j,k]
        # Out_r = L_r(A) * R_r(B)
        # Then C_{ik} = sum_r W[k,i,r] Out_r ?
        # Let's check the block logic.
        # C_{I,K} += coeff_C * M_{u,w}
        # M_{u,w} = sum_v X_{u,v} Y_{v,w}
        # X_{u,v} = sum_{I,J} A_{I,J} coeff_A
        # Y_{v,w} = sum_{J,K} B_{J,K} coeff_B
        # So M_{u,w} = sum_{I,J,v,J',K} A_{I,J} B_{J',K} coeff_A coeff_B delta_{v,?}
        # This is getting complicated to validate analytically.
        
        # Simple validation:
        # Construct A, B.
        # Compute C_true = A @ B.
        # Compute C_decomp.
        # How to compute C_decomp from U, V, W?
        # C_decomp[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) ?
        # NO. That formula is for the rank-1 approximation of the tensor T_{ikj} = T_{ikj}.
        # The matmul tensor is T_{i,j,k} = 1.
        # The decomposition is:
        # sum_{i,j,k} A_{ij} B_{jk} C_{ki} = sum_r L_r(A) R_r(B) W_r(C)
        # This is a scalar identity.
        # To recover C, we need:
        # C_{ik} = sum_r W[k,i,r] (L_r(A) R_r(B)) ?
        # Let's test this hypothesis.
        # If C_{ik} = sum_r W[k,i,r] L_r(A) R_r(B), then:
        # sum_{i,k} C_{ik} D_{ik} = sum_r L_r(A) R_r(B) (sum_{i,k} W[k,i,r] D_{ik})
        # This matches the trace form if D is arbitrary.
        
        # So, to validate:
        # 1. Generate random A, B.
        # 2. Compute C_true = A @ B.
        # 3. For each r, compute L_r = sum U[:, r] * A_flat, R_r = sum V[:, r] * B_flat.
        # 4. Compute C_decomp_flat = sum_r W[:, r] * (L_r * R_r).
        # 5. Reshape C_decomp_flat to (6,6) using W index convention?
        # W is indexed by k*6 + i. So W_flat[k*6+i] corresponds to C[i,k].
        # So C_decomp[i,k] = C_decomp_flat[k*6 + i].
        # 6. Check if C_decomp == C_true.
        
        A = np.random.rand(6, 6)
        B = np.random.rand(6, 6)
        C_true = A @ B
        
        A_flat = A.flatten() # i*6 + j
        B_flat = B.flatten() # j*6 + k
        
        # L_r = U[:, r] dot A_flat
        # R_r = V[:, r] dot B_flat
        # Prod = L_r * R_r
        # C_decomp_flat = W dot Prod
        
        L = U.T @ A_flat  # Shape (rank,)
        R = V.T @ B_flat  # Shape (rank,)
        Prod = L * R      # Shape (rank,)
        C_decomp_flat = W.T @ Prod # Shape (36,)
        
        # Reshape C_decomp_flat to (6,6)
        # Index in flat: k*6 + i
        # So C_decomp[i, k] = C_decomp_flat[k*6 + i]
        C_decomp = np.zeros((6,6))
        for k in range(6):
            for i in range(6):
                C_decomp[i, k] = C_decomp_flat[k*6 + i]
                
        if not np.allclose(C_decomp, C_true):
            # Fallback to Schoolbook 216
            rank_sb = 216
            U_sb = np.zeros((36, rank_sb))
            V_sb = np.zeros((36, rank_sb))
            W_sb = np.zeros((36, rank_sb))
            col = 0
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        U_sb[i*6+j, col] = 1.0
                        V_sb[j*6+k, col] = 1.0
                        W_sb[k*6+i, col] = 1.0
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
            
    except Exception:
        # If validation fails due to error, fallback to Schoolbook
        rank_sb = 216
        U_sb = np.zeros((36, rank_sb))
        V_sb = np.zeros((36, rank_sb))
        W_sb = np.zeros((36, rank_sb))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_sb[i*6+j, col] = 1.0
                    V_sb[j*6+k, col] = 1.0
                    W_sb[k*6+i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
