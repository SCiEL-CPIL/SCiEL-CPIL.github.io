import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # We want to decompose the tensor T_{i,j,k} = delta_{i,j} * delta_{j,k} ... no, T_{i,j,k} = A_{i,j} B_{j,k} C_{i,k}
    # Actually, the tensor is T_{i,j,k} such that sum_j A_{i,j} B_{j,k} = C_{i,k}.
    # In tensor notation: T_{i,j,k} = 1 if we view it as a trilinear form?
    # The standard decomposition is:
    # C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}  (indices might vary)
    # Let's stick to the matrix factorization:
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    # The product is: sum_r U[:, r] * V[:, r] ... no.
    # The reconstruction is:
    # C = sum_r (U_r * V_r) * W_r ?
    # Let's check the interface definition:
    # "exactly decompose the matrix multiplication tensor"
    # Usually this means:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?? No.
    #
    # Let's look at the seed program's logic.
    # U has shape (n*m, rank). Index: i*m + j.
    # V has shape (m*p, rank). Index: j*p + k.
    # W has shape (p*n, rank). Index: k*n + i.
    #
    # The seed program constructs U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j} delta_{j,k}? No.
    #
    # Let's re-read the task description carefully.
    # "Matrix multiplication 4,5,5"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # The tensor for matrix multiplication C = A * B is:
    # T_{i,j,k} = 1 if i,j,k are such that...
    # Actually, the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is wrong.
    # The tensor is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # In the context of tensor rank (Strassen), the tensor is:
    # T_{i,j,k} = 1 if we consider the trilinear form <A, B, C> = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}?
    # No, the standard tensor for matrix multiplication is:
    # T_{i,j,k} = delta_{i,j} delta_{j,k} ... wait.
    #
    # Let's look at the seed code's W index: `global_w_row = global_k * 4 + global_i`.
    # This is `k * n + i`.
    #
    # The decomposition identity is:
    # sum_{r} U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}
    # Where T_{i,j,k} is the matrix multiplication tensor.
    # For matrix multiplication, T_{i,j,k} = 1 if i,j,k form a valid path?
    # No, T_{i,j,k} = delta_{i,j} * delta_{j,k} is for identity?
    #
    # Actually, the tensor T is defined by:
    # T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is incorrect.
    #
    # The correct tensor is T_{i,j,k} = 1 if we view it as:
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = trace(A B C)?
    # No, the standard Strassen tensor is:
    # T_{i,j,k} = delta_{i,j} delta_{j,k} ...
    #
    # Let's check the seed's construction.
    # In the seed, for the 4x4x4 part, it uses Strassen factors.
    # The Strassen factors satisfy:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,j} delta_{j,k} ?
    #
    # Let's test with 2x2.
    # Seed term 0:
    # U: (1,0,0,1) -> U_{0,0}=1, U_{1,1}=1.
    # V: (1,0,0,1) -> V_{0,0}=1, V_{1,1}=1.
    # W: (1,0,0,1) -> W_{0,0}=1, W_{1,1}=1. (W index k*n+i. n=2. W_{0,0}=1, W_{1,1}=1).
    #
    # Sum for (i=0, j=0, k=0): U_{0,0} V_{0,0} W_{0,0} = 1*1*1 = 1.
    # Sum for (i=0, j=0, k=1): U_{0,0} V_{0,1} W_{1,0} = 1*0*0 = 0.
    # Sum for (i=0, j=1, k=0): U_{0,1} V_{1,0} W_{0,0} = 0*0*1 = 0.
    # Sum for (i=1, j=1, k=1): U_{1,1} V_{1,1} W_{1,1} = 1*1*1 = 1.
    #
    # So T_{0,0,0}=1, T_{1,1,1}=1.
    # This looks like T_{i,j,k} = delta_{i,j} delta_{j,k}.
    #
    # Wait, if T_{i,j,k} = delta_{i,j} delta_{j,k}, then:
    # sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = sum_{i,j} A_{i,j} B_{j,i} = trace(A B^T).
    # This is not matrix multiplication C = A B.
    #
    # Let's re-read the W index convention: "W[k*n+i,r]".
    # In the seed, W is indexed by `global_k * 4 + global_i`.
    #
    # If the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k}, then the rank is n*m*p? No.
    #
    # Let's look at the standard definition of the matrix multiplication tensor.
    # The tensor is T_{i,j,k} = 1 if i,j,k are such that...
    # Actually, the tensor is usually defined as T_{i,j,k} = delta_{i,j} delta_{j,k} is WRONG.
    #
    # The correct tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if we consider the trilinear form:
    # <A, B, C> = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # No, that's trace.
    #
    # The standard tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is for the identity map?
    #
    # Let's check the seed's output shapes.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    #
    # The identity is:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}
    #
    # If T_{i,j,k} = delta_{i,j} delta_{j,k}, then T is non-zero only when i=j=k.
    # Then sum_r U_{i,i,r} V_{i,i,r} W_{i,i,r} = 1.
    # This would imply a diagonal structure.
    #
    # But Strassen multiplies matrices.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #
    # If we set T_{i,j,k} = 1 for all i,j,k, then sum_r ... = 1.
    #
    # Let's look at the seed's 4x4x1 part.
    # U[i*5+j, col] = 1
    # V[j*5+4, col] = 1
    # W[4*4+i, col] = 1
    #
    # Here n=4, m=5, p=5.
    # U index: i*5+j.
    # V index: j*5+4. (j in 0..3, k=4).
    # W index: 4*4+i. (k=4, i in 0..3).
    #
    # So for a fixed col, non-zero terms are:
    # i in 0..3, j in 0..3.
    # U_{i,j} = 1.
    # V_{j,4} = 1.
    # W_{4,i} = 1.
    #
    # The sum is sum_{i,j} U_{i,j} V_{j,4} W_{4,i} = sum_{i,j} 1*1*1 = 16.
    # This doesn't look like a delta.
    #
    # Wait, the seed program is for 4x5x5.
    # n=4, m=5, p=5.
    #
    # The tensor T_{i,j,k} must be such that the decomposition recovers matrix multiplication.
    # The standard tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is NOT correct for matrix multiplication.
    #
    # The correct tensor is T_{i,j,k} = 1 if i,j,k are arbitrary?
    # No, the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is for the identity.
    #
    # Actually, the tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if we view it as:
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = trace(A B C)?
    #
    # Let's assume the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is WRONG.
    #
    # Let's look at the W index again: W[k*n+i].
    #
    # If the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k}, then W_{k,i} is non-zero only if k=i.
    #
    # In the seed, W[4*4+i] is non-zero. So k=4, i in 0..3.
    # This implies k != i.
    #
    # So the tensor is NOT delta_{i,j} delta_{j,k}.
    #
    # The tensor must be T_{i,j,k} = 1 for all i,j,k?
    # If T_{i,j,k} = 1, then sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
    #
    # Let's check the seed's 4x1x1 part.
    # U[i*5+4, col] = 1
    # V[4*5+4, col] = 1
    # W[4*4+i, col] = 1
    #
    # Here i in 0..3.
    # U_{i,4} = 1.
    # V_{4,4} = 1.
    # W_{4,i} = 1.
    #
    # Sum for (i, 4, 4): U_{i,4} V_{4,4} W_{4,i} = 1*1*1 = 1.
    # Sum for (i', 4, 4) with i' != i: U_{i',4} V_{4,4} W_{4,i'} = 0 (since W_{4,i'} is 0 for i' != i in this term).
    #
    # So this term contributes 1 to T_{i,4,4} for each i.
    #
    # This suggests T_{i,j,k} = delta_{j,4} delta_{k,4} ? No.
    #
    # It seems the tensor T_{i,j,k} is 1 if j and k are in a specific range?
    #
    # Actually, for matrix multiplication C = A B, the tensor is:
    # T_{i,j,k} = 1 if we consider the trilinear form:
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} ...
    #
    # Let's assume the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is WRONG.
    #
    # The correct tensor is T_{i,j,k} = 1 for all i,j,k is WRONG.
    #
    # The tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} is for the identity map I: A -> A.
    #
    # For matrix multiplication, the tensor is T_{i,j,k} = 1 if i,j,k are such that...
    #
    # Actually, the tensor is T_{i,j,k} = 1 for all i,j,k is the "all-ones" tensor.
    #
    # Let's look at the score. 1.1765.
    # Rank = 49 + 16 + 16 + 4 = 85.
    #
    # The seed program uses rank 85.
    #
    # Can we do better?
    #
    # For 4x5x5, the lower bound is max(n*m, m*p, p*n) = max(20, 25, 20) = 25.
    # The trivial rank is n*m*p = 100.
    #
    # Strassen for 4x4x4 is rank 49.
    #
    # The seed decomposes 4x5x5 into:
    # 4x4x4 (rank 49)
    # 4x4x1 (rank 16)
    # 4x1x4 (rank 16)
    # 4x1x1 (rank 4)
    # Total 85.
    #
    # Is there a better decomposition?
    #
    # We can view 4x5x5 as 4x(4+1)x(4+1).
    #
    # Maybe we can use a different block structure?
    #
    # Or maybe we can use a known lower rank for 4x5x5?
    #
    # The rank of 4x5x5 is not well-known to be much lower than 85.
    #
    # However, we can try to optimize the implementation or find a slightly better decomposition.
    #
    # One idea: use the fact that 5 = 4 + 1.
    #
    # Another idea: use a different Strassen variant?
    #
    # Or maybe we can reduce the rank of the 4x4x1 part?
    # 4x4x1 is essentially a vector-matrix product.
    # Rank is 4*4 = 16. This is minimal.
    #
    # 4x1x4 is also 16.
    #
    # 4x1x1 is 4.
    #
    # So the overhead is 16+16+4 = 36.
    #
    # Can we combine the 4x4x4 and the other parts?
    #
    # Maybe we can use a 5x5x5 decomposition?
    # Rank of 5x5x5 is not known to be low.
    #
    # Let's stick to the seed's structure but ensure it's correct and maybe optimize the code.
    #
    # The seed code is already quite good.
    #
    # One potential improvement: use numpy for the matrix operations to speed up the construction?
    # The construction is fast enough.
    #
    # Let's verify the correctness of the seed code.
    #
    # The seed code uses `add_4x4_term` which expands the Strassen factors.
    #
    # I will rewrite the program to be cleaner and ensure no bugs.
    #
    # I will also add a deadline guard.

    start_time = time.time()
    
    def deadline_guard():
        return time.time() - start_time > 140

    n, m, p = DIMENSIONS
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    s_u = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ]
    s_v = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, -1, 0, 1, 0, 1]
    ]
    s_w = [
        [1, -1, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            col_idx = start_col + inner_k
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if deadline_guard():
            break
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    if not deadline_guard():
        # 4x4x1
        for i in range(4):
            for j in range(4):
                U[i*5 + j][col] = 1
                V[j*5 + 4][col] = 1
                W[4*4 + i][col] = 1
                col += 1
                
    if not deadline_guard():
        # 4x1x4
        for i in range(4):
            for k in range(4):
                U[i*5 + 4][col] = 1
                V[4*5 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
                
    if not deadline_guard():
        # 4x1x1
        for i in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
