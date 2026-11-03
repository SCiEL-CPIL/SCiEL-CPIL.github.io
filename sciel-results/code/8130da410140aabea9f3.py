import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i

    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strategy: Split the 3x6 matrix into two 3x3 blocks
    # B = [B1 | B2] where B1 = B[:, 0:3], B2 = B[:, 3:6]
    # C = A*B1 + A*B2 = C1 + C2
    # where C1 = A*B1 (2x3) and C2 = A*B2 (2x3)
    
    # For each 2x3x3 sub-problem, find a rank-r decomposition
    # Then concatenate to get a rank-2r decomposition of 2x3x6
    
    # Build tensor for 2x3x3 sub-problem
    # Sub-problem 1: k in {0,1,2}, so c = 2k+i in {0,1,2,3,4,5}
    # Sub-problem 2: k in {3,4,5}, so c = 2k+i in {6,7,8,9,10,11}
    
    # For 2x3x3: a = i*3+j (i in {0,1}, j in {0,1,2}), b = j*3+k (j in {0,1,2}, k in {0,1,2})
    # c_sub = 2k+i (k in {0,1,2}, i in {0,1}) -> c_sub in {0,...,5}
    
    sub_valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a = i * m + j
                b = j * 3 + k
                c_sub = 2 * k + i
                sub_valid_triples.add((a, b, c_sub))
    
    sub_num_A = 6
    sub_num_B = 9
    sub_num_C = 6

    # Try to find low-rank decomposition for 2x3x3 sub-problem
    # Schoolbook rank for 2x3x3 is 18
    # Try ranks: 16, 15, 14, ...
    
    found_sub = False
    best_sub_rank = 18
    best_sub_U = best_sub_V = best_sub_W = None

    for target_sub_rank in [16, 15, 14, 13, 12]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(1000 * target_sub_rank + restart)
            U_curr = np.random.randn(sub_num_A, target_sub_rank) * 0.5
            V_curr = np.random.randn(sub_num_B, target_sub_rank) * 0.5
            W_curr = np.random.randn(sub_num_C, target_sub_rank) * 0.5
            
            lr = 0.05
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                    
                # Compute gradients
                grad_U = np.zeros_like(U_curr)
                grad_V = np.zeros_like(V_curr)
                grad_W = np.zeros_like(W_curr)
                residual_sum = 0
                
                # Forward pass: compute current values
                # For efficiency, precompute partial sums
                UV = np.zeros((sub_num_A, sub_num_B, target_sub_rank))
                for r in range(target_sub_rank):
                    UV[:, :, r] = np.outer(U_curr[:, r], V_curr[:, r])
                
                # For each valid triple, compute contribution
                for (a, b, c) in sub_valid_triples:
                    val = np.sum(UV[a, b, :] * W_curr[c, :])
                    err = val - 1.0
                    residual_sum += err * err
                    
                    # Gradient: d/da = V[b,r]*W[c,r]*err
                    for r in range(target_sub_rank):
                        grad_U[a, r] += err * V_curr[b, r] * W_curr[c, r]
                        grad_V[b, r] += err * U_curr[a, r] * W_curr[c, r]
                        grad_W[c, r] += err * U_curr[a, r] * V_curr[b, r]
                
                # Also penalize invalid positions (sample)
                np.random.seed(restart * 1000 + step)
                for _ in range(20):
                    a = np.random.randint(0, sub_num_A)
                    b = np.random.randint(0, sub_num_B)
                    c = np.random.randint(0, sub_num_C)
                    if (a, b, c) not in sub_valid_triples:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        err = val
                        residual_sum += err * err * 0.5
                        for r in range(target_sub_rank):
                            grad_U[a, r] += 0.5 * err * V_curr[b, r] * W_curr[c, r]
                            grad_V[b, r] += 0.5 * err * U_curr[a, r] * W_curr[c, r]
                            grad_W[c, r] += 0.5 * err * U_curr[a, r] * V_curr[b, r]
                
                # Update
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                lr *= 0.995
                
                if residual_sum < 1e-10:
                    found = True
                    break
            
            if found:
                # Verify
                max_err = 0
                for (a, b, c) in sub_valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                # Check some invalid positions
                for _ in range(100):
                    a = np.random.randint(0, sub_num_A)
                    b = np.random.randint(0, sub_num_B)
                    c = np.random.randint(0, sub_num_C)
                    if (a, b, c) not in sub_valid_triples:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        max_err = max(max_err, abs(val))
                
                if max_err < 1e-8:
                    best_sub_rank = target_sub_rank
                    best_sub_U, best_sub_V, best_sub_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    found_sub = True
                    break
        
        if found_sub:
            break
    
    # If we found a sub-decomposition, construct the full 2x3x6 decomposition
    if found_sub:
        sub_rank = best_sub_rank
        full_rank = 2 * sub_rank
        
        U_full = np.zeros((num_A, full_rank))
        V_full = np.zeros((num_B, full_rank))
        W_full = np.zeros((num_C, full_rank))
        
        # First block: k in {0,1,2}
        # a = i*3+j (same as sub), b = j*6+k for k in {0,1,2}, c = 2k+i
        # Map sub indices to full indices
        for r in range(sub_rank):
            U_full[:, r] = best_sub_U[:, r]
            # V: sub b = j*3+k -> full b = j*6+k
            for j in range(3):
                for k in range(3):
                    sub_b = j * 3 + k
                    full_b = j * 6 + k
                    V_full[full_b, r] = best_sub_V[sub_b, r]
            # W: sub c = 2k+i -> full c = 2k+i (same, since k < 3)
            for k in range(3):
                for i in range(2):
                    sub_c = 2 * k + i
                    full_c = 2 * k + i
                    W_full[full_c, r] = best_sub_W[sub_c, r]
        
        # Second block: k in {3,4,5}
        # a = i*3+j (same), b = j*6+k for k in {3,4,5}, c = 2k+i for k in {3,4,5}
        # k_sub = k - 3, so sub b = j*3+k_sub, sub c = 2*k_sub+i
        # full c = 2k+i = 2*(k_sub+3)+i = 2*k_sub+i+6
        offset = sub_rank
        for r in range(sub_rank):
            U_full[:, offset + r] = best_sub_U[:, r]
            for j in range(3):
                for k_sub in range(3):
                    k = k_sub + 3
                    sub_b = j * 3 + k_sub
                    full_b = j * 6 + k
                    V_full[full_b, offset + r] = best_sub_V[sub_b, r]
            for k_sub in range(3):
                k = k_sub + 3
                for i in range(2):
                    sub_c = 2 * k_sub + i
                    full_c = 2 * k + i  # = 2*k_sub + i + 6
                    W_full[full_c, offset + r] = best_sub_W[sub_c, r]
        
        # Verify the full decomposition
        max_err = 0
        for (a, b, c) in valid_triples:
            val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
            max_err = max(max_err, abs(val - 1.0))
        
        # Check invalid positions
        for a in range(num_A):
            for b in range(num_B):
                for c in range(num_C):
                    if (a, b, c) not in valid_triples:
                        val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
                        max_err = max(max_err, abs(val))
        
        if max_err < 1e-8:
            best_rank = full_rank
            best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    
    # As a fallback, also try direct search on the full 2x3x6 problem with lower ranks
    if time.time() < deadline - 5 and best_rank == 36:
        for target_rank in [30, 28, 26, 24]:
            if time.time() > deadline - 5:
                break
            
            found = False
            for restart in range(10):
                if time.time() > deadline - 5:
                    break
                    
                np.random.seed(5000 + target_rank * 10 + restart)
                U_curr = np.random.randn(num_A, target_rank) * 0.3
                V_curr = np.random.randn(num_B, target_rank) * 0.3
                W_curr = np.random.randn(num_C, target_rank) * 0.3
                
                lr = 0.02
                for step in range(300):
                    if time.time() > deadline - 5:
                        break
                        
                    grad_U = np.zeros_like(U_curr)
                    grad_V = np.zeros_like(V_curr)
                    grad_W = np.zeros_like(W_curr)
                    residual_sum = 0
                    
                    # Only use valid triples for gradient
                    for (a, b, c) in valid_triples:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        err = val - 1.0
                        residual_sum += err * err
                        
                        for r in range(target_rank):
                            grad_U[a, r] += err * V_curr[b, r] * W_curr[c, r]
                            grad_V[b, r] += err * U_curr[a, r] * W_curr[c, r]
                            grad_W[c, r] += err * U_curr[a, r] * V_curr[b, r]
                    
                    U_curr -= lr * grad_U
                    V_curr -= lr * grad_V
                    W_curr -= lr * grad_W
                    lr *= 0.997
                    
                    if residual_sum < 1e-10:
                        found = True
                        break
                
                if found:
                    max_err = 0
                    for (a, b, c) in valid_triples:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-8:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                        break
            
            if best_rank == target_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
