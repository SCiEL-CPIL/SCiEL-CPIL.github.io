import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # 1. Solve 3x3x3 sub-problem
    # Target tensor for 3x3x3: T[i*3+j, j*3+k, k*3+i] = 1
    # We want to find U3, V3, W3 of shape (9, r3)
    
    def solve_3x3x3(max_rank=27, min_rank=19, time_limit=10.0):
        # Build target tensor
        T3 = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T3[i * 3 + j, j * 3 + k, k * 3 + i] = 1.0
        
        # Try ranks from min_rank to max_rank
        for r in range(min_rank, max_rank + 1):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize
            np.random.seed(42 + r)
            U = np.random.randn(9, r) * 0.1
            V = np.random.randn(9, r) * 0.1
            W = np.random.randn(9, r) * 0.1
            
            # Gradient descent
            lr = 0.05
            for it in range(2000):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute T_rec
                T_rec = np.zeros((9, 9, 9))
                for r_idx in range(r):
                    T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                
                error = T_rec - T3
                
                # Gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r_idx in range(r):
                    # grad_U[:, r_idx] = sum_{b,c} error[a,b,c] V[b,r] W[c,r]
                    # = (error * V[:, r_idx, None] * W[:, r_idx, None, None]).sum(axis=(1,2))
                    # Using einsum:
                    grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                    grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                    grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 50 == 0 and np.max(np.abs(error)) < 1e-6:
                    return U, V, W, r
        
        # Fallback to schoolbook rank 27
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
        return U, V, W, 27

    # 2. Solve 3x3x1 sub-problem (Matrix-Vector)
    # Target tensor for 3x3x1: T[i*3+j, j*1+0, 0*3+i] = T[i*3+j, j, i] = 1
    # Rank is 9.
    def solve_3x3x1():
        U = np.zeros((9, 9))
        V = np.zeros((3, 9)) # V shape is (m*p, rank) = (3*1, 9) = (3, 9)
        W = np.zeros((3, 9)) # W shape is (p*n, rank) = (1*3, 9) = (3, 9)
        
        col = 0
        for i in range(3):
            for j in range(3):
                U[i * 3 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W, 9

    # Get solutions
    U3, V3, W3, r3 = solve_3x3x3()
    U1, V1, W1, r1 = solve_3x3x1()
    
    # 3. Assemble full 3x3x7 decomposition
    # B is 3x7. Split into B1 (3x3), B2 (3x3), B3 (3x1).
    # V index is j*p + k.
    # For B1 (k=0,1,2): V index is j*7 + k.
    # For B2 (k=3,4,5): V index is j*7 + k.
    # For B3 (k=6): V index is j*7 + 6.
    
    # U is 3x3, so U indices are same for all parts.
    # W index is k*n + i.
    # For B1 (k=0,1,2): W index is k*3 + i.
    # For B2 (k=3,4,5): W index is k*3 + i.
    # For B3 (k=6): W index is 6*3 + i = 18 + i.
    
    # Total rank = 2 * r3 + r1
    total_rank = 2 * r3 + r1
    
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((21, total_rank))
    W_full = np.zeros((21, total_rank))
    
    # Part 1: B1 (k=0,1,2)
    # U_full[:, 0:r3] = U3
    U_full[:, 0:r3] = U3
    # V_full[:, 0:r3] corresponds to V3 but with indices j*7+k instead of j*3+k
    # V3 indices are j*3+k. We need to map them to j*7+k.
    for r_idx in range(r3):
        for j in range(3):
            for k in range(3):
                # V3[j*3+k, r_idx] is the coefficient for B1[j,k]
                V_full[j*7 + k, r_idx] = V3[j*3 + k, r_idx]
    # W_full[:, 0:r3] corresponds to W3 with indices k*3+i.
    # W_full indices are k*3+i for k=0,1,2. This matches W3 indices.
    W_full[0:9, 0:r3] = W3
    
    # Part 2: B2 (k=3,4,5)
    # U_full[:, r3:2*r3] = U3
    U_full[:, r3:2*r3] = U3
    # V_full[:, r3:2*r3] corresponds to V3 but with indices j*7+k for k=3,4,5.
    for r_idx in range(r3):
        for j in range(3):
            for k in range(3):
                # V3[j*3+k, r_idx] is the coefficient for B2[j,k]
                # Map to V_full index j*7 + (k+3)
                V_full[j*7 + (k+3), r3 + r_idx] = V3[j*3 + k, r_idx]
    # W_full[:, r3:2*r3] corresponds to W3 but with indices (k+3)*3+i.
    for r_idx in range(r3):
        for k in range(3):
            for i in range(3):
                # W3[k*3+i, r_idx] is the coefficient for C2[k,i]
                # Map to W_full index (k+3)*3 + i
                W_full[(k+3)*3 + i, r3 + r_idx] = W3[k*3 + i, r_idx]
    
    # Part 3: B3 (k=6)
    # U_full[:, 2*r3:2*r3+r1] = U1
    U_full[:, 2*r3:2*r3+r1] = U1
    # V_full[:, 2*r3:2*r3+r1] corresponds to V1 with indices j*7+6.
    for r_idx in range(r1):
        for j in range(3):
            V_full[j*7 + 6, 2*r3 + r_idx] = V1[j, r_idx]
    # W_full[:, 2*r3:2*r3+r1] corresponds to W1 with indices 6*3+i = 18+i.
    for r_idx in range(r1):
        for i in range(3):
            W_full[18 + i, 2*r3 + r_idx] = W1[i, r_idx]
    
    # Verify decomposition
    # T_rec = sum_r U[:,r] V[:,r] W[:,r]
    # Check a few entries or all
    T_target = np.zeros((9, 21, 21))
    for i in range(3):
        for j in range(3):
            for k in range(7):
                T_target[i * 3 + j, j * 7 + k, k * 3 + i] = 1.0
                
    T_rec = np.zeros((9, 21, 21))
    for r_idx in range(total_rank):
        T_rec += np.einsum('a,b,c->abc', U_full[:, r_idx], V_full[:, r_idx], W_full[:, r_idx])
        
    if not np.allclose(T_rec, T_target, atol=1e-6):
        # Fallback to schoolbook if verification fails
        U_full = np.zeros((9, 63))
        V_full = np.zeros((21, 63))
        W_full = np.zeros((21, 63))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    U_full[i * 3 + j, col] = 1
                    V_full[j * 7 + k, col] = 1
                    W_full[k * 3 + i, col] = 1
                    col += 1
        total_rank = 63

    U_list = [[float(x) for x in row] for row in U_full]
    V_list = [[float(x) for x in row] for row in V_full]
    W_list = [[float(x) for x in row] for row in W_full]
    
    return {"U": U_list, "V": V_list, "W": W_list}
