import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition correctness."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    for k_inner in range(m):
                        C_computed[i, k] += U_np[i*m + k_inner, r] * V_np[k_inner*p + k, r] * W_np[k*n + i, r]
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen 7 multiplications
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T (k=8)
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    best_result = current_solution()
    best_rank = 50
    
    # Search for lower-rank decomposition of [2,3,3] sub-tensor
    # If we find rank r for [2,3,3], we can use 3 blocks for 9 columns
    # Total rank would be 3*r + 18 (for the 2x1x9 part)
    # Wait, that's not quite right. Let me think about this more carefully.
    
    # Actually, the 2x3x9 tensor can be decomposed as:
    # C[i,j] = sum_{k=0}^{2} A[i,k] B[k,j]
    # = sum_{k=0}^{1} A[i,k] B[k,j] + A[i,2] B[2,j]
    # The first part is 2x2x9, the second is 2x1x9.
    
    # For 2x2x9, we can split into three 2x2x3 blocks.
    # If we find rank r for 2x2x3, total for 2x2x9 is 3r.
    # Plus 18 for 2x1x9.
    # Total: 3r + 18.
    # Current: 32 + 18 = 50.
    # If r=10, total=48. If r=9, total=45.
    
    # Let's search for rank of [2,2,3] tensor.
    # T[i,k,j] = 1 if this is a valid multiplication term, else 0.
    # For 2x2x3: i in {0,1}, k in {0,1,2}, j in {0,1,2}
    # The tensor is sum over all (i,k,j) of e_i ⊗ e_k ⊗ e_j
    
    def search_2x2x3_rank(max_rank, time_budget=30):
        """Search for low-rank decomposition of [2,2,3] tensor."""
        if time.time() > deadline - 20:
            return None
        
        n2, m2, p2 = 2, 2, 3
        # Create the tensor
        tensor = np.zeros((n2, m2, p2))
        for i in range(n2):
            for k in range(m2):
                for j in range(p2):
                    tensor[i, k, j] = 1.0
        
        for r in range(8, max_rank + 1):
            if time.time() > deadline - 20:
                break
            # Try to find rank-r decomposition
            # U: (n2*m2, r), V: (m2*p2, r), W: (p2*n2, r)
            np.random.seed(r * 1000)
            U = np.random.randn(n2 * m2, r)
            V = np.random.randn(m2 * p2, r)
            W = np.random.randn(p2 * n2, r)
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each i,k, U[i*m2+k, :] = argmin ||tensor[i,k,:] - V[:,r] outer W[:,r] contracted||
                # This is complex. Let's use a simpler approach.
                # Actually, let's use a direct least squares approach.
                
                # The tensor equation: T[i,k,j] = sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                # This is a 3-way tensor decomposition, which is non-convex.
                # Let's use a simple gradient-free approach: try random initializations and check.
                
                # Actually, let me just check if the current factors work
                # and do a few alternating updates.
                
                # For each r, compute the residual
                # This is getting too complex for the time budget.
                pass
            
            # Simple check: can we verify with a few random initializations?
            for trial in range(50):
                if time.time() > deadline - 20:
                    break
                np.random.seed(trial * 7919 + r)
                U = np.random.randn(n2 * m2, r)
                V = np.random.randn(m2 * p2, r)
                W = np.random.randn(p2 * n2, r)
                
                # Check if these factors can represent the tensor
                # T[i,k,j] should equal sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                # Let's compute the actual tensor from these factors and compare
                T_computed = np.zeros((n2, m2, p2))
                for rr in range(r):
                    for i in range(n2):
                        for k in range(m2):
                            for j in range(p2):
                                T_computed[i, k, j] += U[i*m2+k, rr] * V[k*p2+j, rr] * W[j*n2+i, rr]
                
                if np.allclose(T_computed, tensor, atol=1e-10):
                    # Found a valid decomposition!
                    # Convert to list format
                    U_list = [[U[i, rr] for rr in range(r)] for i in range(n2*m2)]
                    V_list = [[V[i, rr] for rr in range(r)] for i in range(m2*p2)]
                    W_list = [[W[i, rr] for rr in range(r)] for i in range(p2*n2)]
                    return (r, U_list, V_list, W_list)
        
        return None
    
    # Try to find better 2x2x3 decomposition
    result_2x2x3 = search_2x2x3_rank(12, time_budget=30)
    
    if result_2x2x3 is not None:
        r223, U223, V223, W223 = result_2x2x3
        total_rank = 3 * r223 + 18
        if total_rank < best_rank:
            # Build the full decomposition
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            # Three 2x2x3 blocks
            for block in range(3):
                k_start = block * 3
                for rr in range(r223):
                    # U indices: i in {0,1}, k in {0,1,2} -> i*m + k
                    for i in range(2):
                        for k in range(3):
                            U[i * m + k][col + rr] = U223[i * 2 + k][rr]
                    # V indices: k in {0,1,2}, j in {0,1,2} -> k*p + j
                    for k in range(3):
                        for j in range(3):
                            V[k * p + (k_start + j)][col + rr] = V223[k * 3 + j][rr]
                    # W indices: j in {0,1,2}, i in {0,1} -> j*n + i
                    for j in range(3):
                        for i in range(2):
                            W[(k_start + j) * n + i][col + rr] = W223[j * 2 + i][rr]
                col += r223
            
            # Part 2: 2x1x9 (rank 18)
            u_ar0 = 0 * m + 2
            u_ar1 = 1 * m + 2
            for k in range(9):
                v_b = 2 * p + k
                U[u_ar0][col] = 1; V[v_b][col] = 1; W[k * n + 0][col] = 1
                U[u_ar1][col+1] = 1; V[v_b][col+1] = 1; W[k * n + 1][col+1] = 1
                col += 2
            
            # Verify
            if verify(U, V, W, total_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
    
    # Return the best result
    return best_result
