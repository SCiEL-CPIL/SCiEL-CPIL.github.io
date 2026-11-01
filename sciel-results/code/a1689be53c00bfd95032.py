import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    n, m, p = 2, 4, 5
    
    # 1. Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T is 1 if i=i', j=j', k=k' else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # 2. Define a set of candidate "parent" decompositions for larger matrices
    # We try to find a decomposition for 4x4x4 (rank 32 Strassen) or 8x8x8 (rank 70 Winograd-like)
    # and see if we can restrict it to our 2x4x5 sub-problem.
    # However, direct restriction is hard because indices don't align perfectly.
    # Alternative: Use the "Lifting" method.
    # We know that multiplying (2x4) * (4x5) can be seen as part of a larger multiplication.
    # Specifically, if we pad A to 4x4 and B to 4x4, the top-left 2x4 block of the result 
    # corresponds to our desired output IF the padding is chosen correctly (zeros).
    # But Strassen's algorithm mixes blocks.
    
    # A more robust algebraic approach:
    # We look for a decomposition of rank R < 40.
    # The tensor T is sparse. 
    # Let's try a randomized search with a structured initialization based on 
    # the fact that n=2 is small.
    # We can split the problem into two 1x4x5 multiplications? No, that's rank 40.
    # We can split into two 2x2x5? 
    # A (2x4) = [A1 A2] where A1, A2 are 2x2.
    # B (4x5) = [B1; B2] where B1, B2 are 2x5.
    # C = A1*B1 + A2*B2.
    # This is the sum of two 2x2x5 multiplications.
    # Rank of 2x2x5 multiplication?
    # Schoolbook: 2*2*5 = 20.
    # Total rank: 40.
    # Can we do better for 2x2x5?
    # Lower bound for 2x2x5: max(4, 10, 5) = 10.
    # Known best for 2x2x5? 
    # If we can find a decomposition for 2x2x5 with rank < 20, say R_225,
    # then total rank is 2 * R_225.
    # If R_225 = 15, total rank = 30.
    # If R_225 = 14, total rank = 28.
    
    # Let's try to find a low-rank decomposition for 2x2x5.
    # Tensor T_225 of shape (4, 10, 10).
    # T_225[i*2+j, j'*5+k, k'*2+i'] = 1 if i=i', j=j', k=k'.
    
    # We will use a simple optimization: 
    # Initialize U, V, W for 2x2x5 with random small integers.
    # Minimize the error || sum_r U[:,r] V[:,r]^T W[:,r]^T - T_225 ||_F^2
    # Using gradient descent or a simple iterative refinement.
    # Given the time limit, we can try a few random restarts for 2x2x5.
    
    def solve_2x2x5(time_budget):
        n2, m2, p2 = 2, 2, 5
        T_sub = np.zeros((n2*m2, m2*p2, p2*n2), dtype=np.float64)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i * m2 + j
                    b = j * p2 + k
                    c = k * n2 + i
                    T_sub[a, b, c] = 1.0
        
        best_rank = 20 # Schoolbook
        best_U = None
        best_V = None
        best_W = None
        
        # Try ranks from 12 up to 19 (since 20 is schoolbook)
        # Actually, let's just try to find ANY rank < 20.
        # We'll try rank 15 first as a heuristic middle ground.
        
        for rank in [14, 15, 16, 17, 18, 19]:
            if time.time() > time_budget:
                break
            
            # Initialize
            U = np.random.randn(n2*m2, rank) * 0.1
            V = np.random.randn(m2*p2, rank) * 0.1
            W = np.random.randn(p2*n2, rank) * 0.1
            
            # Simple gradient descent on the squared error
            # Error = || sum_r U[:,r] outer V[:,r] outer W[:,r] - T_sub ||^2
            # This is expensive to compute fully each step.
            # Instead, we can use a "tensor train" or "ALS" like approach?
            # Let's use a simple iterative update for each factor.
            
            # Precompute contractions to speed up?
            # For now, just a few steps of gradient descent.
            lr = 0.01
            for step in range(100):
                if time.time() > time_budget:
                    break
                
                # Compute current tensor estimate
                # E = np.einsum('ar,br,cr->abc', U, V, W)
                # This is 4*10*10*rank = 400*rank ops. Very fast.
                E = np.zeros_like(T_sub)
                for r in range(rank):
                    E += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                error = np.linalg.norm(E - T_sub)
                if error < 1e-4:
                    # Found a solution!
                    return rank, U, V, W
                
                # Gradient descent step
                # grad_U = sum_b,c (E-T_sub) V_b W_c
                # This is approximate. Let's do a proper gradient.
                # dE/dr = U_r V_r W_r
                # dError/dU_a = sum_b,c (E_abc - T_abc) V_br W_cr
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                diff = E - T_sub
                
                # grad_U[:, r] = sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
                # This is a matrix product for each r.
                for r in range(rank):
                    # grad_U[:, r] = diff @ (V[:, r] outer W[:, r])? No.
                    # grad_U[:, r]_a = sum_b sum_c diff[a,b,c] V[b,r] W[c,r]
                    # = sum_c (diff @ V[:, r])_c * W[c, r] ? No.
                    # Let M_r be the matrix M[b,c] = V[b,r] * W[c,r]
                    # grad_U[:, r] = diff @ M_r (flattened)
                    
                    M_r = np.outer(V[:, r], W[:, r])
                    # diff is (4, 10, 10). M_r is (10, 10).
                    # grad_U[:, r] = diff @ M_r -> shape (4, 10) @ (10, 10) -> (4, 10)? No.
                    # We want a vector of length 4.
                    # sum_b,c diff[a,b,c] M_r[b,c]
                    # This is the contraction of diff with M_r over b,c.
                    grad_U[:, r] = np.einsum('abc,bc->a', diff, M_r)
                    
                    # Similarly for V
                    # grad_V[:, r]_b = sum_a,c diff[a,b,c] U[a,r] W[c,r]
                    # N_r[a,c] = U[a,r] * W[c,r]
                    N_r = np.outer(U[:, r], W[:, r])
                    grad_V[:, r] = np.einsum('abc,ac->b', diff, N_r)
                    
                    # Similarly for W
                    # grad_W[:, r]_c = sum_a,b diff[a,b,c] U[a,r] V[b,r]
                    # P_r[a,b] = U[a,r] * V[b,r]
                    P_r = np.outer(U[:, r], V[:, r])
                    grad_W[:, r] = np.einsum('abc,ab->c', diff, P_r)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent drift
                for r in range(rank):
                    nU = np.linalg.norm(U[:, r])
                    nV = np.linalg.norm(V[:, r])
                    nW = np.linalg.norm(W[:, r])
                    if nU > 1e-9: U[:, r] /= nU
                    if nV > 1e-9: V[:, r] /= nV
                    if nW > 1e-9: W[:, r] /= nW
                    # Rebalance scaling
                    scale = (nU * nV * nW) ** (1/3)
                    if scale > 1e-9:
                        U[:, r] *= scale
                        V[:, r] *= scale
                        W[:, r] *= scale

        # If no solution found for 2x2x5, fall back to schoolbook for the whole thing
        return None, None, None

    # Try to solve 2x2x5
    rank_225, U_225, V_225, W_225 = solve_2x2x5(deadline - 10)
    
    if rank_225 is not None:
        # Construct the full 2x4x5 decomposition
        # A = [A1 A2], B = [B1; B2], C = A1 B1 + A2 B2
        # We have two independent 2x2x5 multiplications.
        # Total rank = 2 * rank_225
        total_rank = 2 * rank_225
        
        U_full = np.zeros((n*m, total_rank), dtype=np.int64)
        V_full = np.zeros((m*p, total_rank), dtype=np.int64)
        W_full = np.zeros((p*n, total_rank), dtype=np.int64)
        
        # Mapping for first block (A1, B1)
        # A1 is A[0:2, 0:2] -> indices 0,1,2,3 in A_vec (i*4+j)
        # B1 is B[0:2, :] -> indices 0..9 in B_vec (j*5+k)
        # C1 is C[:, :] -> indices 0..9 in C_vec (k*2+i)
        
        # Mapping for second block (A2, B2)
        # A2 is A[0:2, 2:4] -> indices 2,3,6,7 in A_vec
        # B2 is B[2:4, :] -> indices 10..19 in B_vec
        # C2 is C[:, :] -> indices 0..9 in C_vec
        
        # We need to map the 2x2x5 indices to the 2x4x5 indices.
        # 2x2x5 indices:
        # a_sub = i*2+j (i in 0,1; j in 0,1)
        # b_sub = j'*5+k (j' in 0,1; k in 0..4)
        # c_sub = k'*2+i (k' in 0..4; i in 0,1)
        
        # For Block 1:
        # i_full = i_sub
        # j_full = j_sub
        # j'_full = j'_sub
        # k_full = k'_sub
        # a_full = i_full * 4 + j_full
        # b_full = j'_full * 5 + k_full
        # c_full = k'_full * 2 + i_full
        
        # For Block 2:
        # i_full = i_sub
        # j_full = 2 + j_sub
        # j'_full = 2 + j'_sub
        # k_full = k'_sub
        # a_full = i_full * 4 + (2 + j_sub)
        # b_full = (2 + j'_sub) * 5 + k_full
        # c_full = k'_full * 2 + i_full
        
        # We will fill U_full, V_full, W_full
        for r in range(rank_225):
            # Block 1
            col1 = r
            col2 = r + rank_225
            
            # U
            for a_sub in range(4):
                i_sub = a_sub // 2
                j_sub = a_sub % 2
                
                # Block 1
                a_full_1 = i_sub * 4 + j_sub
                U_full[a_full_1, col1] += U_225[a_sub, r]
                
                # Block 2
                a_full_2 = i_sub * 4 + (2 + j_sub)
                U_full[a_full_2, col2] += U_225[a_sub, r]
            
            # V
            for b_sub in range(10):
                j_sub = b_sub // 5
                k_sub = b_sub % 5
                
                # Block 1
                b_full_1 = j_sub * 5 + k_sub
                V_full[b_full_1, col1] += V_225[b_sub, r]
                
                # Block 2
                b_full_2 = (2 + j_sub) * 5 + k_sub
                V_full[b_full_2, col2] += V_225[b_sub, r]
            
            # W
            for c_sub in range(10):
                k_sub = c_sub // 2
                i_sub = c_sub % 2
                
                # Block 1
                c_full_1 = k_sub * 2 + i_sub
                W_full[c_full_1, col1] += W_225[c_sub, r]
                
                # Block 2
                c_full_2 = k_sub * 2 + i_sub
                W_full[c_full_2, col2] += W_225[c_sub, r]
        
        # Verify
        # Check if the decomposition is exact
        T_est = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for r in range(total_rank):
            T_est += np.einsum('a,b,c->abc', U_full[:,r], V_full[:,r], W_full[:,r])
        
        if np.allclose(T_est, T):
            # Success!
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
        else:
            # Verification failed, fall back to schoolbook
            pass

    # Fallback: Schoolbook Rank 40
    rank = 40
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
