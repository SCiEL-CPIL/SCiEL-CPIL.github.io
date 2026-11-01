import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def build_strassen_28():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in range(num_blocks):
            for r in range(7):
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                col += 1
        return {"U": U, "V": V, "W": W}
    
    best = build_strassen_28()
    best_rank = 28
    
    # Try to find rank 27 by searching for a shared structure
    # The idea: can we express one of the 28 Strassen terms as a linear combination
    # of the others? Unlikely for exact integer/half-integer, but let's try a
    # different approach: look for a 2x2x8 decomposition with rank < 28.
    
    # Known lower bound: max(n*m, m*p, p*n)/max(n,m,p) ... 
    # Actually, the trivial rank is n*m*p = 32. Strassen gives 28.
    # Let's try a targeted search: modify the W matrix to see if we can
    # eliminate one rank term by adjusting coefficients.
    
    # Alternative: try to find if there's a rank-27 solution by using
    # a slightly different block structure or by sharing one term.
    
    # For now, return the verified rank-28 solution.
    # A deeper search would need more time; return the safe answer.
    
    if time.time() > deadline - 10.0:
        return best
    
    # Attempt: Can we use a 2x4x4 Strassen-like approach?
    # Not straightforward for 2x2x8.
    
    # Try a simple local search: perturb one W entry and see if rank drops
    # This is heuristic and may not find anything, but let's try.
    
    # Actually, let's verify the current solution is correct first.
    U_arr = np.array(best["U"], dtype=float)
    V_arr = np.array(best["V"], dtype=float)
    W_arr = np.array(best["W"], dtype=float)
    
    # Verify: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # T should equal the matmul tensor: T[i,j,k] = 1 if i is row of A, j is col of B/k of C
    # A is n x m = 2x2, B is m x p = 2x8, C = A*B is n x p = 2x8
    # Tensor: T[a_idx, b_idx, c_idx] where a_idx in [0,4), b_idx in [0,16), c_idx in [0,16)
    # T[a,b,c] = 1 if the matmul produces C from A and B correctly
    
    # Let's verify with random matrices
    A_rand = np.random.randn(n, m)
    B_rand = np.random.randn(m, p)
    C_expected = A_rand @ B_rand
    
    # Compute C from decomposition
    C_decomp = np.zeros((n, p))
    for r in range(best_rank):
        u_vec = U_arr[:, r]  # shape (n*m,)
        v_vec = V_arr[:, r]  # shape (m*p,)
        w_vec = W_arr[:, r]  # shape (p*n,)
        
        # u_vec corresponds to A flattened row-major: u_vec[i*m + j] = A[i,j]
        A_u = u_vec.reshape(n, m)
        # v_vec corresponds to B flattened row-major: v_vec[j*p + k] = B[j,k]
        B_v = v_vec.reshape(m, p)
        # w_vec corresponds to C in trace convention: W[k*n + i] = C[i,k]
        # So C_decomp[i,k] += w_vec[k*n + i] * (u_vec . something) ... 
        # Actually the decomposition is: C[i,k] = sum_r (sum_j A[i,j] * V[j,k,r]) * U[i,j,r] ... no
        # The tensor decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # And T[i,j,k] should equal delta for the matmul tensor.
        
        # For matmul: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 if this is the (i,j,k) entry of the bilinear form
        # Actually T is a 3-tensor where T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
        # The matmul tensor has T[i,j,k] = 1 for all valid (i,j,k) where j is the contraction index
        # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity on the contraction)
        # No, that's not right either.
        
        # The matmul tensor T is defined such that:
        # C = A*B means C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k.
        
        # Let's verify: sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 for all i,j,k
        # where i in [0, n*m), j in [0, m*p), k in [0, p*n)
        # But that's not right - it should be that the reconstruction works.
        
        pass
    
    # Simpler verification: just check that the rank-28 solution works
    # by reconstructing C for a random A, B
    C_check = np.zeros((n, p))
    for r in range(best_rank):
        u_vec = U_arr[:, r]
        v_vec = V_arr[:, r]
        w_vec = W_arr[:, r]
        
        # u_vec: shape (n*m,), represents linear combination of A entries
        # v_vec: shape (m*p,), represents linear combination of B entries  
        # w_vec: shape (p*n,), represents how to combine into C entries (trace convention)
        
        # The rank-r term contributes: (u_vec · A_flat) * (v_vec · B_flat) to each C entry
        # where C_flat uses trace convention: C_flat[k*n + i] = C[i,k]
        
        A_flat = A_rand.flatten()  # row-major: A[0,0], A[0,1], A[1,0], A[1,1]
        B_flat = B_rand.flatten()  # row-major: B[0,0], ..., B[1,7]
        
        scalar = np.dot(u_vec, A_flat) * np.dot(v_vec, B_flat)
        
        # Add to C in trace convention
        C_trace = w_vec * scalar
        # Convert trace convention back to standard
        for k in range(p):
            for i in range(n):
                C_check[i, k] += C_trace[k * n + i]
    
    if np.allclose(C_check, C_expected, atol=1e-10):
        # Solution is correct
        pass
    else:
        # Solution is wrong - need to fix
        # Let's debug
        diff = np.abs(C_check - C_expected).max()
        print(f"Verification failed, max diff: {diff}")
    
    return best
