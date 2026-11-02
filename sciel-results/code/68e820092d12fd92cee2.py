"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Helper to build a block decomposition
    def build_from_blocks(blocks):
        """
        blocks: list of (sub_p, U_sub, V_sub, W_sub)
        """
        total_rank = sum(b[1].shape[1] for b in blocks)
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col_offset = 0
        k_offset = 0
        for sub_p, U_sub, V_sub, W_sub in blocks:
            rank_sub = U_sub.shape[1]
            # U is same for all blocks (A is same)
            U[:, col_offset:col_offset+rank_sub] = U_sub
            # V and W depend on k
            for j in range(m):
                V[j*p + k_offset : j*p + k_offset + sub_p, col_offset:col_offset+rank_sub] = V_sub[j*sub_p : (j+1)*sub_p, :]
            for k in range(sub_p):
                for i in range(n):
                    W[(k_offset + k)*n + i, col_offset:col_offset+rank_sub] = W_sub[k*n + i, :]
            col_offset += rank_sub
            k_offset += sub_p
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Standard blocks
    # M_{2,2,1} Rank 4
    U_1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_1 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_1 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # M_{2,2,2} Rank 7 (Strassen)
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])

    # Fallback: 7 blocks of 2 + 1 block of 1 = 7*7 + 4 = 53
    fallback_blocks = [(2, U_2, V_2, W_2)] * 7 + [(1, U_1, V_1, W_1)]
    fallback_result = build_from_blocks(fallback_blocks)
    
    # Try to find M_{2,2,5} Rank 17
    def find_m225_rank17():
        if time.time() > deadline - 20:
            return None
            
        # Target tensor for M_{2,2,5}
        # U: 4x17, V: 10x17, W: 10x17
        # Residuals: 40
        def objective(x):
            U = x[0:68].reshape(4, 17)
            V = x[68:238].reshape(10, 17)
            W = x[238:408].reshape(10, 17)
            residuals = np.zeros(40)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        val = np.dot(np.dot(U[i*2+j, :], V[j*5+k, :]), W[k*2+i, :])
                        target = 1.0 if i == k else 0.0 # Wait, i is row of A, k is col of B.
                        # C[i,k] = sum_j A[i,j] B[j,k]
                        # Tensor index: (i,j), (j,k), (k,i)
                        # So target is 1 if i==k? No.
                        # The tensor T_{u,v,w} = 1 if u's j == v's j and v's k == w's k and w's i == u's i.
                        # u = i*2+j, v = j*5+k, w = k*2+i.
                        # So we check T[u,v,w].
                        # In the loop, we are effectively computing C[i,k].
                        # The target for the decomposition is that the computed C[i,k] matches the true C[i,k].
                        # The tensor entries are 1 at positions where the multiplication happens.
                        # Actually, the condition "val == target" is wrong for tensor decomposition verification.
                        # The tensor decomposition requires:
                        # sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
                        # T[u,v,w] = 1 if j_u == j_v and k_v == k_w and i_w == i_u.
                        
                        u = i*2+j
                        v = j*5+k
                        w = k*2+i
                        # Check if indices align
                        # j_u = j, j_v = j. Match.
                        # k_v = k, k_w = k. Match.
                        # i_w = i, i_u = i. Match.
                        # So T[u,v,w] is ALWAYS 1 for these indices?
                        # No, the tensor is defined on the full space.
                        # We only need to satisfy the equations for the non-zero entries of the tensor?
                        # No, we must satisfy T[u,v,w] for ALL u,v,w.
                        # But T[u,v,w] is 0 if indices don't align.
                        # The loop above only iterates over valid (i,j,k) triples that form a "path".
                        # We must ensure that for any u,v,w, the sum equals T[u,v,w].
                        # Since T is sparse, we can just check the non-zero entries?
                        # If the decomposition is valid for the non-zero entries, is it valid for the zero entries?
                        # Not necessarily. We must check the zero entries too.
                        # However, usually, we just check the non-zero ones and hope the zeros are satisfied?
                        # No, that's dangerous.
                        # Let's check all 4*10*10 = 400 entries?
                        # That's too slow for least_squares.
                        # But the tensor has a specific structure.
                        # T[u,v,w] is non-zero only if j_u=j_v, k_v=k_w, i_w=i_u.
                        # There are 2*2*5 = 20 non-zero entries.
                        # If we satisfy these 20, do we satisfy the others?
                        # The equations are linear.
                        # Let's just check the 20 non-zero entries.
                        # If the solution is close to integer, it might work.
                        
                        if True: # This loop structure is for non-zero entries
                            residuals[idx] = val - 1.0
                            idx += 1
            return residuals
            
        # This objective function is flawed. It only checks 20 entries.
        # We need to check all 400?
        # Or we can use a property: if we satisfy the non-zero ones, the zeros are likely satisfied if we use a structured initialization?
        # Let's try to check all 400 entries in a vectorized way for speed.
        
        def objective_full(x):
            U = x[0:68].reshape(4, 17)
            V = x[68:238].reshape(10, 17)
            W = x[238:408].reshape(10, 17)
            
            # Compute all entries of the decomposed tensor
            # D[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
            # This is a tensor contraction.
            # We can compute it as:
            # Temp[u,v] = sum_r U[u,r] V[v,r] -> 4x10 matrix
            # Then D[u,v,w] = sum_r Temp[u,v] * W[w,r]? No.
            # D[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
            # This is the trace of U[:,r] V[:,r]^T W[:,r]^T?
            
            # Let's just loop over the 20 non-zero entries first.
            # If we find a solution, we can verify the zeros later.
            residuals = np.zeros(20)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        val = np.dot(np.dot(U[i*2+j, :], V[j*5+k, :]), W[k*2+i, :])
                        residuals[idx] = val - 1.0
                        idx += 1
            return residuals

        np.random.seed(42)
        best_res = np.inf
        best_sol = None
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            scale = 0.5
            U = np.random.randn(4, 17) * scale
            V = np.random.randn(10, 17) * scale
            W = np.random.randn(10, 17) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(objective_full, x0, method='lm', max_nfev=500)
                res = result.fun
                max_res = np.max(np.abs(res))
                
                if max_res < best_res:
                    best_res = max_res
                    best_sol = result.x.copy()
                    
                if max_res < 1e-6:
                    # Check zeros
                    U_opt = result.x[0:68].reshape(4, 17)
                    V_opt = result.x[68:238].reshape(10, 17)
                    W_opt = result.x[238:408].reshape(10, 17)
                    
                    zero_violation = 0
                    for u in range(4):
                        for v in range(10):
                            for w in range(10):
                                # Check if this is a non-zero entry
                                i_u = u // 2
                                j_u = u % 2
                                j_v = v // 5
                                k_v = v % 5
                                k_w = w // 2
                                i_w = w % 2
                                
                                if not (j_u == j_v and k_v == k_w and i_w == i_u):
                                    val = np.dot(np.dot(U_opt[u, :], V_opt[v, :]), W_opt[w, :])
                                    if abs(val) > 1e-4:
                                        zero_violation = max(zero_violation, abs(val))
                                        
                    if zero_violation < 1e-4:
                        # Try rounding
                        U_r = np.round(U_opt * 2) / 2
                        V_r = np.round(V_opt * 2) / 2
                        W_r = np.round(W_opt * 2) / 2
                        
                        # Verify rounded
                        valid = True
                        for i in range(2):
                            for j in range(2):
                                for k in range(5):
                                    val = np.dot(np.dot(U_r[i*2+j, :], V_r[j*5+k, :]), W_r[k*2+i, :])
                                    if abs(val - 1.0) > 1e-6:
                                        valid = False
                        if valid:
                            # Check zeros for rounded
                            for u in range(4):
                                for v in range(10):
                                    for w in range(10):
                                        i_u = u // 2; j_u = u % 2
                                        j_v = v // 5; k_v = v % 5
                                        k_w = w // 2; i_w = w % 2
                                        if not (j_u == j_v and k_v == k_w and i_w == i_u):
                                            val = np.dot(np.dot(U_r[u, :], V_r[v, :]), W_r[w, :])
                                            if abs(val) > 1e-6:
                                                valid = False
                            if valid:
                                return U_r, V_r, W_r
                            
            except Exception:
                continue
                
        return None

    U_5, V_5, W_5 = find_m225_rank17()
    
    if U_5 is not None:
        # Use 3 blocks of 5
        blocks = [(5, U_5, V_5, W_5)] * 3
        result = build_from_blocks(blocks)
        return result
    else:
        return fallback_result

# EVOLVE-BLOCK-END
