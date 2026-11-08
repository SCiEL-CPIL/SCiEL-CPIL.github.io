import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [2, 4, 3, 6, 5, 10, 1, 8, 12, 15, 20]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        # Also try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None

    def get_trivial_solution():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U50, V50, W50 = get_trivial_solution()
    
    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Strategy: Fix V to a canonical sparse form to reduce gauge freedom
    # For each rank r, V[r] has exactly one 1 in some column
    # This means V is a permutation-like matrix with rank ones
    
    # The tensor equation is: sum_r U[:,r] * V[:,r] * W[:,r] = T
    # If V[:,r] = e_{v_r} (standard basis vector), then:
    # sum_r U[:,r] * W[:,r] * e_{v_r} = T
    # This means for each v_idx, the contribution is sum over r where v_r = v_idx of U[:,r] * W[:,r]
    
    # Let's group columns of V by which basis vector they are
    # For a given rank R, we assign each of the R columns to one of V_dim positions
    # Let count[v] = number of columns assigned to position v
    # Then T[:, v, :] = sum_{r: v_r=v} U[:,r] * W[:,r]
    
    # This is a bilinear system. Let's try to solve it by:
    # 1. Choosing a pattern for V (which columns go where)
    # 2. For each group, solving the bilinear system
    
    # Alternative approach: Use the fact that for matrix multiplication,
    # we can use known decomposition structures
    
    # Let's try a structured approach:
    # For 2x5 * 5x5, we can think of it as block multiplication
    # A = [A1; A2] where A1, A2 are 1x5
    # B is 5x5
    # Result C = [A1*B; A2*B] is 2x5
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try rank = 25 (which is m*p)
    # If we set V[j*p+k, r] = 1 when r = j*p+k, else 0
    # Then for each (j,k), we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # This means U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1
    # But this gives rank 25, which is the trivial solution
    
    # Can we do better? Let's try rank < 25
    
    # For rank R, we need to assign R columns to positions in V
    # Let's use a greedy approach with random restarts
    
    def try_rank_with_fixed_V(target_rank):
        """Try to find U, W given a fixed pattern for V"""
        if time.time() > deadline - 5:
            return None, None, None
        
        # Generate random assignment of target_rank columns to V_dim positions
        # Each column of V gets a random position
        v_positions = np.random.randint(0, V_dim, size=target_rank)
        
        # Count how many columns go to each position
        v_counts = np.bincount(v_positions, minlength=V_dim)
        
        # For each position v, we have v_counts[v] columns
        # T[:, v, :] = sum_{r: v_r=v} U[:,r] * W[:,r]
        
        # This is a sum of outer products (element-wise in the 2D sense)
        # Let's set up the linear system for U and W
        
        # For each (v, u_idx, w_idx) where T[u_idx, v, w_idx] = 1:
        # sum_{r: v_r=v} U[u_idx, r] * W[w_idx, r] = 1
        
        # This is still bilinear. Let's try to solve it iteratively
        
        # Initialize U and W randomly
        U = np.zeros((U_dim, target_rank))
        W = np.zeros((W_dim, target_rank))
        
        # For each column r, initialize based on its v position
        for r in range(target_rank):
            v_pos = v_positions[r]
            # This column contributes to T[:, v_pos, :]
            # Let's set U and W to 1 initially for relevant entries
            # Actually, let's just use random initialization
            U[:, r] = np.random.randn(U_dim) * 0.1
            W[:, r] = np.random.randn(W_dim) * 0.1
        
        # V is determined by v_positions
        V = np.zeros((V_dim, target_rank))
        for r in range(target_rank):
            V[v_positions[r], r] = 1.0
        
        # Now optimize U and W to minimize ||sum_r U[:,r] * V[:,r] * W[:,r] - T||^2
        # Since V is fixed, this is: sum_r U[:,r] * (V[:,r] * W[:,r]) - T
        # = sum_r U[:,r] * (W[:,r] * V[:,r]) - T
        # = sum_r U[:,r] * (W[:,r] with only v_positions[r] entry nonzero) - T
        
        # Let's define B_r = V[:,r] * W[:,r], which has only one nonzero entry
        # B_r[v_positions[r], :] = W[:, r] * 1 = W[:, r]
        # Wait, that's not right. V[:,r] is a vector with 1 at v_positions[r]
        # So V[:,r] * W[:,r] is element-wise product, which has W[w_idx, r] at w_idx = v_positions[r]
        
        # Actually, the tensor product is:
        # (U[:,r] * V[:,r] * W[:,r])[u,v,w] = U[u,r] * V[v,r] * W[w,r]
        # = U[u,r] * (1 if v == v_positions[r] else 0) * W[w,r]
        
        # So the contribution to T[u,v,w] from column r is:
        # U[u,r] * W[w,r] if v == v_positions[r], else 0
        
        # Therefore: T[u,v,w] = sum_{r: v_positions[r]==v} U[u,r] * W[w,r]
        
        # Let's set up the optimization problem
        def objective_uw(x):
            U_flat = x[:U_dim * target_rank]
            W_flat = x[U_dim * target_rank:]
            U_mat = U_flat.reshape(U_dim, target_rank)
            W_mat = W_flat.reshape(W_dim, target_rank)
            
            prod = np.zeros((U_dim, V_dim, W_dim))
            for r in range(target_rank):
                v_pos = v_positions[r]
                prod[:, v_pos, :] += U_mat[:, r] * W_mat[:, r]
            
            diff = prod - T
            return np.sum(diff * diff)
        
        x0 = np.concatenate([U.flatten(), W.flatten()])
        
        try:
            res = minimize(
                objective_uw,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            
            if res.fun < 1e-8:
                U_opt = res.x[:U_dim * target_rank].reshape(U_dim, target_rank)
                W_opt = res.x[U_dim * target_rank:].reshape(W_dim, target_rank)
                
                valid, U_s, V_s, W_s = snap_and_validate(U_opt, V, W_opt)
                if valid:
                    return U_s, V_s, W_s
        except Exception:
            pass
        
        return None, None, None

    # Try different ranks
    for target_rank in range(25, 35):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed + target_rank * 100)
            
            U_s, V_s, W_s = try_rank_with_fixed_V(target_rank)
            if U_s is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_s, V_s, W_s
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # If we didn't find anything better, return the trivial solution
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
