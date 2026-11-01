import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Check if the decomposition exactly matches the tensor."""
    # Compute the tensor from the factors
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Construct the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Check for exactness (with small tolerance for floating point)
    return np.allclose(T_comp, T_target, atol=1e-6)

def _optimize_rank(n, m, p, rank, start_time, deadline):
    """
    Attempt to find a valid decomposition for the given rank using Nelder-Mead optimization.
    Returns (U, V, W) if valid, else None.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Objective function: Frobenius norm of the difference between computed and target tensor
    def objective(params):
        # Reshape parameters back to U, V, W
        U = params[0:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
        
        # Compute tensor
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor (pre-computed outside for speed, but passed via closure or global? 
        # For simplicity and safety in this snippet, we recompute or assume it's in scope.
        # Actually, let's pre-compute the target vector to make objective faster.
        # T_target is sparse. We can just compute the difference on the non-zero entries?
        # No, T_comp is dense. We must compare against T_target.
        # To speed up, let's flatten and use a pre-allocated target vector.
        
        return np.sum((T_comp - T_target_vec)**2)

    # Pre-compute target vector
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    T_target_vec = T_target.flatten()
    
    # Closure variable for objective
    def obj_func(params):
        U = params[0:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_comp.flatten() - T_target_vec)**2)

    # Initial guess: Random or Trivial subset
    # A good initial guess is crucial. 
    # If rank >= trivial, we can start with trivial.
    # If rank < trivial, we start with random.
    # Let's try a few random restarts if time permits.
    
    best_U, best_V, best_W = None, None, None
    
    # Number of restarts based on remaining time
    # Each Nelder-Mead run might take 1-5 seconds for small dimensions.
    # We have ~120s. Let's allow up to 5 restarts per rank.
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        # Initialize parameters
        # Strategy: Start with a random initialization, but scale appropriately.
        # Or, if rank is close to trivial, use a subset of the trivial decomposition.
        if rank >= n * m * p:
            # Use trivial decomposition, padded with zeros if rank > trivial? No, rank is fixed.
            # If rank == trivial, just return trivial.
            # If rank > trivial, we can just use trivial and pad? No, we want minimal.
            # But this function is called for specific rank.
            pass
        
        # Random initialization
        # Values around 1.0 or -1.0 seem reasonable for these tensors.
        x0 = np.random.uniform(-1, 1, size=(dim_u + dim_v + dim_w) * rank)
        
        # Run Nelder-Mead
        try:
            res = minimize(
                obj_func, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < 1e-6:
                # Extract factors
                U = res.x[0:dim_u*rank].reshape(dim_u, rank)
                V = res.x[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                W = res.x[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
                
                # Verify exactness
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W
        except Exception:
            continue
            
    return None

def _build_strassen_block():
    """
    Constructs the known rank-50 decomposition using Strassen blocks.
    This serves as a strong baseline and fallback.
    """
    n, m, p = 2, 4, 7
    U = np.zeros((n * m, 50), dtype=float)
    V = np.zeros((m * p, 50), dtype=float)
    W = np.zeros((p * n, 50), dtype=float)
    
    # We define the Strassen 2x2x2 terms locally
    # A indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1) -- Note: W index is k*n+i.
    # For 2x2x2 block at A[rows, cols], B[rows, cols], C[rows, cols]:
    # Local A map: [0,1,2,3] -> global A indices
    # Local B map: [0,1,2,3] -> global B indices
    # Local C map: [0,1,2,3] -> global W indices (k*2+i)
    
    # Strassen Terms (U_local, V_local, W_local)
    # m1 = (a+d)(e+h) -> C[0,0] += 1, C[1,1] += 1
    s1 = (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1]))
    # m2 = (c+d)e -> C[1,0] += 1
    s2 = (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,0]))
    # m3 = a(f-h) -> C[0,1] += 1
    s3 = (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,0]))
    # m4 = d(g-f) -> C[0,0] += 1, C[1,0] += 1
    s4 = (np.array([0,0,0,1]), np.array([0,-1,1,0]), np.array([1,0,1,0]))
    # m5 = (a+b)h -> C[0,1] += 1, C[1,0] -= 1
    s5 = (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([0,1,-1,0]))
    # m6 = (c-a)(e+f) -> C[1,1] += 1
    s6 = (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1]))
    # m7 = (b-d)(g+h) -> C[0,0] += 1
    s7 = (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0]))
    
    strassen_terms = [s1, s2, s3, s4, s5, s6, s7]
    
    # Schoolbook 2x2x1 Terms
    # A (2x2) * B (2x1) -> C (2x1)
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    b1 = (np.array([1,0,0,0]), np.array([1,0]), np.array([1,0]))
    b2 = (np.array([0,1,0,0]), np.array([0,1]), np.array([1,0]))
    b3 = (np.array([0,0,1,0]), np.array([1,0]), np.array([0,1]))
    b4 = (np.array([0,0,0,1]), np.array([0,1]), np.array([0,1]))
    schoolbook_terms = [b1, b2, b3, b4]
    
    rank_idx = 0
    
    def add_block(u_local, v_local, w_local, u_map, v_map, w_map):
        nonlocal rank_idx
        for i, val in enumerate(u_local):
            if val != 0:
                U[u_map[i], rank_idx] = val
        for i, val in enumerate(v_local):
            if val != 0:
                V[v_map[i], rank_idx] = val
        for i, val in enumerate(w_local):
            if val != 0:
                W[w_map[i], rank_idx] = val
        rank_idx += 1
        
    # Mappings
    # A: 2x4. Split into Left (cols 0,1) and Right (cols 2,3)
    u_map_left = [0, 1, 2, 3]
    u_map_right = [4, 5, 6, 7]
    
    # B: 4x7. Split into Top (rows 0,1) and Bottom (rows 2,3)
    # Top: 2x7. Split into 3 blocks of 2x2 (cols 0-1, 2-3, 4-5) and 1 block of 2x1 (col 6)
    v_map_top0 = [0, 1, 2, 3]
    v_map_top1 = [4, 5, 6, 7]
    v_map_top2 = [8, 9, 10, 11]
    v_map_top3 = [12, 13]
    
    # Bottom: 2x7. Split into 3 blocks of 2x2 (cols 0-1, 2-3, 4-5) and 1 block of 2x1 (col 6)
    v_map_bot0 = [14, 15, 16, 17]
    v_map_bot1 = [18, 19, 20, 21]
    v_map_bot2 = [22, 23, 24, 25]
    v_map_bot3 = [26, 27]
    
    # W: 7x2 (k*n+i). 
    # Block 0 (k=0,1): C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    w_map0 = [0, 2, 1, 3]
    # Block 1 (k=2,3): C[0,0]->4, C[0,1]->6, C[1,0]->5, C[1,1]->7
    w_map1 = [4, 6, 5, 7]
    # Block 2 (k=4,5): C[0,0]->8, C[0,1]->10, C[1,0]->9, C[1,1]->11
    w_map2 = [8, 10, 9, 11]
    # Block 3 (k=6, 2x1): C[0,0]->12, C[1,0]->13
    w_map3 = [12, 13]
    
    # A_left * B_top
    for term in strassen_terms:
        add_block(term[0], term[1], term[2], u_map_left, v_map_top0, w_map0)
    for term in strassen_terms:
        add_block(term[0], term[1], term[2], u_map_left, v_map_top1, w_map1)
    for term in strassen_terms:
        add_block(term[0], term[1], term[2], u_map_left, v_map_top2, w_map2)
    for term in schoolbook_terms:
        add_block(term[0], term[1], term[2], u_map_left, v_map_top3, w_map3)
        
    # A_right * B_bottom
    for term in strassen_terms:
        add_block(term[0], term[1], term[2], u_map_right, v_map_bot0, w_map0)
    for term in strassen_terms:
        add_block(term[0], term[1], term[2], u_map_right, v_map_bot1, w_map1)
    for term in strassen_terms:
        add_block(term[0], term[1], term[2], u_map_right, v_map_bot2, w_map2)
    for term in schoolbook_terms:
        add_block(term[0], term[1], term[2], u_map_right, v_map_bot3, w_map3)
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Generate the known Rank-50 baseline
    U_base, V_base, W_base = _build_strassen_block()
    if _verify_decomposition(U_base, V_base, W_base, n, m, p):
        best_U, best_V, best_W = U_base, V_base, W_base
        best_rank = 50
    else:
        # Fallback to trivial if Strassen construction failed (should not happen)
        U_triv, V_triv, W_triv = _build_trivial(n, m, p)
        best_U, best_V, W_base = U_triv, V_triv, W_triv
        best_rank = n * m * p

    # 2. Attempt to find a lower rank using global optimization
    # We try ranks lower than the current best.
    # Theoretical lower bound for 2x4x7 is likely around 40-45.
    # Let's try 42, 43, 44, 45, 46, 47, 48, 49.
    
    # To save time, we prioritize ranks that are "likely" to be minimal.
    # 42 is a strong candidate (14 blocks of 2x2x1 rank 3).
    # Let's iterate from 42 up to best_rank - 1.
    
    for rank in range(42, best_rank):
        if time.time() > deadline - 20: # Keep 20s buffer
            break
            
        result = _optimize_rank(n, m, p, rank, start_time, deadline)
        if result is not None:
            U_opt, V_opt, W_opt = result
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = rank
                break # Found a valid lower rank, stop (since we iterate from low to high)
                
    # 3. If we didn't find anything lower, we still have the baseline.
    # Return the best found.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")

