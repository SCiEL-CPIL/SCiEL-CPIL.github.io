import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = 3, 4, 4
    
    # Build the target tensor: T[a,b,c] = 1 if A[r,i]*B[i,c] contributes to C[r,c]
    # a = r*m + i, b = i'*p + c, c_out = c'*n + r
    # T[a,b,c_out] = 1 if i == i' and the output index matches
    # Specifically: a=(r,i), b=(j,c), c_out=(c',r') => T=1 if i==j and r==r' and c==c'
    
    # For 3x4x4: a in 0..11, b in 0..15, c in 0..11
    # T[a,b,c] = 1 if a%m == b//p and a//m == c%n and b%p == c//n ... 
    # Wait, let me be more careful.
    # A is 3x4: A[r,i] with r in 0..2, i in 0..3
    # B is 4x4: B[i,c] with i in 0..3, c in 0..3
    # C is 3x4: C[r,c] with r in 0..2, c in 0..3
    # C[r,c] = sum_i A[r,i] * B[i,c]
    # 
    # Tensor indices: a = r*m + i (input to U), b = i*p + c (input to V), c_idx = c*n + r (output for W)
    # T[a, b, c_idx] = 1 if the i from a equals the i from b AND the r from a equals the r from c_idx AND the c from b equals the c from c_idx
    # a = r_a * m + i_a, b = i_b * p + c_b, c_idx = c_c * n + r_c
    # T = 1 if i_a == i_b and r_a == r_c and c_b == c_c
    
    def build_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for r in range(n):
            for i in range(m):
                for c in range(p):
                    a = r * m + i
                    b = i * p + c
                    c_idx = c * n + r
                    T[a, b, c_idx] = 1.0
        return T
    
    T = build_target_tensor()
    
    # Helper to verify a full decomposition quickly
    def verify_full(U, V, W):
        rank = U.shape[1]
        # Compute T_reconstructed[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
        # Use einsum for speed
        T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-10)
    
    # Helper to build U,V,W from local terms for a block
    def build_from_local(local_terms, blocks):
        local_rank = len(local_terms)
        rank = len(blocks) * local_rank
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for b_idx, (a_cs, b_rs, b_cs, c_cs) in enumerate(blocks):
            col_offset = b_idx * local_rank
            for l_idx, (u_list, v_list, w_list) in enumerate(local_terms):
                col = col_offset + l_idx
                for idx, val in u_list:
                    r_loc = idx // 2
                    c_loc = idx % 2
                    g_a_idx = r_loc * m + (a_cs + c_loc)
                    U[g_a_idx, col] += val
                for idx, val in v_list:
                    r_loc = idx // 2
                    c_loc = idx % 2
                    g_b_idx = (b_rs + r_loc) * p + (b_cs + c_loc)
                    V[g_b_idx, col] += val
                for idx, val in w_list:
                    r_loc = idx // 2
                    c_loc = idx % 2
                    g_c_idx = (c_cs + c_loc) * n + r_loc
                    W[g_c_idx, col] += val
        return U, V, W
    
    # Known local 3x2x2 decomposition rank 11
    local_terms_11 = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
        ([ (4, 1) ], [ (0, 1) ], [ (4, 1) ]),
        ([ (5, 1) ], [ (2, 1) ], [ (4, 1) ]),
        ([ (4, 1) ], [ (1, 1) ], [ (5, 1) ]),
        ([ (5, 1) ], [ (3, 1) ], [ (5, 1) ]),
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    # First, verify the known solution works
    U, V, W = build_from_local(local_terms_11, blocks)
    best_rank = len(local_terms_11) * len(blocks)  # 44
    best_U, best_V, best_W = U, V, W
    
    if not verify_full(U, V, W):
        # Fallback to trivial
        best_rank = n * m * p
        best_U = np.zeros((n * m, best_rank), dtype=float)
        best_V = np.zeros((m * p, best_rank), dtype=float)
        best_W = np.zeros((p * n, best_rank), dtype=float)
        for r in range(n):
            for i in range(m):
                for c in range(p):
                    k = r * m * p + i * p + c
                    best_U[r * m + i, k] = 1.0
                    best_V[i * p + c, k] = 1.0
                    best_W[c * n + r, k] = 1.0
    
    # Try to search for a better local 3x2x2 decomposition
    # The 3x2x2 matrix multiplication tensor: A is 3x2, B is 2x2, C is 3x2
    # a = r*2+i (r in 0..2, i in 0..1), b = i'*2+c (i' in 0..1, c in 0..1), c_idx = c'*2+r (c' in 0..1, r' in 0..2)
    # T[a,b,c] = 1 if i==i' and r==r' and c==c'
    
    def build_local_tensor():
        T_loc = np.zeros((6, 4, 6))
        for r in range(3):
            for i in range(2):
                for c in range(2):
                    a = r * 2 + i
                    b = i * 2 + c
                    c_idx = c * 2 + r
                    T_loc[a, b, c_idx] = 1.0
        return T_loc
    
    T_loc = build_local_tensor()
    
    def verify_local(U_loc, V_loc, W_loc):
        T_rec = np.einsum('ak,bk,ck->abc', U_loc, V_loc, W_loc)
        return np.allclose(T_rec, T_loc, atol=1e-10)
    
    # Search for rank 10 local decomposition using randomized approach
    # Try random sparse initializations and optimize
    def try_random_local(rank):
        random.seed(42)
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline:
                return None
            # Random sparse initialization
            U_loc = np.zeros((6, rank))
            V_loc = np.zeros((4, rank))
            W_loc = np.zeros((6, rank))
            
            for k in range(rank):
                # Each factor has some random non-zero entries
                n_u = random.randint(1, 3)
                n_v = random.randint(1, 2)
                n_w = random.randint(1, 3)
                
                for idx in random.sample(range(6), n_u):
                    U_loc[idx, k] = random.choice([1, -1, 0.5, -0.5])
                for idx in random.sample(range(4), n_v):
                    V_loc[idx, k] = random.choice([1, -1, 0.5, -0.5])
                for idx in random.sample(range(6), n_w):
                    W_loc[idx, k] = random.choice([1, -1, 0.5, -0.5])
            
            if verify_local(U_loc, V_loc, W_loc):
                return (U_loc, V_loc, W_loc)
        return None
    
    # Try to find rank 10
    if time.time() < deadline:
        result_10 = try_random_local(10)
        if result_10 is not None:
            U_loc, V_loc, W_loc = result_10
            # Convert to local_terms format
            local_terms_10 = []
            for k in range(10):
                u_list = [(i, U_loc[i, k]) for i in range(6) if abs(U_loc[i, k]) > 1e-10]
                v_list = [(i, V_loc[i, k]) for i in range(4) if abs(V_loc[i, k]) > 1e-10]
                w_list = [(i, W_loc[i, k]) for i in range(6) if abs(W_loc[i, k]) > 1e-10]
                local_terms_10.append((u_list, v_list, w_list))
            
            U_new, V_new, W_new = build_from_local(local_terms_10, blocks)
            if verify_full(U_new, V_new, W_new):
                best_rank = 40
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # If we have time, try more systematic search for rank 10
    if best_rank > 40 and time.time() < deadline:
        # Try structured search: use the rank 11 solution as starting point
        # and try to merge or eliminate terms
        U11 = np.zeros((6, 11))
        V11 = np.zeros((4, 11))
        W11 = np.zeros((6, 11))
        for l_idx, (u_list, v_list, w_list) in enumerate(local_terms_11):
            for idx, val in u_list:
                U11[idx, l_idx] += val
            for idx, val in v_list:
                V11[idx, l_idx] += val
            for idx, val in w_list:
                W11[idx, l_idx] += val
        
        # Try removing one term and adjusting others
        np.random.seed(123)
        for remove_idx in range(11):
            if time.time() > deadline:
                break
            U_test = U11.copy()
            V_test = V11.copy()
            W_test = W11.copy()
            U_test[:, remove_idx] = 0
            V_test[:, remove_idx] = 0
            W_test[:, remove_idx] = 0
            
            # The residual that needs to be covered
            T_rec = np.einsum('ak,bk,ck->abc', U_test, V_test, W_test)
            residual = T_loc - T_rec
            
            # Try to cover residual with one additional term (but we're removing, so we need net -1 rank)
            # This approach won't directly work. Instead, let's try random perturbations of rank 10.
            pass
        
        # More aggressive: try many random rank-10 inits with different sparsity patterns
        for trial in range(200):
            if time.time() > deadline:
                break
            np.random.seed(trial)
            U_loc = np.zeros((6, 10))
            V_loc = np.zeros((4, 10))
            W_loc = np.zeros((6, 10))
            
            for k in range(10):
                # Vary sparsity
                if trial % 3 == 0:
                    # Very sparse
                    for idx in random.sample(range(6), 1):
                        U_loc[idx, k] = random.choice([1, -1])
                    for idx in random.sample(range(4), 1):
                        V_loc[idx, k] = random.choice([1, -1])
                    for idx in random.sample(range(6), 1):
                        W_loc[idx, k] = random.choice([1, -1])
                elif trial % 3 == 1:
                    # Medium sparse
                    for idx in random.sample(range(6), 2):
                        U_loc[idx, k] = random.choice([1, -1, 0.5])
                    for idx in random.sample(range(4), 1):
                        V_loc[idx, k] = random.choice([1, -1])
                    for idx in random.sample(range(6), 2):
                        W_loc[idx, k] = random.choice([1, -1, 0.5])
                else:
                    # Denser
                    for idx in random.sample(range(6), 3):
                        U_loc[idx, k] = random.choice([1, -1, 0.5, -0.5])
                    for idx in random.sample(range(4), 2):
                        V_loc[idx, k] = random.choice([1, -1, 0.5])
                    for idx in random.sample(range(6), 3):
                        W_loc[idx, k] = random.choice([1, -1, 0.5, -0.5])
            
            if verify_local(U_loc, V_loc, W_loc):
                local_terms_10 = []
                for k in range(10):
                    u_list = [(i, U_loc[i, k]) for i in range(6) if abs(U_loc[i, k]) > 1e-10]
                    v_list = [(i, V_loc[i, k]) for i in range(4) if abs(V_loc[i, k]) > 1e-10]
                    w_list = [(i, W_loc[i, k]) for i in range(6) if abs(W_loc[i, k]) > 1e-10]
                    local_terms_10.append((u_list, v_list, w_list))
                
                U_new, V_new, W_new = build_from_local(local_terms_10, blocks)
                if verify_full(U_new, V_new, W_new):
                    best_rank = 40
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
