import time
import numpy as np

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                T_recon[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9

def construct_strassen():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_steps = [
        { 'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1} },
        { 'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1} },
        { 'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1} },
        { 'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1} },
        { 'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1} },
        { 'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1} },
        { 'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1} }
    ]

    def place_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_start + lc
                U[gi * m + gj][col] += coeff
            for local_idx, coeff in step['v'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_start + lr
                gj = b_col_start + lc
                V[gi * p + gj][col] += coeff
            for local_idx, coeff in step['w'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gk = c_col_start + lc
                gi = lr
                W[gk * n + gi][col] += coeff

    place_block(0, 0, 0, 0, 0)
    place_block(0, 0, 2, 2, 7)
    place_block(2, 2, 0, 0, 14)
    place_block(2, 2, 2, 2, 21)

    return U, V, W

def try_reduce_rank(U, V, W, n=2, m=4, p=4, max_time=10):
    """
    Attempt to reduce rank by removing columns.
    """
    start_t = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = len(U_arr[0])
    
    # Heuristic: Try removing columns with smallest L2 norm
    norms = []
    for r in range(current_rank):
        norm = np.linalg.norm(U_arr[:, r]) + np.linalg.norm(V_arr[:, r]) + np.linalg.norm(W_arr[:, r])
        norms.append((norm, r))
    norms.sort()
    
    for _, r_to_remove in norms:
        if time.time() - start_t > max_time:
            break
        if current_rank <= 26: # Don't go below 26, unlikely to succeed and save time
            break
            
        mask = [i for i in range(current_rank) if i != r_to_remove]
        new_U = U_arr[:, mask]
        new_V = V_arr[:, mask]
        new_W = W_arr[:, mask]
        
        # Verify
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(mask)):
                        s += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                    T_recon[i, j, k] = s
        
        T_target = np.ones((n, m, p), dtype=float)
        if np.max(np.abs(T_recon - T_target)) < 1e-9:
            U_arr = new_U
            V_arr = new_V
            W_arr = new_W
            current_rank = len(mask)
            # Recalculate norms
            norms = []
            for r in range(current_rank):
                norm = np.linalg.norm(U_arr[:, r]) + np.linalg.norm(V_arr[:, r]) + np.linalg.norm(W_arr[:, r])
                norms.append((norm, r))
            norms.sort()
            
    # Convert back to list of lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    return U_out, V_out, W_out

def try_laderman_embedding(n=2, m=4, p=4):
    """
    Attempt to find a rank-19 decomposition by embedding the 2x4x4 tensor
    into a 3x3x3 tensor (Laderman's algorithm) and projecting.
    
    This is a heuristic attempt. If it fails, we return None.
    """
    # Laderman's algorithm for 3x3x3 has rank 19.
    # The tensor is T[i,j,k] = sum_l A[i,l] B[l,k].
    # We want to map our 2x4x4 tensor T[i,j,k] = A[i,j] B[j,k] (where A is 2x4, B is 4x4, result C is 2x4)
    # into a 3x3x3 tensor.
    
    # Let's try to map indices:
    # Our i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # Laderman i' in {0,1,2}, j' in {0,1,2}, k' in {0,1,2}.
    
    # This mapping is non-trivial and likely won't work directly with simple padding 
    # because the dimensions don't match up neatly (4 vs 3).
    # However, we can try a linear combination approach.
    
    # Actually, a more direct approach to "projection" is to assume the rank-19 
    # decomposition of 3x3x3 can be "stretched" or "compressed". 
    # But given the complexity and low probability of success without a known mapping,
    # and the strict time limit, this specific embedding is risky.
    
    # Let's try a different "projection" idea: 
    # Can we express the 2x4x4 tensor as a sub-tensor of a larger tensor 
    # that has a known low-rank decomposition?
    
    # The 2x4x4 tensor is equivalent to two 2x4x2 tensors or four 2x2x2 tensors.
    # We already used Strassen for the four 2x2x2 parts (rank 28).
    
    # Is there a known rank-27 or lower decomposition for 2x4x4?
    # The lower bound is max(2*4, 4*4, 4*2) = 16? No, that's for bilinear complexity lower bounds.
    # For 2x4x4, the Strassen rank is 28.
    # It is known that the rank of <2,4,4> is at most 28 (Strassen).
    # Is it less? 
    # According to literature, the rank of <2,4,4> is not definitively known to be less than 28 
    # in simple constructions, but 28 is the standard Strassen bound.
    
    # However, we can try to "merge" the Strassen blocks.
    # The 4 blocks are independent in Strassen. 
    # If there are linear dependencies between the 28 rank-1 tensors, we can reduce the rank.
    
    # Let's check if the 28 vectors in the tensor product space are linearly independent.
    # If they are, rank is 28. If not, we can reduce.
    
    # We already did a greedy removal in try_reduce_rank. 
    # Let's try a more systematic linear algebra approach to find the minimal rank 
    # within the span of the Strassen decomposition.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Get the base Strassen decomposition
    U, V, W = construct_strassen()
    
    # Verify base
    if not verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
        
    # Attempt to reduce rank
    U, V, W = try_reduce_rank(U, V, W, n, m, p, max_time=10)
    
    return {"U": U, "V": V, "W": W}
