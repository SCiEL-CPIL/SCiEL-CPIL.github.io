import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # DFT-based approach: exploit cyclic structure of 9 columns
    # The 9 columns can be seen as 3 groups of 3, allowing DFT diagonalization
    # For each DFT frequency, we get a 2x3x3 tensor
    
    # Build the 3 base 2x3x3 tensors for each DFT frequency
    # Using omega = exp(2*pi*i/3)
    omega = np.exp(2j * np.pi / 3)
    
    # T_base[l] is the 2x3x3 tensor for the l-th DFT frequency
    T_bases = []
    for freq in range(3):
        T_base = np.zeros((6, 9, 6), dtype=complex)
        for i in range(2):
            for j in range(3):
                for s in range(3):
                    for l in range(3):
                        # k = 3*l + s, so the DFT coefficient for frequency freq
                        # is sum over l of omega^(freq*l) * T[i*m+j, j*p+k, k*n+i]
                        # where k = 3*l + s
                        coeff = omega ** (freq * l)
                        T_base[i*3+j, j*3+s, s*2+i] += coeff
        T_bases.append(T_base)
    
    # For each frequency, try to find a low-rank decomposition
    # The total rank will be sum of ranks across frequencies
    
    def try_decompose_tensor(T_target, r_max, max_trials=200):
        """Try to find rank-r decomposition of a 6x9x6 tensor for r from 5 to r_max"""
        for r in range(5, r_max + 1):
            if time.time() > deadline:
                return None, None
            for trial in range(max_trials):
                if time.time() > deadline:
                    return None, None
                
                seed = 70000 + r * 1000 + trial
                np.random.seed(seed)
                
                V = np.random.randn(9, r)
                W = np.random.randn(6, r)
                
                # Build Kronecker matrix
                M = np.zeros((54, r))
                for r_idx in range(r):
                    M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
                
                try:
                    # Check if M has full column rank
                    M_rank = np.linalg.matrix_rank(M, tol=1e-10)
                    if M_rank < r:
                        continue
                    
                    # Solve least squares for U
                    M_pinv = np.linalg.pinv(M)
                    
                    U = np.zeros((6, r))
                    for a in range(6):
                        target = T_target[a, :, :].flatten()
                        U[a, :] = M_pinv @ target
                    
                    # Verify
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T_approx, T_target, atol=1e-8):
                        return U, (V, W, r)
                except:
                    continue
        return None, None

    # Try to decompose each base tensor
    found_all = True
    U_parts = []
    V_parts = []
    W_parts = []
    total_rank = 0
    
    for freq in range(3):
        T_base = T_bases[freq]
        
        # Try ranks 5, 6, 7, 8 for this base tensor
        U_base, V_base_info = try_decompose_tensor(T_base, r_max=8, max_trials=150)
        
        if U_base is None:
            found_all = False
            break
        
        V_b, W_b, r_b = V_base_info
        U_parts.append(U_base)
        V_parts.append(V_b)
        W_parts.append(W_b)
        total_rank += r_b
    
    if found_all and total_rank < best_rank:
        # Combine the decompositions
        U_full = np.zeros((6, total_rank))
        V_full = np.zeros((27, total_rank))
        W_full = np.zeros((18, total_rank))
        
        col_offset = 0
        for freq in range(3):
            r_b = V_parts[freq].shape[1]
            U_full[:, col_offset:col_offset+r_b] = U_parts[freq]
            
            # Map the 9 columns of V_base back to 27 columns of V_full
            # V_base has shape (9, r_b) where row index is j*3+s (j=0..2, s=0..2)
            # V_full has shape (27, total_rank) where row index is j*9+k (j=0..2, k=0..8)
            # For the freq-th DFT component, we need to spread V_base across the 3 groups
            for r_idx in range(r_b):
                for j in range(3):
                    for s in range(3):
                        v_val = V_parts[freq][j*3+s, r_idx]
                        # This value contributes to V_full[j*9 + 3*l + s, col_offset + r_idx]
                        # for each l, with DFT coefficient
                        for l in range(3):
                            coeff = omega ** (freq * l)
                            V_full[j*9 + 3*l + s, col_offset + r_idx] += v_val * coeff
            
            # W_base has shape (6, r_b) where row index is s*2+i (s=0..2, i=0..1)
            # W_full has shape (18, total_rank) where row index is k*2+i (k=0..8, i=0..1)
            # For the freq-th DFT component, W_base[s*2+i] maps to W_full[(3*l+s)*2+i]
            # with DFT coefficient
            for r_idx in range(r_b):
                for s in range(3):
                    for i in range(2):
                        w_val = W_parts[freq][s*2+i, r_idx]
                        for l in range(3):
                            coeff = omega ** (freq * l)
                            W_full[(3*l+s)*2 + i, col_offset + r_idx] += w_val * coeff
            
            col_offset += r_b
        
        if verify_solution(U_full, V_full, W_full, T):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = total_rank
    
    # Direct search on full tensor if DFT didn't help enough
    if best_rank > 20:
        for r in [18, 20, 21, 22, 24]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                break
            for trial in range(30):
                if time.time() > deadline:
                    break
                np.random.seed(80000 + r * 1000 + trial)
                
                V = np.random.randn(27, r)
                W = np.random.randn(18, r)
                
                # Build Kronecker matrix
                M = np.zeros((486, r))
                for r_idx in range(r):
                    M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
                
                try:
                    M_rank = np.linalg.matrix_rank(M, tol=1e-10)
                    if M_rank < r:
                        continue
                    
                    M_pinv = np.linalg.pinv(M)
                    
                    U = np.zeros((6, r))
                    for a in range(6):
                        target = T[a, :, :].flatten()
                        U[a, :] = M_pinv @ target
                    
                    if verify_solution(U, V, W, T):
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                        break
                except:
                    continue
            if best_rank == r:
                break

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
