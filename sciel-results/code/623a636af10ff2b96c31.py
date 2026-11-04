"""Matrix multiplication 4x5x6 tensor decomposition using structured block construction."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """
    Returns Strassen's 2x2 matrix multiplication decomposition.
    Rank 7.
    U, V, W are lists of 7 vectors of length 4 (2*2).
    """
    # Standard Strassen factors
    # C = AB
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express this as sum of rank-1 tensors U_k ⊗ V_k ⊗ W_k
    # where T[i,j,k] = sum U[i,j]_k V[j,k]_k W[k,i]_k = 1 if i,j,k correspond to valid mult.
    # Actually, the tensor T is defined by T[i,j,k] = 1 if the entry (i,k) of AB depends on A[i,j] and B[j,k].
    # For matrix mult, T[i,j,k] is 1 if i,j,k form a valid path.
    # Wait, the standard tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    # Entry (i,j) in U, (j,k) in V, (k,i) in W.
    # Sum U[i,j] V[j,k] W[k,i] should be 1 if (i,j,k) is a valid term in C[i,k] = sum_j A[i,j]B[j,k].
    # So for each (i,k), there are m terms (one for each j).
    # The decomposition is a sum of rank 1 tensors.
    # Strassen provides a sum of 7 rank 1 tensors that reconstruct the multiplication.
    
    # Let's define the 7 rank 1 components.
    # Each component corresponds to one of the M's.
    # M1 contributes to C11, C22.
    # This is tricky to map directly to the U,V,W vectors without deriving the linear forms.
    
    # Alternative: Use the known explicit rank-7 decomposition for 2x2.
    # Reference: "Strassen's algorithm"
    
    # Let's use a known explicit list.
    # U vectors (4 entries each: A11, A12, A21, A22)
    # V vectors (4 entries each: B11, B12, B21, B22)
    # W vectors (4 entries each: C11, C12, C21, C22)
    
    # M1: (A11+A22)(B11+B22) -> C11 + C22
    # U1 = [1, 0, 0, 1]
    # V1 = [1, 0, 0, 1]
    # W1 = [1, 0, 0, 1]
    
    # M2: (A21+A22)B11 -> C21
    # U2 = [0, 0, 1, 1]
    # V2 = [1, 0, 0, 0]
    # W2 = [0, 0, 1, 0]
    
    # M3: A11(B12-B22) -> C12
    # U3 = [1, 0, 0, 0]
    # V3 = [0, 1, 0, -1]
    # W3 = [0, 1, 0, 0]
    
    # M4: A22(B21-B11) -> C11, C21
    # U4 = [0, 0, 0, 1]
    # V4 = [0, 0, 1, -1]
    # W4 = [1, 0, 1, 0]
    
    # M5: (A11+A12)B22 -> C11, C12
    # U5 = [1, 1, 0, 0]
    # V5 = [0, 0, 0, 1]
    # W5 = [1, 1, 0, 0]
    
    # M6: (A21-A11)(B11+B12) -> C22
    # U6 = [-1, 0, 1, 0]
    # V6 = [1, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    
    # M7: (A12-A22)(B21+B22) -> C11, C22
    # U7 = [0, 1, 0, -1]
    # V7 = [0, 0, 1, 1]
    # W7 = [1, 0, 0, 1]
    
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    
    return U, V, W

def _schoolbook_decomp(n, m, p):
    """
    Returns schoolbook decomposition for (n,m,p).
    Rank n*m*p.
    """
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify if U, V, W (lists of vectors) form a valid decomposition.
    U: list of (n*m,) vectors
    V: list of (m*p,) vectors
    W: list of (p*n,) vectors
    """
    rank = len(U)
    if rank == 0:
        return False
    
    # Convert to matrices for faster dot product if needed, but rank is small.
    # T[i,j,k] = sum_{r=0}^{rank-1} U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
    
    # We only need to check that T[i,j,k] == 1 for all valid (i,j,k).
    # Note: In matrix multiplication, every (i,j,k) is "valid" in the sense that
    # the tensor entry is 1. The tensor is the structure tensor of the algebra.
    # Yes, for matrix multiplication, the tensor is sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}.
    # So every entry (i,j,k) in the full tensor space is 1.
    
    # Optimization: We can check a subset if rank is large, but for small ranks (like 100)
    # and dimensions 4,5,6 (120 entries), it's fast.
    
    # Precompute U, V, W as numpy arrays for speed
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                
                # sum(U_arr[:, idx_u] * V_arr[:, idx_v] * W_arr[:, idx_w])
                val = np.dot(U_arr[:, idx_u] * V_arr[:, idx_v], W_arr[:, idx_w])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _embed_decomp(U_list, V_list, W_list, n, m, p, N, M, P, offset_i=0, offset_j=0, offset_k=0):
    """
    Embed a decomposition of (n,m,p) into (N,M,P) by padding with zeros.
    Places the sub-problem at the specified offsets.
    U_list: list of (n*m,) vectors
    V_list: list of (m*p,) vectors
    W_list: list of (p*n,) vectors
    Returns U, V, W as lists of vectors of length N*M, M*P, P*N respectively.
    """
    rank = len(U_list)
    U_out = []
    V_out = []
    W_out = []
    
    for k in range(rank):
        u_vec = U_list[k]
        v_vec = V_list[k]
        w_vec = W_list[k]
        
        U_new = [0] * (N*M)
        V_new = [0] * (M*P)
        W_new = [0] * (P*N)
        
        # Place U
        for idx in range(n*m):
            i, j = divmod(idx, m)
            U_new[(i + offset_i)*M + (j + offset_j)] = u_vec[idx]
            
        # Place V
        for idx in range(m*p):
            j, k_idx = divmod(idx, p)
            V_new[(j + offset_j)*P + (k_idx + offset_k)] = v_vec[idx]
            
        # Place W
        for idx in range(p*n):
            k_idx, i = divmod(idx, n)
            W_new[(k_idx + offset_k)*N + (i + offset_i)] = w_vec[idx]
            
        U_out.append(U_new)
        V_out.append(V_new)
        W_out.append(W_new)
        
    return U_out, V_out, W_out

def _combine_decomps(decomps):
    """
    Combine a list of decompositions (each a tuple (U, V, W, n, m, p, off_i, off_j, off_k))
    into a single decomposition for the full space.
    Returns U, V, W lists.
    """
    if not decomps:
        return [], [], []
    
    # Get dimensions from the first decomposition (they should all be the same full N, M, P)
    # Actually, we embed them into the full space first, then just concatenate the lists.
    # The input decomps are already embedded? No, let's assume the caller embeds them.
    # Or better, this function takes pre-embedded decomps.
    
    U_all = []
    V_all = []
    W_all = []
    for U, V, W in decomps:
        U_all.extend(U)
        V_all.extend(V)
        W_all.extend(W)
    return U_all, V_all, W_all

def _recursive_decomp(n, m, p, deadline):
    """
    Recursively decompose T_{n,m,p} by splitting dimensions.
    Returns U, V, W as lists of vectors.
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # Special case: 2x2x2 (Strassen)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook for this sub-problem
        return _schoolbook_decomp(n, m, p)
    
    best_rank = float('inf')
    best_result = None
    
    # Try splitting N
    if n > 1:
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = _recursive_decomp(n1, m, p, deadline)
            U2, V2, W2 = _recursive_decomp(n2, m, p, deadline)
            
            # Combine by concatenation (direct sum)
            U = U1 + U2
            V = V1 + V2
            W = W1 + W2
            
            rank = len(U)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
            
    # Try splitting M
    if m > 1:
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = _recursive_decomp(n, m1, p, deadline)
            U2, V2, W2 = _recursive_decomp(n, m2, p, deadline)
            
            U = U1 + U2
            V = V1 + V2
            W = W1 + W2
            
            rank = len(U)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
            
    # Try splitting P
    if p > 1:
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = _recursive_decomp(n, m, p1, deadline)
            U2, V2, W2 = _recursive_decomp(n, m, p2, deadline)
            
            U = U1 + U2
            V = V1 + V2
            W = W1 + W2
            
            rank = len(U)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
            
    # Fallback
    return _schoolbook_decomp(n, m, p)

def _to_matrices(U, V, W):
    """Convert lists of vectors to matrices of shape (dim, rank)."""
    if not U:
        return [], [], []
    rank = len(U)
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    U_list, V_list, W_list = _recursive_decomp(n, m, p, deadline)
    
    # Verify
    if not _verify_decomp(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook
        U_list, V_list, W_list = _schoolbook_decomp(n, m, p)
        
    U, V, W = _to_matrices(U_list, V_list, W_list)
    
    return {"U": U, "V": V, "W": W}
