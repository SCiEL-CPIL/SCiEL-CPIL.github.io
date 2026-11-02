import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 10]

def verify_decomposition(U, V, W, n, m, p, seed=12345):
    """Verify that the decomposition is correct."""
    np.random.seed(seed)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    # Compute using decomposition
    # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    R = U.shape[1]
    
    # For each rank term r, we compute sum_r (U^T A_vec) * (V^T B_vec) * W[:,r]
    # where A_vec is flattened A, B_vec is flattened B
    
    A_flat = A.flatten()  # shape (n*m,)
    B_flat = B.flatten()  # shape (m*p,)
    
    # U @ A_flat gives shape (R,)
    # V @ B_flat gives shape (R,)
    u_contrib = U.T @ A_flat  # (R,)
    v_contrib = V.T @ B_flat  # (R,)
    
    # C_decomp[i,k] = sum_r u_contrib[r] * v_contrib[r] * W[k*n+i, r]
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_decomp[i, k] = np.dot(u_contrib * v_contrib, W[k*n + i, :])
    
    return np.allclose(C_true, C_decomp, atol=1e-10)

def strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2."""
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    return terms

def build_decomposition_from_terms(terms, n, m, p, block_sizes):
    """Build U, V, W matrices from a list of (a_vec, b_vec, c_vec) terms.
    block_sizes specifies how many columns of B each term's b_vec covers.
    The b_vec is flattened as B[0,0], B[0,1], ..., B[1,0], B[1,1], ...
    For p columns, B has indices (j,k) with j in 0..m-1, k in 0..p-1.
    The W index is k*n+i.
    """
    R = len(terms)
    N = n * m
    M = m * p
    P = p * n
    
    U = np.zeros((N, R), dtype=float)
    V = np.zeros((M, R), dtype=float)
    W = np.zeros((P, R), dtype=float)
    
    for r, (a_vec, b_vec, c_vec) in enumerate(terms):
        # a_vec has n*m entries
        for idx, val in enumerate(a_vec):
            U[idx, r] = val
        # b_vec has m*p entries
        for idx, val in enumerate(b_vec):
            V[idx, r] = val
        # c_vec has p*n entries
        for idx, val in enumerate(c_vec):
            W[idx, r] = val
    
    return U, V, W

def find_rank10_2x2x3(deadline):
    """Try to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) of shapes (4,10), (6,10), (6,10) or None if not found."""
    n, m, p = 2, 2, 3
    R = 10
    N, M, P = n*m, m*p, p*n  # 4, 6, 6
    
    # The tensor T has entries T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[(i,j),(j',k),(i',k')]
    # Non-zero entries: for each (i,j,k), T[(i,j),(j,k),(i,k)] = 1
    # There are 2*2*3 = 12 such entries
    
    # Build the system: for each of the 12 (i,j,k) tuples, we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i',j',k') not of the form (i,j,k) with j=j', we need 0
    
    # Actually, the tensor is: T[a, b, c] where a indexes A (n*m=4), b indexes B (m*p=6), c indexes C (p*n=6)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k; else 0
    
    # Let's enumerate all non-zero entries
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nonzero_entries.append((a, b, c))
    
    # We have 12 equations of the form: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # Plus many more equations = 0 for other (a,b,c)
    # Total possible (a,b,c): 4*6*6 = 144
    # We need all 144 equations to be satisfied
    
    # This is a system of 144 equations in 3*R = 30 unknowns (but they're coupled via products)
    # We'll use random initialization and gradient-free search
    
    best_error = float('inf')
    best_solution = None
    
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Initialize randomly
        U = np.random.randn(N, R) * 0.5
        V = np.random.randn(M, R) * 0.5
        W = np.random.randn(P, R) * 0.5
        
        # Compute the residual
        def compute_residual(U, V, W):
            residual = 0.0
            for (a, b, c) in nonzero_entries:
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                residual += (val - 1.0) ** 2
            # Also check some zero entries
            for a in range(N):
                for b in range(M):
                    for c in range(P):
                        if (a, b, c) not in nonzero_entries:
                            val = np.dot(U[a, :] * V[b, :], W[c, :])
                            residual += val ** 2
            return residual
        
        # Simple iterative improvement: try to reduce residual by adjusting one term at a time
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            err = compute_residual(U, V, W)
            if err < best_error:
                best_error = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-12:
                break
            
            # Try random perturbations
            improved = False
            for _ in range(20):
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                # Perturb one random entry
                mat = np.random.choice(['U', 'V', 'W'])
                idx1 = np.random.randint(0, N if mat == 'U' else M if mat == 'V' else P)
                idx2 = np.random.randint(0, R)
                delta = np.random.randn() * 0.1
                
                if mat == 'U':
                    U_new[idx1, idx2] += delta
                elif mat == 'V':
                    V_new[idx1, idx2] += delta
                else:
                    W_new[idx1, idx2] += delta
                
                new_err = compute_residual(U_new, V_new, W_new)
                if new_err < err:
                    U, V, W = U_new, V_new, W_new
                    err = new_err
                    improved = True
                    break
        
        if not improved and err < 1e-8:
            break
    
    if best_solution is not None and best_error < 1e-10:
        return best_solution
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 20
    P = p * n  # 20
    
    # First, try to find rank-10 decomposition for 2x2x3
    rank10_2x2x3 = None
    if time.time() < deadline - 30:
        rank10_2x2x3 = find_rank10_2x2x3(deadline - 10)
    
    if rank10_2x2x3 is not None:
        # Use two 2x2x3 blocks (rank 10 each) and one 2x2x4 block (rank 14)
        # Total rank: 10 + 10 + 14 = 34
        U3, V3, W3 = rank10_2x2x3
        
        # Strassen for 2x2x2, extended to 2x2x4 by using two independent 2x2x2 blocks
        strassen_terms = strassen_2x2x2()
        
        # Build full decomposition
        # Block 1: columns 0-2 of B (2x2x3), rank 10
        # Block 2: columns 3-5 of B (2x2x3), rank 10  
        # Block 3: columns 6-9 of B (2x2x4), rank 14 (two Strassen 2x2x2)
        
        R = 34
        U = np.zeros((N, R), dtype=float)
        V = np.zeros((M, R), dtype=float)
        W = np.zeros((P, R), dtype=float)
        
        col = 0
        # Block 1: 2x2x3 for columns 0,1,2
        for r in range(10):
            U[:, col] = U3[:, r]
            # V3 has shape (6, 10), indices (j,k) with k in 0..2
            for j in range(m):
                for k in range(3):
                    V[j*p + k, col] = V3[j*3 + k, r]
            # W3 has shape (6, 10), indices (k,i) with k in 0..2
            for k in range(3):
                for i in range(n):
                    W[k*n + i, col] = W3[k*n + i, r]
            col += 1
        
        # Block 2: 2x2x3 for columns 3,4,5
        for r in range(10):
            U[:, col] = U3[:, r]
            for j in range(m):
                for k in range(3):
                    V[j*p + (3+k), col] = V3[j*3 + k, r]
            for k in range(3):
                for i in range(n):
                    W[(3+k)*n + i, col] = W3[k*n + i, r]
            col += 1
        
        # Block 3: 2x2x4 for columns 6,7,8,9 (two Strassen 2x2x2)
        for sub_block in range(2):
            base_col = 6 + sub_block * 2
            for a_vec, b_vec, c_vec in strassen_terms:
                for idx, val in enumerate(a_vec):
                    U[idx, col] = val
                # b_vec has 4 entries for 2x2: (0,0),(0,1),(1,0),(1,1)
                b_vec_expanded = [0.0] * (m * 4)
                b_vec_expanded[0] = b_vec[0]   # (0,0)
                b_vec_expanded[1] = b_vec[1]   # (0,1)
                b_vec_expanded[4] = b_vec[2]   # (1,0)
                b_vec_expanded[5] = b_vec[3]   # (1,1)
                for j in range(m):
                    for k in range(4):
                        V[j*p + (base_col+k), col] = b_vec_expanded[j*4 + k]
                # c_vec has 4 entries for 2x2: (0,0),(0,1),(1,0),(1,1) meaning (k,i)
                c_vec_expanded = [0.0] * (4 * n)
                c_vec_expanded[0] = c_vec[0]   # (k=0,i=0)
                c_vec_expanded[1] = c_vec[1]   # (k=0,i=1)
                c_vec_expanded[2] = c_vec[2]   # (k=1,i=0)
                c_vec_expanded[3] = c_vec[3]   # (k=1,i=1)
                for k in range(4):
                    for i in range(n):
                        W[(base_col+k)*n + i, col] = c_vec_expanded[k*n + i]
                col += 1
        
        # Verify
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 35 using 5 independent Strassen blocks
    strassen_terms = strassen_2x2x2()
    num_blocks = p // 2  # 5
    R = 7 * num_blocks  # 35
    
    U = np.zeros((N, R), dtype=float)
    V = np.zeros((M, R), dtype=float)
    W = np.zeros((P, R), dtype=float)
    
    col = 0
    for q in range(num_blocks):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx, val in enumerate(a_vec):
                U[idx, col] = val
            # b_vec for 2x2: (0,0),(0,1),(1,0),(1,1)
            V[0*p + 2*q, col] = b_vec[0]
            V[0*p + 2*q+1, col] = b_vec[1]
            V[1*p + 2*q, col] = b_vec[2]
            V[1*p + 2*q+1, col] = b_vec[3]
            # c_vec for 2x2: (k,i) with k in 0,1 and i in 0,1
            W[2*q*n + 0, col] = c_vec[0]
            W[2*q*n + 1, col] = c_vec[1]
            W[(2*q+1)*n + 0, col] = c_vec[2]
            W[(2*q+1)*n + 1, col] = c_vec[3]
            col += 1
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Last resort: schoolbook
    R_school = n * m * p
    U = np.zeros((N, R_school), dtype=float)
    V = np.zeros((M, R_school), dtype=float)
    W = np.zeros((P, R_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
