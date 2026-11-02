import time
import numpy as np
from itertools import product

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _verify_decomposition(U, V, W, T_exp):
    r = U.shape[1]
    T_check = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.array_equal(T_check, T_exp)

def _build_rank44_fallback(n, m, p):
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    col_offset = 0
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        # Block 1: A[:, 0] * B[0, k0:k1] - rank 4
        U[0, col_offset] = 1
        V[k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        U[0, col_offset] = 1
        V[k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        U[3, col_offset] = 1
        V[k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        U[3, col_offset] = 1
        V[k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        # Block 2: A[:, 1:3] * B[1:3, k0:k1] - Strassen rank 7
        U[1, col_offset] = 1
        U[4, col_offset] = 1
        V[p+k0, col_offset] = 1
        V[2*p+k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        U[3, col_offset] = 1
        U[4, col_offset] = 1
        V[p+k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = -1
        col_offset += 1
        U[1, col_offset] = 1
        V[p+k1, col_offset] = 1
        V[2*p+k1, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        U[4, col_offset] = 1
        V[2*p+k0, col_offset] = 1
        V[p+k0, col_offset] = -1
        W[k0*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        U[1, col_offset] = 1
        U[2, col_offset] = 1
        V[2*p+k1, col_offset] = 1
        W[k0*n + 0, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        U[3, col_offset] = 1
        U[1, col_offset] = -1
        V[p+k0, col_offset] = 1
        V[p+k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        U[2, col_offset] = 1
        U[4, col_offset] = -1
        V[2*p+k0, col_offset] = 1
        V[2*p+k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
    return U, V, W

def _try_rank10_block_search(deadline):
    """Search for a rank-10 decomposition of the 2x3x3x2 block tensor."""
    T_block = np.zeros((6, 6, 4), dtype=np.int64)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T_block[i*3+j, j*2+k, k*2+i] = 1
    
    # Strategy: Try to find U (6,10), V (6,10), W (4,10) with small integer entries
    # such that sum_d U[:,d] outer V[:,d] outer W[:,d] = T_block
    
    # Use a structured ansatz based on combining the 4-term and 7-term solutions
    # with potential sharing.
    
    # The 4 terms handle A[:,0] * B[0,:]:
    # t1: U=[1,0,0,0,0,0], V=[1,0,0,0,0,0], W=[1,0,0,0]
    # t2: U=[1,0,0,0,0,0], V=[0,1,0,0,0,0], W=[0,1,0,0]
    # t3: U=[0,0,0,1,0,0], V=[1,0,0,0,0,0], W=[0,0,1,0]
    # t4: U=[0,0,0,1,0,0], V=[0,1,0,0,0,0], W=[0,0,0,1]
    
    # The 7 Strassen terms handle A[:,1:3] * B[1:3,:]:
    # Let's try to merge some terms.
    
    # Alternative approach: use the fact that we have 12 equations (one per nonzero entry)
    # and 10 variables (rank-10 means 10 terms, each with 6*6*4=144 entries but constrained).
    
    # Let me try a direct parameterized search.
    # Assume entries are in {-1, 0, 1} and try random searches.
    
    best_rank = 11  # fallback
    
    # Try random search for rank 10
    np.random.seed(42)
    for attempt in range(500):
        if time.time() > deadline:
            break
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(6, 10))
        V = np.random.randint(-2, 3, size=(6, 10))
        W = np.random.randint(-2, 3, size=(4, 10))
        
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        if np.array_equal(T_check, T_block):
            return U, V, W
    
    # If random search fails, try a more structured approach
    # Use the known rank-11 solution and try to merge two terms
    
    # Rank-11 solution:
    U11 = np.zeros((6, 11), dtype=np.int64)
    V11 = np.zeros((6, 11), dtype=np.int64)
    W11 = np.zeros((4, 11), dtype=np.int64)
    
    col = 0
    # 4 terms for A[:,0]*B[0,:]
    U11[0, col] = 1; V11[0, col] = 1; W11[0, col] = 1; col += 1
    U11[0, col] = 1; V11[1, col] = 1; W11[1, col] = 1; col += 1
    U11[3, col] = 1; V11[0, col] = 1; W11[2, col] = 1; col += 1
    U11[3, col] = 1; V11[1, col] = 1; W11[3, col] = 1; col += 1
    
    # 7 Strassen terms for A[:,1:3]*B[1:3,:]
    U11[1, col] = 1; U11[4, col] = 1; V11[3, col] = 1; V11[5, col] = 1; W11[0, col] = 1; W11[3, col] = 1; col += 1
    U11[3, col] = 1; U11[4, col] = 1; V11[3, col] = 1; W11[2, col] = 1; W11[3, col] = -1; col += 1
    U11[1, col] = 1; V11[4, col] = 1; V11[5, col] = -1; W11[1, col] = 1; W11[3, col] = 1; col += 1
    U11[4, col] = 1; V11[5, col] = 1; V11[3, col] = -1; W11[0, col] = 1; W11[2, col] = 1; col += 1
    U11[1, col] = 1; U11[2, col] = 1; V11[5, col] = 1; W11[0, col] = -1; W11[1, col] = 1; col += 1
    U11[3, col] = 1; U11[1, col] = -1; V11[3, col] = 1; V11[4, col] = 1; W11[3, col] = 1; col += 1
    U11[2, col] = 1; U11[4, col] = -1; V11[5, col] = 1; V11[4, col] = 1; W11[0, col] = 1; col += 1
    
    # Verify rank-11
    T_check = np.einsum('ad,bd,cd->abc', U11, V11, W11)
    if not np.array_equal(T_check, T_block):
        return None
    
    # Try to merge pairs of terms to get rank 10
    # Two terms d1 and d2 can be merged if there exist coefficients such that
    # the sum of their outer products equals a single outer product.
    # This is generally not possible, but we can try linear combinations.
    
    # Try: for each pair (i,j), see if U[:,i] and U[:,j] are proportional, etc.
    for i in range(11):
        for j in range(i+1, 11):
            if time.time() > deadline:
                break
            # Check if we can combine terms i and j
            # New term: try U_new = U[:,i] + c*U[:,j] for various c
            for c in range(-3, 4):
                U_new = U11[:, i] + c * U11[:, j]
                if np.all(U_new == 0):
                    continue
                # Similarly for V and W
                for d in range(-3, 4):
                    V_new = V11[:, i] + d * V11[:, j]
                    if np.all(V_new == 0):
                        continue
                    for e in range(-3, 4):
                        W_new = W11[:, i] + e * W11[:, j]
                        if np.all(W_new == 0):
                            continue
                        # Check if U_new outer V_new outer W_new = U_i outer V_i outer W_i + U_j outer V_j outer W_j
                        T_old = np.einsum('a,b,c->abc', U11[:,i], V11[:,i], W11[:,i]) + \
                               np.einsum('a,b,c->abc', U11[:,j], V11[:,j], W11[:,j])
                        T_new = np.einsum('a,b,c->abc', U_new, V_new, W_new)
                        if np.array_equal(T_old, T_new):
                            # Build rank-10 solution
                            U10 = np.zeros((6, 10), dtype=np.int64)
                            V10 = np.zeros((6, 10), dtype=np.int64)
                            W10 = np.zeros((4, 10), dtype=np.int64)
                            col10 = 0
                            for k in range(11):
                                if k == i or k == j:
                                    continue
                                U10[:, col10] = U11[:, k]
                                V10[:, col10] = V11[:, k]
                                W10[:, col10] = W11[:, k]
                                col10 += 1
                            U10[:, col10] = U_new
                            V10[:, col10] = V_new
                            W10[:, col10] = W_new
                            # Verify
                            T_check = np.einsum('ad,bd,cd->abc', U10, V10, W10)
                            if np.array_equal(T_check, T_block):
                                return U10, V10, W10
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    T_exp = _build_target_tensor(n, m, p)
    
    # Try to find a rank-40 solution (rank-10 per block)
    result = _try_rank10_block_search(deadline)
    if result is not None:
        U_block, V_block, W_block = result
        U = np.zeros((n*m, 40), dtype=np.int64)
        V = np.zeros((m*p, 40), dtype=np.int64)
        W = np.zeros((p*n, 40), dtype=np.int64)
        for block in range(4):
            k0 = block * 2
            col_start = block * 10
            for d in range(10):
                # Map block-local indices to global
                # U indices: i*3+j where i in 0..1, j in 0..2 -> global row i*3+j
                # V indices: j*2+k where j in 0..2, k in 0..1 -> global row j*8 + (k0+k)
                # W indices: k*2+i where k in 0..1, i in 0..1 -> global row (k0+k)*2 + i
                for a in range(6):
                    U[a, col_start+d] = U_block[a, d]
                for b in range(6):
                    j = b // 2
                    k = b % 2
                    V[j*8 + k0 + k, col_start+d] = V_block[b, d]
                for c in range(4):
                    k = c // 2
                    i = c % 2
                    W[(k0+k)*2 + i, col_start+d] = W_block[c, d]
        
        if _verify_decomposition(U, V, W, T_exp):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-44
    U, V, W = _build_rank44_fallback(n, m, p)
    if _verify_decomposition(U, V, W, T_exp):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}


