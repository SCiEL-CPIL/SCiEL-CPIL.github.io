"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x7 matrix.
    
    Strategy: Use Strassen's 7-term decomposition for 2x2x2 sub-blocks where possible,
    and schoolbook for the remaining singleton dimensions. Then apply linear dependency
    removal to further reduce rank.
    
    The tensor T[i,j,k] for matrix multiplication satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all (i,j,k)
    
    We build the decomposition by partitioning the index space into blocks and
    applying known low-rank decompositions to each block.
    """
    t0 = time.time()
    deadline = t0 + 130  # safety margin before 150s kill
    
    n, m, p = 3, 5, 7
    
    def elapsed():
        return time.time() - t0
    
    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    # For multiplying [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # The 7 products are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(e+f)
    # m7 = (b-c)(g+h)
    # 
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # In our tensor notation for a 2x2x2 block with local indices (i', j', k') in {0,1}:
    # U_local[i'*2+j', c], V_local[j'*2+k', c], W_local[k'*2+i', c]
    # We need: sum_c U[i',j',c] V[j',k',c] W[k',i',c] = 1 for all (i',j',k')
    #
    # The 7 terms give:
    # Term 1: U has (1,0,0,1) for (i',j') in {(0,0),(1,1)}, V has (1,0,1,0) for (j',k') in {(0,0),(1,0)}, W has (1,0,0,1) for (k',i') in {(0,0),(1,1)}
    # Wait, let me be more careful.
    #
    # For the 2x2x2 matmul tensor, T[i',j',k'] = 1 for all 8 combinations (it's the identity tensor).
    # Actually no - the matmul tensor T_{ijk} where i indexes rows of A, j columns of A / rows of B, k columns of B.
    # T_{ijk} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k) - it's the "multiplication" tensor.
    #
    # For 2x2: T[i',j',k'] = 1 for all (i',j',k') in {0,1}^3. That's 8 entries, all equal to 1.
    #
    # Strassen gives rank 7 < 8. The 7 rank-1 terms:
    # Each term c has U_c (2x2 flattened to 4), V_c (4), W_c (4) such that
    # sum_c U_c[i',j'] * V_c[j',k'] * W_c[k',i'] = 1 for all (i',j',k')
    
    def strassen_2x2x2():
        """Return the 7 terms of Strassen's decomposition for 2x2x2.
        Each term is (u, v, w) where u, v, w are length-4 arrays.
        u[i*2+j], v[j*2+k], w[k*2+i]
        """
        terms = []
        # m1 = (a00+a11)(b00+b11): 
        # u: a00=1, a11=1 -> u = [1,0,0,1]
        # v: b00=1, b11=1 -> v = [1,0,0,1]  
        # w: c00=1, c11=1 -> w = [1,0,0,1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # m2 = (a10+a11)*b00:
        # u: a10=1, a11=1 -> u = [0,0,1,1]
        # v: b00=1 -> v = [1,0,0,0]
        # w: c10=1 -> w = [0,1,0,0]
        terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
        
        # m3 = a00*(b01-b11):
        # u: a00=1 -> u = [1,0,0,0]
        # v: b01=1, b11=-1 -> v = [0,1,0,-1]
        # w: c01=1 -> w = [0,0,1,0]
        terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))
        
        # m4 = a11*(b10-b00):
        # u: a11=1 -> u = [0,0,0,1]
        # v: b10=1, b00=-1 -> v = [-1,0,1,0]
        # w: c10=1 -> w = [0,1,0,0]
        terms.append(([0,0,0,1], [-1,0,1,0], [0,1,0,0]))
        
        # m5 = (a00+a01)*b11:
        # u: a00=1, a01=1 -> u = [1,1,0,0]
        # v: b11=1 -> v = [0,0,0,1]
        # w: c01=1 -> w = [0,0,1,0]
        terms.append(([1,1,0,0], [0,0,0,1], [0,0,1,0]))
        
        # m6 = (a10-a11)*(b00+b01):
        # u: a10=1, a11=-1 -> u = [0,0,1,-1]
        # v: b00=1, b01=1 -> v = [1,1,0,0]
        # w: c11=1 -> w = [0,0,0,1]
        terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))
        
        # m7 = (a01-a10)*(b10+b11):
        # u: a01=1, a10=-1 -> u = [0,1,-1,0]
        # v: b10=1, b11=1 -> v = [0,0,1,1]
        # w: c00=1 -> w = [1,0,0,0]
        terms.append(([0,1,-1,0], [0,0,1,1], [1,0,0,0]))
        
        return terms
    
    def verify_2x2x2(terms):
        """Verify that the 7 terms correctly decompose the 2x2x2 matmul tensor."""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for u, v, w in terms:
                        s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify Strassen
    strassen_terms = strassen_2x2x2()
    assert verify_2x2x2(strassen_terms), "Strassen verification failed"
    
    # Now build the full 3x5x7 decomposition.
    # Partition:
    # i in {0,1,2}: pair (0,1) as Strassen, 2 as singleton
    # j in {0,1,2,3,4}: pairs (0,1), (2,3) as Strassen, 4 as singleton
    # k in {0,1,2,3,4,5,6}: pairs (0,1), (2,3), (4,5) as Strassen, 6 as singleton
    
    # We'll build the decomposition term by term.
    # Each global term has U (15,), V (35,), W (21,)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_strassen_block(i_start, j_start, k_start):
        """Add a Strassen 2x2x2 block at the given offset."""
        for u_loc, v_loc, w_loc in strassen_terms:
            u_vec = [0.0] * (n * m)  # 15
            v_vec = [0.0] * (m * p)  # 35
            w_vec = [0.0] * (p * n)  # 21
            
            for di in range(2):
                for dj in range(2):
                    u_vec[(i_start + di) * m + (j_start + dj)] = u_loc[di*2+dj]
            
            for dj in range(2):
                for dk in range(2):
                    v_vec[(j_start + dj) * p + (k_start + dk)] = v_loc[dj*2+dk]
            
            for dk in range(2):
                for di in range(2):
                    w_vec[(k_start + dk) * n + (i_start + di)] = w_loc[dk*2+di]
            
            add_term(u_vec, v_vec, w_vec)
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term for (i,j,k)."""
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[i * m + j] = 1.0
        v_vec[j * p + k] = 1.0
        w_vec[k * n + i] = 1.0
        add_term(u_vec, v_vec, w_vec)
    
    # Partition the index space:
    # i: {0,1} (Strassen pair), {2} (singleton)
    # j: {0,1} (pair), {2,3} (pair), {4} (singleton)
    # k: {0,1} (pair), {2,3} (pair), {4,5} (pair), {6} (singleton)
    
    # Strassen blocks (2x2x2):
    # i_pair x j_pair x k_pair combinations:
    # (0,1) x (0,1) x (0,1)
    # (0,1) x (0,1) x (2,3)
    # (0,1) x (0,1) x (4,5)
    # (0,1) x (2,3) x (0,1)
    # (0,1) x (2,3) x (2,3)
    # (0,1) x (2,3) x (4,5)
    
    strassen_blocks = [
        (0, 0, 0),   # i={0,1}, j={0,1}, k={0,1}
        (0, 0, 2),   # i={0,1}, j={0,1}, k={2,3}
        (0, 0, 4),   # i={0,1}, j={0,1}, k={4,5}
        (0, 2, 0),   # i={0,1}, j={2,3}, k={0,1}
        (0, 2, 2),   # i={0,1}, j={2,3}, k={2,3}
        (0, 2, 4),   # i={0,1}, j={2,3}, k={4,5}
    ]
    
    for (i_s, j_s, k_s) in strassen_blocks:
        if elapsed() > deadline - 20:
            break
        add_strassen_block(i_s, j_s, k_s)
    
    # Schoolbook terms for remaining (i,j,k) combinations:
    # i=2 (singleton) with all j, k:
    # j=4 (singleton) with i in {0,1}, all k
    # k=6 (singleton) with all i, j
    
    # Let's enumerate all (i,j,k) not covered by Strassen blocks:
    covered = set()
    for (i_s, j_s, k_s) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i_s+di, j_s+dj, k_s+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if elapsed() > deadline - 20:
                        break
                    add_schoolbook_term(i, j, k)
            if elapsed() > deadline - 20:
                break
        if elapsed() > deadline - 20:
            break
    
    # Convert to numpy arrays
    U = np.array(U_cols, dtype=float).T  # (n*m, rank)
    V = np.array(V_cols, dtype=float).T  # (m*p, rank)
    W = np.array(W_cols, dtype=float).T  # (p*n, rank)
    
    rank = U.shape[1]
    
    # Verify the full decomposition
    def verify_full(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_full(U, V, W):
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_schoolbook_term(i, j, k)
        U = np.array(U_cols, dtype=float).T
        V = np.array(V_cols, dtype=float).T
        W = np.array(W_cols, dtype=float).T
    
    # Try to reduce rank via linear dependency removal
    if elapsed() < deadline - 30:
        U, V, W = reduce_rank(U, V, W, n, m, p, deadline)
    
    # Convert to lists of integer/half-integer values
    U_list = [[_to_half_int(x) for x in row] for row in U]
    V_list = [[_to_half_int(x) for x in row] for row in V]
    W_list = [[_to_half_int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _to_half_int(x):
    """Convert to integer or half-integer (multiple of 0.5)."""
    # Round to nearest half-integer
    hx = round(x * 2) / 2.0
    if abs(hx - round(hx)) < 1e-9:
        return int(round(hx))
    return hx

def reduce_rank(U, V, W, n, m, p, deadline):
    """Try to remove linearly dependent columns to reduce rank."""
    rank = U.shape[1]
    if rank <= 7:
        return U, V, W
    
    # Try greedy column removal: check if removing column c still gives valid decomposition
    # This is expensive, so we limit the search
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    
    current_rank = U_work.shape[1]
    removed = [False] * current_rank
    
    # Try removing columns one at a time
    max_attempts = 50
    attempts = 0
    
    for _ in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        found_removal = False
        for c in range(current_rank):
            if removed[c]:
                continue
            
            # Try removing column c
            mask = [i for i in range(current_rank) if not removed[i] and i != c]
            if not mask:
                continue
            
            U_test = U_work[:, mask]
            V_test = V_work[:, mask]
            W_test = W_work[:, mask]
            
            # Quick check: verify a subset of (i,j,k)
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :])
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Remove column c
                removed[c] = True
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if any(removed):
        keep = [i for i in range(current_rank) if not removed[i]]
        U = U_work[:, keep]
        V = V_work[:, keep]
        W = W_work[:, keep]
    
    return U, V, W

# EVOLVE-BLOCK-END
