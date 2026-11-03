import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140.0  # 10s safety margin
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    # Build the initial rank-72 solution (same as parent)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u.copy())
        V_cols.append(v.copy())
        W_cols.append(w.copy())

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M2 = (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai+1)] = 1
        w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)

        # M3 = a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M4 = d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)

        # M5 = (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = -1
        w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)

        # M6 = (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M7 = (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)

    # Build initial solution
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)
    
    initial_rank = len(U_cols)
    
    # Build the multiplication tensor to verify
    # T[i,j,k,l] = delta[i,k] * delta[j,l] for (i,j) in A, (j,k) in B, (k,i) in C
    # Actually, the matrix multiplication tensor T satisfies:
    # T[(i,j),(j,k),(k,i)] = 1
    # In our indexing:
    # U has shape (n*m, rank) - index (i,j)
    # V has shape (m*p, rank) - index (j,k)  
    # W has shape (p*n, rank) - index (k,i)
    # The decomposition says: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = delta[j,k] if we're multiplying A[i,j] * B[j,k] = C[i,k]
    # Wait, let me reconsider. For A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta[j,j] which is always 1... 
    # No, the tensor decomposition is: T[i,j,k] such that C[i,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Actually the standard form: the multiplication tensor has entries T[(i,j),(j,k),(k,i)] = 1
    # where (i,j) indexes A, (j,k) indexes B, (k,i) indexes C
    # And we need: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if we think of it as A[i,j]*B[j,k] contributing to C[i,k]
    
    # Let's verify the decomposition by checking all (i,j,k) triples
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that U*V*W decomposes the matrix multiplication tensor."""
        # U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
        # For each (i,j,k): check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # if j is in range [0,m), else 0... 
        # Actually for matrix multiplication A(nxm) * B(mx p) = C(n x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k
        # In our flat indexing: U index = i*m+j, V index = j*p+k, W index = k*n+i
        # We need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Compute the contraction
                    val = np.sum(U_mat[u_idx, :] * V_mat[v_idx, :] * W_mat[w_idx, :])
                    if abs(val - 1) > 1e-10:
                        return False
        return True

    # Convert to matrices
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    best_rank = initial_rank
    best_U = U_mat.copy()
    best_V = V_mat.copy()
    best_W = W_mat.copy()
    
    # Try to reduce rank by removing linearly dependent terms
    # and searching for better decompositions
    rng = random.Random(42)
    
    def try_remove_term(U_mat, V_mat, W_mat, term_idx):
        """Try to remove term term_idx by redistributing its contribution."""
        rank = U_mat.shape[1]
        if rank <= 1:
            return None, None, None, False
        
        # Remove the term
        mask = np.ones(rank, dtype=bool)
        mask[term_idx] = False
        U_new = U_mat[:, mask]
        V_new = V_mat[:, mask]
        W_new = W_mat[:, mask]
        
        # Check if the remaining terms still span the tensor
        if not verify_decomposition(U_new, V_new, W_new):
            return None, None, None, False
        
        return U_new, V_new, W_new, True
    
    def try_merge_terms(U_mat, V_mat, W_mat, idx1, idx2):
        """Try to merge two terms by finding if their combined effect can be represented with one term."""
        rank = U_mat.shape[1]
        if rank <= 2:
            return None, None, None, False
        
        # The idea: if term1 and term2 can be combined into a single term
        # We need to find u, v, w such that u*v*w = u1*v1*w1 + u2*v2*w2
        # This is generally not possible, but we can try
        # Remove both terms
        mask = np.ones(rank, dtype=bool)
        mask[idx1] = False
        mask[idx2] = False
        U_new = U_mat[:, mask]
        V_new = V_mat[:, mask]
        W_new = W_mat[:, mask]
        
        # Check if remaining terms still work (they won't, since we removed two terms)
        # Instead, we need to see if the sum of the two terms can be absorbed
        # This is complex, so we'll use a different approach
        
        return None, None, None, False
    
    # Strategy: try removing individual terms
    # First, try simple removal
    U_cur = best_U.copy()
    V_cur = best_V.copy()
    W_cur = best_W.copy()
    
    # Try removing each term one at a time
    removed = []
    rank = U_cur.shape[1]
    for idx in range(rank):
        if time.time() > deadline - 5:
            break
        if idx in removed:
            continue
        U_new, V_new, W_new, ok = try_remove_term(U_cur, V_cur, W_cur, idx)
        if ok:
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            removed.append(idx)
            best_rank = U_cur.shape[1]
            best_U = U_cur.copy()
            best_V = V_cur.copy()
            best_W = W_cur.copy()
    
    # Try a more aggressive approach: random restarts with local search
    # Build the tensor T as a 3D array
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. Let me reconsider.
    
    # For matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T is defined such that the bilinear map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i, j, k] where i indexes rows of A and C, j indexes cols of A and rows of B, k indexes cols of B and C
    # The tensor entry T[i, j, k] = 1 (it's the identity in the middle index)
    # The decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r has entries u_r[i, j], v_r has entries v_r[j, k], w_r has entries w_r[k, i]
    # And we need: for all i, j, k: sum_r u_r[i, j] * v_r[j, k] * w_r[k, i] = 1
    
    # So the tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is just the all-ones tensor! No wait, that's not right either.
    
    # Let me think again. The matrix multiplication tensor is:
    # μ = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the map is (A, B) -> C
    # Actually, the standard tensor for matrix multiplication <n,m,p> has:
    # T[(i,j),(j,k),(k,i)] = 1
    # In flat form with our indexing:
    # For U: index i*m + j (i in [0,n), j in [0,m))
    # For V: index j*p + k (j in [0,m), k in [0,p))
    # For W: index k*n + i (k in [0,p), i in [0,n))
    # The tensor value at (i,j,k) is 1
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is correct. The tensor is the all-ones tensor in the sense that T[i,j,k] = 1 for all valid indices.
    
    # Let's do a local search: try to find a lower rank decomposition
    # by perturbing the current solution
    
    def local_search(U_cur, V_cur, W_cur, max_iter=1000):
        """Try to reduce rank via local search."""
        rank = U_cur.shape[1]
        best = (rank, U_cur.copy(), V_cur.copy(), W_cur.copy())
        
        for iter in range(max_iter):
            if time.time() > deadline - 5:
                break
            if rank <= 1:
                break
            
            # Try removing a random term
            idx = rng.randint(0, rank - 1)
            U_new, V_new, W_new, ok = try_remove_term(U_cur, V_cur, W_cur, idx)
            if ok:
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                rank = U_cur.shape[1]
                best = (rank, U_cur.copy(), V_cur.copy(), W_cur.copy())
                continue
            
            # Try swapping: replace one term with a perturbed version
            # This is complex, so we'll just try removals for now
            pass
        
        return best
    
    # Run local search
    best_rank, best_U, best_V, best_W = local_search(U_cur, V_cur, W_cur, max_iter=500)
    
    # Try a few random restarts
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        # Start from the initial solution and do a different random search
        U_rst = best_U.copy()
        V_rst = best_V.copy()
        W_rst = best_W.copy()
        
        # Try removing terms in random order
        order = list(range(U_rst.shape[1]))
        rng.shuffle(order)
        removed = set()
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if idx in removed:
                continue
            U_new, V_new, W_new, ok = try_remove_term(U_rst, V_rst, W_rst, idx)
            if ok:
                U_rst = U_new
                V_rst = V_new
                W_rst = W_new
                removed.add(idx)
        
        if U_rst.shape[1] < best_rank:
            best_rank = U_rst.shape[1]
            best_U = U_rst.copy()
            best_V = V_rst.copy()
            best_W = W_rst.copy()
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # Fall back to the known valid solution
        U_mat = np.array(U_cols).T
        V_mat = np.array(V_cols).T
        W_mat = np.array(W_cols).T
    else:
        U_mat = best_U
        V_mat = best_V
        W_mat = best_W
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
# EVOLVE-BLOCK-END
