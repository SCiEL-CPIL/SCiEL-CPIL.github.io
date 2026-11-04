"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def check_validity_fast(U, V, W, tol=1e-8):
        """Fast validity check using vectorized operations."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: for all i in [0,n), k in [0,p):
        # (U[i*m:(i+1)*m, :] * W[k*n+i, :][None, :]) @ V_block.T == I_m
        # where V_block is formed by rows V[j*p+k, :] for j in [0,m)
        for i in range(n):
            for k in range(p):
                A_ik = U[i*m:(i+1)*m, :]  # (m, rank)
                w_ik = W[k*n+i, :]  # (rank,)
                # Construct B_ik.T which is (rank, m)
                # B_ik[j, :] = V[j*p + k, :]
                # So B_ik.T[:, j] = V[j*p + k, :]
                # We can compute (A_ik * w_ik[None, :]) @ B_ik.T
                # Let M = A_ik * w_ik[None, :]  (m, rank)
                # Result = M @ B_ik.T
                # Result[a, b] = sum_c M[a, c] * B_ik[b, c]
                #             = sum_c U[i*m+a, c] * W[k*n+i, c] * V[b*p+k, c]
                # We need Result == I_m, i.e., Result[a, b] = delta_ab
                
                M = A_ik * w_ik[None, :]
                # Efficiently compute M @ B_ik.T
                # B_ik.T is (rank, m). Col j is V[j*p+k, :].T
                # We can form B_ik.T as a matrix
                # Indices for V: [0*p+k, 1*p+k, ..., (m-1)*p+k]
                v_indices = [j*p + k for j in range(m)]
                B_ik_T = V[v_indices, :].T  # (rank, m)
                
                result = M @ B_ik_T  # (m, m)
                if np.max(np.abs(result - np.eye(m))) > tol:
                    return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_2x2():
        """
        Rank-7 tensor decomposition for 2x2 matrix multiplication.
        Returns U, V, W of shape (4, 7).
        Indices: U/V/W row = i*2+j (0:00, 1:01, 2:10, 3:11)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+g)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def strassen_4x4_4x9():
        """
        Use Strassen-like block decomposition for 4x4 * 4x9.
        Split B into B_left (4x4) and B_right (4x5).
        C_left = A * B_left: Strassen 4x4*4x4, rank 49
        C_right = A * B_right: schoolbook 4x4*4x5, rank 80
        Total: 129
        """
        U2, V2, W2 = strassen_2x2()
        
        U_left = np.zeros((16, 49))
        V_left = np.zeros((16, 49))
        W_left = np.zeros((16, 49))
        
        def block_indices(bi, bj, dim=4):
            return [(bi*2+lr)*dim + (bj*2+lc) for lr in range(2) for lc in range(2)]
        
        A_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,1)],
            2: [(0,0,1)],
            3: [(1,1,1)],
            4: [(0,0,1), (0,1,1)],
            5: [(1,0,1), (0,0,-1)],
            6: [(0,1,1), (1,1,-1)]
        }
        B_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(0,0,1)],
            2: [(0,1,1), (1,1,-1)],
            3: [(1,0,1), (0,0,-1)],
            4: [(1,1,1)],
            5: [(0,0,1), (0,1,1)],
            6: [(1,0,1), (1,1,1)]
        }
        C_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,-1)],
            2: [(0,1,1), (1,1,1)],
            3: [(0,0,1), (1,0,1)],
            4: [(0,0,-1), (0,1,1)],
            5: [(1,1,-1)],
            6: [(0,0,1)]
        }
        
        for k in range(7):
            for s in range(7):
                col = k * 7 + s
                for (bi, bj, coeff) in A_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if U2[l, s] != 0:
                            U_left[idxs[l], col] += coeff * U2[l, s]
                for (bk, bj, coeff) in B_blocks[k]:
                    idxs = block_indices(bk, bj, dim=4)
                    for l in range(4):
                        if V2[l, s] != 0:
                            V_left[idxs[l], col] += coeff * V2[l, s]
                for (bi, bj, coeff) in C_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if W2[l, s] != 0:
                            W_left[idxs[l], col] += coeff * W2[l, s]
        
        # Schoolbook for right part: 4x4 * 4x5, rank 80
        U_right = np.zeros((16, 80))
        V_right = np.zeros((20, 80))
        W_right = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_right[i*4+j, col] = 1
                    V_right[j*5+k, col] = 1
                    W_right[k*4+i, col] = 1
                    col += 1
        
        rank_total = 49 + 80
        U_full = np.zeros((16, rank_total))
        V_full = np.zeros((36, rank_total))
        W_full = np.zeros((36, rank_total))
        
        U_full[:, :49] = U_left
        for j in range(4):
            for k in range(4):
                V_full[j*9+k, :49] = V_left[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, :49] = W_left[k*4+i, :]
        
        U_full[:, 49:] = U_right
        for j in range(4):
            for k in range(5):
                V_full[j*9+4+k, 49:] = V_right[j*5+k, :]
        for k in range(5):
            for i in range(4):
                W_full[k*4+i, 49:] = W_right[k*4+i, :]
        
        assert check_validity_fast(U_full, V_full, W_full), "Combined 4x4*4x9 failed"
        return U_full, V_full, W_full, rank_total

    def try_random_rank(r, num_restarts=10):
        """
        Try to find a valid decomposition with rank r using random U,V and solving for W.
        Uses sparse linear algebra for efficiency.
        """
        if r < 36:
            return None
            
        # Precompute tensor structure
        # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0
        # We can precompute for each w, the list of (u, v, target)
        # But actually, the system for W[w, :] is:
        # sum_c (U[u,c] V[v,c]) W[w,c] = T[u,v,w]
        # This is a linear system A_w @ W[w,:] = b_w
        # where A_w is a matrix with rows indexed by (u,v) and cols by c.
        # A_w[(u,v), c] = U[u,c] * V[v,c]
        # b_w[(u,v)] = T[u,v,w]
        
        # Note that A_w is the Khatri-Rao product of U and V (column-wise Kronecker).
        # Actually, it's the matrix where row (u,v) is U[u,:] * V[v,:] (element-wise product).
        # This is the same for all w! Let's call it M.
        # M is of size (n*m * m*p) x r = (16 * 36) x r = 576 x r.
        # M[(u*p*m + v), c] = U[u,c] * V[v,c]  -- wait, indexing.
        # Let's flatten (u,v) to index idx = u * (m*p) + v.
        # M[idx, c] = U[u, c] * V[v, c]
        
        # So we compute M once for given U, V.
        # Then for each w, we solve M @ W[w,:] = b_w.
        # b_w is a vector of size 576.
        # b_w[idx] = 1 if (u,v,w) is a valid tensor entry, else 0.
        
        # Precompute b_w for all w
        b_all = np.zeros((p*n, n*m*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    idx = u * (m*p) + v
                    b_all[w, idx] = 1.0

        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(restart * 7919 + r * 131 + int(time.time()*1000))
            
            # Generate sparse random U and V
            # Density needs to be high enough for M to have full column rank r
            # M has rank <= min(rank(U), rank(V)) * something? No.
            # M is the Khatri-Rao product. Rank(M) <= min(rank(U), rank(V)) is false.
            # Rank of Khatri-Rao product is generally min(rank(U), rank(V)) if generic?
            # Actually, for random matrices, M will have full column rank r with high probability
            # if U and V have sufficient rank.
            # Let's use density 0.5 to be safe.
            
            density = 0.5
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            
            # Random sparse entries
            u_indices = np.random.choice(n*m*r, int(n*m*r*density), replace=False)
            U_flat = U.flatten()
            U_flat[u_indices] = np.random.choice([-1, 1], size=len(u_indices))
            
            v_indices = np.random.choice(m*p*r, int(m*p*r*density), replace=False)
            V_flat = V.flatten()
            V_flat[v_indices] = np.random.choice([-1, 1], size=len(v_indices))
            
            # Build M
            # M[idx, c] = U[u,c] * V[v,c]
            # idx = u * (m*p) + v
            # We can compute this efficiently
            # M = np.zeros((n*m*m*p, r))
            # for u in range(n*m):
            #   for v in range(m*p):
            #     M[u*(m*p)+v, :] = U[u,:] * V[v,:]
            
            # Vectorized:
            # U_expanded = U[:, None, :]  # (n*m, 1, r)
            # V_expanded = V[None, :, :]  # (1, m*p, r)
            # M = (U_expanded * V_expanded).reshape(n*m*m*p, r)
            
            M = (U[:, None, :] * V[None, :, :]).reshape(n*m*m*p, r)
            
            # Check if M has full column rank
            # If not, we cannot solve for W uniquely, and likely no solution exists
            # for arbitrary b_w.
            # We can try to solve M @ W[w,:] = b_w using least squares.
            # If residual is non-zero, it's invalid.
            
            # Solve for all w at once?
            # W is (p*n, r).
            # M @ W.T = b_all.T
            # This is a system of p*n equations? No.
            # For each w, M @ W[w,:] = b_all[w,:]
            # So M @ W.T = b_all.T is correct.
            # M is (576, r), W.T is (r, 36), b_all.T is (576, 36).
            
            try:
                # Use lstsq
                W_T, residuals, rank_M, sv = np.linalg.lstsq(M, b_all.T, rcond=None)
                W = W_T.T
                
                # Check residuals
                if len(residuals) > 0:
                    # residuals are sum of squares of residuals for each column?
                    # np.linalg.lstsq returns residuals only if M is full rank.
                    # If M is not full rank, residuals is empty.
                    # We need to check the actual residual norm.
                    pass
                
                # Compute actual residual
                # Res = M @ W.T - b_all.T
                # If max abs is > tol, invalid.
                # This is expensive (576*36*r ops).
                # But r is ~100, so 576*36*100 ~ 2M ops. Fast.
                
                Res = M @ W.T - b_all.T
                if np.max(np.abs(Res)) > 1e-8:
                    continue
                
                # If residual is small, check validity
                # Note: M @ W[w,:] = b_all[w,:] implies
                # sum_c U[u,c] V[v,c] W[w,c] = T[u,v,w] for all u,v,w.
                # This is exactly the tensor decomposition condition.
                # So if residual is 0, it's valid.
                
                # Double check with check_validity_fast just in case
                if check_validity_fast(U, V, W):
                    return U, V, W, r
                    
            except Exception:
                continue
        
        return None

    # Get schoolbook solution as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try Strassen-based decomposition (rank 129)
    if time.time() < deadline - 30:
        try:
            U_str, V_str, W_str, rank_str = strassen_4x4_4x9()
            if rank_str < best_rank:
                best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
        except Exception:
            pass
    
    # Try lower ranks via optimized random search
    # The linear algebra approach allows us to test many ranks quickly.
    # We search from 128 down to 100.
    # If we find a solution at r, we stop (since we want minimal).
    # But we might miss a solution at r-1 if we don't search enough.
    # So we search descending.
    
    search_ranks = list(range(128, 99, -1))
    for r in search_ranks:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        # Increase restarts for lower ranks as they are harder
        restarts = 5 if r > 110 else 10
        
        result = try_random_rank(r, num_restarts=restarts)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            # We could break here, but maybe we can find even lower?
            # The loop goes down, so we will try lower ranks next.
            # But if we find 120, we will try 119, 118...
            # This is good.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
