"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that U, V, W give an exact tensor decomposition."""
    r = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[i,j,k] should equal sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But actually the tensor for matrix mult (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has T[i,j,k] = 1 if this contributes to (i,k) output
    # More precisely: T is a 3-index tensor where T[i,j,k] = delta(i,k) for the (i,k) output
    # Wait, let me reconsider.
    
    # For C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the identity in the j dimension for each (i,k))
    # Actually T[i,j,k] = delta_{i,k} doesn't make sense...
    # T[i,j,k] should be such that sum_j A[i,j] B[j,k] = T[i,j,k] contracted
    # The multiplication tensor: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1
    # for each (i,j,k) triple. Wait no.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T satisfies: T[i,j,k] = 1 for all i,j,k? No...
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # That means T[i,j,k] = 1 for all i, j, k.
    # 
    # Wait, that's not right either. Let me think again.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form: for fixed i,k, it's linear in A and B.
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # Actually no. The tensor decomposition is:
    # T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 
    # 
    # Let me use the standard formulation. The matrix multiplication tensor for 
    # multiplying (n×m) by (m×p) to get (n×p) is:
    # T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # T[a,b,c] where a indexes (i,j) of A, b indexes (j',k) of B, c indexes (i',k') of C
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k), else 0
    # i.e., T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k', else 0
    # 
    # So T is actually a very sparse tensor! For each (i,j,k), there's exactly one nonzero:
    # T[(i,j),(j,k),(i,k)] = 1
    # 
    # The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is n×m, V_r is m×p, W_r is n×p
    # T[(i,j),(j',k),(i',k')] = sum_r U_r[i,j] * V_r[j',k] * W_r[i',k']
    # 
    # So we need: for all i,j,j',k,k':
    # sum_r U_r[i,j] V_r[j',k] W_r[i',k'] = 1 if (i,j,j',k,k') = (i,j,j,k,i,k) i.e. j=j', i=i', k=k'
    # = 0 otherwise
    # 
    # In matrix form with the given interface:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # 
    # The tensor element T[(i,j),(j',k),(i',k')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    # This should equal 1 if i=i', j=j', k=k', else 0.
    
    # Let's verify by checking all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            # Check T[(i,j),(jp,k),(kp,ip)]
                            val = 0.0
                            for r in range(r):
                                val += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            if abs(val - expected) > tol:
                                return False
    return True

def _verify_fast(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Faster verification using numpy."""
    r = len(U[0]) if U else 0
    U_mat = np.array(U)  # (n*m, r)
    V_mat = np.array(V)  # (m*p, r)
    W_mat = np.array(W)  # (p*n, r)
    
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_3d = U_mat.reshape(n, m, r)
    V_3d = V_mat.reshape(m, p, r)
    W_3d = W_mat.reshape(p, n, r)
    
    # T[(i,j),(jp,k),(kp,ip)] = sum_r U_3d[i,j,r] * V_3d[jp,k,r] * W_3d[kp,ip,r]
    # We need this to equal 1 if i=ip, j=jp, k=kp, else 0
    
    # Let's compute the full tensor and check
    # Actually, let's check the condition more efficiently.
    # For the decomposition to be valid:
    # sum_r U[i,j,r] V[j',k,r] W[k',i',r] = delta(i,i') delta(j,j') delta(k,k')
    
    # This means: for each r, the outer product structure must work out.
    # Let's just compute and check.
    
    # Build T as a 6D array: T[i,j,jp,k,kp,ip]
    # T[i,j,jp,k,kp,ip] = sum_r U_3d[i,j,r] * V_3d[jp,k,r] * W_3d[kp,ip,r]
    
    # Use einsum: T[i,j,jp,k,kp,ip] = U[i,j,r] * V[jp,k,r] * W[kp,ip,r]
    T = np.einsum('ijr,jkr,pir->ijjkip', U_3d, V_3d, W_3d)
    
    # Expected: T[i,j,jp,k,kp,ip] = 1 if i==ip and j==jp and k==kp, else 0
    # Let's build the expected tensor
    expected = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, j, k, k, i] = 1.0
    
    return np.allclose(T, expected, atol=tol, rtol=0)

def _construct_schoolbook(n=3, m=3, p=5):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def _try_random_rank(rank, n=3, m=3, p=5, max_iters=200, tol=1e-9, rng=None):
    """Try to find a decomposition of given rank using random init + LSQ."""
    if rng is None:
        rng = random.Random()
    
    U_mat = np.random.randn(n * m, rank)
    V_mat = np.random.randn(m * p, rank)
    W_mat = np.random.randn(p * n, rank)
    
    U_3d = U_mat.reshape(n, m, rank)
    V_3d = V_mat.reshape(m, p, rank)
    W_3d = W_mat.reshape(p, n, rank)
    
    for it in range(max_iters):
        # Fix U and V, solve for W
        # For each (kp, ip), we need: sum_r V[jp,k,r] * W[kp,ip,r] * U[i,j,r] = delta...
        # This is complex. Let's use a different approach.
        
        # Alternative: use the fact that the tensor must satisfy specific constraints.
        # Let's just check if current solution is valid
        T = np.einsum('ijr,jkr,pir->ijjkip', U_3d, V_3d, W_3d)
        expected = np.zeros((n, m, m, p, p, n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i, j, j, k, k, i] = 1.0
        
        err = np.linalg.norm(T - expected)
        if err < tol:
            return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
        
        # Update W: for each (kp, ip), minimize ||sum_r U[i,j,r] V[jp,k,r] W[kp,ip,r] - target||
        # This is a linear system in W[kp,ip,:]
        # For fixed kp, ip, the contribution to T[i,j,jp,k,kp,ip] is:
        # sum_r U[i,j,r] * V[jp,k,r] * W[kp,ip,r]
        # Let A[i,j,jp,k] = U[i,j,:] * V[jp,k,:] (element-wise product, shape rank)
        # Then T[i,j,jp,k,kp,ip] = dot(A[i,j,jp,k], W[kp,ip,:])
        
        # For each (kp, ip), we have a linear system:
        # sum over (i,j,jp,k) of [target - current_other] * A[i,j,jp,k] * W[kp,ip,r] = 0
        # Actually, let's set up: for each (kp,ip), we want to find W[kp,ip,:] such that
        # for all (i,j,jp,k): dot(U[i,j,:]*V[jp,k,:], W[kp,ip,:]) = target[i,j,jp,k,kp,ip]
        # where target[i,j,jp,k,kp,ip] = 1 if i==ip and j==jp and k==kp, else 0
        
        # This is overdetermined in general. Use least squares.
        # Build the system: for each (kp, ip):
        # M_{(i,j,jp,k), r} = U[i,j,r] * V[jp,k,r]
        # b_{(i,j,jp,k)} = 1 if i==ip and j==jp and k==kp, else 0
        # Solve: M @ W[kp,ip,:] = b
        
        for kp in range(p):
            for ip in range(n):
                # Build M and b
                rows = []
                b_vec = []
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                row = U_mat[i*m+j] * V_mat[jp*p+k]
                                rows.append(row)
                                b_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec.append(b_val)
                M = np.array(rows)
                b = np.array(b_vec)
                # Least squares
                W_kp_ip, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                W_mat[kp*n+ip] = W_kp_ip
        
        # Update V similarly
        for jp in range(m):
            for k in range(p):
                rows = []
                b_vec = []
                for i in range(n):
                    for j in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                row = U_mat[i*m+j] * W_mat[kp*n+ip]
                                rows.append(row)
                                b_val = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                b_vec.append(b_val)
                M = np.array(rows)
                b = np.array(b_vec)
                V_jp_k, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                V_mat[jp*p+k] = V_jp_k
        
        # Update U similarly
        for i in range(n):
            for j in range(m):
                rows = []
                b_vec = []
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                row = V_mat[jp*p+k] * W_mat[kp*n+ip]
                                rows.append(row)
                                b_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec.append(b_val)
                M = np.array(rows)
                b = np.array(b_vec)
                U_i_j, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                U_mat[i*m+j] = U_i_j
        
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
    
    # Final check
    T = np.einsum('ijr,jkr,pir->ijjkip', U_3d, V_3d, W_3d)
    expected = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, j, k, k, i] = 1.0
    if np.linalg.norm(T - expected) < tol:
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    return None

def _try_structured_rank(rank, n=3, m=3, p=5, max_iters=100, tol=1e-9, rng=None):
    """Try structured initialization based on known algorithms."""
    if rng is None:
        rng = random.Random()
    
    # Start from schoolbook and try to reduce rank by merging
    # Or use a parametric family
    
    # For now, try random with different seeds
    for seed in range(max_iters):
        rng2 = random.Random(seed)
        result = _try_random_rank(rank, n, m, p, max_iters=50, tol=tol, rng=rng2)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Known lower bounds: for 3x3 x 3x5, the rank is at least...
    # Let's try ranks from a reasonable lower bound up
    
    # The rank of matrix multiplication (n,m,p) is at least max(n*m, m*p, n*p) / something
    # For our case: at least 15 (since output is 15 elements)
    # Actually, the rank must be at least the number of independent outputs = n*p = 15
    
    # Try ranks from 15 to best_rank-1
    for rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_random_rank(rank, n, m, p, max_iters=30, tol=1e-9)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Try even lower
            continue
    
    return best
# EVOLVE-BLOCK-END
