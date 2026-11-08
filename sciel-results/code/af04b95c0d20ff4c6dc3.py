import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off + 1 * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def solve_fixed_v_milp(R, timeout=25):
        """
        Fix V to canonical sparse form (each row has exactly one 1), then solve
        for U and W using MILP with half-integer constraints.
        
        V is fixed: V[b, assign[b]] = 1 for each row b.
        Then T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                     = U[a, assign[b]] * W[c, assign[b]]
        
        So for each b: T[:,b,:] = U[:,assign[b]] * W[:,assign[b]].T
        
        This means U[:,r] * W[:,r].T must equal T[:,b,:] for all b with assign[b]=r.
        If multiple b's map to the same r, they must all give the same outer product.
        
        We need to find U (U_dim x R) and W (W_dim x R) with half-integer entries
        such that for all a,b,c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        With V fixed (one-hot per row), this becomes:
        For each b: U[:,assign[b]] * W[:,assign[b]].T = T[:,b,:]
        
        Let's enumerate possible assignments and use MILP.
        """
        if time.time() > deadline - timeout:
            return None
        
        # For each row b of V, assign it to a column r in [0, R)
        # V[b, assign[b]] = 1, all other entries 0
        
        # The constraint is: for all a,c: T[a,b,c] = U[a, assign[b]] * W[c, assign[b]]
        # If multiple b's have the same assign value r, then T[:,b1,:] = T[:,b2,:] for those b's
        
        # Strategy: Try different assignment patterns
        # For small R, we can be systematic
        
        # First, check which (a,c) pairs have T[a,b,c]=1 for each b
        # T[a,b,c]=1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        
        # For a given b = j*p+k, the non-zero entries in T[:,b,:] are:
        # T[i*m+j, b, k*n+i] = 1 for i=0,1
        # So T[:,b,:] has non-zeros at positions (i*m+j, k*n+i) for i=0,1
        
        # If two rows b1, b2 are assigned to the same column r, then
        # T[:,b1,:] and T[:,b2,:] must have the same "structure" in terms of which
        # outer product they represent.
        
        # Let's try: assign each row b to a unique column (if R >= V_dim)
        # Then U[:,r] * W[:,r].T = T[:,b,:] where b is the row assigned to r
        
        # For R < V_dim, multiple rows share columns.
        
        # Let's try R from 45 down to 40
        
        # For each assignment, solve:
        # Variables: U[a,r] for a in range(U_dim), r in range(R)
        #            W[c,r] for c in range(W_dim), r in range(R)
        # Constraints: For each b: U[:,assign[b]] * W[:,assign[b]].T = T[:,b,:]
        
        # This is still bilinear. Let's use a different approach:
        # Fix the assignment, then for each column r, we need:
        # U[:,r] * W[:,r].T = M_r where M_r is determined by the rows assigned to r
        
        # If exactly one row b is assigned to r, then M_r = T[:,b,:]
        # If multiple rows b1,b2,... are assigned to r, we need T[:,b1,:]=T[:,b2,:]=...=M_r
        
        # Let's check if T[:,b1,:]=T[:,b2,:] is possible.
        # T[:,b,:] depends on b = j*p+k. The non-zero pattern is:
        # Row indices: i*m+j for i=0,1 -> j, m+j
        # Col indices: k*n+i for i=0,1 -> k, n+k
        
        # So T[:,b,:] is a 10x10 matrix with 1s at (j,k), (j,n+k), (m+j,k), (m+j,n+k)
        # Wait, let me recompute. b = j*p+k.
        # T[a,b,c] = 1 when a = i*m+j, b=j*p+k, c=k*n+i
        # So for fixed b=j*p+k:
        #   i=0: a=j, c=k
        #   i=1: a=m+j, c=n+k
        # So T[:,b,:] has 1s at (j,k) and (m+j,n+k), all other entries 0.
        
        # Two rows b1=j1*p+k1 and b2=j2*p+k2 give the same T[:,b,:] iff:
        # {j1, m+j1} = {j2, m+j2} AND {k1, n+k1} = {k2, n+k2}
        # Since 0<=j<5 and 0<=k<5, and n=2:
        # {j, j+5} determines j uniquely (since j in [0,4], j+5 in [5,9])
        # {k, k+2} determines k: k=0->{0,2}, k=1->{1,3}, k=2->{2,4}, k=3->{3,5}, k=4->{4,6}
        # So {k,k+2} is unique for each k in [0,4].
        
        # Therefore T[:,b1,:]=T[:,b2,:] iff j1=j2 AND k1=k2, i.e., b1=b2.
        
        # This means each column r can be assigned to at most ONE row b!
        # So we need R >= V_dim = 25.
        
        # But we want R < 46. Since V_dim=25, we need R >= 25.
        # Let's try R=25 first: each row gets its own column.
        
        # For R=25: assign[b] = b for all b.
        # Then U[:,b] * W[:,b].T = T[:,b,:] for each b.
        # T[:,b,:] has 1s at (j,k) and (m+j,n+k) where b=j*p+k.
        # So U[:,b] * W[:,b].T must have 1s at those positions and 0 elsewhere.
        # This means U[:,b] and W[:,b] must be such that their outer product is T[:,b,:].
        # T[:,b,:] = e_j * e_k^T + e_{m+j} * e_{n+k}^T
        # This is a rank-2 matrix, so it cannot be written as a single outer product u * w^T
        # unless one of the vectors is zero, which it's not.
        
        # So R=25 doesn't work with this simple assignment.
        
        # We need a different approach. Let's allow V to have more flexibility.
        
        # Alternative: Allow V[b,r] to be in {-1, 0, 1} and solve the full system.
        
        # Let's use a greedy/iterative approach:
        # 1. Start with a known rank-46 solution
        # 2. Try to eliminate columns by finding dependencies
        
        # Or: use the fact that we need half-integer solutions.
        
        # Let's try a direct approach for R in [40, 45]:
        # Randomly initialize V with sparse patterns, then use ALS to find U, W,
        # then snap to half-integers.
        
        best = None
        
        for attempt in range(3):
            if time.time() > deadline - timeout:
                break
            
            np.random.seed(123 + attempt * 7)
            
            # Create V with sparse structure: each column has 2-3 non-zeros
            V = np.zeros((V_dim, R))
            # Each row should have at least one non-zero
            for r in range(R):
                # Pick 2-3 rows for this column
                num_nz = np.random.choice([2, 3], p=[0.7, 0.3])
                rows = np.random.choice(V_dim, size=num_nz, replace=False)
                for row in rows:
                    V[row, r] = np.random.choice([-1.0, 1.0])
            
            # Ensure each row has at least one non-zero
            for b in range(V_dim):
                if V[b, :].sum() == 0:
                    V[b, 0] = 1.0
            
            # ALS
            U = np.random.randn(U_dim, R) * 0.1
            W = np.random.randn(W_dim, R) * 0.1
            
            for iteration in range(200):
                if time.time() > deadline - timeout:
                    break
                
                # Solve for U given W
                U_new = np.zeros((U_dim, R))
                for a in range(U_dim):
                    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # For fixed a: T[a,:,:] = U[a,:] @ (V.T * W) element-wise
                    # M_a[b*W_dim+c, r] = V[b,r] * W[c,r]
                    M_a = np.zeros((V_dim * W_dim, R))
                    for b in range(V_dim):
                        for c in range(W_dim):
                            M_a[b * W_dim + c, :] = V[b, :] * W[c, :]
                    try:
                        U_new[a, :] = np.linalg.lstsq(M_a, T[a, :, :].flatten(), rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                
                # Solve for W given U
                W_new = np.zeros((W_dim, R))
                for c in range(W_dim):
                    M_c = np.zeros((U_dim * V_dim, R))
                    for a in range(U_dim):
                        for b in range(V_dim):
                            M_c[a * V_dim + b, :] = U[a, :] * V[b, :]
                    try:
                        W_new[c, :] = np.linalg.lstsq(M_c, T[:, :, c].flatten(), rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                
                diff_U = np.linalg.norm(U_new - U)
                diff_W = np.linalg.norm(W_new - W)
                U = U_new
                W = W_new
                
                if diff_U + diff_W < 1e-9:
                    break
            
            # Check and snap
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            loss = np.sum((prod - T) ** 2)
            
            if loss < 1e-4:
                valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
                if valid:
                    best = (U_s, V_s, W_s)
                    break
        
        return best

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    
    # Try to find lower rank
    for target_rank in [45, 44, 43, 42]:
        if time.time() > deadline - 20:
            break
        
        result = solve_fixed_v_milp(target_rank, timeout=20)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
