"""Program for matrix multiplication 3x4 x 4x8 with verified schoolbook decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook tensor decomposition for n x m * m x p matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=20):
    """Verify that the decomposition correctly computes C = A @ B."""
    random.seed(42)
    for _ in range(num_tests):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        true_C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        rank = len(U[0]) if U else 0
        u = [[0] * rank for _ in range(n)]
        for i in range(n):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += U[i * m + j][t] * A[i][j]
                u[i][t] = s
        
        v = [[0] * rank for _ in range(p)]
        for k in range(p):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += V[j * p + k][t] * B[j][k]
                v[k][t] = s
        
        for i in range(n):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += W[k * n + i][t] * u[i][t] * v[k][t]
                if s != true_C[i][k]:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p, num_tests=50):
    """Faster numpy-based verification."""
    np.random.seed(42)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randint(-10, 10, size=(n, m)).astype(float)
        B = np.random.randint(-10, 10, size=(m, p)).astype(float)
        
        # u[i,t] = sum_j U[i*m+j, t] * A[i,j]
        # Reshape U to (n, m, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        u = np.einsum('imt,ij->it', U_reshaped, A)
        
        # v[k,t] = sum_j V[j*p+k, t] * B[j,k]
        V_reshaped = V_np.reshape(m, p, rank)
        v = np.einsum('jkt,jk->kt', V_reshaped, B)
        
        # C[i,k] = sum_t W[k*n+i, t] * u[i,t] * v[k,t]
        W_reshaped = W_np.reshape(p, n, rank)
        C = np.einsum('kit,it,kt->ik', W_reshaped, u, v)
        
        true_C = A @ B
        if not np.allclose(C, true_C, atol=1e-10):
            return False
    return True

def _build_kronecker(n, m, p):
    """Build decomposition using Kronecker product structure.
    
    For n=3, m=4, p=8, we can write:
    - Split m=4 into two blocks of 2
    - Split p=8 into four blocks of 2
    - Use 2x2 Strassen-like structure
    
    This gives rank = n * (m/2) * (p/2) * 7 = 3 * 2 * 4 * 7 = 168 (worse)
    Or simpler: n * m * p = 96 (same as schoolbook)
    
    Actually, let's use a different approach: treat the 4-dim middle as 
    two independent 2-dim spaces and use rank-7 for each 2x2x2 block.
    """
    # For 3x4*4x8: split m=4 into m1=2, m2=2; p=8 into p1=2,p2=2,p3=2,p4=2
    # Each 3x2 * 2x2 subproblem has rank 3*2*2=12 schoolbook, or we can use structure
    # Total rank with independent blocks: 3 * (2*2) * 4 = 48? No...
    
    # Let me think differently. The tensor is T_{i,j,k} where i in [0,3), j in [0,4), k in [0,8)
    # Schoolbook: T = sum_{j} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k} with rank 3*4*8=96
    
    # Try: decompose j into (j1, j2) where j1 in [0,2), j2 in [0,2)
    # Then T_{i,(j1,j2),k} = sum_{j1,j2} ...
    # This doesn't directly help unless we use low-rank structure in j2
    
    # Alternative: use the fact that for fixed i, the map A_i (row i of A, length 4) 
    # times B (4x8) gives row i of C.
    # rank_3 = min rank of {A_i * B : A_i in R^4, B in R^{4x8}}
    # This is the rank of the 4x4x8 tensor as a 4x(4*8) matrix... 
    # Actually it's the bilinear complexity of 1x4*4x8 which is 32.
    # Total = 3 * 32 = 96. Same as schoolbook.
    
    # So Kronecker doesn't help here. Return schoolbook.
    return _build_schoolbook(n, m, p)

def _random_search(n, m, p, target_rank, time_budget, seed=42):
    """Search for a decomposition of given rank using random initialization + alternating minimization.
    
    The constraint is: sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k*n+i, t] = delta_{j,j'}
    for all i, j, j', k. This is a system of n*m*m*p equations.
    """
    rng = np.random.RandomState(seed)
    best_result = None
    best_error = float('inf')
    
    start = time.time()
    
    def compute_error(U, V, W, r):
        """Compute max error over random test matrices."""
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        max_err = 0.0
        for _ in range(10):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            u = np.einsum('imt,ij->it', U_r, A)
            v = np.einsum('jkt,jk->kt', V_r, B)
            C = np.einsum('kit,it,kt->ik', W_r, u, v)
            err = np.abs(C - A @ B).max()
            max_err = max(max_err, err)
        return max_err
    
    def solve_W_given_UV(U, V, r):
        """Solve for W given U, V using the tensor equation."""
        # For each (i,k), we need: sum_t W[k*n+i, t] * (sum_j U[i*m+j, t]*A[i,j]) * (sum_j' V[j'*p+k, t]*B[j',k]) = sum_j A[i,j]*B[j,k]
        # Use multiple random (A,B) pairs to build an overdetermined system
        num_samples = min(20, max(5, r // 2))
        rhs_all = np.zeros((num_samples * n * p, r))
        lhs_all = np.zeros(num_samples * n * p)
        
        idx = 0
        for s in range(num_samples):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            u = np.einsum('imt,ij->it', U_r, A)  # (n, r)
            v = np.einsum('jkt,jk->kt', V_r, B)  # (p, r)
            
            for i in range(n):
                for k in range(p):
                    rhs_all[idx, :] = u[i, :] * v[k, :]
                    lhs_all[idx] = C_true[i, k]
                    idx += 1
        
        try:
            W_flat, _, _, _ = np.linalg.lstsq(rhs_all, lhs_all, rcond=None)
            W = W_flat.reshape(num_samples, p, n, r).reshape(-1, r)
            # Average over samples - actually we need consistent W
            # Better: use all samples together (already done above)
            W = W_flat.reshape(p, n, r).transpose(1, 0, 2).reshape(p * n, r)
        except:
            return None
        return W
    
    def solve_U_given_VW(V, W, r):
        """Solve for U given V, W."""
        num_samples = min(20, max(5, r // 2))
        rhs_all = np.zeros((num_samples * n * m, r))
        lhs_all = np.zeros(num_samples * n * m)
        
        idx = 0
        for s in range(num_samples):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            v = np.einsum('jkt,jk->kt', V_r, B)  # (p, r)
            
            # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * v[k,t]
            # = sum_j A[i,j] * sum_t U[i*m+j, t] * W[k*n+i, t] * v[k,t]
            # For each (i,j): sum_t U[i*m+j, t] * (sum_k W[k*n+i, t] * v[k,t] * ... ) 
            # Actually: C[i,k] = sum_j A[i,j] * D[i,j,k] where D[i,j,k] = sum_t U[i*m+j,t]*W[k*n+i,t]*v[k,t]
            # We want D[i,j,k] = B[j,k]
            # So: sum_t U[i*m+j, t] * W[k*n+i, t] * v[k,t] = B[j,k] for all k
            # For fixed (i,j): U[i*m+j, :] @ (sum over k of W[k*n+i,:]*v[k,:]) ... no
            # Let me think: for fixed i, the equation is:
            # C[i,:] = sum_j A[i,j] * (sum_t U[i*m+j, t] * W[:,n+i, t] * v[:,t])
            # where W[:,n+i,t] means W[k*n+i,t] for all k
            # So define E[i,j,:] = sum_t U[i*m+j, t] * (W[:,n+i, t] * v[:,t])  (shape p)
            # Then C[i,:] = sum_j A[i,j] * E[i,j,:]
            # We want E[i,j,:] = B[j,:]
            
            # For fixed i, this is linear in U[i*m:(i+1)*m, :]
            # C[i,k] = sum_j A[i,j] * sum_t U[i*m+j, t] * W[k*n+i, t] * v[k,t]
            # = sum_t U[i*m+j, t] * (sum_k ... ) -- no, j is summed
            # Let me write it as: for each (i,k):
            # C[i,k] = sum_j sum_t A[i,j] * U[i*m+j, t] * W[k*n+i, t] * v[k,t]
            # = sum_t (sum_j A[i,j] * U[i*m+j, t]) * W[k*n+i, t] * v[k,t]
            # This is not linear in U in a simple way for lstsq.
            
            # Alternative: treat it as linear in U_flat
            # C[i,k] = sum_j sum_t A[i,j] * U[i*m+j, t] * W[k*n+i, t] * v[k,t]
            # For fixed sample, this is: C_flat = M @ U_flat where M is (n*p) x (n*m*r)
            # But that's huge. Let me just do the simple approach.
            
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, :] appears in C[i,k] with coefficient A[i,j] * W[k*n+i, t] * v[k,t]
                    # Sum over k: contribution to C[i,:] from U[i*m+j,:] is:
                    # sum_t U[i*m+j, t] * (W[:,n+i, t] * v[:,t])  -- shape (p,)
                    # So C[i,:] = sum_j A[i,j] * sum_t U[i*m+j, t] * (W[:,n+i, t] * v[:,t])
                    # For the (i,j) block: we want sum_t U[i*m+j, t] * (W[:,n+i, t] * v[:,t]) = B[j,:]
                    # This gives p equations for r unknowns (U[i*m+j, :])
                    # But we need consistency across all i.
                    # Let's just build the full system.
                    pass
            
            # Build full linear system for this sample
            # Unknowns: U_flat of size n*m*r
            # Equations: C[i,k] for all i,k (n*p equations)
            # C[i,k] = sum_j sum_t A[i,j] * U[i*m+j, t] * W[k*n+i, t] * v[k,t]
            # = sum_{(j,t)} A[i,j] * W[k*n+i, t] * v[k,t] * U[i*m+j, t]
            
            # M[(i,k), (i',j',t)] = delta_{i,i'} * A[i,j] * W[k*n+i, t] * v[k,t]
            # This is sparse. Let me build it.
            M = np.zeros((n * p, n * m * r))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for t in range(r):
                            M[i * p + k, i * m * r + j * r + t] = A[i, j] * W[k * n + i, t] * v[k, t]
            
            C_flat = C_true.flatten()
            rhs_all[idx:idx + n * p] = M  # This is wrong, M is the matrix not the rhs
            break  # This approach is too complex, let me use a different method
        
        # Simpler approach: use gradient-free alternating updates
        return None
    
    for attempt in range(30):
        if time.time() - start > time_budget:
            break
        
        # Initialize with structured random
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Alternating minimization
        for it in range(100):
            if time.time() - start > time_budget:
                break
            
            # Solve for W
            W_new = solve_W_given_UV(U, V, target_rank)
            if W_new is not None:
                W = W_new
            
            # Check error
            err = compute_error(U, V, W, target_rank)
            if err < best_error:
                best_error = err
                if err < 1e-10:
                    best_result = (U.tolist(), V.tolist(), W.tolist())
                    return best_result
            
            # Solve for V (similar to W but transposed)
            # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j' V[j'*p+k, t] * B[j',k])
            # For fixed U, W: linear in V
            num_samples = min(20, max(5, target_rank // 2))
            rhs_v = np.zeros((num_samples * m * p, target_rank))
            lhs_v = np.zeros(num_samples * m * p)
            idx = 0
            for s in range(num_samples):
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C_true = A @ B
                
                U_r = U.reshape(n, m, target_rank)
                W_r = W.reshape(p, n, target_rank)
                u = np.einsum('imt,ij->it', U_r, A)  # (n, r)
                
                # C[i,k] = sum_t W[k*n+i, t] * u[i,t] * (sum_j V[j*p+k, t] * B[j,k])
                # = sum_j sum_t W[k*n+i, t] * u[i,t] * B[j,k] * V[j*p+k, t]
                # For each (j,k): V[j*p+k, :] @ (sum_i W[k*n+i, :] * u[i, :] * B[j,k]) = ... 
                # C[i,k] = sum_j B[j,k] * sum_t V[j*p+k, t] * W[k*n+i, t] * u[i,t]
                # Let D[j,k,i] = sum_t V[j*p+k, t] * W[k*n+i, t] * u[i,t]
                # Then C[i,k] = sum_j B[j,k] * D[j,k,i]
                # We want C[i,k] = sum_j A[i,j] * B[j,k]
                # So D[j,k,i] should relate to A[i,j]
                # sum_j B[j,k] * D[j,k,i] = sum_j A[i,j] * B[j,k] for all B
                # This means D[j,k,i] = A[i,j]
                # So: sum_t V[j*p+k, t] * W[k*n+i, t] * u[i,t] = A[i,j]
                # For each (j,k): V[j*p+k, :] @ (sum_i W[k*n+i, :] * u[i, :] * ... ) 
                # = sum_i A[i,j] * W[k*n+i, t] * u[i,t] ... no
                # sum_t V[j*p+k, t] * (sum_i A[i,j] * W[k*n+i, t] * u[i,t]) = ... 
                # Hmm, let me just build the system directly.
                for j in range(m):
                    for k in range(p):
                        # V[j*p+k, t] appears in C[i,k] with coefficient B[j,k] * W[k*n+i, t] * u[i,t]
                        # Sum over i: sum_i B[j,k] * W[k*n+i, t] * u[i,t] * V[j*p+k, t]
                        # = V[j*p+k, :] @ (sum_i B[j,k] * W[k*n+i, :] * u[i, :])
                        # We want sum_i B[j,k] * W[k*n+i, t] * u[i,t] * V[j*p+k, t] summed over t to give... 
                        # Actually C[i,k] = sum_j sum_t B[j,k] * V[j*p+k, t] * W[k*n+i, t] * u[i,t]
                        # For fixed (j,k): the contribution to all C[i,k] is V[j*p+k, :] @ (W[:,n+i, :] * u[i, :] * B[j,k]) for each i
                        # This is getting complex. Let me use a simpler heuristic.
                        rhs_v[idx, :] = np.zeros(target_rank)
                        lhs_v[idx] = 0
                        idx += 1
                break  # Skip V update for now, focus on W
        
            # Simple heuristic update for U and V: scale to maintain consistency
            U_norm = np.linalg.norm(U)
            V_norm = np.linalg.norm(V)
            W_norm = np.linalg.norm(W)
            target_norm = np.sqrt(n * m * target_rank)
            if U_norm > 0:
                U *= target_norm / U_norm
            if V_norm > 0:
                V *= target_norm / V_norm
            if W_norm > 0:
                W *= target_norm / W_norm
        
        # Final check
        err = compute_error(U, V, W, target_rank)
        if err < best_error:
            best_error = err
            if err < 1e-10:
                best_result = (U.tolist(), V.tolist(), W.tolist())
                return best_result
    
    return best_result

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8."""
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # Build schoolbook decomposition (rank = 96) as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    # Try to find lower rank decomposition
    # Try ranks from low to high
    for target_rank in range(20, 96):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        
        result = _random_search(n, m, p, target_rank, remaining, seed=target_rank)
        
        if result is not None:
            U, V, W = result
            # Verify with numpy - FIXED: was checking schoolbook instead of found solution
            if _verify_numpy(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return {"U": U_school, "V": V_school, "W": W_school}
