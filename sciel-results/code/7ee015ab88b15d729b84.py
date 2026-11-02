import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    schoolbook = get_schoolbook()
    best_rank = n * m * p
    best_solution = schoolbook

    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for 120s limit

    # Precompute target tensor slices for efficient access
    # T_target[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    # Precompute for each u, the set of (v, w) pairs that should be 1
    target_slices_u = []
    for u in range(dim_u):
        i = u // m
        j = u % m
        pairs = []
        for v in range(dim_v):
            jp = v // p
            kp = v % p
            if jp == j:
                for w in range(dim_w):
                    kn = w // n
                    in_ = w % n
                    if kn == kp and in_ == i:
                        pairs.append((v, w))
        target_slices_u.append(pairs)
    
    # Precompute for each v
    target_slices_v = []
    for v in range(dim_v):
        j = v // p
        k = v % p
        pairs = []
        for u in range(dim_u):
            i = u // m
            ju = u % m
            if ju == j:
                for w in range(dim_w):
                    kw = w // n
                    iw = w % n
                    if kw == k and iw == i:
                        pairs.append((u, w))
        target_slices_v.append(pairs)
    
    # Precompute for each w
    target_slices_w = []
    for w in range(dim_w):
        k = w // n
        i = w % n
        pairs = []
        for u in range(dim_u):
            iu = u // m
            ju = u % m
            if iu == i:
                for v in range(dim_v):
                    jv = v // p
                    kv = v % p
                    if jv == ju and kv == k:
                        pairs.append((u, v))
        target_slices_w.append(pairs)

    def compute_error(U, V, W, R):
        """Compute the error of the decomposition."""
        error = 0.0
        for u in range(dim_u):
            recon = np.zeros((dim_v, dim_w))
            for r in range(R):
                recon += U[u, r] * np.outer(V[:, r], W[:, r])
            for v, w in target_slices_u[u]:
                error += (recon[v, w] - 1.0) ** 2
            # Check zeros where target is 0
            for v in range(dim_v):
                for w in range(dim_w):
                    if (v, w) not in target_slices_u[u]:
                        error += recon[v, w] ** 2
        return error

    def check_half_integer(mat, tol=1e-4):
        """Check if all entries are close to half-integers."""
        for val in mat:
            closest_half = round(val * 2) / 2
            if abs(val - closest_half) > tol:
                return False
        return True

    def snap_to_half_integer(mat):
        """Snap values to nearest half-integer."""
        return np.round(mat * 2) / 2

    def als_solve(U, V, W, R, max_iters=100, tol=1e-8):
        """Run ALS iterations to converge."""
        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for u in range(dim_u):
                A_u = np.zeros((dim_v * dim_w, R))
                for r in range(R):
                    A_u[:, r] = np.kron(V[:, r], W[:, r])
                T_vec = np.zeros(dim_v * dim_w)
                for v, w in target_slices_u[u]:
                    T_vec[v * dim_w + w] = 1.0
                try:
                    U_new, residuals, rank_a, s = np.linalg.lstsq(A_u, T_vec, rcond=None)
                    U[:, u] = U_new
                except:
                    pass

            # Update V
            for v in range(dim_v):
                A_v = np.zeros((dim_u * dim_w, R))
                for r in range(R):
                    A_v[:, r] = np.kron(U[:, r], W[:, r])
                T_vec = np.zeros(dim_u * dim_w)
                for u, w in target_slices_v[v]:
                    T_vec[u * dim_w + w] = 1.0
                try:
                    V_new, residuals, rank_a, s = np.linalg.lstsq(A_v, T_vec, rcond=None)
                    V[:, v] = V_new
                except:
                    pass

            # Update W
            for w in range(dim_w):
                A_w = np.zeros((dim_u * dim_v, R))
                for r in range(R):
                    A_w[:, r] = np.kron(U[:, r], V[:, r])
                T_vec = np.zeros(dim_u * dim_v)
                for u, v in target_slices_w[w]:
                    T_vec[u * dim_v + v] = 1.0
                try:
                    W_new, residuals, rank_a, s = np.linalg.lstsq(A_w, T_vec, rcond=None)
                    W[:, w] = W_new
                except:
                    pass
            
            # Compute error
            error = compute_error(U, V, W, R)
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        return error

    # Try structured initializations based on known patterns
    # For 2x3 x 3x5, try various rank values with different initialization strategies
    
    # Strategy 1: Try ranks from 29 down to 20 with multiple random restarts
    for R in range(29, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        num_attempts = 30 if R >= 25 else 50
        
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Random initialization
                U = np.random.randn(dim_u, R)
                V = np.random.randn(dim_v, R)
                W = np.random.randn(dim_w, R)
            elif attempt % 3 == 1:
                # Sparse-like initialization
                U = np.random.choice([-1, 0, 1], size=(dim_u, R))
                V = np.random.choice([-1, 0, 1], size=(dim_v, R))
                W = np.random.choice([-1, 0, 1], size=(dim_w, R))
            else:
                # Small random initialization
                U = np.random.randn(dim_u, R) * 0.1
                V = np.random.randn(dim_v, R) * 0.1
                W = np.random.randn(dim_w, R) * 0.1
            
            error = als_solve(U, V, W, R, max_iters=80)
            
            if error < 1e-6:
                if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                    U_snapped = snap_to_half_integer(U)
                    V_snapped = snap_to_half_integer(V)
                    W_snapped = snap_to_half_integer(W)
                    # Verify the snapped solution
                    verify_error = compute_error(U_snapped, V_snapped, W_snapped, R)
                    if verify_error < 1e-6:
                        best_rank = R
                        best_solution = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        found = True
                        break
        
        if found:
            break

    # Strategy 2: Try a few more ranks with different approach - use gradient-free local search
    for R in range(29, 24, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Start from near schoolbook and perturb
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # Run ALS
            error = als_solve(U, V, W, R, max_iters=60)
            
            if error < 1e-5:
                # Try to polish with tighter tolerance
                error = als_solve(U, V, W, R, max_iters=50, tol=1e-10)
                if error < 1e-6 and check_half_integer(U, tol=1e-3) and check_half_integer(V, tol=1e-3) and check_half_integer(W, tol=1e-3):
                    U_snapped = snap_to_half_integer(U)
                    V_snapped = snap_to_half_integer(V)
                    W_snapped = snap_to_half_integer(W)
                    verify_error = compute_error(U_snapped, V_snapped, W_snapped, R)
                    if verify_error < 1e-6:
                        best_rank = R
                        best_solution = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        found = True
                        break
        
        if found:
            break

    return best_solution
# EVOLVE-BLOCK-END
