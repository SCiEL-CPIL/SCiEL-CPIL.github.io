import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def make_schoolbook(rank):
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
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > 1e-6):
            return False
        return True
    
    def is_valid_solution(U, V, W, rank):
        """Full validation: exact decomposition + half-integer entries."""
        if not verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
            return False
        if not check_half_integer(U.tolist()):
            return False
        if not check_half_integer(V.tolist()):
            return False
        if not check_half_integer(W.tolist()):
            return False
        return True
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    rng = np.random.RandomState(42)
    
    # Strategy: For each candidate rank, try structured sparse patterns for U and V,
    # then solve the linear system for W. The key is that for fixed U,V, the problem
    # is linear in W.
    
    # The linear system: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # Unknowns: W[k*n+i,r] for all k,i,r -> p*n*r variables
    # Equations: n*m*p = 45
    
    def solve_for_W(U, V, r):
        """Given U and V, solve for W using the linear system."""
        num_eqs = n * m * p
        num_unk = p * n * r
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        # W[k*n+i, r_idx] is at position (k*n+i)*r + r_idx
                        col = (k * n + i) * r + r_idx
                        A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                    eq_idx += 1
        
        # Solve using least squares
        try:
            w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        # Check if solution is exact
        if residuals.size > 0:
            if residuals[0] > 1e-10:
                return None
        
        w_vec = w_vec.reshape(p * n, r)
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_vec[k * n + i, r_idx]
                    if abs(val - 1) > 1e-10:
                        return None
        
        return w_vec
    
    # Try ranks from 44 down to 36
    for target_rank in range(rank_schoolbook - 1, 35, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Multiple strategies for generating U and V
        num_attempts = 200
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # Strategy varies by attempt number
            strategy = attempt % 5
            
            if strategy == 0:
                # Dense-ish with ±1 entries
                for r_idx in range(target_rank):
                    u_rows = rng.choice(n * m, size=3, replace=False)
                    for row in u_rows:
                        U[int(row), r_idx] = rng.choice([-1, 1])
                    
                    v_rows = rng.choice(m * p, size=3, replace=False)
                    for row in v_rows:
                        V[int(row), r_idx] = rng.choice([-1, 1])
                        
            elif strategy == 1:
                # Structured: use block patterns
                for r_idx in range(target_rank):
                    # U: pick 2-3 rows from one block of 3
                    block = r_idx % 3
                    u_rows = rng.choice([block*3, block*3+1, block*3+2], size=2, replace=False)
                    for row in u_rows:
                        U[int(row), r_idx] = rng.choice([-1, 1])
                    
                    v_rows = rng.choice(m * p, size=2, replace=False)
                    for row in v_rows:
                        V[int(row), r_idx] = rng.choice([-1, 1])
                        
            elif strategy == 2:
                # Sparse: 1-2 nonzeros per column
                for r_idx in range(target_rank):
                    u_rows = rng.choice(n * m, size=1, replace=False)
                    U[int(u_rows[0]), r_idx] = rng.choice([-1, 1])
                    
                    v_rows = rng.choice(m * p, size=1, replace=False)
                    V[int(v_rows[0]), r_idx] = rng.choice([-1, 1])
                    
            elif strategy == 3:
                # Medium density
                for r_idx in range(target_rank):
                    u_rows = rng.choice(n * m, size=4, replace=False)
                    for row in u_rows:
                        U[int(row), r_idx] = rng.choice([-1, 1])
                    
                    v_rows = rng.choice(m * p, size=4, replace=False)
                    for row in v_rows:
                        V[int(row), r_idx] = rng.choice([-1, 1])
                        
            else:
                # Very sparse
                for r_idx in range(target_rank):
                    u_rows = rng.choice(n * m, size=2, replace=False)
                    for row in u_rows:
                        U[int(row), r_idx] = rng.choice([-1, 1])
                    
                    v_rows = rng.choice(m * p, size=2, replace=False)
                    for row in v_rows:
                        V[int(row), r_idx] = rng.choice([-1, 1])
            
            # Solve for W
            W = solve_for_W(U, V, target_rank)
            if W is None:
                continue
            
            # Check half-integer constraint
            if not check_half_integer(W.tolist()):
                continue
            if not check_half_integer(U.tolist()):
                continue
            if not check_half_integer(V.tolist()):
                continue
            
            # Final verification
            if is_valid_solution(U, V, W, target_rank):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
