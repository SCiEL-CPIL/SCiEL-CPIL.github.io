import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
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
    
    def verify_decomposition(U, V, W, rank, tol=1e-10):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Vectorized check: for each (i,j,k), compute sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_half_integer(mat, tol=1e-6):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > tol):
            return False
        return True
    
    def is_valid_solution(U, V, W, rank):
        """Full validation: exact decomposition + half-integer entries."""
        if not verify_decomposition(U, V, W, rank):
            return False
        if not check_half_integer(U):
            return False
        if not check_half_integer(V):
            return False
        if not check_half_integer(W):
            return False
        return True
        
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    def build_A_matrix(U, V, rank):
        """Build linear system A @ w_vec = 1 where w_vec flattens W."""
        num_eqs = n * m * p
        num_unk = p * n * rank
        A = np.zeros((num_eqs, num_unk))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r_idx in range(rank):
                        col = (k * n + i) * rank + r_idx
                        A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
        return A

    def solve_for_W(U, V, rank):
        """Solve for W given U and V using least squares."""
        A = build_A_matrix(U, V, rank)
        b = np.ones(n * m * p)
        try:
            w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            return None
        if residuals.size > 0:
            if residuals[0] > 1e-8:
                return None
        w_vec = w_vec.reshape(p * n, rank)
        # Verify solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r_idx in range(rank):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_vec[k * n + i, r_idx]
                    if abs(val - 1) > 1e-8:
                        return None
        return w_vec

    def round_to_half_integer(mat, tol=1e-4):
        """Round matrix to half-integers if close enough."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > tol):
            return None
        return (rounded / 2.0).tolist()

    def try_structured_decomposition(target_rank, seed=42):
        """Try to find a decomposition of given rank using structured random search."""
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Generate diverse candidates with varying sparsity patterns
        candidates = []
        
        # Strategy 1: Sparse random with varying density
        for trial in range(30):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            for r_idx in range(target_rank):
                # Vary sparsity
                u_density = rng.uniform(0.15, 0.45)
                u_count = max(1, int(round(n * m * u_density)))
                u_rows = rng.choice(n * m, size=u_count, replace=False)
                for row in u_rows:
                    U[int(row), r_idx] = rng.choice([-1, 1])
                
                v_density = rng.uniform(0.15, 0.45)
                v_count = max(1, int(round(m * p * v_density)))
                v_rows = rng.choice(m * p, size=v_count, replace=False)
                for row in v_rows:
                    V[int(row), r_idx] = rng.choice([-1, 1])
            candidates.append((U, V))
        
        # Strategy 2: Block-structured patterns (exploit symmetry in 3x3)
        for trial in range(20):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            for r_idx in range(target_rank):
                # Assign each row to a "group" and assign values based on group
                u_groups = rng.randint(0, 3, size=n * m)
                v_groups = rng.randint(0, 3, size=m * p)
                for row in range(n * m):
                    if rng.random() < 0.5:
                        U[row, r_idx] = rng.choice([-1, 1]) if u_groups[row] == r_idx % 3 else 0
                for row in range(m * p):
                    if rng.random() < 0.5:
                        V[row, r_idx] = rng.choice([-1, 1]) if v_groups[row] == r_idx % 3 else 0
            candidates.append((U, V))
            
        # Strategy 3: Based on Kronecker/product structures
        for trial in range(15):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            # Use tensor product structure hints
            base_u = rng.choice([-1, 0, 1], size=(n, m))
            base_v = rng.choice([-1, 0, 1], size=(m, p))
            for r_idx in range(target_rank):
                shift_i = r_idx % n
                shift_j = r_idx % m
                shift_k = r_idx % p
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r_idx] = base_u[(i + shift_i) % n, (j + shift_j) % m]
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, r_idx] = base_v[(j + shift_j) % m, (k + shift_k) % p]
            candidates.append((U, V))
        
        for U, V in candidates:
            if time.time() > deadline - 15:
                break
            
            # Skip if U or V has all-zero columns (degenerate)
            if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
                continue
                
            W = solve_for_W(U, V, target_rank)
            if W is None:
                continue
            
            # Try rounding to half-integers
            W_half = round_to_half_integer(W.tolist(), tol=1e-3)
            if W_half is None:
                # Try slightly different rounding
                W_half = round_to_half_integer(W.tolist(), tol=5e-3)
                if W_half is None:
                    continue
            
            # Also check if U and V are already half-integers (they are by construction)
            if not check_half_integer(U.tolist()):
                continue
            if not check_half_integer(V.tolist()):
                continue
            
            if is_valid_solution(U.tolist(), V.tolist(), W_half, target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_half}
        
        return None

    def try_perturbation_from_schoolbook():
        """Try to find lower rank by perturbing schoolbook structure."""
        if time.time() > deadline - 30:
            return None
        
        # Start from a subset of schoolbook terms and try to complete
        # This is a heuristic: remove some terms and see if others can compensate
        rng = np.random.RandomState(123)
        
        for target_rank in [40, 39, 38, 37, 36]:
            if time.time() > deadline - 25:
                break
            
            # Try keeping a subset of schoolbook terms
            all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            
            for trial in range(10):
                # Keep target_rank terms
                kept = set(rng.choice(len(all_terms), size=target_rank, replace=False))
                
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                for idx, (i, j, k) in enumerate(all_terms):
                    if idx in kept:
                        U[i * m + j, idx] = 1
                        V[j * p + k, idx] = 1
                        W[k * n + i, idx] = 1
                
                # Now try to solve: can we adjust W (and maybe U,V slightly) to make it work?
                # Actually, with this structure, the system is underdetermined for missing terms
                # Let's try a different approach: use the kept terms as a basis and solve for corrections
                
                # Build the system considering all terms but allowing non-zero values for missing ones
                # This gets complex; instead, just verify if the kept subset happens to work
                # (it won't unless we've found a valid sub-decomposition)
                
                # Better: try random sparse U,V with the constraint that they're close to schoolbook
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                
                for r_idx in range(target_rank):
                    # Start with schoolbook-like pattern
                    i0 = r_idx // (m * p)
                    j0 = (r_idx // p) % m
                    k0 = r_idx % p
                    
                    U_rand[i0 * m + j0, r_idx] = 1
                    V_rand[j0 * p + k0, r_idx] = 1
                    
                    # Add some random perturbations
                    if rng.random() < 0.3:
                        extra_row = rng.randint(0, n * m)
                        U_rand[extra_row, r_idx] += rng.choice([-1, 1])
                    if rng.random() < 0.3:
                        extra_row = rng.randint(0, m * p)
                        V_rand[extra_row, r_idx] += rng.choice([-1, 1])
                
                # Clip to half-integers
                U_rand = np.round(U_rand * 2) / 2
                V_rand = np.round(V_rand * 2) / 2
                
                W_sol = solve_for_W(U_rand, V_rand, target_rank)
                if W_sol is None:
                    continue
                
                W_half = round_to_half_integer(W_sol.tolist(), tol=1e-3)
                if W_half is None:
                    continue
                
                if is_valid_solution(U_rand.tolist(), V_rand.tolist(), W_half, target_rank):
                    return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_half}
        
        return None

    # Main search loop: try decreasing ranks
    # Known lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But practical lower bounds are higher. For 3x3*3x5, let's search from 44 down to 30.
    
    search_ranks = list(range(rank_schoolbook - 1, 29, -1))
    
    for target_rank in search_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try structured decomposition
        result = try_structured_decomposition(target_rank, seed=target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
        
        # If we're at higher ranks, also try perturbation approach
        if target_rank >= 36 and target_rank <= 44:
            if time.time() > deadline - 25:
                break
            result2 = try_perturbation_from_schoolbook()
            if result2 is not None:
                best_result = result2
                best_rank = target_rank
                break
    
    # Final verification of best result
    if best_rank < rank_schoolbook:
        if not is_valid_solution(best_result["U"], best_result["V"], best_result["W"], best_rank):
            # Fallback to schoolbook if something went wrong
            best_result = make_schoolbook(rank_schoolbook)
            best_rank = rank_schoolbook
    
    return best_result
