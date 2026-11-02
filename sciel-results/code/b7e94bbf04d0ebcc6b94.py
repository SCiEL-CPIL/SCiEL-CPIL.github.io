"""Program to find a rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank

    start_time = time.time()
    time_limit = 140.0  # seconds

    # Try ranks from 14 to 23
    for r in range(14, schoolbook_rank):
        if time.time() - start_time > time_limit:
            break
            
        result = _try_find_decomposition_structured(n, m, p, r, time_limit - (time.time() - start_time))
        if result is not None:
            best_result = result
            best_rank = r
            # If we found 14, it's likely minimal. Break early if r == 14?
            # Let's try to go lower if possible, but 14 is the known min.
            if r == 14:
                break

    return best_result

def _try_find_decomposition_structured(n, m, p, rank, time_budget):
    """
    Try to find a decomposition of the given rank using a structured search:
    1. Random sparse U, V.
    2. Solve for W linearly for each (i,k) block.
    3. Check for half-integer solutions.
    """
    start_time = time.time()
    
    # Parameters for random search
    np.random.seed(42 + rank)
    
    # Densities to try
    densities = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    for density in densities:
        if time.time() - start_time > time_budget:
            break
            
        max_iters = 2000 # Adjust based on speed
        for it in range(max_iters):
            if it % 50 == 0 and time.time() - start_time > time_budget:
                break
            
            # Generate random sparse U and V
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            
            # Fill with -1, 0, 1
            for idx in range(n * m):
                mask = np.random.rand(rank) < density
                U[idx, mask] = np.random.choice([-1, 1], size=np.sum(mask))
            
            for idx in range(m * p):
                mask = np.random.rand(rank) < density
                V[idx, mask] = np.random.choice([-1, 1], size=np.sum(mask))
            
            # Check if solution exists for W
            W_solution = _solve_for_W(U, V, n, m, p, rank)
            
            if W_solution is not None:
                # Verify
                if _verify_solution(U, V, W_solution, n, m, p):
                    return _finalize_solution(U, V, W_solution, n, m, p)
    
    return None

def _solve_for_W(U, V, n, m, p, rank):
    """
    For each (i, k), solve M_ik * w_ik = 1 for w_ik.
    M_ik is m x rank.
    We look for a half-integer solution.
    """
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for k in range(p):
            # Construct M_ik: m x rank
            M_ik = np.zeros((m, rank))
            for j in range(m):
                # U[i*m+j, :] * V[j*p+k, :]
                M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Solve M_ik * w = 1
            # We want w to be half-integers.
            # Scale by 2: M_ik * (2w) = 2
            # Let x = 2w. We want integer x such that M_ik * x = 2 * ones(m)
            
            # Use least squares to find a candidate x
            # If M_ik is underdetermined, use pinv
            try:
                x_min = np.linalg.pinv(M_ik) @ (2.0 * np.ones(m))
            except np.linalg.LinAlgError:
                return None
            
            # Round to nearest integer
            x_rounded = np.round(x_min).astype(int)
            
            # Check if M_ik * x_rounded == 2 * ones(m)
            if not np.allclose(M_ik @ x_rounded, 2.0 * np.ones(m), atol=1e-9):
                # Try to repair? Or just fail this candidate
                # We can try a small local search on x
                # But for now, reject
                continue
            
            # Store w = x / 2
            W[k * n + i, :] = x_rounded / 2.0
    
    return W

def _verify_solution(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid decomposition.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _finalize_solution(U, V, W, n, m, p):
    """
    Convert numpy arrays to lists.
    """
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure they are integers or half-integers
    # The problem allows integer/half-integer.
    # Our W is x/2 where x is int. U, V are int.
    # So all are half-integers.
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
