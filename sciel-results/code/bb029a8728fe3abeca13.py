import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
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
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56

    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is valid."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_random_integer_search(rank, max_samples=10000, seed=0):
        """
        Structural approach:
        1. Randomly initialize U and V with small integers.
        2. Solve linear systems for W given U and V.
        3. Check if W has integer (or half-integer) entries.
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # We want to find U (nm x r), V (mp x r) such that 
        # for all i, k, the system M_ik W_ik^T = 1_m has an integer solution.
        # M_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
        
        # To increase the chance of integer solutions, we can restrict U and V entries to small integers.
        # Let's try entries from {-1, 0, 1} or {-1, 1}.
        
        for sample in range(max_samples):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V with random small integers
            # Using {-1, 1} is often good for tensor decompositions (like Strassen)
            # But {-1, 0, 1} allows more flexibility.
            U = np.random.randint(-1, 2, size=(n*m, rank))
            V = np.random.randint(-1, 2, size=(m*p, rank))
            
            # Ensure the matrices are not trivially zero
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            W = np.zeros((p*n, rank), dtype=float)
            valid = True
            
            # Solve for each (i, k) pair
            for i in range(n):
                for k in range(p):
                    # Construct M_ik: shape (m, rank)
                    # M_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # U rows involved: i*m to i*m+m-1
                    # V rows involved: j*p+k for j in 0..m-1
                    # V indices: k, p+k, 2p+k, ... (m-1)p+k
                    
                    # Vectorized construction of M_ik
                    # U_block = U[i*m:(i+1)*m, :]  # shape (m, rank)
                    # V_block = V[k::p, :]         # shape (m, rank) -- wait, V is (mp, rank). 
                    # V[j*p+k] for j=0..m-1 are rows k, p+k, ...
                    # So V_block = V[k::p] is correct if p is the stride.
                    
                    U_block = U[i*m:(i+1)*m, :]
                    V_block = V[k::p, :] # This grabs rows k, k+p, k+2p... which correspond to j=0,1,2...
                    
                    M_ik = U_block * V_block # shape (m, rank)
                    
                    # Solve M_ik @ W_ik^T = ones(m)
                    # W_ik is a row vector of length rank.
                    # This is a linear system A x = b where A is (m, rank) and x is (rank,).
                    # If rank >= m, we have a underdetermined system (if rank > m) or square (if rank == m).
                    # We want an integer solution.
                    
                    # Use lstsq to find the minimum norm solution first.
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if sol is close to integers or half-integers
                    # The problem allows integer/half-integer entries.
                    # Let's check if 2*sol is close to integers.
                    sol_2 = 2.0 * sol
                    if np.max(np.abs(sol_2 - np.round(sol_2))) < 1e-6:
                        W_ik = np.round(sol_2) / 2.0
                        W[k*n+i, :] = W_ik
                    else:
                        # Try to find a nearby integer solution? 
                        # For now, reject this sample if not close.
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Verify the solution
            # Convert to lists for verification function or verify directly
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks starting from a lower bound.
    # The rank of matrix multiplication is at least max(n, m, p) * something?
    # Actually, rank >= (nm+mp+np)/2 is a known lower bound? No.
    # Lower bound is often related to omega.
    # For 2x4x7, schoolbook is 56.
    # Let's try reducing rank by 1, 2, ...
    
    # We can also try specific known structures if any, but random search is more general.
    
    # To improve efficiency, we can try smaller ranks first.
    # If r < m (4), the system M_ik (4 x r) cannot generally have a solution for arbitrary U, V.
    # So r must be at least m=4? No, the system is m equations in r unknowns.
    # If r < m, it's overdetermined and likely no solution unless U, V are specially chosen.
    # So we should start r from max(m, p, n) or higher.
    # Let's start from 20 to be safe, then go up.
    
    # Actually, let's start from a reasonable lower bound.
    # Rank of 2x4x7 is likely > 20.
    
    ranks_to_try = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # More samples for smaller ranks? Or less?
        # Smaller ranks are harder to hit, so maybe more samples?
        # But computation per sample is O(n*p*m*r^2) roughly.
        # Let's use a fixed budget of time per rank.
        
        found = try_random_integer_search(r, max_samples=5000, seed=r)
        if found is not None:
            best_rank = r
            best_result = found
            return best_result
            
    return best_result
# EVOLVE-BLOCK-END
