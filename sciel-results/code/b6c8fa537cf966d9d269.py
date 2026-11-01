import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build the target tensor indices
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_triples.append((a, b, c))
    
    # Try to find a lower-rank decomposition using a greedy approach
    # We'll try ranks from 42 down to 30, and for each rank, try to find
    # a valid decomposition using a systematic search with small rational values
    
    import random
    import time
    start_time = time.time()
    random.seed(12345)
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check all target triples
        for (a, b, c) in target_triples:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        # Check a sample of non-target triples
        # Total triples: (n*m)*(m*p)*(p*n) = 6*21*14 = 1764
        # Check all for safety
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Compute expected value
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition with rank < 42
    # Use a randomized approach with verification
    
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for rank in range(42, 25, -1):
        if time.time() - start_time > 100:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() - start_time > 105:
                break
            
            # Generate random V and W with small rational values
            V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(m * p)], dtype=float)
            W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(p * n)], dtype=float)
            
            # Try to solve for U
            # For each a, we need: for all (b,c), sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is a linear system in U[a,:]
            # Let's form the system: for fixed a, unknowns are U[a,0],...,U[a,rank-1]
            # Equations: for each (b,c): sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            
            # Number of equations: (m*p) * (p*n) = 21 * 14 = 294
            # Number of unknowns: rank
            # If rank < 294, the system is overdetermined. We need exact solution.
            
            # Use least squares and check if residual is zero
            U_candidate = np.zeros((n * m, rank), dtype=float)
            valid = True
            
            for a in range(n * m):
                # Form the matrix A where A[b,c,r] = V[b,r] * W[c,r]
                # Reshape to (294, rank)
                A = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, rank)
                # Form the target vector
                target_vec = np.zeros(m * p * p * n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i * m + j == a:
                                b = j * p + k
                                c = k * n + i
                                target_vec[b * (p * n) + c] = 1.0
                
                # Solve least squares
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(A, target_vec, rcond=None)
                    U_candidate[a, :] = sol
                    # Check if the solution is exact
                    residual = np.linalg.norm(A @ sol - target_vec)
                    if residual > 1e-9:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Verify the full decomposition
                if verify_decomposition(U_candidate, V, W, rank):
                    # Success! Convert to lists
                    U_list = [list(row) for row in U_candidate]
                    V_list = [list(row) for row in V]
                    W_list = [list(row) for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
