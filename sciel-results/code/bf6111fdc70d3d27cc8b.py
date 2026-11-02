"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Tensor dimensions
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8
    
    # Build the list of (i, j, k) triples that must satisfy the constraint
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U, V, W):
        """Check if decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for (i, j, k) in triples:
            s = 0.0
            for r in range(len(U[0])):
                s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def verify_with_matrices(U, V, W):
        """Faster verification using matrix operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        # For each triple (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for (i, j, k) in triples:
            u_row = U_arr[i*m+j, :]
            v_row = V_arr[j*p+k, :]
            w_row = W_arr[k*n+i, :]
            s = np.dot(u_row * v_row, w_row)
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def solve_for_W(U, V):
        """Given U and V, solve for W entries.
        For each (i,k): W[k*n+i, r] must satisfy
        sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This is: for each (i,k), we have a linear system in W[k*n+i, :]
        Let M_{i,k} be a matrix where M_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r]
        Then we need M_{i,k} @ W[k*n+i, :]^T ... no, that's not right.
        
        Actually: sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        Let a_r = sum_j U[i*m+j, r] * V[j*p+k, r]
        Then sum_r a_r * W[k*n+i, r] = 1
        
        This is a single linear equation: a^T @ W_row = 1
        Underdetermined if r > 1. We can pick the minimum-norm solution or add constraints.
        """
        r = U.shape[1]
        W = np.zeros((dim_w, r))
        for i in range(n):
            for k in range(p):
                # a_r = sum_j U[i*m+j, r] * V[j*p+k, r]
                a = np.zeros(r)
                for j in range(m):
                    a += U[i*m+j, :] * V[j*p+k, :]
                # Need a^T @ W[k*n+i, :] = 1
                # Minimum norm solution: W_row = a / (a^T @ a) if a^T@a != 0
                norm_sq = np.dot(a, a)
                if norm_sq > 1e-12:
                    W[k*n+i, :] = a / norm_sq
                else:
                    # a is zero vector, no solution exists for this (i,k)
                    return None
        return W
    
    # Try to find a decomposition of rank < 32
    best_rank = n * m * p  # schoolbook = 32
    best_result = None
    
    # First, always have the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try ranks from low to high, with random initializations
    np.random.seed(12345)
    
    for rank in range(16, best_rank + 1):  # Try from 16 up to 31
        if time.time() > deadline - 10:
            break
        
        max_attempts = 50 if rank <= 24 else 30
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization of U and V
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            
            # Solve for W
            W = solve_for_W(U, V)
            if W is None:
                continue
            
            # Verify
            if verify_with_matrices(U.tolist(), V.tolist(), W.tolist()):
                # Found a valid decomposition!
                # Convert to list of lists with exact values
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to clean values if close to integers/half-integers
                U_list = [[round(x, 10) if abs(x - round(x)) < 1e-9 else (round(x*2)/2 if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in U_list]
                V_list = [[round(x, 10) if abs(x - round(x)) < 1e-9 else (round(x*2)/2 if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in V_list]
                W_list = [[round(x, 10) if abs(x - round(x)) < 1e-9 else (round(x*2)/2 if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in W_list]
                
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
