import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110
    
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
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        for row in mat:
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) > 1e-6:
                    return False
        return True
    
    # Strategy: Try ranks from 44 down to 40
    # For each rank, try multiple random initializations of U and V
    # Then solve for W using the linear system approach
    
    for target_rank in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Generate structured random U and V
            # Use a mix of sparse and denser patterns
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # Different strategies for different attempts
            if attempt % 3 == 0:
                # Sparse: 2 non-zeros per column
                for r in range(target_rank):
                    u_rows = random.sample(range(n * m), 2)
                    for row in u_rows:
                        U[row, r] = random.choice([-1, 1])
                    
                    v_rows = random.sample(range(m * p), 2)
                    for row in v_rows:
                        V[row, r] = random.choice([-1, 1])
            elif attempt % 3 == 1:
                # Medium sparse: 3 non-zeros per column
                for r in range(target_rank):
                    u_rows = random.sample(range(n * m), 3)
                    for row in u_rows:
                        U[row, r] = random.choice([-1, 1])
                    
                    v_rows = random.sample(range(m * p), 3)
                    for row in v_rows:
                        V[row, r] = random.choice([-1, 1])
            else:
                # Denser: 4 non-zeros per column
                for r in range(target_rank):
                    u_rows = random.sample(range(n * m), 4)
                    for row in u_rows:
                        U[row, r] = random.choice([-1, 1])
                    
                    v_rows = random.sample(range(m * p), 4)
                    for row in v_rows:
                        V[row, r] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p * n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute A_ik[r] = sum_j U[i*m+j,r] * V[j*p+k,r]
                    A_ik = np.zeros(target_rank)
                    for j in range(m):
                        A_ik += U[i * m + j] * V[j * p + k]
                    
                    denom = np.dot(A_ik, A_ik)
                    if abs(denom) < 1e-10:
                        valid = False
                        break
                    
                    # Minimum norm solution: w = A_ik / (A_ik @ A_ik)
                    w = A_ik / denom
                    W[k * n + i] = w
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the solution
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, target_rank):
                # Check half-integer constraint
                if check_half_integer(U_list) and check_half_integer(V_list) and check_half_integer(W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Try to round to half-integers and re-verify
                    U_rounded = [[round(val * 2) / 2 for val in row] for row in U_list]
                    V_rounded = [[round(val * 2) / 2 for val in row] for row in V_list]
                    W_rounded = [[round(val * 2) / 2 for val in row] for row in W_list]
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, target_rank):
                        best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    return best_result
