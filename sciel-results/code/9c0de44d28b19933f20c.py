import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 145  # 145s to leave safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use block decomposition
    # 3x3 x 3x8 = (3x3 x 3x3) + (3x3 x 3x3) + (3x3 x 3x2)
    
    # Helper to find decomposition for a, b, c matrix multiplication
    def find_decomp(a, b, c, max_rank, time_budget, max_attempts=500):
        """Find a decomposition for a x b times b x c matrix multiplication."""
        end_time = start_time + time_budget
        for rank in range(6, max_rank + 1):
            if time.time() > end_time:
                break
            for attempt in range(max_attempts):
                if time.time() > end_time:
                    break
                # Use values from {-1, -0.5, 0, 0.5, 1}
                U = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], (a * b, rank))
                V = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], (b * c, rank))
                W = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], (c * a, rank))
                
                # Quick check using numpy
                valid = True
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            u_idx = i * b + j
                            v_idx = j * c + k
                            w_idx = k * a + i
                            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return U_list, V_list, W_list, rank
        return None, None, None, None
    
    # Try to find decompositions for sub-problems
    # 3x3 x 3x3: try rank 7-9
    U33, V33, W33, r33 = find_decomp(3, 3, 3, 9, 15, 300)
    # 3x3 x 3x2: try rank 5-6
    U32, V32, W32, r32 = find_decomp(3, 3, 2, 6, 15, 300)
    
    if U33 is not None and U32 is not None:
        # Combine: 3x3 x 3x8 = (3x3 x 3x3) + (3x3 x 3x3) + (3x3 x 3x2)
        total_rank = r33 + r33 + r32
        U_full = [[0.0] * total_rank for _ in range(9)]
        V_full = [[0.0] * total_rank for _ in range(24)]
        W_full = [[0.0] * total_rank for _ in range(24)]
        
        # First 3x3 x 3x3 block (columns 0-2 of B)
        for c in range(r33):
            for i in range(9):
                U_full[i][c] = U33[i][c]
            for j in range(3):
                for k in range(3):
                    V_full[j*8 + k][c] = V33[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_full[k*3 + i][c] = W33[k*3 + i][c]
        
        # Second 3x3 x 3x3 block (columns 3-5 of B)
        offset = r33
        for c in range(r33):
            for i in range(9):
                U_full[i][offset + c] = U33[i][c]
            for j in range(3):
                for k in range(3):
                    V_full[j*8 + (k+3)][offset + c] = V33[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3 + i][offset + c] = W33[k*3 + i][c]
        
        # Third 3x3 x 3x2 block (columns 6-7 of B)
        offset2 = 2 * r33
        for c in range(r32):
            for i in range(9):
                U_full[i][offset2 + c] = U32[i][c]
            for j in range(3):
                for k in range(2):
                    V_full[j*8 + (k+6)][offset2 + c] = V32[j*2 + k][c]
            for k in range(2):
                for i in range(3):
                    W_full[(k+6)*3 + i][offset2 + c] = W32[k*3 + i][c]
        
        # Verify
        if verify_decomposition(U_full, V_full, W_full, total_rank) and total_rank < best_rank:
            best_rank = total_rank
            best_result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Also try direct random search for the full problem
    for rank in range(min(best_rank - 1, 40), 24, -1):
        if time.time() > deadline:
            break
        attempts = 100
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            U = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], (9, rank))
            V = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], (24, rank))
            W = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], (24, rank))
            
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(8):
                        val = np.dot(U[i*3+j], V[j*8+k] * W[k*3+i])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result
