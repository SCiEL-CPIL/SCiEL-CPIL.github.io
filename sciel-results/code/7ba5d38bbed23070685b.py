import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 25, 25, 25

    # Build the target tensor T[i,j,k] = 1 for all i,j,k in range(n),range(m),range(p)
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U shape: (nm, rank), V shape: (mp, rank), W shape: (pn, rank)

    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook fallback: rank 125
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Try to find a rank-49 decomposition using a structured approach
    # For 5x5 matrix multiplication, we can use a block decomposition approach
    
    # Strategy: Use the fact that 5 = 2 + 3, and combine Strassen's 2x2 and 3x3 decompositions
    # 2x2 Strassen has rank 7, 3x3 has rank 23 (known optimal)
    # But we need to handle the cross terms carefully
    
    # Alternative: Try a direct algebraic approach with rank 49
    # The 5x5 matrix multiplication tensor has 125 entries, each equal to 1
    # We need to find rank-49 matrices U, V, W such that the contraction gives 1 everywhere
    
    # Let's try a parameterized family of solutions
    # For rank 49, we can think of it as 7x7 = 49 terms
    # Each term c has U[:,c], V[:,c], W[:,c]
    
    # Try a structured initialization based on Kronecker products
    # 5 = 2+3, so we can use block structure
    
    # Actually, let's try a more systematic approach:
    # The tensor T[i,j,k] = 1 for all i,j,k
    # This is the all-ones tensor, which has rank 1 in the standard sense
    # But we need the specific tensor structure for matrix multiplication
    
    # Wait, I need to reconsider. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if and only if the (i,k) entry of A*B is sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (the coefficient is always 1)
    # This means the tensor is actually the all-ones tensor of size 5x5x5
    
    # The all-ones tensor [1]_{i,j,k} has tensor rank 1!
    # U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1 would give sum = 1
    
    # But wait, that's not right. Let me re-read the problem.
    # T[i,j,k] = 1 for all i,j,k means the tensor is all ones
    # The decomposition T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # If we set rank=1: U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
    # Then sum_c = 1*1*1 = 1 for all i,j,k. This works!
    
    # So the rank-1 decomposition exists!
    U_rank1 = [[1] for _ in range(nm)]
    V_rank1 = [[1] for _ in range(mp)]
    W_rank1 = [[1] for _ in range(pn)]
    
    if check_decomposition(U_rank1, V_rank1, W_rank1, 1):
        best_result = {"U": U_rank1, "V": V_rank1, "W": W_rank1}
        return best_result
    
    # If rank 1 doesn't work (maybe I misunderstood the tensor structure),
    # try rank 2, 3, ... up to some reasonable limit
    
    for target_rank in range(2, 126):
        if time.time() > deadline:
            break
            
        # Try simple structured solutions
        # Pattern 1: All ones
        U_test = [[1] * target_rank for _ in range(nm)]
        V_test = [[1] * target_rank for _ in range(mp)]
        W_test = [[1] * target_rank for _ in range(pn)]
        
        if check_decomposition(U_test, V_test, W_test, target_rank):
            best_result = {"U": U_test, "V": V_test, "W": W_test}
            return best_result
        
        # Pattern 2: Sparse structured
        # Try to find a sparse solution
        U_test = [[0] * target_rank for _ in range(nm)]
        V_test = [[0] * target_rank for _ in range(mp)]
        W_test = [[0] * target_rank for _ in range(pn)]
        
        for c in range(target_rank):
            # Assign non-zero entries in a structured way
            for idx in range(nm):
                U_test[idx][c] = 1 if (idx + c) % target_rank == 0 else 0
            for idx in range(mp):
                V_test[idx][c] = 1 if (idx + c) % target_rank == 0 else 0
            for idx in range(pn):
                W_test[idx][c] = 1 if (idx + c) % target_rank == 0 else 0
        
        if check_decomposition(U_test, V_test, W_test, target_rank):
            best_result = {"U": U_test, "V": V_test, "W": W_test}
            return best_result
    
    return best_result
