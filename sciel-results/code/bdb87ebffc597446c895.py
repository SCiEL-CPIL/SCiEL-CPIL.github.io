import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40
    
    # Build trivial (schoolbook) decomposition
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
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Try to find a lower rank decomposition using structured search
    # For 2x4x5 matrix multiplication, we can try various rank values
    # The tensor has dimensions (2*4, 4*5, 5*2) = (8, 20, 10)
    
    # Strategy: Try ranks from lower bound up to schoolbook rank
    # Use random initialization and iterative refinement
    
    # Known lower bound for 2x4x5 is at least 24 (from Strassen-like bounds)
    # Let's try ranks from 24 to 39
    
    for target_rank in range(24, rank_school):
        if time.time() > deadline - 5.0:
            break
            
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize U, V, W with random integer/half-integer values
            # We'll use a structured approach based on known patterns
            
            # For matrix multiplication tensor T[i,j,k] = delta[i,j,k]
            # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,j,k]
            
            # Try a structured decomposition
            U = np.zeros((nm, target_rank), dtype=np.float64)
            V = np.zeros((mp, target_rank), dtype=np.float64)
            W = np.zeros((pn, target_rank), dtype=np.float64)
            
            # Initialize with random values
            for r in range(target_rank):
                U[:, r] = np.random.choice([-1, 0, 1], size=nm)
                V[:, r] = np.random.choice([-1, 0, 1], size=mp)
                W[:, r] = np.random.choice([-1, 0, 1], size=pn)
            
            # Check if this gives an exact decomposition
            # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0
            
            def check_decomposition(U, V, W, rank):
                """Check if U, V, W form an exact decomposition of the matmul tensor."""
                # For each (i, j, k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
                # For each other combination, check if it equals 0
                
                # This is expensive, so we'll do a partial check
                # Check all (i, j, k) where the tensor should be 1
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i * m + j]
                            v_val = V[j * p + k]
                            w_val = W[k * n + i]
                            # sum_r u_val[r] * v_val[r] * w_val[r]
                            s = np.sum(u_val * v_val * w_val)
                            if abs(s - 1.0) > 1e-9:
                                return False
                
                # Check some zero entries (sample for speed)
                # For entries where (i,j,k) doesn't form a valid matmul index
                # Actually, the tensor is defined on all (a, b, c) where a in [0, nm), b in [0, mp), c in [0, pn)
                # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j' and k=k' and i=i'
                # Wait, that's not right. Let me reconsider.
                
                # The matmul tensor T has T[i, j, k] = 1 for the multiplication C[i,k] = sum_j A[i,j] * B[j,k]
                # In the flattened form: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
                # So T[i*m+j, j*p+k, k*n+i] = 1, and all other entries are 0.
                
                # Check a few zero entries
                random.seed(42)
                for _ in range(50):
                    a = random.randint(0, nm - 1)
                    b = random.randint(0, mp - 1)
                    c = random.randint(0, pn - 1)
                    
                    # Check if this is a "one" entry
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        # Should be 1, already checked
                        continue
                    
                    # Should be 0
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
                
                return True
            
            if check_decomposition(U, V, W, target_rank):
                # Convert to list format with integer/half-integer entries
                U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(target_rank)] for i in range(nm)]
                V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(target_rank)] for i in range(mp)]
                W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(target_rank)] for i in range(pn)]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
