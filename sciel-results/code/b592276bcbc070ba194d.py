import random
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to improve using block decomposition with Strassen
    # Partition: n=2, m=4, p=7
    # Split m=4 into 2+2: A = [A1 A2], B = [B1; B2]
    # A1, A2: 2x2, B1, B2: 2x7
    # C = A1*B1 + A2*B2
    
    # For (2x2)*(2x7): split p=7 into 2+2+2+1
    # (2x2)*(2x7) = (2x2)*(2x2) + (2x2)*(2x2) + (2x2)*(2x2) + (2x2)*(2x1)
    # Each (2x2)*(2x2) uses Strassen rank 7
    # (2x2)*(2x1) uses rank 4 (no savings)
    # Total for (2x2)*(2x7): 3*7 + 4 = 25
    # Total: 2 * 25 = 50
    
    # But we can do better by sharing Strassen products between the two (2x2)*(2x7) blocks
    
    # Let me try a direct construction with rank around 40
    
    # Use the following approach:
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    
    # I'll use a known decomposition strategy:
    # Group the j indices: {0,1} and {2,3}
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    # For the first group (j=0,1): this is (2x2)*(2x7)
    # For the second group (j=2,3): this is (2x2)*(2x7)
    
    # Each (2x2)*(2x7) can be decomposed using Strassen on the 2x2 part
    # Strassen for (2x2)*(2x2): 7 products
    # For (2x2)*(2x7), we can view it as 7 independent (2x2)*(2x1) multiplications
    # But (2x2)*(2x1) has rank 4, so 7*4 = 28 per (2x2)*(2x7)
    # Total: 2*28 = 56 (same as schoolbook)
    
    # Better: use Strassen for the 2x2 matrix multiplication
    # (2x2)*(2x7): Let M = [[a,b],[c,d]], N = [[n00,n01,...,n06],[n10,n11,...,n16]]
    # MN = [[a*n0j + b*n1j for j], [c*n0j + d*n1j for j]]
    # This is 2*7 = 14 bilinear forms, each requiring 2 products in schoolbook
    # Total 28 products
    
    # With Strassen-like: define S1 = (a+b)(n0j+n1j), S2 = (a-b)(n0j-n1j), etc.
    # But Strassen works for square matrices. For (2x2)*(2x7), we can apply Strassen to each 2x2 block of N
    # Split N into 2x2 + 2x2 + 2x2 + 2x1
    # (2x2)*(2x2) uses 7 products, (2x2)*(2x1) uses 4 products
    # Total: 3*7 + 4 = 25
    
    # So (2x2)*(2x7) has rank at most 25
    # And we have two such blocks, but we can potentially share some products
    
    # Let me just implement the rank 50 decomposition and see if I can improve
    
    # Actually, let me try a completely different approach: use the fact that
    # the problem is small enough to try a systematic search for low-rank decompositions
    
    # I'll use a heuristic: try to find a decomposition with rank R by solving
    # a system of equations where the entries are restricted to {-1, 0, 1}
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try various ranks
    for target_rank in range(35, 57):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random sparse initialization with entries in {-1, 0, 1}
            U_t = np.zeros((n*m, target_rank), dtype=int)
            V_t = np.zeros((m*p, target_rank), dtype=int)
            W_t = np.zeros((p*n, target_rank), dtype=int)
            
            for c in range(target_rank):
                # Randomly place non-zeros
                nz_U = random.sample(range(n*m), min(3, n*m))
                nz_V = random.sample(range(m*p), min(4, m*p))
                nz_W = random.sample(range(p*n), min(3, p*n))
                
                for idx in nz_U:
                    U_t[idx, c] = random.choice([-1, 1])
                for idx in nz_V:
                    V_t[idx, c] = random.choice([-1, 1])
                for idx in nz_W:
                    W_t[idx, c] = random.choice([-1, 1])
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U_t[i*m+j, c] * V_t[j*p+k, c] * W_t[k*n+i, c]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_t.tolist()
                best_V = V_t.tolist()
                best_W = W_t.tolist()
                found = True
                break
        
        if found:
            break  # Found the minimum rank in this range
    
    return {"U": best_U, "V": best_V, "W": best_W}
