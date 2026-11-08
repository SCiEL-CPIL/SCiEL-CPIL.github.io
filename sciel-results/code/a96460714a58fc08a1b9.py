import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Function to verify a 3x3x3 decomposition
    def verify_333(U, V, W):
        # U, V, W are 9xR
        R = len(U[0])
        # Convert to numpy for speed
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # The tensor T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
        # We need T[i,j,k] == 1 for all i,j,k in 0..2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(R):
                        s += U_arr[i*3+j, r] * V_arr[j*3+k, r] * W_arr[k*3+i, r]
                    if s != 1:
                        return False
        return True

    def find_333_decomp(max_rank=26, time_limit=20):
        # Try to find a decomposition for 3x3x3 with rank <= max_rank
        # Start with max_rank and go down
        for R in range(max_rank, 22, -1):
            if time.time() > time.time() + time_limit:
                break
            # Try a few random initializations
            for trial in range(100):
                if time.time() > time.time() + time_limit:
                    break
                U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(9)]
                V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(9)]
                W = [[random.randint(-2, 2) for _ in range(R)] for _ in range(9)]
                
                # Local search
                for step in range(1000):
                    if time.time() > time.time() + time_limit:
                        break
                    if verify_333(U, V, W):
                        return U, V, W, R
                    
                    # Pick a random unsatisfied constraint
                    # Simplified: just pick random i,j,k
                    i = random.randint(0, 2)
                    j = random.randint(0, 2)
                    k = random.randint(0, 2)
                    
                    # Calculate current sum
                    s = 0
                    for r in range(R):
                        s += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                    
                    if s == 1:
                        continue
                    
                    # Try to adjust one random element
                    # This is a very basic local search
                    # We need delta such that new_s = 1
                    # This is hard with integers.
                    # Instead, let's just try flipping signs or adding 1
                    # This search is likely to fail for R=23 quickly.
                    # It might find R=26 or 27.
                    
                    # Given the difficulty, let's just return None if not found quickly
                    # and rely on schoolbook.
                    pass
        
        return None

    # Schoolbook for 3x3x3
    def schoolbook_333():
        R = 27
        U = [[0]*R for _ in range(9)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W, R

    # Schoolbook for 3x3x1
    def schoolbook_331():
        R = 9
        U = [[0]*R for _ in range(9)]
        V = [[0]*R for _ in range(3)] # m*p = 3*1 = 3
        W = [[0]*R for _ in range(3)] # p*n = 1*3 = 3
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(1):
                    U[i*3+j][col] = 1
                    V[j*1+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W, R

    # Get 3x3x3 decomp
    decomp_333 = find_333_decomp(max_rank=26, time_limit=10)
    if decomp_333 is None:
        decomp_333 = schoolbook_333()
    
    U1, V1, W1, R1 = decomp_333
    
    # Get 3x3x1 decomp
    U2, V2, W2, R2 = schoolbook_331()
    
    # Construct 3x3x7
    # Part 1: 3x3x3 (cols 0-2 of B)
    # Part 2: 3x3x3 (cols 3-5 of B) -> Same as Part 1 but shifted indices in V and W?
    # No, V and W indices depend on k.
    # For Part 2, k' = k - 3.
    # We need to map the indices.
    
    # Easier: Just concatenate the tensor slices.
    # Total Rank = R1 + R1 + R2
    Total_R = R1 + R1 + R2
    
    U_final = [[0]*Total_R for _ in range(9)]
    V_final = [[0]*Total_R for _ in range(21)] # 3*7
    W_final = [[0]*Total_R for _ in range(21)] # 7*3
    
    # Fill Part 1 (k=0,1,2)
    for r in range(R1):
        for i in range(3):
            for j in range(3):
                U_final[i*3+j][r] = U1[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_final[j*7+k][r] = V1[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_final[k*3+i][r] = W1[k*3+i][r]
                
    # Fill Part 2 (k=3,4,5)
    # This is another 3x3x3 block.
    # We can reuse the same U1, V1, W1 structure?
    # Yes, because the tensor is the same, just shifted in k.
    # U is the same.
    # V indices: j*7 + (k+3).
    # W indices: (k+3)*3 + i.
    offset1 = R1
    for r in range(R1):
        for i in range(3):
            for j in range(3):
                U_final[i*3+j][offset1 + r] = U1[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_final[j*7 + (k+3)][offset1 + r] = V1[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_final[(k+3)*3 + i][offset1 + r] = W1[k*3+i][r]
                
    # Fill Part 3 (k=6)
    # This is 3x3x1.
    offset2 = R1 + R1
    for r in range(R2):
        for i in range(3):
            for j in range(3):
                U_final[i*3+j][offset2 + r] = U2[i*3+j][r]
        for j in range(3):
            for k in range(1):
                V_final[j*7 + 6][offset2 + r] = V2[j*1+k][r]
        for k in range(1):
            for i in range(3):
                W_final[6*3 + i][offset2 + r] = W2[k*3+i][r]
                
    return {"U": U_final, "V": V_final, "W": W_final}
