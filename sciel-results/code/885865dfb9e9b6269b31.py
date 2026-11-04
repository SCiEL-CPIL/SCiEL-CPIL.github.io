import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using the S-vector property
    # We look for U, V with entries in {-1, 0, 1} such that for all i,k,
    # S[i,k,r] = sum_j U[i,j,r] V[j,k,r] has at least one r with S[i,k,r] in {-1, 1}.
    # Then we can construct W with integer entries.
    
    target_ranks = [20, 24, 28, 32, 40, 50, 60, 70, 80]
    
    for R in target_ranks:
        if time.time() > deadline - 10:
            break
            
        if R >= best_rank:
            continue
            
        # Run a heuristic search for this rank
        found = False
        max_iterations = 50000  # Adjust based on time
        
        # Initialize U and V randomly
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        
        # Fill with random -1, 0, 1
        for r in range(R):
            for idx in range(n * m):
                U[idx, r] = random.choice([-1, 0, 1])
            for idx in range(m * p):
                V[idx, r] = random.choice([-1, 0, 1])
        
        # Precompute indices for S calculation
        # S[i, k, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        def check_and_build(U, V, R):
            # Calculate S
            S = np.zeros((n, p, R), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(R):
                        s_val = 0
                        for j in range(m):
                            s_val += U[i * m + j, r] * V[j * p + k, r]
                        S[i, k, r] = s_val
            
            # Check condition: for all i,k, exists r s.t. S[i,k,r] in {-1, 1}
            W = np.zeros((p * n, R), dtype=int)
            for i in range(n):
                for k in range(p):
                    found_r = -1
                    for r in range(R):
                        if S[i, k, r] == 1 or S[i, k, r] == -1:
                            found_r = r
                            break
                    if found_r == -1:
                        return None # Failed
                    # Set W[k*n+i, found_r] = 1 / S[i,k,found_r]
                    # Since S is +/- 1, W is +/- 1
                    W[k * n + i, found_r] = 1 / S[i, k, found_r]
                    # Ensure integer
                    if W[k * n + i, found_r] != int(W[k * n + i, found_r]):
                         return None
                    W[k * n + i, found_r] = int(W[k * n + i, found_r])
            
            # Verify the full decomposition
            # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1
            # Note: W index is k*n+i
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(R):
                            total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if total != 1:
                            return None
                            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Heuristic Search
        # We want to maximize the number of (i,k) pairs that have a valid r.
        # Metric: count of (i,k) where min_r |S[i,k,r]| == 1 (or 0? no, must be 1)
        
        def get_score(U, V, R):
            S = np.zeros((n, p, R), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(R):
                        s_val = 0
                        for j in range(m):
                            s_val += U[i * m + j, r] * V[j * p + k, r]
                        S[i, k, r] = s_val
            
            score = 0
            for i in range(n):
                for k in range(p):
                    for r in range(R):
                        if S[i, k, r] == 1 or S[i, k, r] == -1:
                            score += 1
                            break
            return score

        best_score = -1
        best_U = U.copy()
        best_V = V.copy()
        
        current_score = get_score(U, V, R)
        if current_score == n * p:
            res = check_and_build(U, V, R)
            if res:
                best_result = res
                best_rank = R
                found = True
                break
        
        for iter in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # Pick a random element in U or V to flip
            if random.random() < 0.5:
                idx = random.randint(0, n * m - 1)
                r = random.randint(0, R - 1)
                old_val = U[idx, r]
                U[idx, r] = random.choice([-1, 0, 1])
                if U[idx, r] == old_val:
                    continue
            else:
                idx = random.randint(0, m * p - 1)
                r = random.randint(0, R - 1)
                old_val = V[idx, r]
                V[idx, r] = random.choice([-1, 0, 1])
                if V[idx, r] == old_val:
                    continue
            
            new_score = get_score(U, V, R)
            
            if new_score > best_score:
                best_score = new_score
                best_U = U.copy()
                best_V = V.copy()
                
                if new_score == n * p:
                    res = check_and_build(U, V, R)
                    if res:
                        best_result = res
                        best_rank = R
                        found = True
                        break
            else:
                # Revert (simple hill climbing)
                if random.random() < 0.5:
                    U[idx, r] = old_val
                else:
                    V[idx, r] = old_val

        if found:
            break

    return best_result

