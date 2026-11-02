import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    rank_school = nm * mp * pn // m  # Actually nm * p = 84? No, nm * mp * pn / m = 12*28*21/4 = 1764. Wrong.
    # Schoolbook rank is n*m*p = 84
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
    
    # Try to find a better decomposition using randomized search
    # The minimal rank for 3x4x7 is believed to be 72 (or close to it)
    # We can try random rank-72 decompositions and check if they work
    
    # But that's a huge search space. Instead, let's try a structured approach.
    # We can use the fact that matrix multiplication tensors have symmetry.
    
    # Actually, for this small case, let's try a few random low-rank decompositions
    # and verify them exactly.
    
    for target_rank in [72, 73, 74, 75, 76, 77, 78, 79, 80]:
        if time.time() > deadline - 5:
            break
            
        attempts = 0
        max_attempts = 50 if target_rank <= 75 else 20
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Generate random factor matrices with small integer/half-integer entries
            U_rand = np.zeros((nm, target_rank), dtype=np.int64)
            V_rand = np.zeros((mp, target_rank), dtype=np.int64)
            W_rand = np.zeros((pn, target_rank), dtype=np.int64)
            
            # Use small random integers
            for r in range(target_rank):
                U_rand[:, r] = np.random.randint(-1, 2, nm)
                V_rand[:, r] = np.random.randint(-1, 2, mp)
                W_rand[:, r] = np.random.randint(-1, 2, pn)
            
            # Verify: for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            # This is expensive: 3*4*7 * target_rank = 84 * 72 ~ 6000 operations per check
            # But we can vectorize somewhat.
            
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        u_val = U_rand[i * m + j]
                        v_val = V_rand[j * p + k]
                        w_val = W_rand[k * n + i]
                        # Compute dot product
                        prod = np.sum(u_val * v_val * w_val)
                        if prod != 1:
                            valid = False
                            break
            
            if valid:
                # Convert to lists of integers
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
    
    return best_result
