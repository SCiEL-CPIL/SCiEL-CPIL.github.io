import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline for 2x3x9
    rank_54 = n * m * p
    U_54 = [[0] * rank_54 for _ in range(nm)]
    V_54 = [[0] * rank_54 for _ in range(mp)]
    W_54 = [[0] * rank_54 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_54[i * m + j][col] = 1
                V_54[j * p + k][col] = 1
                W_54[k * n + i][col] = 1
                col += 1
    best = {"U": [row[:] for row in U_54], "V": [row[:] for row in V_54], "W": [row[:] for row in W_54]}
    best_rank = rank_54
    
    # Try to find a better decomposition for 2x3x3
    # If we find rank R for 2x3x3, we can use 3*R for 2x3x9
    # Target rank for 2x3x3: 17
    
    n2, m2, p2 = 2, 3, 3
    nm2 = n2 * m2
    mp2 = m2 * p2
    pn2 = p2 * n2
    target_rank_233 = 17
    
    def verify_233(U, V, W, rank):
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = 0
                    for r in range(rank):
                        s += U[i*m2+j][r] * V[j*p2+k][r] * W[k*n2+i][r]
                    if s != 1:
                        return False
        return True

    found_233 = None
    # Search for rank 17 solution for 2x3x3
    # Use random initialization and local search
    for attempt in range(20):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Initialize with random small integers
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank_233)] for _ in range(nm2)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank_233)] for _ in range(mp2)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank_233)] for _ in range(pn2)]
        
        # Local search: perturb and check
        # To speed up, we can check the error instead of exact verification
        def error_233(U, V, W, rank):
            err = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0
                        for r in range(rank):
                            s += U[i*m2+j][r] * V[j*p2+k][r] * W[k*n2+i][r]
                        err += (s - 1) ** 2
            return err
        
        current_err = error_233(U, V, W, target_rank_233)
        
        # Gradient-free local search
        for step in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
            if current_err == 0:
                if verify_233(U, V, W, target_rank_233):
                    found_233 = (U, V, W, target_rank_233)
                    break
                break
            
            # Pick a random variable to change
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm2-1)
                col = random.randint(0, target_rank_233-1)
                old_val = U[row][col]
                U[row][col] += random.choice([-1, 1])
            elif mat == 'V':
                row = random.randint(0, mp2-1)
                col = random.randint(0, target_rank_233-1)
                old_val = V[row][col]
                V[row][col] += random.choice([-1, 1])
            else:
                row = random.randint(0, pn2-1)
                col = random.randint(0, target_rank_233-1)
                old_val = W[row][col]
                W[row][col] += random.choice([-1, 1])
                
            new_err = error_233(U, V, W, target_rank_233)
            if new_err < current_err:
                current_err = new_err
            else:
                # Revert
                if mat == 'U':
                    U[row][col] = old_val
                elif mat == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
                    
        if found_233:
            break
    
    if found_233:
        U_233, V_233, W_233, rank_233 = found_233
        total_rank = 3 * rank_233
        
        # Construct 2x3x9 solution
        U_9 = [[0] * total_rank for _ in range(nm)]
        V_9 = [[0] * total_rank for _ in range(mp)]
        W_9 = [[0] * total_rank for _ in range(pn)]
        
        for block in range(3):
            offset = block * rank_233
            for i in range(n2):
                for j in range(m2):
                    for r in range(rank_233):
                        U_9[i*m2+j][offset+r] = U_233[i*m2+j][r]
                        
            for j in range(m2):
                for k in range(p2):
                    for r in range(rank_233):
                        # V index: j*p + k
                        # In 2x3x3, V index is j*p2 + k
                        # In 2x3x9, V index is j*9 + (block*3 + k)
                        V_9[j*p + (block*p2 + k)][offset+r] = V_233[j*p2+k][r]
                        
            for k in range(p2):
                for i in range(n2):
                    for r in range(rank_233):
                        # W index: k*n + i
                        # In 2x3x3, W index is k*n2 + i
                        # In 2x3x9, W index is (block*3 + k)*n + i
                        W_9[(block*p2 + k)*n + i][offset+r] = W_233[k*n2+i][r]
                        
        # Verify the full solution
        def verify_full(U, V, W, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            return False
            return True
            
        if verify_full(U_9, V_9, W_9, total_rank):
            best = {"U": U_9, "V": V_9, "W": W_9}
            best_rank = total_rank
        else:
            # Fallback
            pass
            
    return best
