import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Initialize with schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = schoolbook_rank
    
    def check_333(U, V, W, rank):
        """Check 3x3x3 decomposition: T[i,j,k]=sum_r U[i*3+j,r]*V[j,r]*W[k,r] == 1"""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = np.sum(U_np[i * 3 + j, :] * V_np[j, :] * W_np[k, :])
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def check_332(U, V, W, rank):
        """Check 3x3x2 decomposition: T[i,j,k]=sum_r U[i*3+j,r]*V[j*2+k,r]*W[k*3+i,r] == 1"""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    s = np.sum(U_np[i * 3 + j, :] * V_np[j * 2 + k, :] * W_np[k * 3 + i, :])
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def check_full(U, V, W, rank):
        """Check full 3x3x8 decomposition."""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :])
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def search_333(target_rank, time_limit=5, seed=42):
        """Search for rank decomposition of 3x3x3 using structured random search."""
        start = time.time()
        random.seed(seed)
        vals = [-1, 0, 1]
        
        for attempt in range(200000):
            if time.time() > start + time_limit:
                break
            
            # Use sparse random matrices
            U = [[random.choice(vals) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(3)]
            W = [[random.choice(vals) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(3)]
            
            if check_333(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None
    
    def search_332(target_rank, time_limit=5, seed=43):
        """Search for rank decomposition of 3x3x2."""
        start = time.time()
        random.seed(seed)
        vals = [-1, 0, 1]
        
        for attempt in range(200000):
            if time.time() > start + time_limit:
                break
            
            U = [[random.choice(vals) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(6)]
            W = [[random.choice(vals) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(6)]
            
            if check_332(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None
    
    found_333 = None
    found_332 = None
    
    # Try to find 3x3x3 decomposition at various ranks
    for tr in [23, 24, 25, 26, 27, 28]:
        if time.time() > deadline - 20:
            break
        remaining = deadline - 20 - time.time()
        if remaining < 1:
            break
        res = search_333(tr, time_limit=min(5, remaining), seed=42 + tr)
        if res:
            found_333 = res
            break
    
    # Try to find 3x3x2 decomposition at various ranks
    for tr in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - 10 - time.time()
        if remaining < 1:
            break
        res = search_332(tr, time_limit=min(4, remaining), seed=43 + tr)
        if res:
            found_332 = res
            break
    
    # Compose full decomposition from blocks if found
    # 3x3x8 = 3x3x3 (k=0,1,2) + 3x3x3 (k=3,4,5) + 3x3x2 (k=6,7)
    if found_333 and found_332:
        U1, V1, W1, r1 = found_333
        U2, V2, W2, r2 = found_333  # Reuse same decomposition
        U3, V3, W3, r3 = found_332
        
        total_rank = r1 + r2 + r3
        
        if total_rank < best_rank:
            U_full = np.zeros((9, total_rank))
            V_full = np.zeros((24, total_rank))
            W_full = np.zeros((24, total_rank))
            
            # Block 1: k=0,1,2
            U_np1 = np.array(U1)
            V_np1 = np.array(V1)
            W_np1 = np.array(W1)
            for i in range(9):
                U_full[i, :r1] = U_np1[i, :]
            for j in range(3):
                for k in range(3):
                    V_full[j * 8 + k, :r1] = V_np1[j, :]
            for k in range(3):
                for i in range(3):
                    W_full[k * 3 + i, :r1] = W_np1[k, :]
            
            # Block 2: k=3,4,5
            offset1 = r1
            U_np2 = np.array(U2)
            V_np2 = np.array(V2)
            W_np2 = np.array(W2)
            for i in range(9):
                U_full[i, offset1:offset1+r2] = U_np2[i, :]
            for j in range(3):
                for k in range(3):
                    V_full[j * 8 + (3 + k), offset1:offset1+r2] = V_np2[j, :]
            for k in range(3):
                for i in range(3):
                    W_full[k * 3 + i, offset1:offset1+r2] = W_np2[k, :]
            
            # Block 3: k=6,7
            offset2 = r1 + r2
            U_np3 = np.array(U3)
            V_np3 = np.array(V3)
            W_np3 = np.array(W3)
            for i in range(9):
                U_full[i, offset2:offset2+r3] = U_np3[i, :]
            for j in range(3):
                for k in range(2):
                    V_full[j * 8 + (6 + k), offset2:offset2+r3] = V_np3[j * 2 + k, :]
            for k in range(2):
                for i in range(3):
                    W_full[k * 3 + i, offset2:offset2+r3] = W_np3[k * 3 + i, :]
            
            # Verify
            if check_full(U_full, V_full, W_full, total_rank):
                best_result = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                best_rank = total_rank
    
    # If no block decomposition found, try direct random search on full tensor
    if best_rank == schoolbook_rank and time.time() < deadline - 5:
        for tr in [60, 55, 50]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            
            random.seed(100 + tr)
            vals = [-1, 0, 1]
            found = False
            for attempt in range(50000):
                if time.time() > start_time + remaining:
                    break
                
                U = [[random.choice(vals) if random.random() < 0.15 else 0 for _ in range(tr)] for _ in range(9)]
                V = [[random.choice(vals) if random.random() < 0.15 else 0 for _ in range(tr)] for _ in range(24)]
                W = [[random.choice(vals) if random.random() < 0.15 else 0 for _ in range(tr)] for _ in range(24)]
                
                if check_full(U, V, W, tr):
                    best_result = {
                        "U": U,
                        "V": V,
                        "W": W
                    }
                    best_rank = tr
                    found = True
                    break
            
            if found:
                break
    
    return best_result
