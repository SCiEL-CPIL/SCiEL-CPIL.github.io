import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition using block method
    # Block 1: 3x3x3
    # Block 2: 3x3x4
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_sb = [[0] * (n*m*p) for _ in range(n*m)]
    V_sb = [[0] * (n*m*p) for _ in range(m*p)]
    W_sb = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_rank = n*m*p
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find low rank decomposition for 3x3x3
    # We search for rank r in [23, 30]
    # If found, combine with schoolbook for 3x3x4
    
    # Helper to verify decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Construct tensor
        T = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            u = np.array(U)[:, c]
            v = np.array(V)[:, c]
            w = np.array(W)[:, c]
            # u is nm x 1, v is mp x 1, w is pn x 1
            # T += u[:, None, None] * v[None, :, None] * w[None, None, :]
            T += np.einsum('a,b,c->abc', u, v, w)
        
        # Check against standard multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    d = k*n + i
                    if T[a, b, d] != 1:
                        return False
                    # Other entries should be 0?
                    # The tensor T has 1s at (i*m+j, j*p+k, k*n+i)
                    # Are there other non-zero entries?
                    # No, the multiplication tensor is sparse.
        # Check that other entries are 0
        # This is expensive, so we might skip or sample
        return True

    # Search for 3x3x3 decomposition
    found_333 = False
    U_333 = None
    V_333 = None
    W_333 = None
    rank_333 = 9*3*3 # fallback
    
    # Try ranks 23 to 25
    for r in range(23, 26):
        if time.time() > deadline - 20: break
        # Random search
        for _ in range(1000):
            if time.time() > deadline - 20: break
            U_rand = np.random.randint(-1, 2, size=(9, r))
            V_rand = np.random.randint(-1, 2, size=(9, r)) # 3x3 = 9
            W_rand = np.random.randint(-1, 2, size=(9, r)) # 3x3 = 9
            
            # Verify
            T = np.zeros((9, 9, 9))
            for c in range(r):
                T += np.einsum('a,b,c->abc', U_rand[:, c], V_rand[:, c], W_rand[:, c])
            
            # Check non-zero entries
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        a = i*3 + j
                        b = j*3 + k
                        d = k*3 + i
                        if T[a, b, d] != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: continue
            
            # Check zero entries (sample)
            # There are 720 zero entries
            # We can check a few
            for _ in range(10):
                a = random.randint(0, 8)
                b = random.randint(0, 8)
                d = random.randint(0, 8)
                # Check if this is a non-zero entry
                # a = i*3+j, b=j*3+k, d=k*3+i
                # i = a//3, j = a%3
                # j = b//3, k = b%3
                # k = d//3, i = d%3
                i1 = a // 3
                j1 = a % 3
                j2 = b // 3
                k1 = b % 3
                k2 = d // 3
                i2 = d % 3
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue # This is a non-zero entry
                if T[a, b, d] != 0:
                    valid = False
                    break
            if not valid: continue
            
            # Found a valid decomposition
            U_333 = U_rand.tolist()
            V_333 = V_rand.tolist()
            W_333 = W_rand.tolist()
            rank_333 = r
            found_333 = True
            break
        if found_333:
            break
            
    if found_333:
        # Combine with schoolbook for 3x3x4
        # 3x3x4 has rank 36
        # Total rank = rank_333 + 36
        total_rank = rank_333 + 36
        if total_rank < best_rank:
            # Construct full U, V, W
            # U has 9 rows, total_rank cols
            # V has 21 rows, total_rank cols
            # W has 21 rows, total_rank cols
            
            U_full = [[0] * total_rank for _ in range(9)]
            V_full = [[0] * total_rank for _ in range(21)]
            W_full = [[0] * total_rank for _ in range(21)]
            
            # Copy 3x3x3 part
            for i in range(9):
                for c in range(rank_333):
                    U_full[i][c] = U_333[i][c]
            for i in range(9):
                for c in range(rank_333):
                    V_full[i][c] = V_333[i][c]
            for i in range(9):
                for c in range(rank_333):
                    W_full[i][c] = W_333[i][c]
                    
            # Add 3x3x4 part
            # Indices for 3x3x4:
            # A is 3x3 (9)
            # B is 3x4 (12)
            # C is 4x3 (12)
            # But in the full problem, B is 3x7 and C is 7x3.
            # The 3x3x3 part corresponds to columns 0,1,2 of B and rows 0,1,2 of C.
            # The 3x3x4 part corresponds to columns 3,4,5,6 of B and rows 3,4,5,6 of C.
            
            # Schoolbook for 3x3x4
            # j in 0..2, k in 3..6
            # a = i*3 + j
            # b = j*3 + k  (Note: in full problem, b = j*7 + k. Here k is local 0..3, so global k = 3+k_local)
            # d = k*3 + i  (Note: in full problem, d = k*3 + i. Here k is global 3..6)
            
            col_offset = rank_333
            for i in range(3):
                for j in range(3):
                    for k_local in range(4):
                        k_global = 3 + k_local
                        a = i*3 + j
                        b = j*7 + k_global
                        d = k_global*3 + i
                        U_full[a][col_offset] = 1
                        V_full[b][col_offset] = 1
                        W_full[d][col_offset] = 1
                        col_offset += 1
                        
            best_rank = total_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full
            
    # Return best found
    return {"U": best_U, "V": best_V, "W": best_W}
