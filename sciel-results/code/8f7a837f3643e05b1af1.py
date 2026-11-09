import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = rank_schoolbook
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
    best_rank = rank_schoolbook
    
    def verify_decomposition_fast(U, V, W, rank):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(val - 1) > 1e-6:
                        return False
        return True
    
    # Construct explicit rank-28 decomposition using Strassen on 2x2x2 core
    # A (2x3) @ B (3x5) = C (2x5)
    # 
    # Partition: A = [A_left (2x2) | A_right (2x1)]
    #            B = [B_top (2x5); B_bottom (1x5)]
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # For A_left @ B_top (2x2 @ 2x5):
    #   Split B_top into B_top[:, 0:2] (2x2) and B_top[:, 2:5] (2x3)
    #   A_left @ B_top[:, 0:2] (2x2 @ 2x2): use 7 Strassen terms
    #   A_left @ B_top[:, 2:5] (2x2 @ 2x3): use 12 schoolbook terms
    #   Subtotal: 7 + 12 = 19
    #
    # For A_right @ B_bottom (2x1 @ 1x5): 10 schoolbook terms
    #
    # But we also need cross terms:
    # Actually C = A_left @ B_top + A_right @ B_bottom is exact if we define
    # A_left = A[:, 0:2], A_right = A[:, 2], B_top = B[0:2, :], B_bottom = B[2, :]
    # Then C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k] which is correct.
    #
    # So: A_left @ B_top (2x2 @ 2x5) + A_right @ B_bottom (2x1 @ 1x5)
    # For 2x2 @ 2x5: we can split into 5 separate 2x2 @ 2x1 problems,
    # but that's 5*4=20 schoolbook. Using Strassen on the 2x2 @ 2x2 part:
    #   A_left @ B_top[:, 0:2] (2x2 @ 2x2): 7 Strassen terms
    #   A_left @ B_top[:, 2:5] (2x2 @ 2x3): 12 schoolbook terms
    #   Total for this part: 19
    # A_right @ B_bottom (2x1 @ 1x5): 10 terms
    # Grand total: 29. Close to 28 but not quite.
    
    # Let me try a different split to get exactly 28.
    # Split B into B[:, 0:3] (3x3) and B[:, 3:5] (3x2)
    # A @ B[:, 0:3] (2x3 @ 3x3): 18 schoolbook, or can we do better?
    # A @ B[:, 3:5] (2x3 @ 3x2): 12 schoolbook, or 8 with known rank?
    # If rank(2x3x2) = 8, then 18 + 8 = 26? That seems too low.
    
    # I think the known rank-28 construction is more subtle.
    # Let me just do a search with good initializations.
    
    for target_rank in range(27, 31):
        if time.time() > deadline:
            break
        
        max_trials = 400 if target_rank <= 28 else 200
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(99999 + target_rank * 10000 + trial)
            
            if trial % 5 == 0:
                # Sparse with 1-3 nonzeros per column
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                for c in range(target_rank):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    U[i*m + j, c] = random.choice([-1, 1])
                    V[j*p + k, c] = random.choice([-1, 1])
                    W[k*n + i, c] = random.choice([-1, 1])
                    
                    for _ in range(random.randint(0, 2)):
                        i2 = random.randint(0, n-1)
                        j2 = random.randint(0, m-1)
                        U[i2*m + j2, c] += random.choice([-1, 1])
                    
                    for _ in range(random.randint(0, 2)):
                        j2 = random.randint(0, m-1)
                        k2 = random.randint(0, p-1)
                        V[j2*p + k2, c] += random.choice([-1, 1])
                    
                    for _ in range(random.randint(0, 2)):
                        k2 = random.randint(0, p-1)
                        i2 = random.randint(0, n-1)
                        W[k2*n + i2, c] += random.choice([-1, 1])
                        
            elif trial % 5 == 1:
                # Schoolbook base + perturbations
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            U[i*m + j, col] = 1
                            V[j*p + k, col] = 1
                            W[k*n + i, col] = 1
                            col += 1
                            if col >= target_rank:
                                break
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                for c in range(target_rank):
                    if random.random() < 0.5:
                        idx = random.randint(0, n*m-1)
                        U[idx, c] += random.choice([-1, 1])
                    if random.random() < 0.5:
                        idx = random.randint(0, m*p-1)
                        V[idx, c] += random.choice([-1, 1])
                    if random.random() < 0.5:
                        idx = random.randint(0, p*n-1)
                        W[idx, c] += random.choice([-1, 1])
                        
            elif trial % 5 == 2:
                # Strassen on 2x2x2 core + schoolbook for rest
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # 7 Strassen terms for A[:, 0:2] @ B[0:2, 0:2]
                strassen = [
                    ([(0,1),(4,1)], [(0,1),(2,1)], [(0,1),(4,1)]),
                    ([(2,1),(4,1)], [(0,1)], [(2,1),(4,1)]),
                    ([(0,1)], [(2,1),(0,-1)], [(0,1),(4,1)]),
                    ([(4,1)], [(0,1),(2,-1)], [(2,1),(4,1)]),
                    ([(0,1),(2,1)], [(2,1)], [(0,1),(4,1)]),
                    ([(4,1),(0,-1)], [(2,1)], [(2,1),(4,1)]),
                    ([(1,1),(2,-1)], [(1,1)], [(0,1),(4,1)]),
                ]
                
                for c in range(min(7, target_rank)):
                    for idx, sign in strassen[c][0]:
                        U[idx, c] += sign
                    for idx, sign in strassen[c][1]:
                        V[idx, c] += sign
                    for idx, sign in strassen[c][2]:
                        W[idx, c] += sign
                
                # Schoolbook for remaining terms
                # Need to cover: A[:, 0:2]@B[0:2, 2:5], A[:, 0:2]@B[2, :],
                #               A[:, 2]@B[0:2, :], A[:, 2]@B[2, :]
                remaining_terms = []
                for i in range(n):
                    for j in range(2):
                        for k in range(2, p):
                            remaining_terms.append((i, j, k))
                for i in range(n):
                    for j in range(2):
                        for k in range(p):
                            if k >= 2:
                                pass
                    remaining_terms.append((i, 2, random.randint(0, p-1)))
                
                # Just add random terms for the rest
                for c in range(7, target_rank):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    U[i*m + j, c] = random.choice([-1, 1])
                    V[j*p + k, c] = random.choice([-1, 1])
                    W[k*n + i, c] = random.choice([-1, 1])
                    
            elif trial % 5 == 3:
                # Dense random small
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
            else:
                # Hybrid
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                half = target_rank // 2
                for c in range(half):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
                    
                for c in range(half, target_rank):
                    U[:, c] = np.random.randn(n * m) * 0.2
                    V[:, c] = np.random.randn(m * p) * 0.2
                    W[:, c] = np.random.randn(p * n) * 0.2
            
            if verify_decomposition_fast(U, V, W, target_rank):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    if target_rank <= 27:
                        break
        else:
            continue
        break
    
    return best_result
