import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target rank to search for
    target_rank = 10
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds
    
    # Initialize best result with schoolbook
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1

    # Try to find a lower rank decomposition
    # We will use a simple local search to minimize the error
    # The tensor T is defined by T[i, j, k] = 1 if it's a multiplication term
    # Actually, the tensor is T[i, j, k] = delta_{j, j'} where A is n x m, B is m x p
    # The decomposition is C = A B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T has entries T[i, j, k] = 1 if the term A[i, j] B[j, k] is present.
    # So T[i, j, k] = 1 for all i, j, k.
    
    # We want to find U, V, W such that sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # Note: W index is k*n + i.
    
    def error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    err += (val - 1.0) ** 2
        return err

    # Try rank 10
    for rank in [10, 11]:
        if time.time() > deadline:
            break
            
        # Random restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Local search
            best_err = error(U.tolist(), V.tolist(), W.tolist())
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Perturb one element
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, n * m - 1)
                    r = random.randint(0, rank - 1)
                    current_U[idx, r] += random.gauss(0, 0.1)
                    new_err = error(current_U.tolist(), current_V.tolist(), current_W.tolist())
                    if new_err < best_err:
                        best_err = new_err
                    else:
                        current_U[idx, r] -= random.gauss(0, 0.1)
                elif mat == 'V':
                    idx = random.randint(0, m * p - 1)
                    r = random.randint(0, rank - 1)
                    current_V[idx, r] += random.gauss(0, 0.1)
                    new_err = error(current_U.tolist(), current_V.tolist(), current_W.tolist())
                    if new_err < best_err:
                        best_err = new_err
                    else:
                        current_V[idx, r] -= random.gauss(0, 0.1)
                else:
                    idx = random.randint(0, p * n - 1)
                    r = random.randint(0, rank - 1)
                    current_W[idx, r] += random.gauss(0, 0.1)
                    new_err = error(current_U.tolist(), current_V.tolist(), current_W.tolist())
                    if new_err < best_err:
                        best_err = new_err
                    else:
                        current_W[idx, r] -= random.gauss(0, 0.1)
            
            # Check if error is small enough to round to half-integers
            if best_err < 1e-6:
                # Round to half-integers
                U_rounded = np.round(current_U * 2) / 2
                V_rounded = np.round(current_V * 2) / 2
                W_rounded = np.round(current_W * 2) / 2
                
                # Verify exactness
                if error(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()) < 1e-9:
                    # Convert to lists of lists
                    best_rank = rank
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    break
            
            if best_rank == rank:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
