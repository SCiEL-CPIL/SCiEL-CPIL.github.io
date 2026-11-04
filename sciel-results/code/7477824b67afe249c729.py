import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Exact verification using integer arithmetic."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                    if s != 1:
                        return False
        return True

    def error_count(U, V, W, rank):
        """Count how many (i,j,k) entries are wrong."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        wrong = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                    if s != 1:
                        wrong += 1
        return wrong

    def local_search(U, V, W, rank, max_iters=5000):
        """
        Coordinate descent local search on integer entries.
        Try changing one entry at a time to reduce error count.
        """
        current_err = error_count(U, V, W, rank)
        if current_err == 0:
            return U, V, W
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Precompute contributions for speed
        # For each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if current_err == 0:
                return U, V, W
            
            # Pick a random matrix and element
            mat = random.choice(['U', 'V', 'W'])
            
            if mat == 'U':
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                c = random.randint(0, rank-1)
                old = U3[i,j,c]
                # Try new values: old-1, old+1, and maybe 0, 1, -1
                candidates = [old-1, old+1]
                if old != 0:
                    candidates.append(0)
                if old != 1:
                    candidates.append(1)
                if old != -1:
                    candidates.append(-1)
                best_new = None
                best_err = current_err
                for new_val in candidates:
                    U3[i,j,c] = new_val
                    err = error_count(U, V, W, rank)
                    if err < best_err:
                        best_err = err
                        best_new = new_val
                if best_new is not None:
                    U3[i,j,c] = best_new
                    current_err = best_err
                else:
                    U3[i,j,c] = old
            elif mat == 'V':
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                c = random.randint(0, rank-1)
                old = V3[j,k,c]
                candidates = [old-1, old+1]
                if old != 0:
                    candidates.append(0)
                if old != 1:
                    candidates.append(1)
                if old != -1:
                    candidates.append(-1)
                best_new = None
                best_err = current_err
                for new_val in candidates:
                    V3[j,k,c] = new_val
                    err = error_count(U, V, W, rank)
                    if err < best_err:
                        best_err = err
                        best_new = new_val
                if best_new is not None:
                    V3[j,k,c] = best_new
                    current_err = best_err
                else:
                    V3[j,k,c] = old
            else:
                k = random.randint(0, p-1)
                i = random.randint(0, n-1)
                c = random.randint(0, rank-1)
                old = W3[k,i,c]
                candidates = [old-1, old+1]
                if old != 0:
                    candidates.append(0)
                if old != 1:
                    candidates.append(1)
                if old != -1:
                    candidates.append(-1)
                best_new = None
                best_err = current_err
                for new_val in candidates:
                    W3[k,i,c] = new_val
                    err = error_count(U, V, W, rank)
                    if err < best_err:
                        best_err = err
                        best_new = new_val
                if best_new is not None:
                    W3[k,i,c] = best_new
                    current_err = best_err
                else:
                    W3[k,i,c] = old
        
        if current_err == 0:
            return U, V, W
        return None

    # Try ranks from low to high
    for target_rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random integers
            U = np.random.randint(-1, 2, size=(nm, target_rank)).astype(np.int64)
            V = np.random.randint(-1, 2, size=(mp, target_rank)).astype(np.int64)
            W = np.random.randint(-1, 2, size=(pn, target_rank)).astype(np.int64)
            
            # Run local search
            res = local_search(U, V, W, target_rank, max_iters=2000)
            if res:
                best = {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    return best
# EVOLVE-BLOCK-END
