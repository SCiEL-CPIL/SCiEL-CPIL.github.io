import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    def _count_bad_sets(A):
        n = A.shape[0]
        bad_count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if k >= n-1: continue
                    di_jk = A[i, j] + A[i, k]
                    dj_ik = A[j, i] + A[j, k]
                    dk_ij = A[k, i] + A[k, j]
                    
                    l_indices = np.arange(k+1, n)
                    if len(l_indices) == 0: continue
                    A_il = A[i, l_indices]
                    A_jl = A[j, l_indices]
                    A_kl = A[k, l_indices]
                    A_li = A[l_indices, i]
                    A_lj = A[l_indices, j]
                    A_lk = A[l_indices, k]
                    
                    bad_mask = (
                        (di_jk + A_il <= 2) & 
                        (dj_ik + A_jl <= 2) & 
                        (dk_ij + A_kl <= 2) & 
                        (A_li + A_lj + A_lk <= 2)
                    )
                    bad_count += np.sum(bad_mask)
        return bad_count

    def _paley_orientation(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        A = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(p):
                if i != j:
                    if (j - i) % p in squares:
                        A[i, j] = 1
                    else:
                        A[i, j] = 0
        return A

    def _A_to_orientation(A):
        n = A.shape[0]
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                orientation.append(int(A[i, j]))
        return orientation

    best_n = 67
    best_A = _paley_orientation(67)
    
    primes = [3, 7, 11, 19, 23, 31, 43, 47, 59, 61, 67]
    for p in primes:
        if time.time() > deadline: break
        A = _paley_orientation(p)
        if _count_bad_sets(A) == 0:
            best_n = p
            best_A = A
            break

    return {'n': best_n, 'orientation': _A_to_orientation(best_A)}
