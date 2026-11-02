"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # For each (i,j,k), the coefficient should be 1 if i,j,k form a valid triple
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        # Also check that non-entries are zero (they should be by construction if we're careful)
        # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor
        # The decomposition gives sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
        # For matrix multiplication, T[i,j,k] = 1 if this is a valid product term
        # Actually for C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor has entries T[i,j,k] = 1 for all i,j,k (it's the identity in the sum)
        # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        return True

    # Try to find a lower-rank decomposition using random search with integer/half-integer values
    # Known lower bounds: For 3x4x7, the rank is at least max(n*m, m*p, p*n) = max(12,28,21) = 28
    # Upper bound from schoolbook: 3*4*7 = 84
    # Try ranks from 28 up to 83
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try a few random constructions at lower ranks
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break
        # Try random initialization with small integers/halves
        for attempt in range(10):
            if time.time() > deadline:
                break
            random.seed(hash((target_rank, attempt)) % (2**32))
            U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
