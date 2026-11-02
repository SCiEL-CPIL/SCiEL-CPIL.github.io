"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
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

    def verify_numpy(U, V, W, atol=1e-9):
        """Vectorized verification using numpy."""
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            if not np.allclose(T, 1.0, atol=atol):
                return False
            return True
        except Exception:
            return False

    def check_half_integer(values):
        """Check if all values are integers or half-integers."""
        for v in values:
            if not np.isfinite(v):
                return False
            if abs(v * 2 - round(v * 2)) > 1e-9:
                return False
        return True

    def is_valid_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition with integer/half-integer entries."""
        for mat in [U, V, W]:
            for row in mat:
                if not check_half_integer(row):
                    return False
        return verify_numpy(U, V, W)

    # Structured block construction
    # Split m=4 into two blocks of size 2
    # A (3x4) -> A1 (3x2), A2 (3x2)
    # B (4x7) -> B1 (2x7), B2 (2x7)
    # C = A1 B1 + A2 B2
    # Rank of M_{3,2,7} is 21 (lower bound max(6, 14, 21)=21)
    # We need a rank-21 decomposition for M_{3,2,7}
    
    def construct_M327():
        # M_{3,2,7}
        # i in 0..2, j in 0..1, k in 0..6
        # T[i,j,k] = 1
        # We can use the fact that M_{3,1,7} has rank 7?
        # No, M_{3,2,7} can be decomposed into 2 copies of M_{3,1,7}
        # M_{3,1,7} has rank 7?
        # Lower bound max(3, 7, 7) = 7.
        # So rank is 7.
        # Total rank 14?
        # Let's check M_{3,1,7}.
        # C[i,k] = A[i,0] B[0,k]
        # This is an outer product.
        # Rank 1?
        # No, the tensor T[i,0,k] = 1.
        # Rank 1?
        # U[i,0,r] V[0,r] W[k,r]
        # If r=1, U[i,0,1]=1, V[0,1]=1, W[k,1]=1.
        # Then T[i,0,k] = 1.
        # So M_{3,1,7} has rank 1?
        # This is the same paradox.
        # The rank of M_{3,1,7} is 7.
        # Why?
        # Because the bilinear map is (A, B) -> A B^T.
        # A is 3x1, B is 1x7.
        # Output is 3x7.
        # The rank of this bilinear map is 7.
        # So we need 7 multiplications.
        # So M_{3,2,7} has rank 14.
        
        # Let's construct a rank-14 decomposition for M_{3,2,7}
        # We need 14 terms.
        # For each r in 0..13:
        #   U[i,j,r], V[j,k,r], W[k,i,r]
        
        # Actually, we can just use schoolbook for M_{3,2,7} which has rank 3*2*7 = 42.
        # Total rank 84.
        # This doesn't help.
        
        # Let's try to find a better decomposition.
        # Known: Rank of M_{2,2,7} is 14.
        # Rank of M_{3,2,7} is at most 3/2 * 14 = 21.
        # Let's assume rank 21.
        
        return None

    # Given the complexity, we stick to the schoolbook decomposition which is guaranteed to work.
    # Random search for lower ranks is unlikely to succeed in the time limit.
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Try a few random searches for lower ranks just in case
    value_pool = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    for target_rank in range(40, best_rank):
        if time.time() > deadline:
            break
        for attempt in range(10):
            if time.time() > deadline:
                break
            random.seed(hash((target_rank, attempt)) % (2**32))
            U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            if is_valid_decomposition(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        if best_rank == target_rank:
            continue

    return best_result
# EVOLVE-BLOCK-END
