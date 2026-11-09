"""Tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n, m, p = 3, 4, 6
    schoolbook_rank = n * m * p  # 72

    # Lower bound
    lower_bound = max(n * m, m * p, p * n)  # max(12, 24, 18) = 24

    def verify(U, V, W, rank):
        """Verify decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k"""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)

        # U: (n*m, rank) -> (n, m, rank)
        U_3d = U_arr.reshape(n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        V_3d = V_arr.reshape(m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        W_3d = W_arr.reshape(p, n, rank)

        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Using einsum: 'ijr,jkr,kir->ijk'
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        # Check all entries equal 1
        if not np.allclose(result, 1.0, atol=1e-9):
            return False
        return True

    def verify_exact(U, V, W, rank):
        """Exact verification using integer arithmetic if possible."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)

        # Check if all entries are integers
        if not (np.all(U_arr == np.round(U_arr)) and
                np.all(V_arr == np.round(V_arr)) and
                np.all(W_arr == np.round(W_arr))):
            return verify(U, V, W, rank)

        U_int = U_arr.astype(np.int64)
        V_int = V_arr.astype(np.int64)
        W_int = W_arr.astype(np.int64)

        U_3d = U_int.reshape(n, m, rank)
        V_3d = V_int.reshape(m, p, rank)
        W_3d = W_int.reshape(p, n, rank)

        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        return np.all(result == 1)

    # Schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    # Strategy: Use structured random search with small integer values
    # Focus on ranks near the lower bound
    # Use a smarter initialization based on block structure

    def random_search(rank, max_attempts=200, val_range=3):
        """Random search for a decomposition of given rank."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            # Use small integers
            U = [[random.randint(-val_range, val_range) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-val_range, val_range) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-val_range, val_range) for _ in range(rank)] for _ in range(p * n)]

            if verify_exact(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        return None

    def structured_search(rank, max_attempts=100):
        """Structured search using block decomposition."""
        # Split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2
        # This gives a 3 x 2 x 3 grid of 2x2x2 sub-tensors
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            # For each of the 3*2*3 = 18 sub-tensors, we need a 2x2x2 decomposition
            # Use Strassen (rank 7) for each, but share structure
            
            # Actually, let's try a different approach:
            # Generate U, V, W with a specific structure
            
            # Method: Use the fact that we can have "sparse" decompositions
            # Each rank-1 term contributes to specific (i,j,k) entries
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Try to build the decomposition rank by rank
            # This is a greedy approach
            remaining = np.ones((n, m, p), dtype=np.float64)
            
            for r in range(rank):
                if time.time() > deadline - 5:
                    return None
                
                # Pick a random (i,j) for U, (j,k) for V, (k,i) for W
                # and try to cancel part of the remaining tensor
                
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                # Set U[i,j,r] = 1, V[j,k,r] = 1, W[k,i,r] = 1
                # This contributes 1 to remaining[i,j,k]
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                remaining[i, j, k] -= 1
            
            # Check if remaining is all zeros
            if np.allclose(remaining, 0.0, atol=1e-9):
                return {"U": U, "V": V, "W": W}
            
            # Try another approach: use negative values to cancel
            # This is complex, so let's just do random search with wider range
        return None

    # Try ranks from lower bound upward
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break

        # Try structured search first
        result = structured_search(rank, max_attempts=50)
        if result is not None:
            best_result = result
            best_rank = rank
            break

        # Try random search
        result = random_search(rank, max_attempts=100, val_range=2)
        if result is not None:
            best_result = result
            best_rank = rank
            break

    return best_result
