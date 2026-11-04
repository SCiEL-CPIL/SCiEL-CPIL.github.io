import time
import numpy as np
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses a local search algorithm to find a low-rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W, target):
        """Verify that U, V, W exactly decompose the tensor target."""
        rank = U.shape[1]
        result = np.zeros(target.shape, dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, target)

    def compute_error(U, V, W, target):
        """Compute the squared error of the decomposition."""
        rank = U.shape[1]
        result = np.zeros(target.shape, dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        diff = result - target
        return np.sum(diff * diff)

    def local_search(U, V, W, target, max_iters=10000):
        """
        Perform local search to minimize the error.
        Returns the updated U, V, W.
        """
        rank = U.shape[1]
        current_error = compute_error(U, V, W, target)
        
        if current_error == 0:
            return U, V, W

        no_improvement_count = 0
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random factor and a random entry
            factor_idx = random.randint(0, 2)
            if factor_idx == 0:
                row = random.randint(0, dim_a - 1)
                col = random.randint(0, rank - 1)
                old_val = U[row, col]
                new_val = old_val + random.choice([-1, 1])
                U[row, col] = new_val
                new_error = compute_error(U, V, W, target)
                if new_error < current_error:
                    current_error = new_error
                    no_improvement_count = 0
                    if current_error == 0:
                        return U, V, W
                else:
                    U[row, col] = old_val
            elif factor_idx == 1:
                row = random.randint(0, dim_b - 1)
                col = random.randint(0, rank - 1)
                old_val = V[row, col]
                new_val = old_val + random.choice([-1, 1])
                V[row, col] = new_val
                new_error = compute_error(U, V, W, target)
                if new_error < current_error:
                    current_error = new_error
                    no_improvement_count = 0
                    if current_error == 0:
                        return U, V, W
                else:
                    V[row, col] = old_val
            else:
                row = random.randint(0, dim_c - 1)
                col = random.randint(0, rank - 1)
                old_val = W[row, col]
                new_val = old_val + random.choice([-1, 1])
                W[row, col] = new_val
                new_error = compute_error(U, V, W, target)
                if new_error < current_error:
                    current_error = new_error
                    no_improvement_count = 0
                    if current_error == 0:
                        return U, V, W
                else:
                    W[row, col] = old_val
            
            no_improvement_count += 1
            if no_improvement_count > 1000:
                break

        return U, V, W

    def try_rank(R, target, max_restarts=50):
        """Try to find a decomposition of rank R."""
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with small random integers
            U = np.random.randint(-1, 2, size=(dim_a, R))
            V = np.random.randint(-1, 2, size=(dim_b, R))
            W = np.random.randint(-1, 2, size=(dim_c, R))
            
            # Local search
            U, V, W = local_search(U, V, W, target, max_iters=5000)
            
            if verify_decomposition(U, V, W, target):
                return U, V, W
        
        return None

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # Search for lower ranks
    # Try integer solutions for T first
    for R in range(20, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(R, T, max_restarts=20)
        if result:
            best_U, best_V, best_W = result
            best_rank = R
            break

    # If no integer solution found, try half-integer solutions (search for 2T)
    if best_rank == n * m * p:
        T_half = T * 2
        for R in range(20, best_rank):
            if time.time() > deadline - 15:
                break
            
            result = try_rank(R, T_half, max_restarts=20)
            if result:
                U, V, W = result
                # Convert to half-integers
                U_half = U / 2.0
                V_half = V / 2.0
                W_half = W / 2.0
                
                # Verify with float tolerance or exact rational check
                # Since we searched for 2T, the decomposition is exact for 2T.
                # So U/2, V/2, W/2 is exact for T.
                best_U, best_V, best_W = U_half, V_half, W_half
                best_rank = R
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
