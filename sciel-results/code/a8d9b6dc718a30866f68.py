import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (150s hard kill)

    # 1. Build the target tensor T of shape (n*m, m*p, p*n)
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        U_f = np.array(U, dtype=float)
        V_f = np.array(V, dtype=float)
        W_f = np.array(W, dtype=float)
        T_approx = np.einsum('ic,jc,kc->ijk', U_f, V_f, W_f)
        return np.allclose(T_approx, T, atol=1e-9)

    def schoolbook():
        """Fallback schoolbook decomposition (rank 18)."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: Model the matrix multiplication tensor as a linear code over GF(2)
    # by mapping integer entries to their parity, then search for a rank-16 decomposition
    # using a probabilistic algorithm that minimizes the Hamming weight of the tensor's
    # codewords to identify a valid sparse support structure.
    
    # Step 1: Map the tensor to GF(2) by taking parity (mod 2) of entries
    T_gf2 = np.zeros_like(T, dtype=int)
    T_gf2[T != 0] = 1  # All non-zero entries are 1, so parity is 1

    # Step 2: Search for a low-rank decomposition over GF(2) first
    # A rank-r decomposition over GF(2) means T_gf2[i,j,k] = sum_{r=1}^R U[i,r] * V[j,r] * W[k,r] mod 2
    
    def find_gf2_decomposition(rank, max_iterations=5000):
        """
        Find a rank-r decomposition over GF(2) using randomized local search.
        Returns (U, V, W) as binary matrices, or None if not found.
        """
        if time.time() > deadline - 15:
            return None
        
        # Initialize with random binary matrices
        U = np.random.randint(0, 2, size=(n*m, rank))
        V = np.random.randint(0, 2, size=(m*p, rank))
        W = np.random.randint(0, 2, size=(p*n, rank))
        
        def calc_gf2_error(U, V, W):
            """Calculate Hamming distance between T_gf2 and the decomposed tensor."""
            # Compute T_approx over GF(2)
            T_approx = np.zeros((n*m, m*p, p*n), dtype=int)
            for r in range(rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            T_approx = T_approx % 2
            return np.sum(T_approx != T_gf2)
        
        current_error = calc_gf2_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Simulated annealing over GF(2)
        T_start = 1.0
        T_end = 1e-4
        alpha = 0.995
        
        for it in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            T_curr = T_start * (alpha ** it)
            
            # Pick a random variable to flip
            choice = np.random.randint(0, 3)
            if choice == 0:
                i, r = np.random.randint(n*m), np.random.randint(rank)
                U[i, r] = 1 - U[i, r]  # Flip bit
            elif choice == 1:
                j, r = np.random.randint(m*p), np.random.randint(rank)
                V[j, r] = 1 - V[j, r]
            else:
                k, r = np.random.randint(p*n), np.random.randint(rank)
                W[k, r] = 1 - W[k, r]
            
            new_error = calc_gf2_error(U, V, W)
            delta_e = new_error - current_error
            
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T_curr):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        return best_U, best_V, best_W
            else:
                # Revert
                if choice == 0:
                    U[i, r] = 1 - U[i, r]
                elif choice == 1:
                    V[j, r] = 1 - V[j, r]
                else:
                    W[k, r] = 1 - W[k, r]
        
        if best_error == 0:
            return best_U, best_V, best_W
        return None

    # Step 3: Lift the GF(2) solution to integer/half-integer domain
    def lift_gf2_to_integer(U_gf2, V_gf2, W_gf2, rank):
        """
        Lift a GF(2) decomposition to integer/half-integer values.
        We try small integer values: -1, 0, 1, 2, -2, 0.5, -0.5
        """
        if time.time() > deadline - 10:
            return None
        
        # Try to find integer coefficients that match the GF(2) pattern
        # Start with the GF(2) solution and try small perturbations
        best_U, best_V, best_W = U_gf2.astype(float), V_gf2.astype(float), W_gf2.astype(float)
        
        def calc_real_error(U, V, W):
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            return np.sum((T - T_approx)**2)
        
        current_error = calc_real_error(best_U, best_V, best_W)
        
        # Local search in real domain starting from GF(2) lift
        for it in range(2000):
            if time.time() > deadline - 10:
                break
            
            choice = np.random.randint(0, 3)
            if choice == 0:
                i, r = np.random.randint(n*m), np.random.randint(rank)
                old_val = best_U[i, r]
                # Try small changes
                delta = np.random.choice([-0.5, 0.5, -1.0, 1.0])
                new_val = old_val + delta
                # Keep values in a reasonable range
                if -3 <= new_val <= 3:
                    best_U[i, r] = new_val
            elif choice == 1:
                j, r = np.random.randint(m*p), np.random.randint(rank)
                old_val = best_V[j, r]
                delta = np.random.choice([-0.5, 0.5, -1.0, 1.0])
                new_val = old_val + delta
                if -3 <= new_val <= 3:
                    best_V[j, r] = new_val
            else:
                k, r = np.random.randint(p*n), np.random.randint(rank)
                old_val = best_W[k, r]
                delta = np.random.choice([-0.5, 0.5, -1.0, 1.0])
                new_val = old_val + delta
                if -3 <= new_val <= 3:
                    best_W[k, r] = new_val
            
            new_error = calc_real_error(best_U, best_V, best_W)
            if new_error < current_error:
                current_error = new_error
            else:
                # Revert
                if choice == 0:
                    best_U[i, r] = old_val
                elif choice == 1:
                    best_V[j, r] = old_val
                else:
                    best_W[k, r] = old_val
        
        if current_error < 1e-9:
            return best_U, best_V, best_W
        return None

    # Try to find a rank-16 decomposition
    found = False
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(123 + restart)
        gf2_result = find_gf2_decomposition(16)
        
        if gf2_result is not None:
            U_gf2, V_gf2, W_gf2 = gf2_result
            lifted = lift_gf2_to_integer(U_gf2, V_gf2, W_gf2, 16)
            
            if lifted is not None:
                U, V, W = lifted
                if verify_decomposition(U, V, W):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = 16
                    found = True
                    break

    if not found:
        # Try rank 17 as a fallback
        if time.time() < deadline - 15:
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                np.random.seed(200 + restart)
                gf2_result = find_gf2_decomposition(17)
                
                if gf2_result is not None:
                    U_gf2, V_gf2, W_gf2 = gf2_result
                    lifted = lift_gf2_to_integer(U_gf2, V_gf2, W_gf2, 17)
                    
                    if lifted is not None:
                        U, V, W = lifted
                        if verify_decomposition(U, V, W):
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                            best_rank = 17
                            found = True
                            break

    return best_result
# EVOLVE-BLOCK-END
