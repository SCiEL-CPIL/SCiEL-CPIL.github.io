import time
import numpy as np
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (150s hard kill)

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = Fraction(1, 1)

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T using exact arithmetic."""
        rank = U.shape[1]
        # Compute sum over r of U[i,r] * V[j,r] * W[k,r]
        # and check against T[i,j,k]
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = Fraction(0, 1)
                    for r in range(rank):
                        val += U[i, r] * V[j, r] * W[k, r]
                    if val != T[i, j, k]:
                        return False
        return True

    def schoolbook():
        """Fallback schoolbook decomposition (rank 18)."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=object)
        V = np.zeros((m*p, rank), dtype=object)
        W = np.zeros((p*n, rank), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = Fraction(1, 1)
                    V[j*p + k, col] = Fraction(1, 1)
                    W[k*n + i, col] = Fraction(1, 1)
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # ====================================================================
    # STRATEGIC APPROACH: Block decomposition
    # ====================================================================
    # The 2x3 x 3x3 matrix multiplication can be viewed as:
    #   A (2x3) * B (3x3) = C (2x3)
    #
    # We can partition this into overlapping 2x2 x 2x2 sub-problems.
    # If we can find a rank-7 decomposition for 2x2 x 2x2 (Strassen),
    # and combine multiple such blocks cleverly, we may achieve rank < 18.
    #
    # The known result: the rank of matrix multiplication tensor
    # M_{2,3,3} is at most 17 (and known lower bounds are around 16-17).
    #
    # Approach: Use a structured search over low-rank decompositions.
    # We parameterize U, V, W as sparse matrices with entries from
    # {0, 1, -1, 1/2, -1/2} and search for an exact decomposition.
    #
    # Key insight: We can use the fact that the tensor has a specific
    # structure. The non-zero entries of T are at positions
    # (i*m+j, j*p+k, k*n+i) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    #
    # Let's enumerate these 18 positions:
    # i=0, j=0, k=0: (0, 0, 0)
    # i=0, j=0, k=1: (0, 1, 1)
    # i=0, j=0, k=2: (0, 2, 2)
    # i=0, j=1, k=0: (1, 0, 0)
    # i=0, j=1, k=1: (1, 1, 1)
    # i=0, j=1, k=2: (1, 2, 2)
    # i=0, j=2, k=0: (2, 0, 0)
    # i=0, j=2, k=1: (2, 1, 1)
    # i=0, j=2, k=2: (2, 2, 2)
    # i=1, j=0, k=0: (3, 0, 0)
    # i=1, j=0, k=1: (3, 1, 1)
    # i=1, j=0, k=2: (3, 2, 2)
    # i=1, j=1, k=0: (4, 0, 0)
    # i=1, j=1, k=1: (4, 1, 1)
    # i=1, j=1, k=2: (4, 2, 2)
    # i=1, j=2, k=0: (5, 0, 0)
    # i=1, j=2, k=1: (5, 1, 1)
    # i=1, j=2, k=2: (5, 2, 2)
    
    # ====================================================================
    # DIRECT CONSTRUCTION OF A RANK-17 DECOMPOSITION
    # ====================================================================
    # Based on known results in the literature, the border rank of M_{2,3,3}
    # is 17, and the tensor rank is also 17 (or possibly 16).
    #
    # We will attempt to construct an explicit rank-17 decomposition.
    # The idea is to use the Strassen decomposition for the 2x2 block
    # involving rows 0,1 and columns 0,1 of B, and handle the remaining
    # terms efficiently.
    #
    # Let's define:
    #   A = [[a, b, c], [d, e, f]]  (2x3)
    #   B = [[g, h, i], [j, k, l], [m, n, o]]  (3x3)
    #   C = A * B = [[C00, C01, C02], [C10, C11, C12]]
    #
    # Where:
    #   C00 = ag + bj + cm
    #   C01 = ah + bk + cn
    #   C02 = ai + bl + co
    #   C10 = dg + ej + fm
    #   C11 = dh + ek + fn
    #   C12 = di + el + fo
    #
    # We need 17 scalar multiplications to compute all 6 output entries.
    
    # Let's try a specific construction based on the known rank-17 result.
    # Reference: The rank of matrix multiplication for small matrices.
    
    # We'll use the following 17 products:
    # P1 = (a+d)(g+e)        -> contributes to C00, C10, C01, C11
    # P2 = (a-d)(g-e)        -> contributes to C00, C10
    # P3 = (b+e)(h+k)        -> contributes to C01, C11
    # P4 = (b-e)(h-k)        -> contributes to C01, C11
    # P5 = (c+f)(m+o)        -> contributes to C02, C12
    # P6 = (c-f)(m-o)        -> contributes to C02, C12
    # P7 = a*h                -> contributes to C01
    # P8 = d*h                -> contributes to C11
    # P9 = a*i                -> contributes to C02
    # P10 = d*i               -> contributes to C12
    # P11 = b*j               -> contributes to C01
    # P12 = e*j               -> contributes to C11
    # P13 = c*n               -> contributes to C02
    # P14 = f*n               -> contributes to C12
    # P15 = b*l               -> contributes to C02
    # P16 = e*l               -> contributes to C12
    # P17 = (a+b+c+d+e+f)(g+h+i+j+k+l+m+n+o) -> This is too large
    
    # This approach is getting complicated. Let's try a more systematic search.
    
    # ====================================================================
    # SYSTEMATIC SEARCH FOR RANK-17 DECOMPOSITION
    # ====================================================================
    # We'll use a simulated annealing approach with exact rational arithmetic.
    # The search space is: U, V, W with entries from a small set of rationals.
    
    candidate_values = [Fraction(0,1), Fraction(1,1), Fraction(-1,1), 
                       Fraction(1,2), Fraction(-1,2)]
    
    def random_matrix(rows, cols, sparse_density=0.3):
        """Generate a random sparse matrix with entries from candidate_values."""
        M = np.zeros((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                if np.random.random() < sparse_density:
                    M[i, j] = np.random.choice(candidate_values)
        return M
    
    def compute_tensor(U, V, W):
        """Compute the tensor from U, V, W using exact arithmetic."""
        rank = U.shape[1]
        result = {}
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = Fraction(0, 1)
                    for r in range(rank):
                        val += U[i, r] * V[j, r] * W[k, r]
                    if val != 0:
                        result[(i, j, k)] = val
        return result
    
    def tensor_error(T_comp):
        """Compute the error between computed tensor and target tensor."""
        # Check all target positions
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pos = (i*m+j, j*p+k, k*n+i)
                    expected = Fraction(1, 1)
                    actual = T_comp.get(pos, Fraction(0, 1))
                    if actual != expected:
                        error += 1
        # Check that no extra non-zero entries exist
        for pos in T_comp:
            i, j, k = pos
            # Check if this position is a target position
            is_target = False
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        if (ii*m+jj == i and jj*p+kk == j and kk*n+ii == k):
                            is_target = True
                            break
                    if is_target:
                        break
                if is_target:
                    break
            if not is_target:
                error += 1
        return error
    
    # Try to find a rank-17 decomposition using simulated annealing
    found_rank17 = False
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(42 + restart)
            
            rank = 17
            U = random_matrix(n*m, rank, 0.25)
            V = random_matrix(m*p, rank, 0.25)
            W = random_matrix(p*n, rank, 0.25)
            
            current_error = tensor_error(compute_tensor(U, V, W))
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Simulated annealing
            T_start = 1.0
            T_end = 1e-3
            alpha = 0.998
            
            for it in range(3000):
                if time.time() > deadline - 25:
                    break
                
                T_curr = T_start * (alpha ** it)
                
                # Pick a random element to change
                choice = np.random.randint(0, 3)
                if choice == 0:
                    i, r = np.random.randint(n*m), np.random.randint(rank)
                    old_val = U[i, r]
                    U[i, r] = np.random.choice(candidate_values)
                elif choice == 1:
                    j, r = np.random.randint(m*p), np.random.randint(rank)
                    old_val = V[j, r]
                    V[j, r] = np.random.choice(candidate_values)
                else:
                    k, r = np.random.randint(p*n), np.random.randint(rank)
                    old_val = W[k, r]
                    W[k, r] = np.random.choice(candidate_values)
                
                new_error = tensor_error(compute_tensor(U, V, W))
                delta_e = new_error - current_error
                
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(T_curr, 1e-10)):
                    current_error = new_error
                    if new_error < best_error:
                        best_error = new_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_error == 0:
                            break
                else:
                    # Revert
                    if choice == 0:
                        U[i, r] = old_val
                    elif choice == 1:
                        V[j, r] = old_val
                    else:
                        W[k, r] = old_val
            
            if best_error == 0:
                if verify_decomposition(best_U, best_V, best_W):
                    best_result = {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
                    best_rank = 17
                    found_rank17 = True
                    break

    # If rank-17 not found, try rank-16 with a different strategy
    if not found_rank17 and time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(100 + restart)
            
            rank = 16
            U = random_matrix(n*m, rank, 0.2)
            V = random_matrix(m*p, rank, 0.2)
            W = random_matrix(p*n, rank, 0.2)
            
            current_error = tensor_error(compute_tensor(U, V, W))
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            T_start = 1.0
            T_end = 1e-3
            alpha = 0.998
            
            for it in range(2000):
                if time.time() > deadline - 15:
                    break
                
                T_curr = T_start * (alpha ** it)
                
                choice = np.random.randint(0, 3)
                if choice == 0:
                    i, r = np.random.randint(n*m), np.random.randint(rank)
                    old_val = U[i, r]
                    U[i, r] = np.random.choice(candidate_values)
                elif choice == 1:
                    j, r = np.random.randint(m*p), np.random.randint(rank)
                    old_val = V[j, r]
                    V[j, r] = np.random.choice(candidate_values)
                else:
                    k, r = np.random.randint(p*n), np.random.randint(rank)
                    old_val = W[k, r]
                    W[k, r] = np.random.choice(candidate_values)
                
                new_error = tensor_error(compute_tensor(U, V, W))
                delta_e = new_error - current_error
                
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(T_curr, 1e-10)):
                    current_error = new_error
                    if new_error < best_error:
                        best_error = new_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_error == 0:
                            break
                else:
                    if choice == 0:
                        U[i, r] = old_val
                    elif choice == 1:
                        V[j, r] = old_val
                    else:
                        W[k, r] = old_val
            
            if best_error == 0:
                if verify_decomposition(best_U, best_V, best_W):
                    best_result = {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
                    best_rank = 16
                    break

    return best_result
# EVOLVE-BLOCK-END
