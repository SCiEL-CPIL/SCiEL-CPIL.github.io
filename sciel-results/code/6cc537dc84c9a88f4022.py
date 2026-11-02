"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Returns (U, V, W) as numpy arrays of shape (4,14), (8,14), (8,14).
        """
        R = 14
        U = np.zeros((4, R), dtype=int)
        V = np.zeros((8, R), dtype=int)
        W = np.zeros((8, R), dtype=int)

        # Left block (cols 0,1 of B): Strassen on A * B_left where B_left is 2x2
        # B_left[0,0]=b00, B_left[0,1]=b01, B_left[1,0]=b10, B_left[1,1]=b11
        # V indices: j*p+k -> j*4+k
        #   b00 -> V[0], b01 -> V[1], b10 -> V[4], b11 -> V[5]
        # W indices: k*n+i -> k*2+i
        #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]

        # L1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[5,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # L2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1

        # L3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[5,2] = -1
        W[2,2] = 1; W[3,2] = 1

        # L4: a11(b10-b00)
        U[3,3] = 1
        V[4,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1

        # L5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[5,4] = 1
        W[0,4] = -1; W[2,4] = 1

        # L6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # L7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[4,6] = 1; V[5,6] = 1
        W[0,6] = 1

        # Right block (cols 2,3 of B): Strassen on A * B_right where B_right is 2x2
        # B_right[0,0]=b02, B_right[0,1]=b03, B_right[1,0]=b12, B_right[1,1]=b13
        # V indices:
        #   b02 -> V[2], b03 -> V[3], b12 -> V[6], b13 -> V[7]
        # W indices:
        #   C[0,2] -> W[4], C[0,3] -> W[5], C[1,2] -> W[6], C[1,3] -> W[7]

        # R1: (a00+a11)(b02+b13)
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[7,7] = 1

        # R2: (a10+a11)b02
        U[2,8] = 1; U[3,8] = 1
        V[2,8] = 1
        W[5,8] = 1; W[7,8] = -1

        # R3: a00(b03-b13)
        U[0,9] = 1
        V[3,9] = 1; V[7,9] = -1
        W[6,9] = 1; W[7,9] = 1

        # R4: a11(b12-b02)
        U[3,10] = 1
        V[6,10] = 1; V[2,10] = -1
        W[4,10] = 1; W[5,10] = 1

        # R5: (a00+a01)b13
        U[0,11] = 1; U[1,11] = 1
        V[7,11] = 1
        W[4,11] = -1; W[6,11] = 1

        # R6: (a10-a00)(b02+b03)
        U[2,12] = 1; U[0,12] = -1
        V[2,12] = 1; V[3,12] = 1
        W[7,12] = 1

        # R7: (a01-a11)(b12+b13)
        U[1,13] = 1; U[3,13] = -1
        V[6,13] = 1; V[7,13] = 1
        W[4,13] = 1

        return U, V, W

    def verify_fast(U, V, W):
        """
        Fast verification using matrix multiplication.
        For each rank-r term: u (n x m), v (m x p), w (p x n)
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        We can verify by checking T equals 1 everywhere.
        """
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        
        # Build the tensor T
        T = np.zeros((n, m, p), dtype=float)
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_einsum(U, V, W):
        """Verification using einsum for speed."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=1e-9)

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def try_rank13_search(max_iterations=5000):
        """
        Try to find a rank-13 decomposition by searching over structured patterns.
        Uses the fact that rank-13 would require some sharing between the two Strassen blocks.
        """
        random.seed(42)
        
        # The rank-14 solution has two independent Strassen blocks.
        # For rank-13, we'd need to merge one term from each block or find a different structure.
        
        # Strategy: Try random perturbations of the rank-14 solution and check if we can
        # eliminate one rank-1 term while maintaining correctness.
        
        U14, V14, W14 = get_rank14()
        
        # Try removing each rank-1 term and see if the remaining 13 terms can be adjusted
        for r_to_remove in range(14):
            if time.time() > deadline - 30:
                break
                
            # Create a copy without the r_to_remove term
            U = np.zeros((4, 13), dtype=float)
            V = np.zeros((8, 13), dtype=float)
            W = np.zeros((8, 13), dtype=float)
            
            col = 0
            for r in range(14):
                if r != r_to_remove:
                    U[:, col] = U14[:, r]
                    V[:, col] = V14[:, r]
                    W[:, col] = W14[:, r]
                    col += 1
            
            # Check if the remaining 13 terms already work (unlikely but check)
            if verify_einsum(U.tolist(), V.tolist(), W.tolist()):
                return U, V, W
            
            # Try small integer adjustments to the remaining terms
            # This is a heuristic search - try changing one entry at a time
            for _ in range(100):
                if time.time() > deadline - 30:
                    break
                
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                
                # Pick a random position to modify
                matrix_choice = random.randint(0, 2)
                if matrix_choice == 0:
                    row = random.randint(0, 3)
                    col = random.randint(0, 12)
                    delta = random.choice([-1, 1])
                    U_test[row, col] += delta
                elif matrix_choice == 1:
                    row = random.randint(0, 7)
                    col = random.randint(0, 12)
                    delta = random.choice([-1, 1])
                    V_test[row, col] += delta
                else:
                    row = random.randint(0, 7)
                    col = random.randint(0, 12)
                    delta = random.choice([-1, 1])
                    W_test[row, col] += delta
                
                # Check if all entries are integers or half-integers
                valid = True
                for arr in [U_test, V_test, W_test]:
                    for val in arr.flat:
                        if abs(val * 2 - round(val * 2)) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                if verify_einsum(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                    # Convert to integer/half-integer representation
                    U_int = (2 * U_test).astype(int) // 2
                    V_int = (2 * V_test).astype(int) // 2
                    W_int = (2 * W_test).astype(int) // 2
                    return U_int, V_int, W_int
        
        return None

    # First, try the known rank-14 solution
    U, V, W = get_rank14()
    
    # Verify the rank-14 solution
    if verify_einsum(U.tolist(), V.tolist(), W.tolist()):
        rank14_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank14_solution = None
        # Fallback to naive if rank-14 fails verification
        rank14_solution = naive_decomp()

    # Try to find a rank-13 solution (improvement)
    if time.time() < deadline - 35:
        result13 = try_rank13_search()
        if result13 is not None:
            U13, V13, W13 = result13
            if verify_einsum(U13.tolist(), V13.tolist(), W13.tolist()):
                return {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}

    # Return the best solution found (rank-14 or fallback)
    return rank14_solution

# EVOLVE-BLOCK-END
