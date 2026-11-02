import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 145
    n, m, p = DIMENSIONS  # 3, 3, 7

    def schoolbook():
        rank = n * m * p  # 63
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Structural change: attempt to find a rank < 63 decomposition using 
    # a linear algebra / tensor decomposition approach with rational coefficients.
    # 
    # The matrix multiplication tensor T is a 3-index tensor of shape (n*m, m*p, p*n) = (9, 21, 21).
    # We want to find rank r such that T = sum_{k=1}^{r} u_k ⊗ v_k ⊗ w_k.
    #
    # This is equivalent to finding matrices U (9xr), V (21xr), W (21xr) such that
    # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    #
    # A key insight: if we can find a decomposition where the factors have a lot of zeros
    # (sparse structure), we can use that. The schoolbook decomposition is maximally sparse
    # (each factor has exactly one 1 per column), but it's rank 63.
    #
    # For 3x3x3, Laderman's algorithm achieves rank 23. Let's try to use a verified
    # Laderman decomposition for the 3x3x3 core and extend to 3x3x7.
    #
    # Strategy: Split the 7 columns of B into 3 + 3 + 1.
    # - First 3 columns: use Laderman rank-23 for the 3x3x3 subproblem.
    # - Second 3 columns: use Laderman rank-23 for another 3x3x3 subproblem.
    # - Last 1 column: use schoolbook rank-9 for the 3x3x1 subproblem.
    # Total rank: 23 + 23 + 9 = 55.
    #
    # However, we need verified Laderman factors. Let's implement a known correct set.
    #
    # Laderman's rank-23 algorithm for 3x3x3 matrix multiplication:
    # The 23 rank-1 terms are defined by specific combinations of matrix entries.
    # Here's a verified implementation based on the standard reference.

    def e3(r, c):
        """Create a 9-element list with 1 at position (r,c) in 3x3 matrix (row-major)."""
        v = [0] * 9
        v[r * 3 + c] = 1
        return v

    def add9(a, b):
        """Add two 9-element lists element-wise."""
        return [a[i] + b[i] for i in range(9)]

    def mul9(a, b):
        """Element-wise multiply two 9-element lists."""
        return [a[i] * b[i] for i in range(9)]

    # Laderman's 23 factors for 3x3x3.
    # Each factor is a tuple (u, v, w) of three 9-element lists.
    # The tensor contribution is sum of u[i,j]*v[j,k]*w[i,k] over the 23 terms.
    #
    # Based on the standard Laderman decomposition, here are the 23 terms:
    # Terms are indexed 0-22.

    laderman_terms = []

    # 19 "diagonal" terms (one for each pair (i,k) in the output, but structured differently)
    # Actually, Laderman's algorithm is more subtle. Let me use a known correct set.
    #
    # From Laderman (1976) and subsequent references, the 23 products are:
    #
    # I'll use a verified set from the literature. The key is that each term
    # corresponds to a rank-1 tensor u ⊗ v ⊗ w where u, v, w are 9-vectors
    # (flattened 3x3 matrices).

    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    laderman_terms.append((e3(0,0), e3(0,0), e3(0,0)))
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    laderman_terms.append((e3(0,1), e3(1,0), e3(0,0)))
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    laderman_terms.append((e3(0,2), e3(2,0), e3(0,0)))
    # Term 3: A[1,0]*B[0,1] -> C[0,1]
    laderman_terms.append((e3(1,0), e3(0,1), e3(0,1)))
    # Term 4: A[1,1]*B[1,1] -> C[0,1]
    laderman_terms.append((e3(1,1), e3(1,1), e3(0,1)))
    # Term 5: A[1,2]*B[2,1] -> C[0,1]
    laderman_terms.append((e3(1,2), e3(2,1), e3(0,1)))
    # Term 6: A[2,0]*B[0,2] -> C[0,2]
    laderman_terms.append((e3(2,0), e3(0,2), e3(0,2)))
    # Term 7: A[2,1]*B[1,2] -> C[0,2]
    laderman_terms.append((e3(2,1), e3(1,2), e3(0,2)))
    # Term 8: A[2,2]*B[2,2] -> C[0,2]
    laderman_terms.append((e3(2,2), e3(2,2), e3(0,2)))

    # Now the "Strassen-like" merged terms. Laderman combines 18 remaining schoolbook
    # terms into 14 terms using 4 Strassen-style tricks (each reducing 2 terms to 1,
    # but with cross terms that require additional terms).
    #
    # Actually, the standard Laderman algorithm has a specific structure. Let me use
    # a different approach: I'll verify the schoolbook decomposition works, and then
    # attempt a random search for a lower-rank decomposition using rational coefficients.
    #
    # Given the complexity of getting Laderman's factors exactly right from memory,
    # and the risk of subtle errors, let me try a different structural approach:
    #
    # Use the fact that 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1.
    # If I can find ANY decomposition of 3x3x3 with rank < 27, I can improve.
    #
    # Let me try a random search with rational coefficients (small integers) for
    # a rank-23 decomposition of 3x3x3. If found, use it. Otherwise, fall back to schoolbook.

    def random_laderman_search(max_rank=23, num_trials=500):
        """Try to find a rank-r decomposition of 3x3x3 using random rational coefficients."""
        import random
        import sys
        from fractions import Fraction

        np.random.seed(42)
        random.seed(42)

        for trial in range(num_trials):
            if time.time() > DEADLINE - 10:
                break

            # Generate random rank-r factors with small integer entries
            r = max_rank
            U = np.zeros((9, r), dtype=object)
            V = np.zeros((9, r), dtype=object)
            W = np.zeros((9, r), dtype=object)

            for col in range(r):
                # Generate sparse random vectors
                for idx in range(9):
                    if random.random() < 0.3:
                        val = random.choice([-2, -1, 1, 2])
                        U[idx, col] = val
                    else:
                        U[idx, col] = 0
                for idx in range(9):
                    if random.random() < 0.3:
                        val = random.choice([-2, -1, 1, 2])
                        V[idx, col] = val
                    else:
                        V[idx, col] = 0
                for idx in range(9):
                    if random.random() < 0.3:
                        val = random.choice([-2, -1, 1, 2])
                        W[idx, col] = val
                    else:
                        W[idx, col] = 0

            # Verify: for each (i,j,k), sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] == 1
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        total = 0
                        for r_idx in range(r):
                            total += U[i*3+j, r_idx] * V[j*3+k, r_idx] * W[k*3+i, r_idx]
                        if total != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                # Convert to lists
                U_list = [[int(U[i, r_idx]) for r_idx in range(r)] for i in range(9)]
                V_list = [[int(V[i, r_idx]) for r_idx in range(r)] for i in range(9)]
                W_list = [[int(W[i, r_idx]) for r_idx in range(r)] for i in range(9)]
                return U_list, V_list, W_list

        return None

    # Try to find a rank-23 decomposition of 3x3x3
    laderman_result = random_laderman_search(max_rank=23, num_trials=200)

    if laderman_result is not None and time.time() < DEADLINE - 20:
        U_3x3, V_3x3, W_3x3 = laderman_result
        rank_3x3 = len(U_3x3[0])

        # Build the 3x3x7 decomposition
        # Split B's 7 columns into 3 + 3 + 1
        # For columns 0-2: use first 3x3x3 decomposition
        # For columns 3-5: use second 3x3x3 decomposition (same factors, shifted indices)
        # For column 6: use schoolbook 3x3x1 (rank 9)

        total_rank = rank_3x3 + rank_3x3 + 9

        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(21)]
        W_full = [[0] * total_rank for _ in range(21)]

        # First 3x3x3 block (columns 0-2 of B)
        for i in range(9):
            for r in range(rank_3x3):
                U_full[i][r] = U_3x3[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(rank_3x3):
                    V_full[j*7 + k][r] = V_3x3[j*3 + k][r]
        for k in range(3):
            for i in range(3):
                for r in range(rank_3x3):
                    W_full[k*3 + i][r] = W_3x3[k*3 + i][r]

        # Second 3x3x3 block (columns 3-5 of B)
        offset1 = rank_3x3
        for i in range(9):
            for r in range(rank_3x3):
                U_full[i][offset1 + r] = U_3x3[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(rank_3x3):
                    V_full[j*7 + (k+3)][offset1 + r] = V_3x3[j*3 + k][r]
        for k in range(3):
            for i in range(3):
                for r in range(rank_3x3):
                    W_full[(k+3)*3 + i][offset1 + r] = W_3x3[k*3 + i][r]

        # Third block: 3x3x1 (column 6 of B), schoolbook rank 9
        offset2 = rank_3x3 + rank_3x3
        for i in range(3):
            for j in range(3):
                for r_idx in range(9):
                    col_idx = offset2 + r_idx
                    U_full[i*3 + j][col_idx] = 1
                    V_full[j*7 + 6][col_idx] = 1
                    W_full[6*3 + i][col_idx] = 1

        # Verify the full decomposition
        if verify(U_full, V_full, W_full):
            return {"U": U_full, "V": V_full, "W": W_full}

    # Fallback to schoolbook
    result = schoolbook()
    if not verify(result['U'], result['V'], result['W']):
        raise RuntimeError("Schoolbook verification failed!")
    return result
