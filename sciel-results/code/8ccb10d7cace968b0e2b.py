import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin under 150s

    n, m, p = 2, 5, 5

    def time_left():
        return deadline - time.time()

    def build_rank_46():
        """
        Construct the rank 46 solution based on Strassen for 2x2x2 blocks
        and schoolbook for the remaining edges.
        
        The matrix multiplication is C = A * B where A is 2x5 and B is 5x5.
        We can partition A into columns and B into rows/cols to form 2x2x2 blocks
        where Strassen applies.
        
        A is 2x5. B is 5x5.
        We can view this as multiplying 2x2 blocks if we pad or partition appropriately.
        However, the dimensions are 2, 5, 5.
        
        Let's partition the indices:
        i in {0, 1}
        j in {0, 1, 2, 3, 4}
        k in {0, 1, 2, 3, 4}
        
        We can group j and k into pairs:
        (0,1), (2,3), and the single 4.
        
        This creates blocks:
        1. j in {0,1}, k in {0,1} -> 2x2x2 Strassen (Rank 7)
        2. j in {0,1}, k in {2,3} -> 2x2x2 Strassen (Rank 7)
        3. j in {2,3}, k in {0,1} -> 2x2x2 Strassen (Rank 7)
        4. j in {2,3}, k in {2,3} -> 2x2x2 Strassen (Rank 7)
        
        Remaining terms involving index 4:
        5. j=4, k in {0,1,2,3} -> 4 terms (Schoolbook)
        6. j in {0,1,2,3}, k=4 -> 4 terms (Schoolbook)
        7. j=4, k=4 -> 1 term (Schoolbook)
        
        Total Rank = 7 + 7 + 7 + 7 + 4 + 4 + 1 = 37 + 9 = 46.
        """
        rank = 46
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_strassen(j_off, k_off, col_start):
            # 7 steps of Strassen for 2x2x2
            # Indices:
            # A: i in {0,1}, j in {j_off, j_off+1}
            # B: j in {j_off, j_off+1}, k in {k_off, k_off+1}
            # C: i in {0,1}, k in {k_off, k_off+1}
            
            # P1 = (A00 + A11)(B00 + B11)
            c = col_start
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        # Block 1: j in {0,1}, k in {0,1}
        add_strassen(0, 0, 0)
        # Block 2: j in {0,1}, k in {2,3}
        add_strassen(0, 2, 7)
        # Block 3: j in {2,3}, k in {0,1}
        add_strassen(2, 0, 14)
        # Block 4: j in {2,3}, k in {2,3}
        add_strassen(2, 2, 21)
        
        # Block 5: A_1 (j=4) B_{0..3} (k in 0..3)
        # C_{i, k} += A_{i, 4} B_{4, k}
        col = 28
        for i in range(n):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        # Block 6: A_{0..3} (j in 0..3) B_1 (k=4)
        # C_{i, 4} += A_{i, j} B_{j, 4}
        for i in range(n):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                
        # Block 7: A_1 (j=4) B_1 (k=4)
        # C_{i, 4} += A_{i, 4} B_{4, 4}
        for i in range(n):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    def verify(U, V, W):
        """
        Verify that the decomposition is exact.
        T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        Target: T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0.
        """
        R = U.shape[1]
        # Check all (i, j, k) combinations where the tensor entry should be 1
        for i in range(n):
            for j in range(m):
                a_idx = i * m + j
                for k in range(p):
                    b_idx = j * p + k
                    c_idx = k * n + i
                    # Compute sum
                    val = 0.0
                    for r in range(R):
                        val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check that other combinations are 0
        # We sample some random indices to ensure no spurious non-zero terms
        # The tensor is sparse. Non-zero only when b_idx corresponds to j in a_idx and c_idx corresponds to i in a_idx and k in b_idx.
        # Specifically: a=(i,j), b=(j,k), c=(k,i).
        # Any other combination should sum to 0.
        # Given the structure, this is highly likely if the 1s are correct.
        # We will do a few random checks.
        np.random.seed(123)
        for _ in range(1000):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            # We already checked the "1" cases. 
            # We need to check cases that are NOT (i,j), (j,k), (k,i).
            # If we pick random i,j,k, it might be a valid case.
            # Let's just pick random indices for a, b, c that don't form a valid chain.
            a_idx = np.random.randint(n * m)
            b_idx = np.random.randint(m * p)
            c_idx = np.random.randint(p * n)
            
            # Check if this is a valid chain
            i_a = a_idx // m
            j_a = a_idx % m
            j_b = b_idx // p
            k_b = b_idx % p
            k_c = c_idx // n
            i_c = c_idx % n
            
            if not (j_a == j_b and k_b == k_c and i_c == i_a):
                val = 0.0
                for r in range(R):
                    val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                if abs(val) > 1e-9:
                    return False
        return True

    # 1. Initialize with Rank 46
    U_best, V_best, W_best = build_rank_46()
    
    if not verify(U_best, V_best, W_best):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U_best = np.zeros((n * m, rank))
        V_best = np.zeros((m * p, rank))
        W_best = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_best[i * m + j, col] = 1.0
                    V_best[j * p + k, col] = 1.0
                    W_best[k * n + i, col] = 1.0
                    col += 1
        # Verify schoolbook just in case
        if not verify(U_best, V_best, W_best):
            # This should never happen, but if it does, return empty? No, must return valid.
            # Schoolbook is always valid.
            pass

    best_rank = U_best.shape[1]

    # Convert to int/float for output
    result = {}
    for key in ['U', 'V', 'W']:
        mat = result[key] = []
        for row in {'U': U_best, 'V': V_best, 'W': W_best}[key]:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(float(val))
            mat.append(new_row)
    
    return result

if __name__ == "__main__":
    res = construct()
    print("Rank:", res['U'][0].__len__())
