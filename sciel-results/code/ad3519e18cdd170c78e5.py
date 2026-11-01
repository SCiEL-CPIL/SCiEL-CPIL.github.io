"""Matrix multiplication 2x4 x 4x4 tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def build_strassen_28():
        """Build rank-28 decomposition using Strassen on 2x2 blocks."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_U = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):1},
            2: {(0,0):1},
            3: {(1,1):1},
            4: {(0,0):1, (0,1):1},
            5: {(1,0):1, (0,0):-1},
            6: {(0,1):1, (1,1):-1},
        }
        strassen_V = {
            0: {(0,0):1, (1,1):1},
            1: {(0,0):1},
            2: {(1,0):1, (1,1):-1},
            3: {(0,1):1, (0,0):-1},
            4: {(1,1):1},
            5: {(0,0):1, (0,1):1},
            6: {(1,0):1, (1,1):1},
        }
        strassen_W = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):-1},
            2: {(0,1):1, (1,1):1},
            3: {(0,0):1, (1,0):1},
            4: {(0,0):-1, (0,1):1},
            5: {(1,1):1},
            6: {(0,0):1},
        }
        
        col = 0
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[r * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[ccol * 2 + r][c] += coeff
        col += 7
        
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[(r + 2) * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[ccol * 2 + r][c] += coeff
        col += 7
        
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[r * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[(ccol + 2) * 2 + r][c] += coeff
        col += 7
        
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[(r + 2) * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[(ccol + 2) * 2 + r][c] += coeff
        col += 7
        
        return {"U": U, "V": V, "W": W}

    def build_shared_strassen_27():
        """
        Build rank-27 decomposition by sharing Strassen terms across block products.
        
        Key insight: Instead of 4 independent 7-term Strassen decompositions (rank 28),
        we can structure the 2x4 x 4x4 product to share some bilinear terms.
        
        Approach: View the product as computing C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2,3}.
        
        We partition B into left half B_L (cols 0,1) and right half B_R (cols 2,3),
        and A into rows A_0 (row 0) and A_1 (row 1).
        
        For row 0 of C: C[0,k] = sum_j A[0,j]*B[j,k]
        For row 1 of C: C[1,k] = sum_j A[1,j]*B[j,k]
        
        Use a 4-term "mini-Strassen" on the 1x4 x 4x4 product per row,
        but share the B-side terms between rows.
        
        Alternative: Use the fact that we can write the 2x4*4x4 tensor as a sum
        of rank-1 tensors more efficiently than 28.
        
        Let's try a direct algebraic construction:
        C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
        C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
        C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2] + A[0,3]B[3,2]
        C[0,3] = A[0,0]B[0,3] + A[0,1]B[1,3] + A[0,2]B[2,3] + A[0,3]B[3,3]
        (similarly for row 1)
        
        We'll use a structured approach: 14 terms for row 0 (using Strassen-like
        on 1x4*4x4 viewed as 1x2*2x2 blocks) + 13 terms for row 1 sharing some.
        
        Actually, let's try a cleaner approach: 
        - Use 7 Strassen terms for A[0,:] * B (1x4 * 4x4)
        - Use 7 Strassen terms for A[1,:] * B (1x4 * 4x4)  
        - That gives 14, but we need to handle the 4x4 B differently.
        
        Better: Think of it as 2 independent 1x4*4x4 multiplications.
        Each 1x4*4x4 can be done in rank... let's think.
        
        1x4 * 4x4: C[k] = sum_j A[j]*B[j,k] for k=0..3
        This is 4 dot products of length 4. Schoolbook: 16 terms.
        Can we do better? The tensor is (1,4,4) with 16 nonzero entries.
        Minimum rank for 1x4*4x4: This is the "outer product" structure.
        Actually for 1xn*nxn, the rank is exactly n^2 because each output
        C[k] depends on all A[j]*B[j,k] independently.
        Wait no - the rank of the 1xn*nxn multiplication tensor is n^2.
        Proof: The tensor T_{j,k} = A_j * B_{j,k} has rank n^2 because
        the slices are linearly independent.
        
        So 1x4*4x4 has rank 16, and two of them would be 32. That's not helpful.
        
        Let me try a different decomposition. View 2x4*4x4 as:
        A is 2x4, B is 4x4. 
        Partition A into two 2x2 blocks: A = [A_L | A_R]
        Partition B into four 2x2 blocks: B = [[B_00, B_01], [B_10, B_11]]
        C = [A_L*B_00 + A_R*B_10 | A_L*B_01 + A_R*B_11]
        
        Each 2x2*2x2 needs rank 7 (Strassen). We have 4 such products.
        But we can share: the A_L terms appear in two products, A_R in two.
        
        Shared approach:
        Let M1..M7 be Strassen terms for (A_L, B_00)
        Let N1..N7 be Strassen terms for (A_R, B_10)
        Let P1..P7 be Strassen terms for (A_L, B_01)  
        Let Q1..Q7 be Strassen terms for (A_R, B_11)
        
        Total: 28. To reduce, we need to find linear dependencies.
        
        Key observation: The Strassen formula for X*Y uses:
        M1 = (X11+X22)(Y11+Y22)
        ...
        
        If we use the SAME A block (A_L) with two different B blocks (B_00 and B_01),
        we can potentially share the A-side linear combinations.
        
        Specifically, for A_L * B_00 and A_L * B_01:
        Both need A_L11, A_L12, A_L21, A_L22 and combinations like
        A_L11+A_L22, A_L11+A_L12, etc.
        
        We can precompute these A-side combinations and reuse them!
        
        Let's define shared A-side terms:
        a1 = A_L11 + A_L22  (used in M1 for both B_00 and B_01)
        a2 = A_L21 + A_L22  (used in M2 for both)
        a3 = A_L11          (used in M3 for both)
        a4 = A_L22          (used in M4 for both)
        a5 = A_L11 + A_L12  (used in M5 for both)
        a6 = A_L21 - A_L11  (used in M6 for both)
        a7 = A_L12 - A_L22  (used in M7 for both)
        
        Similarly for A_R:
        b1 = A_R11 + A_R22
        b2 = A_R21 + A_R22
        b3 = A_R11
        b4 = A_R22
        b5 = A_R11 + A_R12
        b6 = A_R21 - A_R11
        b7 = A_R12 - A_R22
        
        And B-side terms for each block are independent.
        
        This doesn't directly reduce the count because each product still needs
        7 separate scalar multiplications. The sharing is in the linear combinations,
        not in the number of scalar multiplications.
        
        Let me try yet another approach: use the Winograd-style or a direct
        rank-27 construction from the literature.
        
        For 2x4*4x4, the known minimum rank is actually 24 or 25 (I believe).
        Let me try a systematic construction.
        
        Actually, let's try: view 2x4*4x4 as a special case.
        C[i,k] = sum_j A[i,j] B[j,k]
        
        We can write this using 24 terms by clever grouping.
        Here's a construction:
        
        Group the 16 B entries into pairs and use Strassen-like tricks.
        
        Let me just try rank 27 by modifying the 28-term construction:
        In the 4 Strassen blocks, check if any two terms can be merged.
        
        For now, let's return the rank-28 construction and try random perturbation
        to find rank-27.
        """
        # For now, return None to indicate we need to search
        return None

    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check diagonal (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U_arr[i*m+j], V_arr[j*p+k]), W_arr[k*n+i])
                    if abs(s - 1) > 1e-9:
                        return False
        # Check off-diagonal (should be 0)
        for a in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b_idx // p
                    k1 = b_idx % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = np.dot(np.dot(U_arr[a], V_arr[b_idx]), W_arr[c_idx])
                    if abs(s) > 1e-9:
                        return False
        return True

    def validate_fast(U, V, W, rank):
        """Faster validation using matrix operations."""
        U_arr = np.array(U, dtype=float)  # (nm, rank)
        V_arr = np.array(V, dtype=float)  # (mp, rank)
        W_arr = np.array(W, dtype=float)  # (pn, rank)
        
        # For each column c, the contribution to the tensor is:
        # T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We need sum_c T_c[i,j,k] = 1 if (i,j,k) is a "matching" triple, else 0.
        
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposition tensor
        # Efficient: for each c, compute outer product structure
        # T = sum_c U[:,c] * V[:,c] * W[:,c] (element-wise in 3D)
        
        # Reshape for batch computation
        # T[a,b,c] = sum_col U[a,col]*V[b,col]*W[c,col]
        # = sum_col (U[a,col]*V[b,col]) * W[c,col]
        
        # Compute U*V element-wise: (nm, mp, rank)
        UV = U_arr[:, np.newaxis, :] * V_arr[np.newaxis, :, :]  # (nm, mp, rank)
        # Then T = UV * W broadcast: (nm, mp, rank) * (pn, rank) -> need to sum over rank
        # T[a,b,c] = sum_r UV[a,b,r] * W[c,r]
        
        # This is a batched matmul: for each (a,b), dot product with W[c,:]
        # T = UV @ W.T -> (nm, mp, pn)
        T = np.einsum('abr,cr->abc', UV, W_arr)
        
        return np.allclose(T, target, atol=1e-9)

    # Build and validate the rank-28 Strassen decomposition
    result_28 = build_strassen_28()
    valid_28 = validate_fast(result_28["U"], result_28["V"], result_28["W"], 28)
    
    if valid_28:
        best_result = result_28
        best_rank = 28
    else:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to find lower rank using structured search
    # Strategy: Start from rank-28 and try to find a rank-27 by
    # solving for one column as a function of others, or random search.
    
    found_lower = False
    
    # Try rank 27: structured perturbation of Strassen-28
    if time.time() < deadline - 10:
        U28 = np.array(best_result["U"], dtype=float)
        V28 = np.array(best_result["V"], dtype=float)
        W28 = np.array(best_result["W"], dtype=float)
        
        # Approach: try to find a rank-27 by random search with structure
        # Initialize from Strassen-28, drop one column, and solve
        # for corrections in remaining columns.
        
        # Simpler: random search with half-integer values, biased toward Strassen structure
        random.seed(42)
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate structured random rank-27 matrices
            # Start with Strassen-28 first 27 columns, add random perturbation
            if attempt % 3 == 0:
                # Try dropping each of the 28 columns from Strassen
                drop_col = attempt % 28
                U27 = np.delete(U28, drop_col, axis=1)
                V27 = np.delete(V28, drop_col, axis=1)
                W27 = np.delete(W28, drop_col, axis=1)
                if validate_fast(U27.tolist(), V27.tolist(), W27.tolist(), 27):
                    best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                    best_rank = 27
                    found_lower = True
                    break
            elif attempt % 3 == 1:
                # Random half-integer search
                vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                U = vals[np.random.randint(0, 5, size=(n*m, 27))]
                V = vals[np.random.randint(0, 5, size=(m*p, 27))]
                W = vals[np.random.randint(0, 5, size=(p*n, 27))]
                if validate_fast(U.tolist(), V.tolist(), W.tolist(), 27):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = 27
                    found_lower = True
                    break
            else:
                # Structured: modify Strassen-28 by zeroing one column and
                # adding small corrections
                U27 = U28[:, :27].copy()
                V27 = V28[:, :27].copy()
                W27 = W28[:, :27].copy()
                # Add random ±0.5 to random entries
                for _ in range(5):
                    ri = np.random.randint(0, n*m)
                    ci = np.random.randint(0, 27)
                    U27[ri, ci] += 0.5 * np.random.choice([-1, 1])
                    ri = np.random.randint(0, m*p)
                    ci = np.random.randint(0, 27)
                    V27[ri, ci] += 0.5 * np.random.choice([-1, 1])
                    ri = np.random.randint(0, p*n)
                    ci = np.random.randint(0, 27)
                    W27[ri, ci] += 0.5 * np.random.choice([-1, 1])
                if validate_fast(U27.tolist(), V27.tolist(), W27.tolist(), 27):
                    best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                    best_rank = 27
                    found_lower = True
                    break
    
    # Try rank 26 if we have time and haven't found 27
    if time.time() < deadline - 8 and best_rank > 26:
        found_26 = False
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 4:
                break
            
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = vals[np.random.randint(0, 5, size=(n*m, 26))]
            V = vals[np.random.randint(0, 5, size=(m*p, 26))]
            W = vals[np.random.randint(0, 5, size=(p*n, 26))]
            if validate_fast(U.tolist(), V.tolist(), W.tolist(), 26):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 26
                found_26 = True
                break
    
    return best_result

# EVOLVE-BLOCK-END
