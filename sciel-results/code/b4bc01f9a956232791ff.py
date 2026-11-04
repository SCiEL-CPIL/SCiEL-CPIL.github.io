import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms of the rank-7 decomposition for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) where each vector has length 4 (flattened 2x2).
    U is for A (2x2), V is for B (2x2), W is for C (2x2).
    The tensor is T[i,j,k] = sum_t U_t[i*2+j] * V_t[j*2+k] * W_t[k*2+i] = 1 for all i,j,k in {0,1}.
    """
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    # C = [[i, j], [k, l]] -> indices: i=0, j=1, k=2, l=3
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])  # M7
    ]
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(rank):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomp():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication using Strassen's 2x2 algorithm.
    This treats 4x4 matrices as 2x2 blocks of 2x2 matrices.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    base_terms = _strassen_2x2_terms()
    
    # Block positions for 4x4 as 2x2 blocks:
    # Block (0,0): rows 0-1, cols 0-1
    # Block (0,1): rows 0-1, cols 2-3
    # Block (1,0): rows 2-3, cols 0-1
    # Block (1,1): rows 2-3, cols 2-3
    
    # For block matrix multiplication C = A * B where A, B, C are 4x4:
    # C[block_i, block_j] = sum_block_k A[block_i, block_k] * B[block_k, block_j]
    # But Strassen computes 2x2 block multiplication in 7 products.
    # So we have 7 block-level products, each involving 2x2 blocks.
    # Each block product is a 2x2 * 2x2 = 2x2 multiplication, which itself has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # The 7 Strassen terms for the 2x2 block structure:
    # Let's denote the 2x2 block indices as (r, c) where r, c in {0, 1}.
    # A blocks: A[0,0], A[0,1], A[1,0], A[1,1]
    # B blocks: B[0,0], B[0,1], B[1,0], B[1,1]
    # C blocks: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen's 7 products for the 2x2 block structure:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[1,1] - A[0,1]) * (B[1,0] + B[1,1])
    
    # Each M_t is a 2x2 matrix (block). It contributes to specific C blocks with coefficients.
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    
    # For each block-level Strassen term, we need to expand it into 7 rank-1 terms
    # for the underlying 2x2 matrix multiplication.
    
    # Let's define the block-level Strassen terms with their A and B block coefficients.
    # Each term is: (coeffs_A, coeffs_B, coeffs_C)
    # where coeffs_A is a 2x2 matrix of coefficients for the A blocks,
    # coeffs_B is a 2x2 matrix of coefficients for the B blocks,
    # coeffs_C is a 2x2 matrix of coefficients for the C blocks.
    
    block_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
        # M4
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6
        ([[0, 0], [1, -1]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7
        ([[0, 0], [1, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 1]])
    ]
    
    col = 0
    for t_block in range(7):
        A_coeffs, B_coeffs, C_coeffs = block_terms[t_block]
        for t_inner in range(7):
            U_inner, V_inner, W_inner = base_terms[t_inner]
            
            for r in range(2):
                for s in range(2):
                    if A_coeffs[r][s] != 0:
                        for dr in range(2):
                            for dc in range(2):
                                u_idx = (2*r + dr) * 4 + (2*s + dc)
                                u_inner_idx = dr * 2 + dc
                                U[u_idx][col] += A_coeffs[r][s] * U_inner[u_inner_idx]
            
            for s in range(2):
                for t in range(2):
                    if B_coeffs[s][t] != 0:
                        for dr in range(2):
                            for dc in range(2):
                                v_idx = (2*s + dr) * 4 + (2*t + dc)
                                v_inner_idx = dr * 2 + dc
                                V[v_idx][col] += B_coeffs[s][t] * V_inner[v_inner_idx]
            
            for t in range(2):
                for r in range(2):
                    if C_coeffs[t][r] != 0:
                        for dr in range(2):
                            for dc in range(2):
                                w_idx = (2*t + dc) * 4 + (2*r + dr)
                                w_inner_idx = dr * 2 + dc
                                W[w_idx][col] += C_coeffs[t][r] * W_inner[w_inner_idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_443():
    """
    Schoolbook decomposition for 4x4 * 4x3 matrix multiplication.
    Rank = 4*4*3 = 48.
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
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

def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions:
    decomp1: for n x m * m x p1, with U1 (n*m x r1), V1 (m*p1 x r1), W1 (p1*n x r1)
    decomp2: for n x m * m x p2, with U2 (n*m x r2), V2 (m*p2 x r2), W2 (p2*n x r2)
    
    Result: for n x m * m x (p1+p2), with U (n*m x (r1+r2)), V (m*(p1+p2) x (r1+r2)), W ((p1+p2)*n x (r1+r2))
    
    U = [U1, U2] (horizontal concat)
    V = [V1_top, V2_top; V1_bottom, V2_bottom] -> actually V1 is (m*p1 x r1), V2 is (m*p2 x r2)
    We need V to be (m*(p1+p2) x (r1+r2)).
    The first m*p1 rows of V correspond to B's first p1 columns, and the next m*p2 rows to B's last p2 columns.
    So V = [[V1, 0], [0, V2]]
    
    W = [W1, W2] (vertical concat) -> W1 is (p1*n x r1), W2 is (p2*n x r2)
    W = [[W1, 0], [0, W2]]
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * (p1 + p2))]
    W = [[0] * r for _ in range((p1 + p2) * n)]
    
    # Copy U1 and U2
    for i in range(n * m):
        for c in range(r1):
            U[i][c] = decomp1["U"][i][c]
        for c in range(r2):
            U[i][r1 + c] = decomp2["U"][i][c]
    
    # Copy V1 and V2
    for i in range(m * p1):
        for c in range(r1):
            V[i][c] = decomp1["V"][i][c]
    for i in range(m * p2):
        for c in range(r2):
            V[m * p1 + i][r1 + c] = decomp2["V"][i][c]
    
    # Copy W1 and W2
    for i in range(p1 * n):
        for c in range(r1):
            W[i][c] = decomp1["W"][i][c]
    for i in range(p2 * n):
        for c in range(r2):
            W[p1 * n + i][r1 + c] = decomp2["W"][i][c]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_447():
    """
    Schoolbook decomposition for 4x4 * 4x7 matrix multiplication.
    Rank = 4*4*7 = 112.
    """
    n, m, p = 4, 4, 7
    rank = n * m * p
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

def _search_2x2x7_decomp(deadline):
    """
    Search for a low-rank decomposition of 2x2 * 2x7 matrix multiplication.
    Schoolbook rank = 2*2*7 = 28.
    We try to find a decomposition with rank < 28 using random initialization and iterative refinement.
    
    Returns a dict with U (4 x rank), V (14 x rank), W (14 x rank) or None if not found.
    """
    n, m, p = 2, 2, 7
    
    # Try ranks from 20 to 27
    for target_rank in range(20, 28):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Iterative refinement
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Compute the current tensor approximation
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            errors.append((i, j, k, s - 1.0))
                
                max_err = max(abs(e[3]) for e in errors)
                if max_err < 1e-9:
                    # Found a valid decomposition
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                # Simple gradient-free update: perturb the variable with the largest error
                worst = max(errors, key=lambda e: abs(e[3]))
                i, j, k, err = worst
                # Perturb one of the three vectors
                var_choice = random.randint(0, 2)
                if var_choice == 0:
                    idx = i*m + j
                    U[idx, random.randint(0, target_rank-1)] += random.choice([-1, 1])
                elif var_choice == 1:
                    idx = j*p + k
                    V[idx, random.randint(0, target_rank-1)] += random.choice([-1, 1])
                else:
                    idx = k*n + i
                    W[idx, random.randint(0, target_rank-1)] += random.choice([-1, 1])
    
    return None

def _combine_2x2x7_decomps(decomp1, decomp2, n, m, p):
    """
    Combine two 2x2 * 2xp decompositions into a 4x2 * 2xp decomposition.
    decomp1: for rows 0-1 of A, U1 (4 x r1), V1 (2*p x r1), W1 (p*2 x r1)
    decomp2: for rows 2-3 of A, U2 (4 x r2), V2 (2*p x r2), W2 (p*2 x r2)
    
    Result: for 4x2 * 2xp, U (8 x r1+r2), V (2*p x r1+r2), W (p*4 x r1+r2)
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [[0] * r for _ in range(n * m)]  # 8 x r
    V = [[0] * r for _ in range(m * p)]  # 14 x r
    W = [[0] * r for _ in range(p * n)]  # 28 x r
    
    # Copy U1 (rows 0-3) and U2 (rows 4-7)
    for i in range(4):
        for c in range(r1):
            U[i][c] = decomp1["U"][i][c]
    for i in range(4):
        for c in range(r2):
            U[4 + i][r1 + c] = decomp2["U"][i][c]
    
    # Copy V1 and V2 (same shape)
    for i in range(m * p):
        for c in range(r1):
            V[i][c] = decomp1["V"][i][c]
        for c in range(r2):
            V[i][r1 + c] = decomp2["V"][i][c]
    
    # Copy W1 and W2
    # W1 is for p*2 entries (k*n + i where n=2, i in {0,1})
    # W2 is for p*2 entries
    # Combined W is p*4 entries (k*4 + i where i in {0,1,2,3})
    # W1 corresponds to i=0,1 -> indices k*2 + 0, k*2 + 1
    # W2 corresponds to i=2,3 -> indices k*2 + 2, k*2 + 3
    for k in range(p):
        for i in range(2):
            for c in range(r1):
                W[k*4 + i][c] = decomp1["W"][k*2 + i][c]
        for i in range(2):
            for c in range(r2):
                W[k*4 + 2 + i][r1 + c] = decomp2["W"][k*2 + i][r1 + c]
    
    return {"U": U, "V": V, "W": W}

def _search_4x2x7_decomp(deadline):
    """
    Search for a low-rank decomposition of 4x2 * 2x7 matrix multiplication.
    Schoolbook rank = 4*2*7 = 56.
    We try to find a decomposition with rank < 56 using random initialization and iterative refinement.
    
    Returns a dict with U (8 x rank), V (14 x rank), W (28 x rank) or None if not found.
    """
    n, m, p = 4, 2, 7
    
    # Try ranks from 40 to 55
    for target_rank in range(40, 56):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute the current tensor approximation
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            errors.append((i, j, k, s - 1.0))
                
                max_err = max(abs(e[3]) for e in errors)
                if max_err < 1e-9:
                    # Found a valid decomposition
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                # Simple gradient-free update: perturb the variable with the largest error
                worst = max(errors, key=lambda e: abs(e[3]))
                i, j, k, err = worst
                var_choice = random.randint(0, 2)
                if var_choice == 0:
                    idx = i*m + j
                    U[idx, random.randint(0, target_rank-1)] += random.choice([-1, 1])
                elif var_choice == 1:
                    idx = j*p + k
                    V[idx, random.randint(0, target_rank-1)] += random.choice([-1, 1])
                else:
                    idx = k*n + i
                    W[idx, random.randint(0, target_rank-1)] += random.choice([-1, 1])
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Start with schoolbook (rank 112)
    result = _schoolbook_447()
    
    # Try hybrid: Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x3 (rank 48) = rank 97
    try:
        decomp_444 = _strassen_4x4_decomp()
        decomp_443 = _schoolbook_443()
        hybrid = _combine_decomps(decomp_444, decomp_443, n, m, 4, 3)
        
        # Verify
        if _verify_decomposition(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            result = hybrid
    except Exception:
        pass
    
    # Try to improve using the 2x2x7 search strategy
    # Decompose 4x4x7 into two 2x4x7 sub-problems (splitting rows of A)
    # Then further decompose each 2x4x7 into two 2x2x7 sub-problems (splitting columns of A)
    # But this is complex. Let's try a simpler approach:
    # Search for a low-rank 4x2x7 decomposition and combine with Strassen for 4x2x4
    
    if time.time() < deadline - 30:
        try:
            # Search for 4x2x7 decomposition
            decomp_427 = _search_4x2x7_decomp(deadline - 10)
            if decomp_427 is not None:
                # We need to combine this with a 4x2x4 decomposition
                # For 4x2x4, we can use Strassen on 2x2 blocks
                # But we need a 4x2x4 decomposition. Let's use schoolbook for now (rank 32)
                # Or better, use the 2x2x7 search for the 4x2x4 part
                # Actually, let's just use the 4x2x7 result and combine with a 4x2x3 schoolbook
                # Total rank = rank(4x2x7) + 4*2*3 = rank(4x2x7) + 24
                
                # For now, let's just verify if we can use this directly
                # The 4x2x7 decomposition has U (8 x r), V (14 x r), W (28 x r)
                # We need to embed this into the 4x4x7 problem
                # This is non-trivial, so let's skip for now
                pass
        except Exception:
            pass
    
    return result
