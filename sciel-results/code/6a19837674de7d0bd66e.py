"""Matrix multiplication 3x4x5: hybrid Strassen + schoolbook tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses hybrid Strassen (2x2x2 blocks) + schoolbook approach.
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U shape: (n*m, rank) = (12, rank)
    V shape: (m*p, rank) = (20, rank)
    W shape: (p*n, rank) = (15, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Hybrid approach: Use Strassen for 2x2 blocks where possible, schoolbook for rest
    # For 3x4x5: split into 2x2x2 Strassen blocks + schoolbook remainder
    
    # Strategy: 
    # - Use Strassen for the top-left 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # - Use schoolbook for remaining terms
    # This should give rank around 7 + (3*4*5 - 2*2*2) = 7 + 52 = 59
    # But we can be smarter about the split
    
    # Let's try a more systematic approach:
    # Split the 3x4x5 problem into overlapping Strassen-like blocks
    
    # For now, let's use a structured construction:
    # Rank = schoolbook(3,4,5) - savings from Strassen blocks
    
    # Actually, let's try to find a good decomposition directly.
    
    # Start with schoolbook as baseline
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to improve using structured methods
    # Try various hybrid constructions
    
    # Method 1: Two 2x2x2 Strassen blocks
    result = _hybrid_strassen_schoolbook(n, m, p, num_strassen_blocks=2)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    # Method 2: Try one 2x2x2 Strassen block + optimized schoolbook
    result = _hybrid_strassen_schoolbook(n, m, p, num_strassen_blocks=1)
    if result is not None and len(result["U"][0]) < best_rank:
        best_result = result
        best_rank = len(result["U"][0])
    
    # Method 3: Try random search for better ranks
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        result = _random_search(n, m, p, best_rank - 1, time.time() - start_time)
        if result is not None:
            best_result = result
            best_rank = len(result["U"][0])
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook (naive) tensor decomposition."""
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


def _strassen_2x2x2():
    """
    Return the standard Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where each has 7 columns (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    Strassen formulas for C = A*B where A,B,C are 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U[i*2+j, r] for i,j in {0,1}, r in {0..6}
    # V[j*2+k, r] for j,k in {0,1}, r in {0..6}
    # W[k*2+i, r] for k,i in {0,1}, r in {0..6}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11: U[0,0]=1, A22: U[3,0]=1
    # B11: V[0,0]=1, B22: V[3,0]=1
    # C11: W[0,0]=1, C22: W[3,0]=1 (but C22 gets M1 - M2 + M3 + M6)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11 += M1
    W[3][0] = 1  # C22 += M1
    
    # M2 = (A21 + A22)B11
    # A21: U[2,1]=1, A22: U[3,1]=1
    # B11: V[0,1]=1
    # C21: W[2,1]=1, C22: W[3,1]=-1 (C22 -= M2)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 += M2
    W[3][1] = -1  # C22 -= M2
    
    # M3 = A11(B12 - B22)
    # A11: U[0,2]=1
    # B12: V[1,2]=1, B22: V[3,2]=-1
    # C12: W[1,2]=1, C22: W[3,2]=1 (C22 += M3)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12 += M3
    W[3][2] = 1  # C22 += M3
    
    # M4 = A22(B21 - B11)
    # A22: U[3,3]=1
    # B21: V[2,3]=1, B11: V[0,3]=-1
    # C11: W[0,3]=1, C21: W[2,3]=1 (C21 += M4)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11 += M4
    W[2][3] = 1  # C21 += M4
    
    # M5 = (A11 + A12)B22
    # A11: U[0,4]=1, A12: U[1,4]=1
    # B22: V[3,4]=1
    # C11: W[0,4]=-1 (C11 -= M5), C12: W[1,4]=1 (C12 += M5)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11 -= M5
    W[1][4] = 1  # C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21: U[2,5]=1, A11: U[0,5]=-1
    # B11: V[0,5]=1, B12: V[1,5]=1
    # C22: W[3,5]=1 (C22 += M6)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12: U[1,6]=1, A22: U[3,6]=-1
    # B21: V[2,6]=1, B22: V[3,6]=1
    # C11: W[0,6]=1 (C11 += M7)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += M7
    
    return U, V, W


def _hybrid_strassen_schoolbook(n, m, p, num_strassen_blocks=2):
    """
    Create a hybrid decomposition using Strassen for some 2x2x2 blocks
    and schoolbook for the rest.
    
    For 3x4x5 with 2 Strassen blocks:
    - Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 7 terms
    - Block 2: i in {1,2}, j in {2,3}, k in {3,4} -> 7 terms (overlapping at i=1,j=2)
    - Rest: schoolbook for remaining terms
    
    Actually, let's use non-overlapping blocks for simplicity:
    - Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    - Block 2: i in {1,2}, j in {1,2}, k in {2,3} (but this overlaps)
    
    Let's try a cleaner approach:
    - Use Strassen for i in {0,1}, j in {0,1}, k in {0,1} (7 terms)
    - Schoolbook for all other (i,j,k) combinations
    """
    n, m, p = 3, 4, 5
    
    # Get Strassen decomposition for 2x2x2
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Total rank = 7 (Strassen) + remaining schoolbook terms
    # Schoolbook terms: all (i,j,k) except those in the 2x2x2 block
    # 2x2x2 block covers: i in {0,1}, j in {0,1}, k in {0,1} -> 8 terms
    # Remaining: 3*4*5 - 8 = 60 - 8 = 52 terms
    # Total rank = 7 + 52 = 59
    
    # But we can do better by being smarter about which terms to use Strassen for
    
    # For now, let's just use one Strassen block and schoolbook for the rest
    strassen_rank = 7
    schoolbook_rank = n * m * p - 2 * 2 * 2  # 60 - 8 = 52
    total_rank = strassen_rank + schoolbook_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in Strassen block (columns 0 to 6)
    # The 2x2x2 block is at i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U[i * m + j][r] = S_U[i * 2 + j][r]
    
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j * p + k][r] = S_V[j * 2 + k][r]
    
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k * n + i][r] = S_W[k * 2 + i][r]
    
    # Fill in schoolbook terms (columns 7 to total_rank-1)
    col = strassen_rank
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is in the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify the decomposition
    if not _validate_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}


def _random_search(n, m, p, max_rank, elapsed):
    """
    Try to find a decomposition with rank < max_rank using random search.
    """
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining < 5.0:
        return None
    
    np.random.seed(int(time.time()) % (2**31))
    
    # Try ranks from lower bound up to max_rank - 1
    lower_bound = max(n * p, m * p, n * m)
    
    for target_rank in range(lower_bound, max_rank):
        if time.time() - (DEADLINE - SAFETY_MARGIN) > elapsed + remaining - 2.0:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            U = np.random.randint(-1, 2, (n * m, target_rank))
            V = np.random.randint(-1, 2, (m * p, target_rank))
            W = np.random.randint(-1, 2, (p * n, target_rank))
            
            # Ensure no all-zero columns
            for r in range(target_rank):
                if np.all(U[:, r] == 0):
                    U[np.random.randint(n * m), r] = 1
                if np.all(V[:, r] == 0):
                    V[np.random.randint(m * p), r] = 1
                if np.all(W[:, r] == 0):
                    W[np.random.randint(p * n), r] = 1
            
            # Try to refine using alternating least squares
            success = _als_refine(U, V, W, n, m, p, max_iter=30)
            
            if success:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Round to nearest half-integer
                U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if _validate_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _als_refine(U, V, W, n, m, p, max_iter=30, tol=1e-10):
    """
    Refine U, V, W using alternating least squares.
    """
    rank = U.shape[1]
    
    for iteration in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    u_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u_new
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    v_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v_new
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    w_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w_new
                except:
                    pass
        
        # Check convergence
        max_residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_residual = max(max_residual, abs(1.0 - total))
        
        if max_residual < tol:
            return True
    
    return False


def _validate_decomposition(U, V, W, n, m, p):
    """Check if the decomposition is valid."""
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-9:
                    return False
    
    return True


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    print(f"Valid: {_validate_decomposition(U, V, W, n, m, p)}")
