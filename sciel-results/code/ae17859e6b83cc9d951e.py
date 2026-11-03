import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try Strassen-based decomposition
    result = _try_strassen_decomposition(n, m, p)
    if result is not None:
        best_rank = result["rank"]
        best_result = result
        return best_result
        
    return best_result

def _try_strassen_decomposition(n, m, p):
    # Only works for specific dimensions where we have a known construction
    if n != 3 or m != 4 or p != 5:
        return None
        
    # Use 4 disjoint 2x2x2 Strassen boxes + schoolbook for the rest
    # Each Strassen box handles 2x2x2 with rank 7 instead of rank 8
    # 4 boxes save 4 rank units: 60 - 4 = 56
    
    # Define the 4 disjoint 2x2x2 boxes
    # Box 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Box 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Box 3: i in {2}, j in {0,1}, k in {4}
    # Box 4: i in {2}, j in {2,3}, k in {4}
    
    # Actually, let's use a simpler approach:
    # Split the 3x4x5 into four 2x2x2 blocks where possible
    # But 3 is odd, so we can't perfectly split into 2x2x2 blocks.
    
    # Alternative: Use 2x2x2 Strassen for the (i,j,k) subcubes
    # Let's identify all maximal 2x2x2 subcubes that are disjoint
    
    # For i: 3 rows -> can form one 2x2 block (rows 0,1) and one 1x2 block (row 2)
    # For j: 4 cols -> can form two 2x2 blocks (cols 0,1 and cols 2,3)
    # For k: 5 rows -> can form two 2x2 blocks (rows 0,1 and rows 2,3) and one 1x2 block (row 4)
    
    # Disjoint 2x2x2 boxes:
    # Box A: i={0,1}, j={0,1}, k={0,1}
    # Box B: i={0,1}, j={2,3}, k={2,3}
    # These are the only two fully disjoint 2x2x2 boxes we can form.
    
    # For the remaining elements, use schoolbook.
    
    # Let's count:
    # Box A covers: i=0,1; j=0,1; k=0,1 -> 8 elements, rank 7 (Strassen)
    # Box B covers: i=0,1; j=2,3; k=2,3 -> 8 elements, rank 7 (Strassen)
    # Remaining: 60 - 16 = 44 elements, rank 44 (schoolbook)
    # Total rank: 7 + 7 + 44 = 58
    
    # Wait, that's only a saving of 2. Let me reconsider.
    
    # Actually, I can form more boxes by being more careful.
    # Let's use a different strategy: apply Strassen to every possible 2x2x2 subcube,
    # but ensure they don't overlap in the rank dimension.
    
    # Simpler approach: Just use Strassen for two 2x2x2 blocks and schoolbook for the rest.
    # Rank = 7 + 7 + 44 = 58
    
    # But the previous analysis suggested 56. Let me think again.
    
    # Alternative: Use 4 Strassen boxes by splitting differently.
    # Box 1: i={0,1}, j={0,1}, k={0,1}
    # Box 2: i={0,1}, j={0,1}, k={2,3}
    # Box 3: i={0,1}, j={2,3}, k={0,1}
    # Box 4: i={0,1}, j={2,3}, k={2,3}
    # These 4 boxes cover i={0,1}, all j, k={0,1,2,3}
    # That's 2*4*4 = 32 elements, rank 4*7 = 28
    # Remaining: i={2}, all j, all k = 1*4*5 = 20 elements, rank 20
    # Total: 28 + 20 = 48
    
    # That's much better! Let's verify:
    # The 4 boxes are disjoint in the (i,j,k) space.
    # Each box is a 2x2x2 subcube.
    # Strassen gives rank 7 for each.
    # The remaining part is i=2, j=0..3, k=0..4, which is 20 elements.
    # Schoolbook for those 20 elements.
    # Total rank: 4*7 + 20 = 48.
    
    # Let's implement this.
    
    U = np.zeros((n * m, 48))
    V = np.zeros((m * p, 48))
    W = np.zeros((p * n, 48))
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # M1 = (A11+A22)(B22-B21)
    # M2 = (A11+A12)B22
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A21)B11
    # M6 = (A21+A22)(B11-B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_U = np.array([
        [1, 1, 1, 0, 1, 0, 0],  # A11
        [0, 1, 0, 0, 0, 1, 1],  # A12
        [1, 0, 0, 1, 1, 1, -1], # A21
        [1, 0, 0, 1, 0, 1, -1]  # A22
    ])
    
    strassen_V = np.array([
        [-1, 0, 0, -1, 1, -1, 1], # B11
        [1, 0, 1, 0, 0, 1, 1],    # B12
        [-1, 0, 0, 1, 0, 0, 1],   # B21
        [1, 1, -1, 0, 0, 0, 0]    # B22
    ])
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ])
    
    # Box 1: i={0,1}, j={0,1}, k={0,1}
    # Local indices: i'=0,1; j'=0,1; k'=0,1
    # Global: i=i', j=j', k=k'
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + j_local, col + r] = strassen_U[i_local, j_local, r] if False else strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[j_local * p + k_local, col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[k_local * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Box 2: i={0,1}, j={0,1}, k={2,3}
    # Local: i'=0,1; j'=0,1; k'=0,1
    # Global: i=i', j=j', k=k'+2
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + j_local, col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[j_local * p + (k_local + 2), col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[(k_local + 2) * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Box 3: i={0,1}, j={2,3}, k={0,1}
    # Local: i'=0,1; j'=0,1; k'=0,1
    # Global: i=i', j=j'+2, k=k'
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + (j_local + 2), col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[(j_local + 2) * p + k_local, col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[k_local * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Box 4: i={0,1}, j={2,3}, k={2,3}
    # Local: i'=0,1; j'=0,1; k'=0,1
    # Global: i=i', j=j'+2, k=k'+2
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + (j_local + 2), col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[(j_local + 2) * p + (k_local + 2), col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[(k_local + 2) * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Remaining: i=2, j=0..3, k=0..4 (20 elements)
    # Schoolbook for these
    for i in [2]:
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    if col != 48:
        return None
    
    if not _verify(U, V, W, n, m, p):
        return None
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
        "rank": 48
    }

def _verify(U, V, W, n, m, p):
    # Sample verification
    np.random.seed(0)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for r in range(U.shape[1]):
                sum_U = 0.0
                for j in range(m):
                    sum_U += U[i * m + j, r] * A[i, j]
                sum_V = 0.0
                for j in range(m):
                    sum_V += V[j * p + k, r] * B[j, k]
                val += W[k * n + i, r] * sum_U * sum_V
            C_comp[i, k] = val
            
    if not np.allclose(C_true, C_comp, atol=1e-9):
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p):
    return _verify(U, V, W, n, m, p)
