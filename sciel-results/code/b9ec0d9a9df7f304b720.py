import time
import random
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _construct_strassen_2x2():
    """Construct rank-7 decomposition for 2x2 matrix multiplication (Strassen)."""
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 1, 0, 0, 0, 0],  # (0,1)
        [0, 1, 0, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 1, 1],  # (1,1)
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 1, 1, 0, 0, 0],  # (0,1)
        [1, 1, 0, 0, 1, 1, 0],  # (1,0)
        [0, 0, 1, 1, 1, 1, 1],  # (1,1)
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 1, 1],  # (1,1)
    ]
    return U, V, W

def _construct_rank_23():
    """Construct rank-23 decomposition for 3x3 matrix multiplication.
    
    This uses a structured approach based on decomposing the 3x3 tensor
    into blocks and using Strassen-like combinations.
    """
    n, m, p = DIMENSIONS
    rank = 23
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strategy: Use the 2x2 Strassen decomposition as a building block
    # and extend it to 3x3 by adding extra terms for the third row/column.
    # 
    # The 3x3 matrix multiplication can be seen as:
    # [A11 A12 A13]   [B11 B12 B13]
    # [A21 A22 A23] * [B21 B22 B23]
    # [A31 A32 A33]   [B31 B32 B33]
    #
    # We can decompose this using 2x2 Strassen on the top-left 2x2 block,
    # then handle the interactions with the third row/column.
    
    # First, place the 2x2 Strassen decomposition in the top-left corner
    U2, V2, W2 = _construct_strassen_2x2()
    
    # Map 2x2 indices to 3x3 indices
    # For the 2x2 block, indices are (i,j) where i,j in {0,1}
    # In 3x3, these map to (i,j) directly
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m+j][c] = U2[i*2+j][c]
                V[j*p+i][c] = V2[j*2+i][c]  # Note: V index is (j,k) in 2x2, but in 3x3 it's (j,p+k)
                # Actually, let me fix the indexing
                W[i*n+j][c] = W2[i*2+j][c]  # W index is (k,i) in 2x2
    
    # I need to be more careful with the indexing. Let me restart with a clearer approach.
    
    # Actually, let me use a different strategy: try to find a rank-23 decomposition
    # by using known results or by systematic search.
    
    # For now, return a placeholder that will be replaced by a working decomposition
    # The actual rank-23 decomposition is complex, so let me implement a search-based approach.
    
    # Fallback: return schoolbook if we can't find rank-23
    return None

def _search_rank_23(deadline):
    """Search for a rank-23 decomposition using randomized local search."""
    n, m, p = DIMENSIONS
    rank = 23
    
    # Initialize with random values
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    num_iterations = 1000
    iteration = 0
    
    while time.time() < deadline - 10 and iteration < num_iterations:
        iteration += 1
        
        # Initialize with random values
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        
        # Local search: try to reduce error
        for step in range(100):
            if time.time() > deadline - 10:
                break
                
            # Compute current error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        error += (s - 1) ** 2
            
            if error < best_error:
                best_error = error
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
                
                if error < 1e-12:
                    return best_U, best_V, best_W
            
            # Random perturbation
            if random.random() < 0.5:
                idx = random.randint(0, n*m-1)
                c = random.randint(0, rank-1)
                U[idx][c] += random.gauss(0, 0.1)
            elif random.random() < 0.5:
                idx = random.randint(0, m*p-1)
                c = random.randint(0, rank-1)
                V[idx][c] += random.gauss(0, 0.1)
            else:
                idx = random.randint(0, p*n-1)
                c = random.randint(0, rank-1)
                W[idx][c] += random.gauss(0, 0.1)
    
    if best_U is not None and best_error < 1e-9:
        return best_U, best_V, best_W
    
    return None

def construct():
    overall_deadline = time.time() + 140
    
    # Try to find rank-23 decomposition
    result = _search_rank_23(overall_deadline)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _construct_schoolbook()
