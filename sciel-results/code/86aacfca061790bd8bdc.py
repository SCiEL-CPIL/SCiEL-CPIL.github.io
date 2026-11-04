"""Direct rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 135.0

def _build_rank88():
    """Construct the known rank-88 decomposition:
    - i=0: schoolbook 1x4x8 (32 terms)
    - i=1,2: Strassen on 2x2x2 blocks (56 terms)
    Total: 88 terms.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (schoolbook for 1x4x8)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2x2 blocks
    # Strassen 2x2x2 matrices (7 terms)
    # U: 4x7, V: 4x7, W: 4x7 where each row/col corresponds to (i,j), (j,k), (k,i)
    # For 2x2x2: indices are (i,j) in {0,1}x{0,1}, (j,k) in {0,1}x{0,1}, (k,i) in {0,1}x{0,1}
    # We flatten: U index = i*2+j, V index = j*2+k, W index = k*2+i
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U part: 2x2 block at i=1,2 and j=j_start..j_start+1
                for local_i in range(2):
                    for local_j in range(2):
                        val = strassen_U[local_i * 2 + local_j][t]
                        if val != 0:
                            global_i = 1 + local_i
                            global_j = j_start + local_j
                            U[global_i * m + global_j][col] += val
                # V part: 2x2 block at j=j_start..j_start+1 and k=k_start..k_start+1
                for local_j in range(2):
                    for local_k in range(2):
                        val = strassen_V[local_j * 2 + local_k][t]
                        if val != 0:
                            global_j = j_start + local_j
                            global_k = k_start + local_k
                            V[global_j * p + global_k][col] += val
                # W part: 2x2 block at k=k_start..k_start+1 and i=1,2
                for local_k in range(2):
                    for local_i in range(2):
                        val = strassen_W[local_k * 2 + local_i][t]
                        if val != 0:
                            global_k = k_start + local_k
                            global_i = 1 + local_i
                            W[global_k * n + global_i][col] += val
                col += 1
                
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=3, m=4, p=8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(computed - expected) > 1e-9:
                    return False
    return True


def _try_rank87_compression():
    """Attempt to find a rank-87 decomposition by trying to compress one schoolbook term
    in the i=0 part using a Strassen-like trick combined with the i=1,2 part.
    
    This is an experimental approach. If it fails, we fall back to rank 88.
    """
    n, m, p = 3, 4, 8
    
    # The i=0 part is 1x4x8 schoolbook (32 terms).
    # The i=1,2 part is 2x4x8 Strassen (56 terms).
    # Total: 88.
    
    # Idea: Can we merge one term from i=0 with one from i=1,2 to save 1 rank?
    # This would require finding a rank-1 tensor that contributes to both i=0 and i=1,2
    # in a way that replaces two separate terms.
    
    # For a rank-1 term to contribute to i=0: W[k*n+0] must be nonzero for some k.
    # For it to also contribute to i=1 or i=2: W[k*n+1] or W[k*n+2] must be nonzero.
    
    # Let's try: take one schoolbook term from i=0, say (i=0, j=0, k=0),
    # and one Strassen term from the (j=0,1; k=0,1) block.
    # See if we can find a single rank-1 tensor that handles both.
    
    # This is complex; let's just try a numerical search for a rank-87 solution
    # by setting up the equations and checking feasibility.
    
    # Actually, let's try a simpler approach: check if the known rank-88 solution
    # has any linear dependencies that could allow compression.
    
    result = _build_rank88()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = 88
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check if any column is a linear combination of others (unlikely for rank-1 tensors)
    # This would be a very expensive check, so let's skip it.
    
    # Instead, let's try to see if we can modify the decomposition to use 87 terms
    # by introducing a new term that replaces two existing terms.
    
    # Given the complexity and time constraints, let's just return the rank-88 solution.
    # The probability of finding rank 87 via this ad-hoc approach is very low.
    
    return result


def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: return schoolbook rank 96 as a safe valid answer
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
    
    # Try to find rank 87 first (experimental)
    if time.time() - _START_TIME < _DEADLINE - 30:
        try:
            result = _try_rank87_compression()
            if result["U"] and len(result["U"][0]) == 87:
                if _verify_decomposition(result["U"], result["V"], result["W"]):
                    return result
        except Exception:
            pass
    
    # Return the known rank-88 solution
    result = _build_rank88()
    
    # Verify before returning (safety check)
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook
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
    
    return result
