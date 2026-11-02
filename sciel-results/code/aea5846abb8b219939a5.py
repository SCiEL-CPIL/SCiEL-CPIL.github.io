import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for val in mat.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
    
    # Reconstruct tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But wait - the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Rank-1 decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Compute the decomposition tensor
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _round_to_half(vals):
    """Round values to nearest half-integer."""
    return [v / 2.0 for v in np.round(np.array(vals) * 2)]


def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_decomposition_search(n, m, p, target_rank, time_limit=10.0):
    """Try to find a rank-target_rank decomposition using optimization."""
    start = time.time()
    
    # The tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations with 3*rank*unknowns variables
    # But it's nonlinear (product of three variables per term)
    
    # Strategy: fix two matrices, solve for the third, iterate
    # Or use the fact that for small ranks, we can try random initializations
    
    # Let's try: initialize with small random integers, then use gradient-free search
    
    best_score = float('inf')
    best_result = None
    
    n_eq = n * m * p  # number of equations
    
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small integers/half-integers
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        U = np.random.choice(vals, size=(n * m, target_rank))
        V = np.random.choice(vals, size=(m * p, target_rank))
        W = np.random.choice(vals, size=(p * n, target_rank))
        
        # Evaluate error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                    error += (s - 1.0) ** 2
        
        if error < 1e-6:
            # Found a valid decomposition!
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if error < best_score:
            best_score = error
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result


def _build_from_3x3x3_blocks(U3, V3, W3, n, m, p):
    """Build a 3x3x6 decomposition from two 3x3x3 blocks."""
    # U3, V3, W3 are for 3x3x3
    # Block 1: k in [0,3), Block 2: k in [3,6)
    r = len(U3[0])
    total_rank = 2 * r
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Block 1 (k=0,1,2)
    for r_idx in range(r):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r_idx] = U3[i * m + j][r_idx]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r_idx] = V3[j * 3 + k][r_idx]
        for i in range(n):
            for k in range(3):
                W[k * n + i][r_idx] = W3[k * n + i][r_idx]
    
    # Block 2 (k=3,4,5)
    for r_idx in range(r):
        offset = r + r_idx
        for i in range(n):
            for j in range(m):
                U[i * m + j][offset] = U3[i * m + j][r_idx]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3)][offset] = V3[j * 3 + k][r_idx]
        for i in range(n):
            for k in range(3):
                W[(k + 3) * n + i][offset] = W3[k * n + i][r_idx]
    
    return {"U": U, "V": V, "W": W}


def _search_3x3x3_decomposition(target_rank, time_limit=20.0):
    """Search for a rank-target_rank decomposition of 3x3x3 matrix multiplication."""
    start = time.time()
    n = m = p = 3
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5]
        U = np.random.choice(vals, size=(n * m, target_rank))
        V = np.random.choice(vals, size=(m * p, target_rank))
        W = np.random.choice(vals, size=(p * n, target_rank))
        
        # Evaluate
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                    error += (s - 1.0) ** 2
        
        if error < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 120s limit
    
    # First, ensure we have a valid fallback
    fallback = _schoolbook_decomposition(n, m, p)
    
    # Try to find a better decomposition
    
    # Strategy 1: Try two 3x3x3 Strassen-23 blocks (rank 46)
    # Search for 3x3x3 rank-23 decomposition
    if time.time() < deadline - 30:
        decomp_3x3x3 = _search_3x3x3_decomposition(23, time_limit=25.0)
        if decomp_3x3x3 is not None:
            result = _build_from_3x3x3_blocks(decomp_3x3x3["U"], decomp_3x3x3["V"], decomp_3x3x3["W"], n, m, p)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                # Round to half-integers
                for key in ["U", "V", "W"]:
                    result[key] = _round_to_half(np.array(result[key]))
                    result[key] = [[v / 2.0 for v in row] for row in result[key]]
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    return result
    
    # Strategy 2: Direct search for lower rank 3x3x6
    if time.time() < deadline - 20:
        for rank in [44, 45, 46, 47, 48]:
            if time.time() > deadline - 15:
                break
            result = _try_decomposition_search(n, m, p, rank, time_limit=10.0)
            if result is not None:
                # Round to half-integers
                for key in ["U", "V", "W"]:
                    arr = np.array(result[key])
                    result[key] = _round_to_half(arr)
                    result[key] = [[v / 2.0 for v in row] for row in result[key]]
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    return result
    
    # Return fallback
    return fallback
