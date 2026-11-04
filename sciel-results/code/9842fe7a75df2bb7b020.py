"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def _check_time():
    """Check if we've exceeded the time limit."""
    return (time.time() - START_TIME) > TIME_LIMIT - 5.0


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _build_strassen_decomposition(n=2, m=4, p=4):
    """
    Build a rank-28 Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_i = local_a_idx // 2
                local_j = local_a_idx % 2
                global_i = local_i
                global_j = a_co + local_j
                U[global_i * m + global_j][col] = coeff
            
            for local_b_idx, coeff in pattern['b'].items():
                local_j = local_b_idx // 2
                local_k = local_b_idx % 2
                global_j = b_ro + local_j
                global_k = b_co + local_k
                V[global_j * p + global_k][col] = coeff
            
            for local_c_idx, coeff in pattern['c'].items():
                local_i = local_c_idx // 2
                local_k = local_c_idx % 2
                global_i = local_i
                global_k = c_co + local_k
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=4, seed=42, num_tests=5):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(seed)
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_factor = np.dot(U_arr[:, r], A.flatten())
            b_factor = np.dot(V_arr[:, r], B.flatten())
            if a_factor != 0 and b_factor != 0:
                C_computed += W_arr[:, r].reshape(p, n).T * (a_factor * b_factor)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True


def _try_reduce_rank(U, V, W, n=2, m=4, p=4, target_rank=None):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 terms. Returns a new (U, V, W) with potentially lower rank,
    or None if no reduction found.
    """
    if _check_time():
        return None
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if target_rank is None:
        target_rank = rank - 1
    
    # Try to find if any column is a linear combination of others
    # We'll attempt to find a null vector in the combined space
    
    # For each potential removal, check if the remaining terms span the same tensor
    # This is expensive, so we'll use a heuristic: look for columns that are
    # linearly dependent in a combined matrix
    
    # Build a matrix where each row represents a (U_i, V_j, W_k) entry
    # Actually, let's try a different approach: check if any two columns are
    # "similar" in the sense that their outer products are linearly dependent
    
    # A simpler approach: try removing each column and check if the decomposition
    # still works (approximately)
    
    for col_to_remove in range(rank):
        if _check_time():
            return None
        
        # Create reduced matrices
        U_red = U_arr[:, [c for c in range(rank) if c != col_to_remove]]
        V_red = V_arr[:, [c for c in range(rank) if c != col_to_remove]]
        W_red = W_arr[:, [c for c in range(rank) if c != col_to_remove]]
        
        # Quick verification with a few random tests
        np.random.seed(12345)
        valid = True
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank - 1):
                a_factor = np.dot(U_red[:, r], A.flatten())
                b_factor = np.dot(V_red[:, r], B.flatten())
                C_computed += W_red[:, r].reshape(p, n).T * (a_factor * b_factor)
            
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                valid = False
                break
        
        if valid:
            # Found a valid reduction
            U_new = U_red.tolist()
            V_new = V_red.tolist()
            W_new = W_red.tolist()
            return U_new, V_new, W_new
    
    return None


def _attempt_column_elimination(U, V, W, n=2, m=4, p=4, max_iterations=5):
    """
    Iteratively try to eliminate columns from the decomposition.
    """
    if _check_time():
        return None
    
    current_U, current_V, current_W = U, V, W
    current_rank = len(current_U[0])
    
    for iteration in range(max_iterations):
        if _check_time():
            break
        
        result = _try_reduce_rank(current_U, current_V, current_W, n, m, p)
        if result is None:
            break
        
        current_U, current_V, current_W = result
        current_rank = len(current_U[0])
    
    if current_rank < len(U[0]):
        return current_U, current_V, current_W
    
    return None


def _attempt_optimization(U, V, W, n=2, m=4, p=4):
    """
    Try to optimize the decomposition using local search.
    This is a placeholder for more sophisticated optimization.
    """
    if _check_time():
        return None
    
    # For now, just try column elimination
    return _attempt_column_elimination(U, V, W, n, m, p)


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Try to reduce the rank
    optimized = _attempt_optimization(U, V, W, n, m, p)
    if optimized is not None:
        U, V, W = optimized
    
    return {"U": U, "V": V, "W": W}
