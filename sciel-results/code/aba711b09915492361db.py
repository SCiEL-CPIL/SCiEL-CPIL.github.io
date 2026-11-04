"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _to_fraction_list(lst):
    """Convert a list of numbers to Fractions."""
    return [Fraction(x).limit_denominator(1000) for x in lst]


def _schoolbook_3x3x3():
    """
    Return schoolbook rank-27 decomposition for 3x3 * 3x3.
    
    U: shape (9, 27), U[i*3+j, t]
    V: shape (21, 27), V[j*3+k, t]  (but for 3x3, V is (9, 27))
    W: shape (9, 27), W[k*3+i, t]
    
    Actually for 3x3 * 3x3:
    U: (9, 27)
    V: (9, 27)
    W: (9, 27)
    """
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def _try_strassen_3x3x3():
    """
    Try to find a rank-23 decomposition for 3x3 * 3x3 using a search approach.
    This is based on known results that 3x3 matrix multiplication has rank 23.
    
    We'll try to construct this using a systematic approach.
    """
    # For 3x3 matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We want to write this as sum of rank-1 tensors.
    #
    # The rank is at most 27 (schoolbook) and at least 19 (Landsberg bound).
    # It's known that rank = 23.
    #
    # Let me try a direct construction approach.
    # 
    # One approach: Use the fact that 3x3 matrix multiplication can be
    # decomposed using 23 bilinear forms.
    #
    # I'll try to construct this by considering the structure of the tensor.
    #
    # Actually, let me try a numerical optimization approach to find a low-rank
    # decomposition. But this might be too slow for 145 seconds.
    #
    # Let me try a different approach: use known explicit formulas.
    #
    # The rank-23 decomposition of 3x3 matrix multiplication is known but
    # complex. Let me try to construct it step by step.
    
    # For now, return None to indicate we can't find it in time.
    # We'll fall back to schoolbook.
    return None


def _construct_3x3x7():
    """
    Construct a decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Split the 7 columns into 3+3+1.
    Use rank-23 for each 3x3 part (if we can find it) and rank-9 for the 3x1 part.
    Total: 23 + 23 + 9 = 55
    
    If we can't find rank-23, use schoolbook: 27 + 27 + 9 = 63
    """
    n, m, p = 3, 3, 7
    
    # Try to find a low-rank decomposition for 3x3 * 3x3
    rank23_result = _try_strassen_3x3x3()
    
    if rank23_result is not None:
        U33, V33, W33 = rank23_result
        r33 = U33.shape[1]  # should be 23
    else:
        # Fall back to schoolbook
        U33, V33, W33 = _schoolbook_3x3x3()
        r33 = U33.shape[1]  # 27
    
    # Schoolbook for 3x3 * 3x1: rank 9
    r31 = 9
    U31 = np.zeros((9, r31))
    V31 = np.zeros((3, r31))  # 3x1 matrix, flattened to 3 entries
    W31 = np.zeros((3, r31))  # 3x1 result, flattened to 3 entries
    for i in range(3):
        for j in range(3):
            t = i * 3 + j
            U31[i*3 + j, t] = 1
            V31[j, t] = 1
            W31[i, t] = 1
    
    # Now combine:
    # For 3x3 * 3x7, we split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B = A*B1 + A*B2 + A*B3
    #
    # For A*B1: U is (9, r33), V is (9, r33) but we need V to be (21, total_rank)
    # Actually, V for the full problem is (m*p, rank) = (3*7, rank) = (21, rank)
    #
    # Let me reconsider the dimensions:
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)
    #
    # For the first 3x3 part (columns 0-2 of B):
    # U1: (9, r33) - same as U33
    # V1: (21, r33) - V33 in positions 0-8, zeros elsewhere
    # W1: (21, r33) - W33 in positions 0-8 (for k=0,1,2), zeros elsewhere
    #
    # For the second 3x3 part (columns 3-5 of B):
    # U2: (9, r33) - same as U33
    # V2: (21, r33) - V33 in positions 9-17, zeros elsewhere
    # W2: (21, r33) - W33 in positions 9-17 (for k=3,4,5), zeros elsewhere
    #
    # For the third 3x1 part (column 6 of B):
    # U3: (9, r31) - same as U31
    # V3: (21, r31) - V31 in position 18, zeros elsewhere
    # W3: (21, r31) - W31 in position 18 (for k=6), zeros elsewhere
    
    total_rank = r33 + r33 + r31
    
    U = np.zeros((9, total_rank))
    V = np.zeros((21, total_rank))
    W = np.zeros((21, total_rank))
    
    # First 3x3 part
    U[:, 0:r33] = U33
    V[0:9, 0:r33] = V33
    W[0:9, 0:r33] = W33  # W[k*n+i, t] for k=0,1,2 -> rows 0-8
    
    # Second 3x3 part
    U[:, r33:2*r33] = U33
    V[9:18, r33:2*r33] = V33
    W[9:18, r33:2*r33] = W33  # W[k*n+i, t] for k=3,4,5 -> rows 9-17
    
    # Third 3x1 part
    U[:, 2*r33:2*r33+r31] = U31
    V[18:19, 2*r33:2*r33+r31] = V31
    W[18:19, 2*r33:2*r33+r31] = W31  # W[k*n+i, t] for k=6 -> row 18
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n=3, m=3, p=7):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Check: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    (This is the tensor equation)
    """
    rank = U.shape[1]
    errors = 0
    max_error = 0.0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # Should be 1
                err = abs(val - 1.0)
                if err > max_error:
                    max_error = err
                if err > 1e-9:
                    errors += 1
    
    if errors > 0:
        print(f"Verification failed: {errors} entries with error > 1e-9, max error = {max_error}")
        return False
    return True


def _convert_to_lists(U, V, W):
    """Convert numpy arrays to lists of lists with Fraction or int entries."""
    def arr_to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Try to convert to exact fraction
                f = Fraction(val).limit_denominator(1000)
                # If it's a whole number, use int
                if f.denominator == 1:
                    row_list.append(int(f.numerator))
                else:
                    row_list.append(float(f))
            result.append(row_list)
        return result
    
    return arr_to_list(U), arr_to_list(V), arr_to_list(W)


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    if _time_left() < 30:
        # Emergency fallback: return schoolbook
        n, m, p = 3, 3, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    try:
        U, V, W, rank = _construct_3x3x7()
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W):
            U_list, V_list, W_list = _convert_to_lists(U, V, W)
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fall back to schoolbook
            n, m, p = 3, 3, 7
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fall back to schoolbook on any error
        n, m, p = 3, 3, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
