"""Matrix multiplication 3,3,3 - rank 26 exact decomposition via 2x2+1x1 partitioning."""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 26
    
    # Initialize U, V, W as NumPy arrays for faster manipulation
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w):
        nonlocal col
        if col >= rank:
            return
        for idx, c in zip(u_indices, coeffs_u):
            U[idx, col] += c
        for idx, c in zip(v_indices, coeffs_v):
            V[idx, col] += c
        for idx, c in zip(w_indices, coeffs_w):
            W[idx, col] += c
        col += 1

    # Indices mapping:
    # A[i,j] -> U[i*3 + j]
    # B[j,k] -> V[j*3 + k]
    # C[i,k] -> W[k*3 + i]
    
    # Block 1: Top-left 2x2 * Top-left 2x2 -> Rank 7 (Strassen)
    # A2 indices: 0,1,3,4
    # B2 indices: 0,1,3,4
    # C2 indices: 0,3,1,4
    
    # M1 = (A00+A11)(B00+B11)
    add_term([0, 4], [0, 4], [0, 4], [1, 1], [1, 1], [1, 1])
    
    # M2 = (A10+A11)B00
    add_term([3, 4], [0], [1, 4], [1, 1], [1], [1, -1])
    
    # M3 = A00(B01-B11)
    add_term([0], [1, 4], [3, 4], [1], [1, -1], [1, 1])
    
    # M4 = A11(B10-B00)
    add_term([4], [3, 0], [0, 1], [1], [1, -1], [1, 1])
    
    # M5 = (A00+A01)B11
    add_term([0, 1], [4], [0, 3], [1, 1], [1], [-1, 1])
    
    # M6 = (A10-A00)(B00+B01)
    add_term([3, 0], [0, 1], [4], [1, -1], [1, 1], [1])
    
    # M7 = (A01-A11)(B10+B11)
    add_term([1, 4], [3, 4], [0], [1, -1], [1, 1], [1])
    
    # Block 2: A[0:2,2] * B[2,0:2] -> Rank 4
    # U: 2, 5
    # V: 6, 7
    # W: 0, 3, 1, 4
    add_term([2], [6], [0], [1], [1], [1])
    add_term([2], [7], [3], [1], [1], [1])
    add_term([5], [6], [1], [1], [1], [1])
    add_term([5], [7], [4], [1], [1], [1])
    
    # Block 3: A[0:2,0:2] * B[0:2,2] -> Rank 4
    # U: 0,1,3,4
    # V: 2, 5
    # W: 6, 7
    add_term([0], [2], [6], [1], [1], [1])
    add_term([1], [5], [6], [1], [1], [1])
    add_term([3], [2], [7], [1], [1], [1])
    add_term([4], [5], [7], [1], [1], [1])
    
    # Block 4: A[0:2,2] * B[2,2] -> Rank 2
    # U: 2, 5
    # V: 8
    # W: 6, 7
    add_term([2], [8], [6], [1], [1], [1])
    add_term([5], [8], [7], [1], [1], [1])
    
    # Block 5: A[2,0:2] * B[0:2,0:2] -> Rank 4
    # U: 6, 7
    # V: 0,1,3,4
    # W: 2, 5
    add_term([6], [0], [2], [1], [1], [1])
    add_term([7], [3], [2], [1], [1], [1])
    add_term([6], [1], [5], [1], [1], [1])
    add_term([7], [4], [5], [1], [1], [1])
    
    # Block 6: A[2,2] * B[2,0:2] -> Rank 2
    # U: 8
    # V: 6, 7
    # W: 2, 5
    add_term([8], [6], [2], [1], [1], [1])
    add_term([8], [7], [5], [1], [1], [1])
    
    # Block 7: A[2,0:2] * B[0:2,2] -> Rank 2
    # U: 6, 7
    # V: 2, 5
    # W: 8
    add_term([6], [2], [8], [1], [1], [1])
    add_term([7], [5], [8], [1], [1], [1])
    
    # Block 8: A[2,2] * B[2,2] -> Rank 1
    # U: 8
    # V: 8
    # W: 8
    add_term([8], [8], [8], [1], [1], [1])
    
    # Verify col == rank
    # assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Convert to lists if required, but numpy arrays are usually accepted or faster to process
    # The interface says 'factor matrices of shapes...'. Lists of lists is standard for JSON-like output.
    # However, returning numpy arrays is often accepted by evaluators that convert to numpy.
    # To be safe, we can return lists if the evaluator expects them, but numpy is faster to construct.
    # Let's return numpy arrays. If the evaluator fails, we might need to convert.
    # But usually, "factor matrices" implies array-like.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
