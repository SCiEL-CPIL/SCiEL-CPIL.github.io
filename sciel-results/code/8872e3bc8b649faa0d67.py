"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _matmul_tensor_from_factors(U, V, W):
    """Compute the tensor from factor matrices."""
    rank = U.shape[1]
    n_a, n_b, n_c = U.shape[0], V.shape[0], W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Build the rank-17 decomposition (Strassen + direct)
    rank17_terms = []
    
    # Part 1: Strassen for A1(2x2) x B1(2x2) -> C[0:2, 0:2]
    rank17_terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    rank17_terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    rank17_terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    rank17_terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    rank17_terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    rank17_terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    rank17_terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    rank17_terms.append(([0], [1], [2], [1], [4], [1]))
    rank17_terms.append(([1], [1], [5], [1], [4], [1]))
    rank17_terms.append(([3], [1], [2], [1], [5], [1]))
    rank17_terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    rank17_terms.append(([2], [1], [6], [1], [0], [1]))
    rank17_terms.append(([2], [1], [7], [1], [2], [1]))
    rank17_terms.append(([2], [1], [8], [1], [4], [1]))
    rank17_terms.append(([5], [1], [6], [1], [1], [1]))
    rank17_terms.append(([5], [1], [7], [1], [3], [1]))
    rank17_terms.append(([5], [1], [8], [1], [5], [1]))
    
    # Build rank-17 matrices
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(rank17_terms):
        for a, val in zip(u_idx, u_val):
            U17[a, r] = val
        for b, val in zip(v_idx, v_val):
            V17[b, r] = val
        for c, val in zip(w_idx, w_val):
            W17[c, r] = val
    
    # Verify rank-17
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to schoolbook rank 18
        U_fb = np.zeros((n_a, 18))
        V_fb = np.zeros((n_b, 18))
        W_fb = np.zeros((n_c, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find a rank-16 decomposition using a computational approach
    # Strategy: For each pair of terms in the rank-17 decomposition, check if
    # we can replace them with a single term that preserves the tensor.
    # This is equivalent to checking if the sum of two rank-1 tensors is rank-1.
    
    # Alternative approach: Use the fact that the tensor has specific structure.
    # We'll try to find a rank-16 decomposition by solving a system of equations.
    
    # Let's try a different approach: use the known lower bounds and try to
    # find a decomposition with rank 16 by exploiting the structure.
    
    # For now, return the rank-17 decomposition
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
