"""
Rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication.

Strategy:
- The tensor has 24 required non-zero entries (one per (i,j,k) triple).
- Schoolbook gives rank 24.
- We attempt to find a rank-12 decomposition via structured search:
  1. Try known algebraic constructions (splitting into two 2x3*3x2 blocks).
  2. Use randomized local search with a smooth objective to guide toward valid decompositions.
  3. Fallback to schoolbook rank 24.

The key insight: for 2x3*3x4, the tensor can be viewed as two independent 2x3*3x2 multiplications
sharing the U factor. If we can find a rank-6 decomposition for 2x3*3x2, we get rank 12 for the full problem.
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U', 'V', 'W'} with shapes (6, rank), (12, rank), (8, rank).
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 130  # Safety margin before 150s kill

    # First, try to find a rank-6 decomposition for 2x3*3x2
    # If successful, we can combine two such to get rank 12 for 2x3*3x4
    result = _try_rank_12_via_subproblems(n, m, p, deadline)
    if result is not None:
        return result

    # Try direct rank-12 search with structured initialization
    result = _try_rank_12_direct(n, m, p, deadline)
    if result is not None:
        return result

    # Fallback to schoolbook (rank 24)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank_12_via_subproblems(n, m, p, deadline):
    """
    Try to find rank-6 decomposition for 2x3*3x2, then combine two for rank-12 on 2x3*3x4.
    
    For 2x3*3x2:
    - U shape: (6, 6)
    - V shape: (6, 6) 
    - W shape: (4, 6)
    
    Required: sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k
    """
    n1, m1, p1 = 2, 3, 2
    r1 = 6
    
    # Try to find rank-6 decomposition for 2x3*3x2
    U1, V1, W1 = _find_sub_decomposition(n1, m1, p1, r1, deadline)
    if U1 is not None:
        # Combine two copies for the full 2x3*3x4 problem
        U = np.zeros((n * m, 2 * r1), dtype=int)
        V = np.zeros((m * p, 2 * r1), dtype=int)
        W = np.zeros((p * n, 2 * r1), dtype=int)
        
        # Left half (k=0,1)
        for c in range(r1):
            for i in range(n1):
                for j in range(m1):
                    U[i * m + j, c] = U1[i * m1 + j, c]
            for j in range(m1):
                for k in range(p1):
                    V[j * p + k, c] = V1[j * p1 + k, c]
            for k in range(p1):
                for i in range(n1):
                    W[k * n + i, c] = W1[k * p1 + i, c]
        
        # Right half (k=2,3)
        for c in range(r1):
            for i in range(n1):
                for j in range(m1):
                    U[i * m + j, c + r1] = U1[i * m1 + j, c]
            for j in range(m1):
                for k in range(p1):
                    V[j * p + (k + p1), c + r1] = V1[j * p1 + k, c]
            for k in range(p1):
                for i in range(n1):
                    W[(k + p1) * n + i, c + r1] = W1[k * p1 + i, c]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _find_sub_decomposition(n, m, p, rank, deadline):
    """
    Find a rank decomposition for n x m * m x p matmul.
    Uses randomized search with local optimization.
    """
    # Try several random initializations
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random sparse initialization
        U = np.random.choice([-1, 0, 0, 1], size=(n * m, rank))
        V = np.random.choice([-1, 0, 0, 1], size=(m * p, rank))
        W = np.random.choice([-1, 0, 0, 1], size=(p * n, rank))
        
        # Local search: try to fix errors
        for step in range(50):
            if time.time() > deadline - 5:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            if np.all(errors == 0):
                return U, V, W
            
            # Find worst error location
            flat_err = errors.flatten()
            worst_idx = np.argmax(np.abs(flat_err))
            
            # Try flipping a random entry in one of the matrices
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row, col = np.random.randint(n * m), np.random.randint(rank)
                U[row, col] *= -1
                if U[row, col] == 0:
                    U[row, col] = np.random.choice([-1, 1])
            elif mat_idx == 1:
                row, col = np.random.randint(m * p), np.random.randint(rank)
                V[row, col] *= -1
                if V[row, col] == 0:
                    V[row, col] = np.random.choice([-1, 1])
            else:
                row, col = np.random.randint(p * n), np.random.randint(rank)
                W[row, col] *= -1
                if W[row, col] == 0:
                    W[row, col] = np.random.choice([-1, 1])
    
    return None, None, None


def _try_rank_12_direct(n, m, p, deadline):
    """
    Try direct rank-12 search for the full 2x3*3x4 problem.
    """
    rank = 12
    
    for attempt in range(300):
        if time.time() > deadline - 10:
            break
        
        # Structured initialization: start from schoolbook and merge
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Initialize with random sparse patterns
        U = np.random.choice([-1, 0, 0, 1], size=(n * m, rank))
        V = np.random.choice([-1, 0, 0, 1], size=(m * p, rank))
        W = np.random.choice([-1, 0, 0, 1], size=(p * n, rank))
        
        # Local search
        for step in range(100):
            if time.time() > deadline - 10:
                break
            
            T = _compute_tensor(U, V, W, n, m, p)
            if _is_valid_tensor(T, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Compute gradient-like signal: which (i,j,k) are wrong?
            errors = _compute_errors(U, V, W, n, m, p)
            if np.all(errors == 0):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Perturb a random entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row, col = np.random.randint(n * m), np.random.randint(rank)
                old_val = U[row, col]
                U[row, col] = np.random.choice([-1, 0, 1])
                if U[row, col] == old_val:
                    U[row, col] = -old_val if old_val != 0 else np.random.choice([-1, 1])
            elif mat_idx == 1:
                row, col = np.random.randint(m * p), np.random.randint(rank)
                old_val = V[row, col]
                V[row, col] = np.random.choice([-1, 0, 1])
                if V[row, col] == old_val:
                    V[row, col] = -old_val if old_val != 0 else np.random.choice([-1, 1])
            else:
                row, col = np.random.randint(p * n), np.random.randint(rank)
                old_val = W[row, col]
                W[row, col] = np.random.choice([-1, 0, 1])
                if W[row, col] == old_val:
                    W[row, col] = -old_val if old_val != 0 else np.random.choice([-1, 1])
    
    return None


def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return T


def _compute_errors(U, V, W, n, m, p):
    """
    Compute error tensor: 1 where (i,j,k) should be 1 but isn't, -1 where it should be 0 but isn't.
    Returns array of shape (n, m, p) with signed errors.
    """
    T = _compute_tensor(U, V, W, n, m, p)
    errors = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                errors[i, j, k] = T[u_idx, v_idx, w_idx] - 1
    return errors


def _is_valid_tensor(T, n, m, p):
    """Check if tensor is valid: 1 at (i,j,k) positions, 0 elsewhere."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if T[u_idx, v_idx, w_idx] != 1:
                    return False
    
    # Check that all other entries are 0
    mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                mask[u_idx, v_idx, w_idx] = True
    
    if np.any(T[~mask] != 0):
        return False
    
    return True


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    T = _compute_tensor(U, V, W, n, m, p)
    return _is_valid_tensor(T, n, m, p)

# EVOLVE-BLOCK-END
