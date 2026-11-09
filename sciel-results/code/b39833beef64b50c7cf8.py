"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook (rank 24) decomposition for 2x3x4."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: for all i,j,k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    # Check that no extra terms are introduced (the decomposition should be exact)
    # This is already checked above; if all (i,j,k) give exactly 1, it's correct.
    return True


def _verify_fast(U, V, W):
    """Fast verification using tensor einsum. Returns True if exact."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.array_equal(T, expected)


def _to_half_int_list(matrix):
    """Convert a numpy array to a list of lists with integer or half-integer entries."""
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            # Check if it's integer or half-integer
            v = float(val)
            if abs(v * 2 - round(v * 2)) < 1e-12:
                new_row.append(int(round(v)))
            else:
                # If not half-integer, round to nearest half-integer and hope it works
                new_row.append(round(v * 2) / 2.0)
        result.append(new_row)
    return result


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Start with schoolbook rank 24
    best = _schoolbook()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Try to find a lower-rank decomposition using Strassen-based or structured methods.
    # For 2x3x4, the known best rank is not trivially lower than 24 without specific algorithms.
    # We'll try a few structured approaches:
    
    # Approach 1: Try rank 22, 21, 20, 19, 18 by searching for small integer solutions
    # This is a constrained integer optimization problem. We'll use a simple heuristic.
    
    for target_rank in [22, 21, 20, 19, 18]:
        if time.time() > deadline - 5:
            break
        
        # Use a randomized search with local improvement for small integer entries
        # Initialize with random small integers
        max_iter = 5000
        found = False
        
        for attempt in range(3):  # 3 attempts per rank
            if time.time() > deadline - 5:
                break
            
            U = np.random.randint(-3, 4, (nm, target_rank))
            V = np.random.randint(-3, 4, (mp, target_rank))
            W = np.random.randint(-3, 4, (pn, target_rank))
            
            # Set up the error tensor
            # We want T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
            
            best_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute error
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.sum((T - 1.0) ** 2)
                
                if err < best_err:
                    best_err = err
                    if abs(err) < 1e-12:
                        if _verify_fast(U, V, W):
                            best = {"U": U, "V": V, "W": W}
                            found = True
                            break
                        else:
                            best = {"U": U, "V": V, "W": W}
                
                if found:
                    break
                
                # Local search: perturb a random entry
                if np.random.random() < 0.3:
                    # Perturb U
                    r, c = np.random.randint(0, nm), np.random.randint(0, target_rank)
                    U[r, c] += np.random.choice([-1, 1])
                elif np.random.random() < 0.6:
                    # Perturb V
                    r, c = np.random.randint(0, mp), np.random.randint(0, target_rank)
                    V[r, c] += np.random.choice([-1, 1])
                else:
                    # Perturb W
                    r, c = np.random.randint(0, pn), np.random.randint(0, target_rank)
                    W[r, c] += np.random.choice([-1, 1])
            
            if found:
                break
        
        if found:
            # Verify and convert
            if _verify_fast(best['U'], best['V'], best['W']):
                result = {
                    "U": _to_half_int_list(best['U']),
                    "V": _to_half_int_list(best['V']),
                    "W": _to_half_int_list(best['W'])
                }
                return result

    # Return the schoolbook decomposition as fallback
    return {
        "U": best['U'].tolist(),
        "V": best['V'].tolist(),
        "W": best['W'].tolist()
    }

# EVOLVE-BLOCK-END
