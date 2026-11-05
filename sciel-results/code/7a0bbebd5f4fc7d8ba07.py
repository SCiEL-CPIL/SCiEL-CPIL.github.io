import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 115.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    """
    Returns the standard schoolbook decomposition for matrix multiplication.
    Rank is n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return U, V, W

def _verify(n, m, p, U, V, W):
    """
    Verifies if the given factor matrices exactly decompose the matrix multiplication tensor.
    T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    For matrix multiplication, T_{i,j,k} should be 1 for all valid indices.
    """
    rank = len(U[0])
    # Convert to numpy arrays for vectorized verification if inputs are lists
    if isinstance(U, list):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
    # We can check the tensor reconstruction
    # T is a 3D array of shape (n*m, m*p, p*n)
    # But we only care about the "diagonal" entries where the indices align with multiplication.
    # Actually, the tensor T_{a,b,c} is non-zero only if the indices correspond to a valid multiplication step.
    # Specifically, for a = i*m+j, b = j'*p+k, c = k'*n+i', T_{a,b,c} = 1 if i=i', j=j', k=k'.
    # Wait, the standard decomposition is T_{i,j,k} = 1 where the indices are linked.
    # The condition is: for all i, j, k, the sum over r of U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must be 1.
    # And for any other combination of indices (a, b, c) that doesn't correspond to a valid (i,j,k) triple 
    # (i.e., the middle index of U doesn't match the first index of V, or the last index of V doesn't match 
    # the first index of W, etc.), the sum should be 0.
    
    # Let's just check the required non-zero entries.
    # If the decomposition is correct, the sum must be 1 for all (i, j, k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Calculate sum
                # Using dot product of slices
                val = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                # Check if it's close to 1. Since we want exact integer/half-integer, we check strict equality or very high precision
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # We should also check that "wrong" entries are 0.
    # This is crucial for rank minimality correctness, though often implied if the rank is minimal and the main entries are correct?
    # No, a decomposition can have "noise" in the off-diagonals that cancels out or sums to 0.
    # However, for matrix multiplication tensor, the support is specific.
    # Let's check a few random off-diagonal entries to be safe, or all of them if fast enough.
    # Total entries in tensor: (n*m) * (m*p) * (p*n) = 10 * 25 * 10 = 2500.
    # Rank is at most 50.
    # 2500 * 50 operations is 125,000 multiplications. This is very fast in NumPy.
    
    # Let's do a full verification for robustness.
    # Construct the full tensor from factors
    # T_reconstructed[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # This is equivalent to: T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    
    try:
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    except Exception:
        return False
        
    # The true tensor T_true has T_true[i*m+j, j*p+k, k*n+i] = 1, and 0 elsewhere.
    # Let's construct T_true
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Check for exact match (with tolerance for floating point)
    if not np.allclose(T_recon, T_true, atol=1e-9):
        return False
        
    return True

def construct():
    """
    Returns the factor matrices for the matrix multiplication tensor 2x5x5.
    Currently uses the standard schoolbook decomposition with rank 50.
    Includes a verification step to ensure correctness.
    """
    n, m, p = DIMENSIONS
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _schoolbook(n, m, p)
    
    # 2. Verify
    if not _verify(n, m, p, U, V, W):
        # This should never happen for schoolbook, but good practice
        raise ValueError("Schoolbook decomposition failed verification")
        
    # 3. Return as lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    # No print to avoid overhead during eval
    pass
