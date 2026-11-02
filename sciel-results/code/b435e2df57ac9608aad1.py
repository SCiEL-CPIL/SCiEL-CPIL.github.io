import numpy as np

DIMENSIONS = [3, 5, 7]

# Cache for tensor decompositions
_tensor_cache = {}

def _get_strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition."""
    if '2x2x2' in _tensor_cache:
        return _tensor_cache['2x2x2']
        
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: A(0,0)=0, A(0,1)=1, A(1,0)=2, A(1,1)=3
    # B(0,0)=0, B(0,1)=1, B(1,0)=2, B(1,1)=3
    # C(0,0)=0, C(0,1)=1, C(1,0)=2, C(1,1)=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[0, 6] = 1
    
    _tensor_cache['2x2x2'] = (U, V, W)
    return U, V, W

def _get_schoolbook(n, m, p):
    """Returns U, V, W for schoolbook decomposition."""
    key = f"schoolbook_{n}x{m}x{p}"
    if key in _tensor_cache:
        return _tensor_cache[key]
        
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    _tensor_cache[key] = (U, V, W)
    return U, V, W

def _split_p(n, m, p1, p2):
    """Splits p dimension."""
    key = f"split_p_{n}x{m}x{p1}_{p2}"
    if key in _tensor_cache:
        return _tensor_cache[key]
        
    U1, V1, W1 = _get_tensor(n, m, p1)
    U2, V2, W2 = _get_tensor(n, m, p2)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.hstack([U1, U2])
    
    V = np.zeros((m * (p1 + p2), rank))
    V[:m * p1, :rank1] = V1
    V[m * p1:, rank1:] = V2
    
    W = np.zeros(((p1 + p2) * n, rank))
    W[:p1 * n, :rank1] = W1
    W[p1 * n:, rank1:] = W2
    
    _tensor_cache[key] = (U, V, W)
    return U, V, W

def _split_n(n1, n2, m, p):
    """Splits n dimension."""
    key = f"split_n_{n1}_{n2}x{m}x{p}"
    if key in _tensor_cache:
        return _tensor_cache[key]
        
    U1, V1, W1 = _get_tensor(n1, m, p)
    U2, V2, W2 = _get_tensor(n2, m, p)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros(((n1 + n2) * m, rank))
    U[:n1 * m, :rank1] = U1
    U[n1 * m:, rank1:] = U2
    
    V = np.hstack([V1, V2])
    
    W = np.zeros((p * (n1 + n2), rank))
    W[:p * n1, :rank1] = W1
    W[p * n1:, rank1:] = W2
    
    _tensor_cache[key] = (U, V, W)
    return U, V, W

def _split_m(n, m1, m2, p):
    """Splits m dimension."""
    key = f"split_m_{n}x{m1}_{m2}x{p}"
    if key in _tensor_cache:
        return _tensor_cache[key]
        
    U1, V1, W1 = _get_tensor(n, m1, p)
    U2, V2, W2 = _get_tensor(n, m2, p)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros((n * (m1 + m2), rank))
    U[:n * m1, :rank1] = U1
    U[n * m1:, rank1:] = U2
    
    V = np.zeros(((m1 + m2) * p, rank))
    V[:m1 * p, :rank1] = V1
    V[m1 * p:, rank1:] = V2
    
    W = np.hstack([W1, W2])
    
    _tensor_cache[key] = (U, V, W)
    return U, V, W

def _get_tensor(n, m, p):
    key = f"{n}x{m}x{p}"
    if key in _tensor_cache:
        return _tensor_cache[key]
        
    if n == 2 and m == 2 and p == 2:
        return _get_strassen_2x2x2()
        
    if n == 1 or m == 1 or p == 1:
        return _get_schoolbook(n, m, p)
        
    # Custom splitting strategy for 3x5x7
    if n == 3 and m == 5 and p == 7:
        return _split_p(3, 5, 4, 3)
    if n == 3 and m == 5 and p == 4:
        return _split_n(2, 1, 5, 4)
    if n == 2 and m == 5 and p == 4:
        return _split_m(2, 2, 3, 4)
    if n == 2 and m == 2 and p == 4:
        return _split_p(2, 2, 2, 2)
    if n == 2 and m == 3 and p == 4:
        return _split_m(2, 1, 2, 4)
    if n == 3 and m == 5 and p == 3:
        return _split_n(2, 1, 5, 3)
    if n == 2 and m == 5 and p == 3:
        return _split_m(2, 2, 3, 3)
    if n == 2 and m == 2 and p == 3:
        return _split_p(2, 2, 2, 1)
    if n == 2 and m == 3 and p == 3:
        return _split_m(2, 1, 2, 3)
        
    # Fallback to schoolbook
    return _get_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _get_tensor(n, m, p)
    # Convert to list of lists to match interface if required, but numpy arrays are usually fine
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is standard.
    return {"U": U, "V": V, "W": W}
