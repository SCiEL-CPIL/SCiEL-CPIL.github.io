"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _rank32_decomposition(n, m, p):
    """
    Rank-32 decomposition: 2 terms per k.
    
    For each k (0..15), we have 4 tensor entries:
    T[0, k, 2k] = 1     (A[0,0]*B[0,k] -> C[0,k])
    T[1, 16+k, 2k] = 1  (A[0,1]*B[1,k] -> C[0,k])
    T[2, k, 2k+1] = 1   (A[1,0]*B[0,k] -> C[1,k])
    T[3, 16+k, 2k+1] = 1 (A[1,1]*B[1,k] -> C[1,k])
    
    Term 2k: handles j=0 part (B[0,k] * A[:,0])
    Term 2k+1: handles j=1 part (B[1,k] * A[:,1])
    """
    rank = 2 * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    for k in range(p):
        # Term 1: A[0,0] and A[1,0] with B[0,k], contributing to C[0,k] and C[1,k]
        col1 = 2 * k
        U[0][col1] = 1   # A[0,0]
        U[2][col1] = 1   # A[1,0]
        V[k][col1] = 1   # B[0,k] (row 0 of B, column k)
        W[2 * k][col1] = 1     # C[0,k]
        W[2 * k + 1][col1] = 1 # C[1,k]
        
        # Term 2: A[0,1] and A[1,1] with B[1,k], contributing to C[0,k] and C[1,k]
        col2 = 2 * k + 1
        U[1][col2] = 1   # A[0,1]
        U[3][col2] = 1   # A[1,1]
        V[p + k][col2] = 1  # B[1,k] (row 1 of B, column k)
        W[2 * k][col2] = 1     # C[0,k]
        W[2 * k + 1][col2] = 1 # C[1,k]
    
    return U, V, W

def _strassen_block(n, m, p):
    """Fallback: Strassen-based decomposition, rank 56."""
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        v_idx = {'B00': 2*t, 'B01': 2*t+1, 'B10': 16+2*t, 'B11': 16+2*t+1}
        w_idx = {'C00': 4*t, 'C01': 4*t+2, 'C10': 4*t+1, 'C11': 4*t+3}
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1
        W[w_idx['C10']][col] += 1; W[w_idx['C11']][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[v_idx['B01']][col] += 1; V[v_idx['B11']][col] -= 1
        W[w_idx['C01']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[v_idx['B10']][col] += 1; V[v_idx['B00']][col] -= 1
        W[w_idx['C00']][col] += 1; W[w_idx['C10']][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] -= 1; W[w_idx['C01']][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1; U[0][col] -= 1
        V[v_idx['B00']][col] += 1; V[v_idx['B01']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[v_idx['B10']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1
        col += 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_rank16():
    """
    Attempt to find a rank-16 decomposition using random initialization 
    and gradient-free optimization (Nelder-Mead style).
    
    For rank 16, we have 16 terms. For each k, we need to handle 4 entries.
    The key insight: if we can share terms across k values, we might reduce rank.
    
    Actually, let's try a structured approach. For rank 16, we need 1 term per k.
    For term k, we need:
    U[0,k]*V[k,k]*W[2k,k] = 1        (A[0,0]*B[0,k] -> C[0,k])
    U[1,k]*V[16+k,k]*W[2k,k] = 1     (A[0,1]*B[1,k] -> C[0,k])
    U[2,k]*V[k,k]*W[2k+1,k] = 1      (A[1,0]*B[0,k] -> C[1,k])
    U[3,k]*V[16+k,k]*W[2k+1,k] = 1   (A[1,1]*B[1,k] -> C[1,k])
    
    And all other products must be 0.
    
    This is very constrained. Let's see if it's possible.
    
    From the first equation: U[0,k]*V[k,k]*W[2k,k] = 1
    From the third equation: U[2,k]*V[k,k]*W[2k+1,k] = 1
    
    So V[k,k] = 1/(U[0,k]*W[2k,k]) and V[k,k] = 1/(U[2,k]*W[2k+1,k])
    Thus U[0,k]*W[2k,k] = U[2,k]*W[2k+1,k]
    
    Similarly, U[1,k]*W[2k,k] = U[3,k]*W[2k+1,k]
    
    Let's set U[0,k]=1, U[2,k]=1, U[1,k]=1, U[3,k]=1.
    Then W[2k,k] = W[2k+1,k] and W[2k,k] = W[2k+1,k].
    So W[2k,k] = W[2k+1,k] = w_k.
    
    Then V[k,k] = 1/w_k and V[16+k,k] = 1/w_k.
    
    Now check spurious terms. For term k, we have:
    U[0,k]=1, U[1,k]=1, U[2,k]=1, U[3,k]=1
    V[k,k]=1/w_k, V[16+k,k]=1/w_k
    W[2k,k]=w_k, W[2k+1,k]=w_k
    
    Spurious term: U[0,k]*V[16+k,k]*W[2k,k] = 1*(1/w_k)*w_k = 1. But this should be 0!
    T[0, 16+k, 2k] should be 0, but we get 1.
    
    So this simple assignment doesn't work. The spurious terms are a problem.
    
    Let's try a different approach. What if we use complex numbers or other values?
    
    Actually, the issue is that with 1 term per k, each term has 4 non-zero U entries,
    2 non-zero V entries, and 2 non-zero W entries. The spurious terms arise from
    cross-products.
    
    For term k, the non-zero entries are:
    U: [1,1,1,1] (all 4)
    V: [k, 16+k] (2 entries)
    W: [2k, 2k+1] (2 entries)
    
    The valid products are:
    (0,k,2k), (1,16+k,2k), (2,k,2k+1), (3,16+k,2k+1)
    
    The spurious products are:
    (0,16+k,2k), (0,16+k,2k+1), (1,k,2k), (1,k,2k+1), (2,16+k,2k), (2,16+k,2k+1), (3,k,2k), (3,k,2k+1)
    
    That's 8 spurious terms per k, and we need all of them to be 0.
    
    With U all 1s, V[k,k]=V[16+k,k]=1/w_k, W[2k,k]=W[2k+1,k]=w_k:
    All spurious products are (1)*(1/w_k)*(w_k) = 1. Not 0.
    
    So this doesn't work. We need a more clever assignment.
    
    What if we use different values for U? Let U[0,k]=a, U[1,k]=b, U[2,k]=c, U[3,k]=d.
    V[k,k]=e, V[16+k,k]=f.
    W[2k,k]=g, W[2k+1,k]=h.
    
    Valid:
    a*e*g = 1
    b*f*g = 1
    c*e*h = 1
    d*f*h = 1
    
    Spurious:
    a*f*g = 0
    a*f*h = 0
    b*e*g = 0
    b*e*h = 0
    c*f*g = 0
    c*f*h = 0
    d*e*g = 0
    d*e*h = 0
    
    From a*f*g=0: either a=0, f=0, or g=0.
    But a*e*g=1, so a!=0, e!=0, g!=0.
    So f=0.
    
    From b*f*g=1: b*0*g=0, but we need 1. Contradiction!
    
    So rank 16 is impossible with this structure. The spurious terms cannot all be zero.
    
    This confirms that rank 16 is not achievable with 1 term per k.
    
    So the minimum rank is at least 32 (2 terms per k).
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try rank-32 decomposition first
    U, V, W = _rank32_decomposition(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to Strassen (rank 56)
    if time.time() < deadline:
        U, V, W = _strassen_block(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
