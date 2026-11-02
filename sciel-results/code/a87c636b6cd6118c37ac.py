"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 120  # seconds; return best result before this

def _strassen_patterns():
    """Return the 7 Strassen bilinear forms for 2x2 * 2x2."""
    return [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]

def _find_rank3_col3(deadline):
    """
    Find a rank-3 decomposition for the 3rd column of a 2x2 * 2x3 block.
    We need U2 (4x3), V2 (2x3), W2 (2x3) such that for all i,j in {0,1}:
        sum_{r=0}^{2} U2[i,j,r] * V2[j,r] * W2[i,r] = 1
    Returns (U2, V2, W2) as numpy arrays or None if not found.
    """
    # Unknowns: U2 has 4*3=12, V2 has 2*3=6, W2 has 2*3=6 -> 24 unknowns
    # Constraints: 4 equations (one for each (i,j))
    # We can fix some degrees of freedom. Let's set V2 to be identity-like.
    # Set V2 = [[1,0,0],[0,1,0]] (first two columns of identity)
    # Then: U2[i,j,0]*W2[i,0] + U2[i,j,1]*W2[i,1] = 1 for all i,j
    # This gives: for i=0: U2[0,j,0]*W2[0,0] + U2[0,j,1]*W2[0,1] = 1 for j=0,1
    #              for i=1: U2[1,j,0]*W2[1,0] + U2[1,j,1]*W2[1,1] = 1 for j=0,1
    
    # Try: W2 = [[1, 1], [1, 1]] (all ones)
    # Then: U2[i,j,0] + U2[i,j,1] = 1 for all i,j
    # Set U2[i,j,0] = 1, U2[i,j,1] = 0 for all i,j
    # Check: 1*1 + 0*1 = 1 ✓ for all (i,j)
    
    # But wait, we also need U2[i,j,2]*V2[j,2]*W2[i,2] = 0 (since V2[j,2]=0, W2[i,2]=0)
    # So the third column of U2 doesn't matter. Set it to 0.
    
    # Let's verify: V2 = [[1,0,0],[0,1,0]], W2 = [[1,1,0],[1,1,0]]
    # U2[i,j,0] = 1, U2[i,j,1] = 0, U2[i,j,2] = 0
    # For (i=0,j=0): U2[0,0,0]*V2[0,0]*W2[0,0] + U2[0,0,1]*V2[0,1]*W2[0,1] + U2[0,0,2]*V2[0,2]*W2[0,2]
    #   = 1*1*1 + 0*0*1 + 0*0*0 = 1 ✓
    # For (i=0,j=1): 1*0*1 + 0*1*1 + 0*0*0 = 0 ✗
    
    # That doesn't work. Let me reconsider.
    
    # With V2 = [[1,0,0],[0,1,0]]:
    # (i,j) constraint: sum_r U2[i,j,r]*V2[j,r]*W2[i,r] = U2[i,j,0]*V2[j,0]*W2[i,0] + U2[i,j,1]*V2[j,1]*W2[i,1]
    # For j=0: U2[i,0,0]*1*W2[i,0] + U2[i,0,1]*0*W2[i,1] = U2[i,0,0]*W2[i,0] = 1
    # For j=1: U2[i,1,0]*0*W2[i,0] + U2[i,1,1]*1*W2[i,1] = U2[i,1,1]*W2[i,1] = 1
    # So: U2[0,0,0]*W2[0,0] = 1, U2[1,0,0]*W2[1,0] = 1, U2[0,1,1]*W2[0,1] = 1, U2[1,1,1]*W2[1,1] = 1
    # Set W2 = [[1,1],[1,1]], then U2[0,0,0]=1, U2[1,0,0]=1, U2[0,1,1]=1, U2[1,1,1]=1
    # All other U2 entries = 0.
    
    U2 = np.zeros((4, 3))
    V2 = np.zeros((2, 3))
    W2 = np.zeros((2, 3))
    V2[0, 0] = 1
    V2[1, 1] = 1
    W2[0, 0] = 1
    W2[0, 1] = 1
    W2[1, 0] = 1
    W2[1, 1] = 1
    U2[0, 0] = 1  # (i=0,j=0), r=0
    U2[2, 0] = 1  # (i=1,j=0), r=0 -> U2[1*2+0, 0]
    U2[1, 1] = 1  # (i=0,j=1), r=1 -> U2[0*2+1, 1]
    U2[3, 1] = 1  # (i=1,j=1), r=1 -> U2[1*2+1, 1]
    
    # Verify
    for i in range(2):
        for j in range(2):
            total = 0.0
            for r in range(3):
                total += U2[i*2+j, r] * V2[j, r] * W2[i, r]
            if abs(total - 1.0) > 1e-12:
                return None
    return (U2, V2, W2)

def _build_rank53():
    """Build the baseline rank-53 decomposition: 7 Strassen blocks + 1 naive column."""
    n, m, p = 2, 2, 15
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    strassen_patterns = _strassen_patterns()
    
    for block in range(7):
        k1 = 2 * block
        k2 = 2 * block + 1
        for pat in strassen_patterns:
            for (i, j), uval in pat['U'].items():
                U[i * m + j, col_idx] = uval
            for (j, k), val in pat['V'].items():
                k_global = k1 + k
                V[j * p + k_global, col_idx] = val
            for (k, i), wval in pat['W'].items():
                k_global = k1 + k
                W[k_global * n + i, col_idx] = wval
            col_idx += 1
    
    k_last = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_idx] = 1
            V[j * p + k_last, col_idx] = 1
            W[k_last * n + i, col_idx] = 1
            col_idx += 1
    
    return (U, V, W)

def _build_rank50(deadline):
    """
    Try to build a rank-50 decomposition: 5 blocks of 2x2x3, each with rank 10
    (Strassen rank 7 + rank-3 for third column).
    """
    n, m, p = 2, 2, 15
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_patterns = _strassen_patterns()
    
    # Find rank-3 completion for the 3rd column
    rank3 = _find_rank3_col3(deadline)
    if rank3 is None:
        return None
    U2, V2, W2 = rank3
    
    col_idx = 0
    for block in range(5):
        k_base = 3 * block  # columns k_base, k_base+1, k_base+2
        
        # Strassen for first two columns (k_base, k_base+1)
        for pat in strassen_patterns:
            for (i, j), uval in pat['U'].items():
                U[i * m + j, col_idx] = uval
            for (j, k), val in pat['V'].items():
                k_global = k_base + k  # k=0 -> k_base, k=1 -> k_base+1
                V[j * p + k_global, col_idx] = val
            for (k, i), wval in pat['W'].items():
                k_global = k_base + k
                W[k_global * n + i, col_idx] = wval
            col_idx += 1
        
        # Rank-3 for third column (k_base+2)
        k3 = k_base + 2
        for i in range(n):
            for j in range(m):
                for r in range(3):
                    U[i * m + j, col_idx + r] += U2[i * m + j, r]
            for j in range(m):
                for r in range(3):
                    V[j * p + k3, col_idx + r] += V2[j, r]
            for i in range(n):
                for r in range(3):
                    W[k3 * n + i, col_idx + r] += W2[i, r]
        col_idx += 3
    
    if col_idx != rank:
        return None
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return None
    
    return (U, V, W)

def _verify(U, V, W, n=2, m=2, p=15):
    """Verify the decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    Tries rank 50 (5 blocks of 2x2x3 with rank 10 each), falls back to rank 53.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # 5s safety margin
    
    # Try rank 50 first
    result = _build_rank50(deadline)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }
    
    # Fallback to rank 53
    if time.time() > deadline:
        # Just return the baseline without verification
        U, V, W = _build_rank53()
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    U, V, W = _build_rank53()
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
