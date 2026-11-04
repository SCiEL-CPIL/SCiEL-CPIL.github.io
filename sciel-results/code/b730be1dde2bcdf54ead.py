import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# Cache for optimal ranks and decompositions
_decomp_cache = {}

def _get_rank(n, m, p):
    """Calculate the minimal rank for n x m x p using recursive splitting."""
    if (n, m, p) in _decomp_cache:
        return _decomp_cache[(n, m, p)]['rank']
    
    # Base case: schoolbook
    best_rank = n * m * p
    best_split = None
    
    # Try all possible splits
    # We only need to try splits if n, m, p are large enough to benefit
    # For small sizes, schoolbook is often optimal or close to it
    if n > 1 and m > 1 and p > 1:
        for n1 in range(1, n):
            n2 = n - n1
            for m1 in range(1, m):
                m2 = m - m1
                for p1 in range(1, p):
                    p2 = p - p1
                    # Strassen-like combination of 8 blocks
                    # R = R(n1,m1,p1) + R(n1,m1,p2) + R(n1,m2,p1) + R(n1,m2,p2) + 
                    #     R(n2,m1,p1) + R(n2,m1,p2) + R(n2,m2,p1) + R(n2,m2,p2)
                    # Note: This is NOT Strassen. Strassen reduces 8 to 7.
                    # Standard recursive Strassen requires equal splits.
                    # For arbitrary splits, we just sum the ranks of the 8 sub-problems.
                    # This is equivalent to schoolbook if we don't use Strassen's 7-product trick.
                    # So this DP only helps if we have a better algorithm for sub-problems.
                    
                    # Actually, for arbitrary rectangular splits, we cannot use Strassen's 7-product formula directly 
                    # because the blocks don't align for addition.
                    # We MUST use schoolbook for non-power-of-2 or non-equal splits unless we pad.
                    
                    # So, the only way to get better than schoolbook is to pad to powers of 2 and use Strassen,
                    # OR use specific algorithms like Laderman for 3x3x3.
                    
                    pass
    
    # Since generic recursive splitting without padding is just schoolbook,
    # we should check if padding to a power of 2 is beneficial.
    
    # Pad to next power of 2
    def next_pow2(x):
        if x <= 1: return 1
        return 1 << (x - 1).bit_length()
    
    n_pad = next_pow2(n)
    m_pad = next_pow2(m)
    p_pad = next_pow2(p)
    
    if n_pad == n and m_pad == m and p_pad == p:
        # Already power of 2, use Strassen
        if n == 1:
            return 1
        half = n // 2
        r_sub = _get_rank(half, half, half)
        r_strassen = 7 * r_sub
        if r_strassen < best_rank:
            best_rank = r_strassen
    
    # Check Laderman for 3x3x3
    if n == 3 and m == 3 and p == 3:
        if 23 < best_rank:
            best_rank = 23

    _decomp_cache[(n, m, p)] = {'rank': best_rank}
    return best_rank

def _construct_decomposition(n, m, p):
    """Construct U, V, W for n x m x p."""
    if (n, m, p) in _decomp_cache and 'U' in _decomp_cache[(n, m, p)]:
        return _decomp_cache[(n, m, p)]['U'], _decomp_cache[(n, m, p)]['V'], _decomp_cache[(n, m, p)]['W']
    
    # Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    _decomp_cache[(n, m, p)] = {'U': U, 'V': V, 'W': W, 'rank': rank}
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # For 4x5x5, we use the block decomposition:
    # Split n=4 into 2+2.
    # For each 2x5x5, split m=5 into 2+3, p=5 into 2+3.
    # Use Strassen for 2x2x2, Schoolbook for others.
    
    def _strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
        U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=-1
        U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[0,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[3,6]=1
        return U, V, W

    def _schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Build 2x5x5 decomposition
    def _build_2x5x5():
        n1, m1, p1 = 2, 2, 2
        n2, m2, p2 = 2, 3, 3
        # Splits:
        # 1. 2x2x2 (Strassen)
        # 2. 2x2x3 (Schoolbook)
        # 3. 2x3x2 (Schoolbook)
        # 4. 2x3x3 (Schoolbook)
        
        U1, V1, W1 = _strassen_2x2()
        U2, V2, W2 = _schoolbook(2, 2, 3)
        U3, V3, W3 = _schoolbook(2, 3, 2)
        U4, V4, W4 = _schoolbook(2, 3, 3)
        
        total_rank = 7 + 12 + 12 + 18
        U_full = np.zeros((10, total_rank), dtype=float)
        V_full = np.zeros((25, total_rank), dtype=float)
        W_full = np.zeros((10, total_rank), dtype=float)
        
        col = 0
        # 1. 2x2x2
        # A: 0-1, 0-1. B: 0-1, 0-1. C: 0-1, 0-1.
        U_full[0:4, col:col+7] = U1
        V_full[0:4, col:col+7] = V1
        W_full[0:4, col:col+7] = W1
        col += 7
        
        # 2. 2x2x3
        # A: 0-1, 0-1. B: 0-1, 2-4. C: 2-4, 0-1.
        U_full[0:4, col:col+12] = U2
        V_full[0:2, col:col+12] = V2[0:2, :] # B rows 0-1, cols 0-2 (local) -> global cols 2-4
        # V2 is 2x3 schoolbook. V2[j*3+k, c].
        # Global V index: j*5 + (2+k).
        for c in range(12):
            for j in range(2):
                for k in range(3):
                    V_full[j * 5 + (2 + k), col + c] = V2[j * 3 + k, c]
        # W2 is 3x2 schoolbook. W2[k*2+i, c].
        # Global W index: (2+k)*2 + i.
        for c in range(12):
            for k in range(3):
                for i in range(2):
                    W_full[(2 + k) * 2 + i, col + c] = W2[k * 2 + i, c]
        col += 12
        
        # 3. 2x3x2
        # A: 0-1, 2-4. B: 2-4, 0-1. C: 0-1, 0-1.
        U_full[4:10, col:col+12] = U3
        # U3 is 2x3 schoolbook. U3[i*3+j, c].
        # Global U index: i*5 + (2+j).
        for c in range(12):
            for i in range(2):
                for j in range(3):
                    U_full[i * 5 + (2 + j), col + c] = U3[i * 3 + j, c]
        # V3 is 3x2 schoolbook. V3[j*2+k, c].
        # Global V index: (2+j)*5 + k.
        for c in range(12):
            for j in range(3):
                for k in range(2):
                    V_full[(2 + j) * 5 + k, col + c] = V3[j * 2 + k, c]
        # W3 is 2x2 schoolbook. W3[k*2+i, c].
        # Global W index: k*2 + i.
        W_full[0:4, col:col+12] = W3
        col += 12
        
        # 4. 2x3x3
        # A: 0-1, 2-4. B: 2-4, 2-4. C: 2-4, 0-1.
        U_full[4:10, col:col+18] = U4
        for c in range(18):
            for i in range(2):
                for j in range(3):
                    U_full[i * 5 + (2 + j), col + c] = U4[i * 3 + j, c]
        for c in range(18):
            for j in range(3):
                for k in range(3):
                    V_full[(2 + j) * 5 + (2 + k), col + c] = V4[j * 3 + k, c]
        for c in range(18):
            for k in range(3):
                for i in range(2):
                    W_full[(2 + k) * 2 + i, col + c] = W4[k * 2 + i, c]
        col += 18
        
        assert col == total_rank
        return U_full, V_full, W_full

    U1, V1, W1 = _build_2x5x5()
    U2, V2, W2 = _build_2x5x5()
    
    total_rank = 48 + 48
    U_full = np.zeros((20, total_rank), dtype=float)
    V_full = np.zeros((25, total_rank), dtype=float)
    W_full = np.zeros((20, total_rank), dtype=float)
    
    # Block 1: A rows 0-1
    U_full[0:10, 0:48] = U1
    V_full[0:25, 0:48] = V1
    W_full[0:10, 0:48] = W1
    
    # Block 2: A rows 2-3
    U_full[10:20, 48:96] = U2
    V_full[0:25, 48:96] = V2
    # W2 corresponds to C rows 0-1 of the sub-block, which are global C rows 2-3.
    # W_full index: k*4 + i.
    # Sub-block W2 index: k*2 + i.
    # Global: k*4 + (2+i).
    for c in range(48):
        for k in range(5):
            for i in range(2):
                W_full[k * 4 + (2 + i), 48 + c] = W2[k * 2 + i, c]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# EVOLVE-BLOCK-START
def construct():
    return _construct_decomposition(4, 5, 5)
# EVOLVE-BLOCK-END
