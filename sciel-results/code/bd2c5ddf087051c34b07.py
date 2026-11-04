"""Matrix multiplication 3x5x5 - Rank-minimal exact decomposition search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """
    Return the 7 Strassen patterns for 2x2x2 matrix multiplication.
    Each pattern is a dict with 'U', 'V', 'W' lists of (local_i, local_j, value) tuples.
    For U: (i, j, val) means U[i, j] += val in the local 2x2 block.
    For V: (j, k, val) means V[j, k] += val in the local 2x2 block.
    For W: (k, i, val) means W[k, i] += val in the local 2x2 block.
    
    Verified via direct tensor check.
    """
    # Standard Strassen: 7 products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B00-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5-P7, C01=P3+P5, C10=P2+P7, C11=P1-P2+P3+P6
    
    patterns = []
    
    # P1: U: A00=1,A11=1; V: B00=1,B11=1; W: C00=+1,C01=-1,C10=-1,C11=+1
    patterns.append({
        'U': [(0,0,1), (1,1,1)],
        'V': [(0,0,1), (1,1,1)],
        'W': [(0,0,1), (0,1,-1), (1,0,-1), (1,1,1)]
    })
    
    # P2: U: A10=1,A11=1; V: B00=1; W: C10=+1, C11=-1
    patterns.append({
        'U': [(1,0,1), (1,1,1)],
        'V': [(0,0,1)],
        'W': [(1,0,1), (1,1,-1)]
    })
    
    # P3: U: A00=1; V: B10=1,B11=-1; W: C01=+1, C11=+1
    patterns.append({
        'U': [(0,0,1)],
        'V': [(1,0,1), (1,1,-1)],
        'W': [(0,1,1), (1,1,1)]
    })
    
    # P4: U: A11=1; V: B00=1,B10=-1; W: C00=+1
    patterns.append({
        'U': [(1,1,1)],
        'V': [(0,0,1), (1,0,-1)],
        'W': [(0,0,1)]
    })
    
    # P5: U: A00=1,A01=1; V: B11=1; W: C00=-1, C01=+1
    patterns.append({
        'U': [(0,0,1), (0,1,1)],
        'V': [(1,1,1)],
        'W': [(0,0,-1), (0,1,1)]
    })
    
    # P6: U: A10=1,A00=-1; V: B00=1,B01=1; W: C11=+1
    patterns.append({
        'U': [(1,0,1), (0,0,-1)],
        'V': [(0,0,1), (0,1,1)],
        'W': [(1,1,1)]
    })
    
    # P7: U: A01=1,A11=-1; V: B10=1,B11=1; W: C00=-1, C10=+1
    patterns.append({
        'U': [(0,1,1), (1,1,-1)],
        'V': [(1,0,1), (1,1,1)],
        'W': [(0,0,-1), (1,0,1)]
    })
    
    return patterns

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(d, [])
    return results

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits."""
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    rank += 7
                else:
                    rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p, use_strassen=True):
    """Build the full decomposition for given splits."""
    # Calculate offsets for each block
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    rank = _compute_rank(n_splits, m_splits, p_splits)
    
    # Initialize matrices with zeros
    # Using lists of lists for flexibility, will convert to numpy for verification if needed
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_patterns()
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2 and use_strassen:
                    # Apply Strassen patterns
                    for pattern in strassen_patterns:
                        # Fill U
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] += val
                        # Fill V
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] += val
                        # Fill W
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] += val
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] += 1
                                V[gj * p + gk][col] += 1
                                W[gk * n + gi][col] += 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition using efficient block checks."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Efficient verification: for each (i,j,k) check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # And for invalid triples, check they're 0
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Compute sum_r U[u,r]*V[v,r]*W[w,r]
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    
    # Check invalid entries - only need to check a representative subset for speed
    # Invalid: (i,j,k) where the indices don't align for matmul
    # u=(i,j), v=(j2,k), w=(k2,i2) is valid iff i==i2 and j==j2
    # So invalid when i != i2 OR j != j2
    
    # For efficiency, check all combinations but use vectorized ops
    # There are n*m * m*p * p*n = n^2 * m^2 * p^2 total, but only n*m*p are valid
    # For 3x5x5: 225*25*15 = 84375 total, 75 valid, 84300 invalid
    
    # Use a smarter approach: check that the reconstructed matrix product is correct
    # by verifying the tensor equation more cleverly
    
    # Actually, let's just check all invalid entries but with early exit
    for u in range(n * m):
        i = u // m
        j = u % m
        for v in range(m * p):
            j2 = v // p
            k = v % p
            for w in range(p * n):
                k2 = w // n
                i2 = w % n
                if i != i2 or j != j2:
                    val = np.dot(U_np[u, :], V_np[v, :] * W_np[w, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
    
    return True

def _test_strassen():
    """Quick test that Strassen patterns produce valid 2x2x2 decomposition."""
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    patterns = _strassen_patterns()
    for col, pattern in enumerate(patterns):
        for li, lj, val in pattern['U']:
            U[li * m + lj][col] += val
        for lj, lk, val in pattern['V']:
            V[lj * p + lk][col] += val
        for lk, li, val in pattern['W']:
            W[lk * n + li][col] += val
    
    return _verify_decomposition(U, V, W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 150.0  # Use full budget with safety margin
    
    # Quick sanity check on Strassen
    strassen_ok = _test_strassen()
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute schoolbook factors as fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    school_U, school_V, school_W = get_schoolbook()
    best_U, best_V, best_W = school_U, school_V, school_W
    
    # Sort splits by resulting rank to find best first
    def split_key(splits):
        # Prefer splits with more 2's (can use Strassen)
        count_2 = sum(1 for s in splits if s == 2)
        return -count_2
    
    n_splits_list.sort(key=split_key)
    m_splits_list.sort(key=split_key)
    p_splits_list.sort(key=split_key)
    
    # Try all combinations of splits, sorted by expected rank
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank ascending to try best first
    candidates.sort(key=lambda x: x[0])
    
    # Try candidates in order of rank
    for rank, n_splits, m_splits, p_splits in candidates:
        if time.time() > deadline - 20:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p, use_strassen=strassen_ok)
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            # If we found the theoretical minimum, we can stop
            # Theoretical min for 3x5x5 with Strassen on 2x2x2 blocks:
            # Best split of 3: [2,1], 5: [2,2,1], 5: [2,2,1]
            # 2x2x2: 7, 2x2x1: 4, 2x1x2: 4, 2x1x1: 2, 1x2x2: 4, 1x2x1: 2, 1x1x2: 2, 1x1x1: 1
            # For [2,1] x [2,2,1] x [2,2,1]:
            # (2,2,2)=7, (2,2,1)=4, (2,1,2)=4, (2,1,1)=2, (1,2,2)=4, (1,2,1)=2, (1,1,2)=2, (1,1,1)=1
            # Total: 7+4+4+2+4+2+2+1 = 26
            if best_rank <= 26:
                break
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
# EVOLVE-BLOCK-END
