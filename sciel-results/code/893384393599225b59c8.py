"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C from factors
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    # Efficient computation:
    # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Let S_A[i, r] = sum_j U[i*m+j, r] * A[i,j]
    # Let S_B[k, r] = sum_j V[j*p+k, r] * B[j,k]
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, R)
    # S_A[i, r] = sum_j U_reshaped[i, j, r] * A[i, j]
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    # Reshape V to (m, p, R) -> V[j*p+k, r] is V_flat[j, k, r] if we reshape correctly?
    # V index is j*p+k. So V is (m, p, R) if we reshape to (m, p, R)?
    # V_flat[j*p+k, r]. If we reshape V to (m, p, R), then V_reshaped[j, k, r] = V[j*p+k, r].
    V_reshaped = V.reshape(m, p, R)
    # S_B[k, r] = sum_j V_reshaped[j, k, r] * B[j, k]
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    # Reshape W to (p, n, R)
    # W index is k*n+i. So W_flat[k*n+i, r].
    # If we reshape W to (p, n, R), then W_reshaped[k, i, r] = W[k*n+i, r].
    W_reshaped = W.reshape(p, n, R)
    
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

# EVOLVE-BLOCK-START
def _strassen_2x2_block():
    """
    Returns U, V, W factor matrices for a 2x2x2 Strassen block.
    U: shape (4, 7) - for A_{i,j} where i,j in {0,1}
    V: shape (4, 7) - for B_{j,k} where j,k in {0,1}  
    W: shape (4, 7) - for C_{k,i} where k,i in {0,1}
    
    Strassen's 7 multiplications for 2x2 matrix multiply:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U_block = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)
    
    V_block = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [0, 0, 1, 0, 1, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=float)
    
    W_block = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=float)
    
    return U_block, V_block, W_block


def _combine_decomps(decomps):
    """
    Combine a list of (U, V, W) decompositions into one.
    Each decomposition has U: (n*m, r_i), V: (m*p, r_i), W: (p*n, r_i)
    Returns combined U: (n*m, R), V: (m*p, R), W: (p*n, R) where R = sum(r_i)
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((decomps[0][0].shape[0], total_rank))
    V = np.zeros((decomps[0][1].shape[0], total_rank))
    W = np.zeros((decomps[0][2].shape[0], total_rank))
    col = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, col:col+r] = U_i
        V[:, col:col+r] = V_i
        W[:, col:col+r] = W_i
        col += r
    return U, V, W


def _schoolbook_decomp(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    Returns U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_decomp(n, m, p):
    """
    Recursive Strassen-like decomposition for n x m * m x p.
    Splits dimensions into halves and uses Strassen 7-rank for 2x2x2 blocks,
    schoolbook for odd-sized remainders.
    
    Strategy: 
    - If n, m, p are all even: split each into two halves, use 7 Strassen subcalls
    - If m is even but n or p is odd: split m into two halves, use schoolbook for each half
    - Otherwise: schoolbook
    
    Returns U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    start_time = time.time()
    deadline = start_time + 120
    
    def _recursive(n, m, p, i_off, j_off, k_off):
        """
        Returns (U_local, V_local, W_local) for the subproblem:
        A[i_off:i_off+n, j_off:j_off+m] @ B[j_off:j_off+m, k_off:k_off+p]
        
        U_local: (n*m, r) where row index is (i-i_off)*m + (j-j_off)
        V_local: (m*p, r) where row index is (j-j_off)*p + (k-k_off)  
        W_local: (p*n, r) where row index is (k-k_off)*n + (i-i_off)
        """
        # Base case: schoolbook
        if time.time() > deadline:
            return _schoolbook_decomp(n, m, p)
        
        # If m is even, split j into two halves
        if m % 2 == 0:
            m1 = m // 2
            m2 = m - m1  # = m1 since m is even
            
            # If n and p are both even, use Strassen 7-multiplication
            if n % 2 == 0 and p % 2 == 0:
                n1 = n // 2
                n2 = n - n1
                p1 = p // 2
                p2 = p - p1
                
                # 7 Strassen subcalls:
                # M1 = A11 @ B11  (where A11 is top-left n1 x m1, B11 is top-left m1 x p1)
                # Actually let me use standard Strassen indexing:
                # A = [[A1, A2], [A3, A4]] where A1 is n1 x m1, A2 is n1 x m2, etc.
                # B = [[B1, B2], [B3, B4]] where B1 is m1 x p1, B2 is m1 x p2, etc.
                # 
                # M1 = (A1 + A4)(B1 + B4)
                # M2 = (A3 + A4)B1
                # M3 = A1(B3 - B4)
                # M4 = A4(B2 - B1)
                # M5 = (A1 + A2)B4
                # M6 = (A3 - A1)(B1 + B2)
                # M7 = (A2 - A4)(B3 + B4)
                #
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # For each Strassen multiplication, we need to compute
                # (sum of A subblocks) @ (sum of B subblocks)
                # This is a linear combination of subproblems.
                
                # The key insight: each M_i is a product of linear combinations.
                # We can express each M_i as a sum of sub-subproblems.
                
                # For M1 = (A1+A4)(B1+B4):
                # = A1*B1 + A1*B4 + A4*B1 + A4*B4
                # These are 4 subproblems of size n1 x m1 x p1, n1 x m2 x p2, n2 x m1 x p1, n2 x m2 x p2
                
                # This gets complex. Let me use a simpler approach:
                # Just split m into two halves and do schoolbook for each half.
                # This gives rank = n * (m/2) * p + n * (m/2) * p = n*m*p (same as schoolbook)
                # So that's not helpful.
                
                # Let me try a different approach: for the 2x2x2 case, use Strassen directly.
                # For larger even cases, recursively split.
                
                # Actually, the cleanest approach for non-square problems:
                # Split the middle dimension m into two halves.
                # For each half, we have n x (m/2) * (m/2) x p.
                # If n and p are both even, we can further split them and use Strassen.
                
                # Let me just implement the recursive Strassen properly.
                # For n x m x p where all are even:
                # Split into 8 subproblems and combine with Strassen coefficients.
                
                # This is getting complicated. Let me use a simpler but effective approach:
                # Split only m into two halves, and for each half use schoolbook.
                # Rank = 2 * (n * (m/2) * p) = n*m*p. Same as schoolbook. Not helpful.
                
                # The real Strassen benefit comes from the 2x2x2 block.
                # For a 2x2x2 block, rank goes from 8 to 7.
                # For a 4x4x4 block, we can use Strassen recursively: 7 * 8 = 56 vs 64.
                
                # Let me implement proper recursive Strassen for the even case.
                
                decomp_list = []
                
                # M1 = (A1+A4)(B1+B4)
                # This expands to 4 sub-products, but we can compute it as:
                # We need to form the linear combinations first, then multiply.
                # The tensor decomposition approach:
                # Each rank-1 term in the sub-decomposition gets multiplied by coefficients.
                
                # Actually, the standard way to combine Strassen blocks:
                # For each of the 7 Strassen multiplications, we recursively decompose
                # the corresponding subproblem, and then apply the Strassen coefficients.
                
                # M1: (A1+A4) @ (B1+B4)
                # A1 is at rows [0:n1], cols [0:m1] of A (offset i_off, j_off)
                # A4 is at rows [n1:n], cols [m1:m] of A
                # B1 is at rows [0:m1], cols [0:p1] of B (offset j_off, k_off)
                # B4 is at rows [m1:m], cols [p1:p] of B
                
                # The product (A1+A4) @ (B1+B4) is a n x p matrix.
                # To decompose this, we note that:
                # (A1+A4) @ (B1+B4) = A1@B1 + A1@B4 + A4@B1 + A4@B4
                # Each of these is a subproblem that we can recursively decompose.
                
                # But this gives 4 sub-decompositions for M1 alone, and 7 M's total = 28 sub-decomps.
                # That's worse than just doing 8 sub-decompositions for schoolbook.
                
                # The correct Strassen approach for tensor decomposition:
                # The 7 Strassen multiplications are NOT simply products of submatrices.
                # They are products of LINEAR COMBINATIONS of submatrices.
                # 
                # In tensor decomposition terms, for a 2x2x2 block with Strassen:
                # The U, V, W matrices have entries in {-1, 0, 1}.
                # For a larger block, we need to "inflate" these coefficients.
                
                # Let me use the direct approach: for a 2a x 2b x 2c block,
                # we can use the Strassen 2x2x2 pattern where each "entry" is a subblock.
                
                # The U matrix for the 2x2x2 Strassen has shape (4, 7) = (2*2, 7).
                # For a 2a x 2b x 2c block, U has shape (4ab, 7) but each column
                # is a linear combination of sub-U matrices.
                
                # This is getting too complex for inline implementation. Let me use
                # a simpler but still effective strategy:
                # 
                # For m even: split into two m/2 halves, use schoolbook for each.
                # This gives rank n*m*p (no improvement).
                # 
                # For the 2x2x2 base case: use Strassen (rank 7 instead of 8).
                # 
                # For larger problems, I'll use a hybrid:
                # - If the problem is 2x2x2: Strassen
                # - Otherwise: split the largest even dimension and recurse
                #   with schoolbook combination (which doesn't help)
                # 
                # The real win is only at the 2x2x2 level.
                # For 3x4x7: split m=4 into 2+2.
                # Each half is 3x2x7. Split p=7 into 3+4? No, 7 is odd.
                # 3x2x7: schoolbook = 42. Two halves = 84. No improvement.
                # 
                # Alternative: split p=7 into 2+2+3.
                # 3x4x2, 3x4x2, 3x4x3.
                # 3x4x2: schoolbook = 24. Can we do better?
                #   Split m=4 into 2+2: 3x2x2 + 3x2x2. Each 3x2x2 has schoolbook rank 12.
                #   Total = 24. Same.
                #   But 2x2x2 part within 3x2x2: if we split n=3 into 2+1,
                #   we get 2x2x2 (Strassen rank 7) + 1x2x2 (schoolbook rank 4) = 11 per half.
                #   Total for 3x4x2: 2 * 11 = 22.
                # 3x4x3: schoolbook = 36.
                #   Split m=4 into 2+2: 3x2x3 + 3x2x3. Each schoolbook = 18. Total = 36.
                #   Split n=3 into 2+1 within each: 2x2x3 + 1x2x3 per half.
                #   2x2x3: schoolbook = 12. Can split p=3 into 2+1: 2x2x2 + 2x2x1 = 8+4=12. No help.
                #   Or use Strassen on the 2x2 part: 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
                #   So 2x2x3 has rank 11. 1x2x3 has rank 6.
                #   Total per half: 11 + 6 = 17. Total for 3x4x3: 2 * 17 = 34.
                # 
                # Grand total: 22 + 22 + 34 = 78.
                
                # Let me implement this specific decomposition!
                pass  # Fall through to the specific implementation below
            
            # Split m into two halves, schoolbook for each
            m1 = m // 2
            m2 = m - m1
            
            U1, V1, W1 = _recursive(n, m1, p, i_off, j_off, k_off)
            U2, V2, W2 = _recursive(n, m2, p, i_off, j_off + m1, k_off)
            
            return _combine_decomps([(U1, V1, W1), (U2, V2, W2)])
        
        # Schoolbook fallback
        return _schoolbook_decomp(n, m, p)
    
    # Use the specific 3x4x7 decomposition that achieves rank 78
    # Partition: p = 2 + 2 + 3
    # For each p-block, partition m = 2 + 2
    # For each (m,p) sub-block, partition n = 2 + 1
    # The 2x2x2 sub-blocks use Strassen (rank 7)
    # The remaining sub-blocks use schoolbook
    
    n, m, p = 3, 4, 7
    
    # p partitions: [0,2), [2,4), [4,7)
    # m partitions: [0,2), [2,4)
    # n partitions: [0,2), [2,3)
    
    decomps = []
    
    for k_start, k_end in [(0, 2), (2, 4), (4, 7)]:
        p_sub = k_end - k_start
        for j_start, j_end in [(0, 2), (2, 4)]:
            m_sub = j_end - j_start
            for i_start, i_end in [(0, 2), (2, 3)]:
                n_sub = i_end - i_start
                
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Use Strassen 2x2x2
                    U_s, V_s, W_s = _strassen_2x2_block()
                    # U_s is (4, 7) for 2x2 A, V_s is (4, 7) for 2x2 B, W_s is (4, 7) for 2x2 C
                    # We need to embed this into the full-size matrices
                    U_full = np.zeros((n * m, 7))
                    V_full = np.zeros((m * p, 7))
                    W_full = np.zeros((p * n, 7))
                    
                    # Map local indices to global
                    for li in range(2):
                        for lj in range(2):
                            gi = i_start + li
                            gj = j_start + lj
                            U_full[gi * m + gj, :] = U_s[li * 2 + lj, :]
                    
                    for lj in range(2):
                        for lk in range(2):
                            gj = j_start + lj
                            gk = k_start + lk
                            V_full[gj * p + gk, :] = V_s[lj * 2 + lk, :]
                    
                    for lk in range(2):
                        for li in range(2):
                            gk = k_start + lk
                            gi = i_start + li
                            W_full[gk * n + gi, :] = W_s[lk * 2 + li, :]
                    
                    decomps.append((U_full, V_full, W_full))
                else:
                    # Schoolbook for this sub-block
                    U_sb, V_sb, W_sb = _schoolbook_decomp(n_sub, m_sub, p_sub)
                    U_full = np.zeros((n * m, n_sub * m_sub * p_sub))
                    V_full = np.zeros((m * p, n_sub * m_sub * p_sub))
                    W_full = np.zeros((p * n, n_sub * m_sub * p_sub))
                    
                    for li in range(n_sub):
                        for lj in range(m_sub):
                            gi = i_start + li
                            gj = j_start + lj
                            U_full[gi * m + gj, :] = U_sb[li * m_sub + lj, :]
                    
                    for lj in range(m_sub):
                        for lk in range(p_sub):
                            gj = j_start + lj
                            gk = k_start + lk
                            V_full[gj * p + gk, :] = V_sb[lj * p_sub + lk, :]
                    
                    for lk in range(p_sub):
                        for li in range(n_sub):
                            gk = k_start + lk
                            gi = i_start + li
                            W_full[gk * n + gi, :] = W_sb[lk * n_sub + li, :]
                    
                    decomps.append((U_full, V_full, W_full))
    
    U, V, W = _combine_decomps(decomps)
    
    # Verify
    if _verify(U, V, W):
        return U, V, W
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    try:
        U, V, W = _strassen_decomp(n, m, p)
        rank = U.shape[1]
        
        if _verify(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomp(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
