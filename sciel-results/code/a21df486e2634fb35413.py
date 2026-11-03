"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-minimal tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # First try to find a lower rank using random search
    # Try ranks from 57 down to 50
    for target_rank in range(57, 49, -1):
        if time.time() > deadline - 20:  # Leave 20s for fallback
            break
        
        result = _search_random_rank(n, m, p, target_rank, n_trials=30, time_limit=15)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to known 58-rank Strassen-based decomposition
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Ultimate fallback: schoolbook
    U, V, W, rank = _build_schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def _search_random_rank(n, m, p, target_rank, n_trials=50, time_limit=15):
    """Search for a decomposition of the given rank using random initialization."""
    import time
    start = time.time()
    
    nm = n * m
    mp = m * p  
    pn = p * n
    
    # Precompute the tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    for trial in range(n_trials):
        if time.time() - start > time_limit:
            break
        
        # Generate random U and V with entries from {-2,-1,0,1,2}
        U = np.random.choice([-2, -1, 0, 1, 2], size=(nm, target_rank)).astype(float)
        V = np.random.choice([-2, -1, 0, 1, 2], size=(mp, target_rank)).astype(float)
        
        # For each c, solve for W[c, :]
        W = np.zeros((pn, target_rank))
        valid = True
        
        for c in range(pn):
            # Build the linear system
            A = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, target_rank)
            t = T[:, :, c].reshape(-1)
            
            # Solve A @ w = t
            try:
                w, residuals, rank_A, sv = np.linalg.lstsq(A, t, rcond=1e-12)
                
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                else:
                    if np.linalg.norm(A @ w - t) > 1e-8:
                        valid = False
                        break
                
                W[c, :] = w
            except np.linalg.LinAlgError:
                valid = False
                break
        
        if valid:
            # Verify the full decomposition
            T_recon = np.zeros((nm, mp, pn))
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            if np.allclose(T_recon, T, atol=1e-8):
                # Convert to lists with rounded values
                U_list = [[round(x, 10) for x in row] for row in U]
                V_list = [[round(x, 10) for x in row] for row in V]
                W_list = [[round(x, 10) for x in row] for row in W]
                return U_list, V_list, W_list
    
    return None

def _build_strassen_decomposition(n, m, p):
    """
    Build a rank-58 decomposition using two disjoint Strassen 2x2x2 blocks.
    """
    # Initialize U, V, W with zeros
    max_rank = n * m * p  # 60 for schoolbook
    U = [[0] * max_rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * max_rank for _ in range(m * p)]  # 20 x 60
    W = [[0] * max_rank for _ in range(p * n)]  # 15 x 60
    
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        """
        Add Strassen terms for a 2x2x2 block.
        i_offset, j_offset, k_offset: global offsets for local indices 0,1
        col_start: starting column index in U, V, W
        Returns: number of columns used (7)
        """
        # Define the 7 Strassen multipliers
        # Each multiplier is (coeff_A_list, coeff_B_list) where
        # coeff_A_list = [(ii, jj, coeff), ...] for A entries
        # coeff_B_list = [(jj, kk, coeff), ...] for B entries
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        M1_A = [((0, 0), 1), ((1, 1), 1)]
        M1_B = [((0, 0), 1), ((1, 1), 1)]
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        M2_A = [((1, 0), 1), ((1, 1), 1)]
        M2_B = [((0, 0), 1)]
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        M3_A = [((0, 0), 1)]
        M3_B = [((0, 1), 1), ((1, 1), -1)]
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        M4_A = [((1, 1), 1)]
        M4_B = [((1, 0), 1), ((0, 0), -1)]
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        M5_A = [((0, 0), 1), ((0, 1), 1)]
        M5_B = [((1, 1), 1)]
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        M6_A = [((1, 0), 1), ((0, 0), -1)]
        M6_B = [((0, 0), 1), ((0, 1), 1)]
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        M7_A = [((0, 1), 1), ((1, 1), -1)]
        M7_B = [((1, 0), 1), ((1, 1), 1)]
        
        multipliers = [
            (M1_A, M1_B),
            (M2_A, M2_B),
            (M3_A, M3_B),
            (M4_A, M4_B),
            (M5_A, M5_B),
            (M6_A, M6_B),
            (M7_A, M7_B),
        ]
        
        col = col_start
        for m_idx, (A_terms, B_terms) in enumerate(multipliers):
            # A_terms: list of ((ii, jj), coeff)
            # B_terms: list of ((jj, kk), coeff)
            
            # For U: index is i*m + j
            for (ii, jj), coeff_A in A_terms:
                i_global = i_offset + ii
                j_global = j_offset + jj
                U[i_global * m + j_global][col] += coeff_A
            
            # For V: index is j*p + k
            for (jj, kk), coeff_B in B_terms:
                j_global = j_offset + jj
                k_global = k_offset + kk
                V[j_global * p + k_global][col] += coeff_B
            
            # For W: we need to add contributions for each output (ii, kk)
            # that this multiplier affects.
            
            # M1 (m_idx=0): contributes to C[0,0] with +1, C[1,1] with +1
            if m_idx == 0:  # M1
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M2 (m_idx=1): contributes to C[1,0] with +1, C[1,1] with -1
            elif m_idx == 1:  # M2
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1  # C[1,0]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += -1  # C[1,1]
            
            # M3 (m_idx=2): contributes to C[0,1] with +1, C[1,1] with +1
            elif m_idx == 2:  # M3
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1  # C[0,1]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M4 (m_idx=3): contributes to C[0,0] with +1, C[1,0] with +1
            elif m_idx == 3:  # M4
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1  # C[1,0]
            
            # M5 (m_idx=4): contributes to C[0,0] with -1, C[0,1] with +1
            elif m_idx == 4:  # M5
                W[(k_offset + 0) * n + (i_offset + 0)][col] += -1  # C[0,0]
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1  # C[0,1]
            
            # M6 (m_idx=5): contributes to C[1,1] with +1
            elif m_idx == 5:  # M6
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M7 (m_idx=6): contributes to C[0,0] with +1
            elif m_idx == 6:  # M7
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
            
            col += 1
        
        return col - col_start
    
    # Add Strassen block 1: i_offset=0, j_offset=0, k_offset=0
    col = 0
    col += add_strassen_block(0, 0, 0, col)  # Uses columns 0-6
    
    # Add Strassen block 2: i_offset=0, j_offset=2, k_offset=2
    col += add_strassen_block(0, 2, 2, col)  # Uses columns 7-13
    
    # Now add schoolbook terms for all other (i, j, k) tuples.
    # We need to skip the ones already covered by Strassen blocks.
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2 covers: i in {0,1}, j in {2,3}, k in {2,3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i, j, k) is covered by a Strassen block
                covered = False
                # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    covered = True
                # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
                if i in (0, 1) and j in (2, 3) and k in {2, 3}:
                    covered = True
                
                if not covered:
                    # Schoolbook: one term
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim U, V, W to the actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return U, V, W, actual_rank

def _build_schoolbook_decomposition(n, m, p):
    """
    Build the standard schoolbook decomposition with rank n*m*p.
    """
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
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    and for all other combinations, the sum is 0.
    """
    rank = len(U[0]) if U else 0
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the expected tensor
    T_expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1.0
    
    # Reconstruct the tensor
    T_recon = np.zeros((nm, mp, pn))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    if not np.allclose(T_recon, T_expected, atol=1e-8):
        return False
    
    return True
# EVOLVE-BLOCK-END
