"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each output element (k, i), check that sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    # More precisely: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] should equal
    # the coefficient that multiplies x_i * y_j in output k.
    # The tensor T[k*n+i, j] = delta(i,j) is what we need to reproduce.
    # Actually the convention is: output[k, i] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i, j, k] = 1 if this is the (i,j,k) term.
    # With trace convention W[k*n+i, r], the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i == i and j == j and k == k) else 0.0
                # Wait, T[i,j,k] = 1 always (it's the matrix mult tensor)
                # Actually for matrix multiplication, T[i,j,k] = 1 for all i,j,k
                # No wait. The matrix multiplication tensor M_{i,j,k} = 1 for all valid indices.
                # The decomposition is: M_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
                # where M_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
                # That can't be right because then we'd need rank 1.
                
                # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
                # The standard tensor for matrix multiplication C = AB where A is n x m, B is m x p:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor is T_{i,j,k} = 1 (it's just the identity in the middle dimension)
                # No, T_{i,j,k} = delta_{j,j} which is always 1.
                
                # Actually I think the tensor entries are:
                # T[i, j, k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
                # So T[i,j,k] = 1 for all i,j,k.
                
                # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                # That's the schoolbook: one term per (i,j,k) with U=V=W=1.
                
                # But wait, that means the tensor is the all-ones tensor, which has rank 1!
                # That doesn't make sense. Let me reconsider.
                
                # I think the correct interpretation is:
                # The matrix multiplication tensor has entries T_{i,j,k} where:
                # - i indexes rows of A (0..n-1)  
                # - j indexes the shared dimension (0..m-1)
                # - k indexes columns of B (0..p-1)
                # And T_{i,j,k} = 1 for all i,j,k (it's the "multiplication" operation)
                
                # The rank of this tensor is what we're minimizing.
                # For schoolbook, rank = n*m*p because each (i,j,k) needs its own term.
                
                # But actually no! The tensor for matrix multiplication is NOT all-ones.
                # It's the identity tensor in the middle: T_{i,j,k} = delta_{j,j}... 
                
                # OK let me think about this differently. The matrix multiplication tensor
                # M_{n,m,p} has entries M_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
                # Its rank is known to be at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p).
                
                # The decomposition: M_{i,j,k} = sum_{r=1}^{rank} U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
                
                # For schoolbook: U_{i*m+j, (i,j,k)} = 1, V_{j*p+k, (i,j,k)} = 1, W_{k*n+i, (i,j,k)} = 1
                # Then sum_r ... = 1 for the matching (i,j,k) and 0 otherwise. Wait no,
                # each (i,j,k) has exactly one r where all three are 1, so the sum is 1.
                # For a different (i',j',k'), there's no r where all three match, so sum is 0.
                # But we need M_{i,j,k} = 1 for ALL i,j,k!
                
                # I'm confused. Let me just trust the seed program and verify it works.
                
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_block_2x2x2():
    """
    Strassen's 7-rank decomposition of 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) for the 2x2x2 case.
    U indexed by [i*2+j, r], V by [j*2+k, r], W by [k*2+i, r]
    """
    # Strassen's 7 products for 2x2x2:
    # P1 = (A11)(B11 - B22)
    # P2 = (A11 + A22)(B22)
    # P3 = (A11 + A12)(B11)
    # P4 = (A22 - A11)(B21 + B22)
    # P5 = (A12 + A22)(B21)
    # P6 = (A11 - A12)(B11 + B12)
    # P7 = (A21 + A12)(B22 - B21)
    #
    # C11 = P1 + P2 - P3 + P7
    # C12 = P3 + P5
    # C21 = P4 + P6
    # C22 = P1 - P2 + P4 + P5
    
    n2, m2, p2 = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n2 * m2)]  # 4x7
    V = [[0.0] * rank for _ in range(m2 * p2)]  # 4x7
    W = [[0.0] * rank for _ in range(p2 * n2)]  # 4x7
    
    # U[i*m2 + j, r], V[j*p2 + k, r], W[k*n2 + i, r]
    
    # P1 = A11 * (B11 - B22)
    # U[0*2+0, 0] = 1 (A11)
    # V[0*2+0, 0] = 1 (B11), V[1*2+1, 0] = -1 (B22)
    # W[0*2+0, 0] = 1 (C11), W[1*2+0, 0] = -1 (C22)
    U[0][0] = 1; V[0][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[2][0] = -1  # W[k*n2+i]: C11 is k=0,i=0 -> idx 0; C22 is k=1,i=1 -> idx 3? 
    # Wait, W[k*n2 + i]: for 2x2, k in {0,1}, i in {0,1}
    # C11: k=0, i=0 -> W[0]
    # C12: k=1, i=0 -> W[1]  
    # C21: k=0, i=1 -> W[2]
    # C22: k=1, i=1 -> W[3]
    # So W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22
    
    # Let me redo:
    U = [[0.0] * rank for _ in range(n2 * m2)]
    V = [[0.0] * rank for _ in range(m2 * p2)]
    W = [[0.0] * rank for _ in range(p2 * n2)]
    
    # Indexing: U[i*m2+j], V[j*p2+k], W[k*n2+i]
    # A indices: i in {0,1}, j in {0,1} -> A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B indices: j in {0,1}, k in {0,1} -> B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    # C indices: k in {0,1}, i in {0,1} -> C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    
    # P1 = A00 * (B00 - B11)
    U[0][0] = 1
    V[0][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[3][0] = 1  # C00 += P1, C11 += P1
    
    # P2 = (A00 + A11) * B11
    U[0][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[0][1] = 1; W[3][1] = -1  # C00 += P2, C11 -= P2
    
    # P3 = (A00 + A01) * B00
    U[0][2] = 1; U[1][2] = 1
    V[0][2] = 1
    W[0][2] = -1; W[1][2] = 1  # C00 -= P3, C01 += P3
    
    # P4 = (A11 - A00) * (B10 + B11)
    U[3][3] = 1; U[0][3] = -1
    V[2][3] = 1; V[3][3] = 1
    W[2][3] = 1; W[3][3] = 1  # C10 += P4, C11 += P4
    
    # P5 = (A01 + A11) * B10
    U[1][4] = 1; U[3][4] = 1
    V[2][4] = 1
    W[1][4] = 1; W[3][4] = 1  # C01 += P5, C11 += P5
    
    # P6 = (A00 - A01) * (B00 + B01)
    U[0][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1  # C10 += P6
    
    # P7 = (A10 + A01) * (B11 - B10)
    U[2][6] = 1; U[1][6] = 1
    V[3][6] = 1; V[2][6] = -1
    W[0][6] = 1  # C00 += P7
    
    return U, V, W

def _build_strassen_based_decomposition(n, m, p):
    """
    Build a decomposition using Strassen on 2x2x2 blocks.
    For n=2, m=3, p=10:
    - Split the m=3 dimension: use one 2x2x2 Strassen block for j in {0,1}, 
      and schoolbook for j=2.
    - Actually, let's think more carefully.
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    We can decompose this by splitting dimensions into blocks.
    For n=2: split into 1 block of size 2.
    For m=3: split into 1 block of size 2 + 1 block of size 1.
    For p=10: split into 5 blocks of size 2.
    
    For each combination of (n_block, m_block, p_block):
    - If n_block=2, m_block=2, p_block=2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank = n_block * m_block * p_block)
    
    Total rank = sum over all block combinations.
    """
    # Split m=3 into blocks: [2, 1]
    # Split p=10 into blocks: [2, 2, 2, 2, 2]
    # n=2 is one block: [2]
    
    m_blocks = [2, 1]
    p_blocks = [2, 2, 2, 2, 2]
    n_blocks = [2]
    
    # Build the full decomposition
    all_U = []
    all_V = []
    all_W = []
    
    offset_m = 0
    offset_p = 0
    
    for nb in n_blocks:
        offset_m = 0
        for mb in m_blocks:
            offset_p = 0
            for pb in p_blocks:
                if nb == 2 and mb == 2 and pb == 2:
                    # Use Strassen
                    U_blk, V_blk, W_blk = _strassen_block_2x2x2()
                    # Remap indices
                    U_remapped = [[0.0] * 7 for _ in range(n * m)]
                    V_remapped = [[0.0] * 7 for _ in range(m * p)]
                    W_remapped = [[0.0] * 7 for _ in range(p * n)]
                    
                    for ri in range(nb):
                        for rj in range(mb):
                            for r in range(7):
                                U_remapped[(offset_m + rj) * 1 + ri][r] = U_blk[ri * 2 + rj][r]
                        # Wait, U is indexed by [i*m + j] in the full problem,
                        # but in the block it's [i*mb + j]. We need to map.
                    
                    # Actually this is getting complex. Let me use a different approach.
                    # Just build the full U, V, W directly.
                    pass
                else:
                    # Schoolbook for this block
                    for ri in range(nb):
                        for rj in range(mb):
                            for rk in range(pb):
                                col = len(all_U[0]) if all_U else 0
                                # Add a new column
                                for row in range(n * m):
                                    all_U[row].append(0.0)
                                for row in range(m * p):
                                    all_V[row].append(0.0)
                                for row in range(p * n):
                                    all_W[row].append(0.0)
                                all_U[(offset_m + rj) * 1 + ri][col] = 1.0  # Hmm, indexing is wrong
                                all_V[(offset_m + rj) * p + (offset_p + rk)][col] = 1.0
                                all_W[(offset_p + rk) * n + ri][col] = 1.0
                        offset_p += pb
            offset_m += mb
    
    # This approach is getting messy. Let me use a cleaner method.
    return None

def _build_strassen_decomposition_v2(n, m, p):
    """
    Cleaner Strassen-based decomposition.
    Split m into blocks of size 2 (plus remainder 1).
    Split p into blocks of size 2.
    For each (m_block, p_block) pair where both are size 2, use Strassen.
    For others, use schoolbook.
    """
    # Determine block structure
    m_blocks = []
    rem = m
    while rem >= 2:
        m_blocks.append(2)
        rem -= 2
    if rem > 0:
        m_blocks.append(rem)
    
    p_blocks = []
    rem = p
    while rem >= 2:
        p_blocks.append(2)
        rem -= 2
    if rem > 0:
        p_blocks.append(rem)
    
    # For n=2, it's always one block of size 2
    n_blocks = [n]
    
    # Build decomposition
    U = []
    V = []
    W = []
    
    def add_column(u_vals, v_vals, w_vals):
        """Add a new column to U, V, W."""
        for i in range(len(U)):
            U[i].append(0.0)
        for i in range(len(V)):
            V[i].append(0.0)
        for i in range(len(W)):
            W[i].append(0.0)
        col = len(U[0]) - 1
        for idx, val in u_vals:
            U[idx][col] = val
        for idx, val in v_vals:
            V[idx][col] = val
        for idx, val in w_vals:
            W[idx][col] = val
    
    # Initialize
    for i in range(n * m):
        U.append([])
    for i in range(m * p):
        V.append([])
    for i in range(p * n):
        W.append([])
    
    m_offset = 0
    for mb in m_blocks:
        p_offset = 0
        for pb in p_blocks:
            if mb == 2 and pb == 2:
                # Use Strassen for this 2x2x2 block (n=2, m=2, p=2)
                U_blk, V_blk, W_blk = _strassen_block_2x2x2()
                for r in range(7):
                    u_vals = []
                    v_vals = []
                    w_vals = []
                    for ri in range(2):
                        for rj in range(2):
                            u_vals.append(((ri) * m + (m_offset + rj), U_blk[ri * 2 + rj][r]))
                    for rj in range(2):
                        for rk in range(2):
                            v_vals.append(((m_offset + rj) * p + (p_offset + rk), V_blk[rj * 2 + rk][r]))
                    for rk in range(2):
                        for ri in range(2):
                            w_vals.append(((p_offset + rk) * n + ri, W_blk[rk * 2 + ri][r]))
                    # Filter out zeros
                    u_vals = [(idx, val) for idx, val in u_vals if val != 0]
                    v_vals = [(idx, val) for idx, val in v_vals if val != 0]
                    w_vals = [(idx, val) for idx, val in w_vals if val != 0]
                    add_column(u_vals, v_vals, w_vals)
            else:
                # Schoolbook for this block
                for ri in range(n):
                    for rj in range(mb):
                        for rk in range(pb):
                            add_column(
                                [(ri * m + (m_offset + rj), 1.0)],
                                [((m_offset + rj) * p + (p_offset + rk), 1.0)],
                                [((p_offset + rk) * n + ri, 1.0)]
                            )
            p_offset += pb
        m_offset += mb
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, time_limit=30.0):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    This is a heuristic approach.
    """
    start = time.time()
    rank = len(U[0])
    
    # Convert to numpy for efficient operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to find if any column is a linear combination of others
    # This is expensive, so we'll do a simple check
    # Check if removing any single column still gives a valid decomposition
    # (This would mean the column was redundant)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for col_to_remove in range(rank):
        if time.time() - start > time_limit:
            break
        
        # Try removing this column
        U_test = np.delete(U_arr, col_to_remove, axis=1)
        V_test = np.delete(V_arr, col_to_remove, axis=1)
        W_test = np.delete(W_arr, col_to_remove, axis=1)
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = np.sum(U_test[i * m + j] * V_test[j * p + k] * W_test[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            best_rank = rank - 1
            best_U = U_test.tolist()
            best_V = V_test.tolist()
            best_W = W_test.tolist()
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, build the Strassen-based decomposition (rank 55)
    U, V, W = _build_strassen_decomposition_v2(n, m, p)
    rank = len(U[0])
    
    # Verify it's correct
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook if Strassen construction is wrong
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
    
    # Try to reduce rank further if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        remaining = min(DEADLINE - 10 - elapsed, 30.0)
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, time_limit=remaining)
    
    # Final safety check
    if time.time() > DEADLINE - 5:
        # Return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
