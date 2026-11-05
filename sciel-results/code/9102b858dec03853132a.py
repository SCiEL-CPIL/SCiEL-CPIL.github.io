<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_block_decomposition(n, m, p):
    # Returns U, V, W for the whole problem using block decomposition
    U = np.zeros((n*m, 84))
    V = np.zeros((m*p, 84))
    W = np.zeros((p*n, 84))
    
    rank_offset = 0
    
    # Define blocks: (i_indices, j_indices, k_indices, use_strassen)
    # We need to map local indices to global indices
    
    # Block 1-3: I1={0,1}, J1={0,1}, K1={0,1}, K2={2,3}, K3={4,5}
    # Strassen
    
    # Helper to add a block
    def add_block(i_idx, j_idx, k_idx, is_strassen):
        nonlocal rank_offset
        n_b = len(i_idx)
        m_b = len(j_idx)
        p_b = len(k_idx)
        
        if is_strassen:
            if n_b != 2 or m_b != 2 or p_b != 2:
                raise ValueError("Strassen requires 2x2x2")
            Ub, Vb, Wb = _strassen_222()
            r_b = 7
        else:
            Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
            r_b = n_b * m_b * p_b
            
        # Map rows
        for r in range(r_b):
            for i_local, i_global in enumerate(i_idx):
                for j_local, j_global in enumerate(j_idx):
                    U[i_global*m + j_global, rank_offset + r] += Ub[i_local*m_b + j_local, r]
                    
            for j_local, j_global in enumerate(j_idx):
                for k_local, k_global in enumerate(k_idx):
                    V[j_global*p + k_global, rank_offset + r] += Vb[j_local*p_b + k_local, r]
                    
            for k_local, k_global in enumerate(k_idx):
                for i_local, i_global in enumerate(i_idx):
                    W[k_global*n + i_global, rank_offset + r] += Wb[k_local*n_b + i_local, r]
                    
        rank_offset += r_b

    # Partition indices
    I1 = [0, 1]
    I2 = [2]
    J1 = [0, 1]
    J2a = [2, 3]
    J2b = [4]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4, 5]
    
    # Blocks for J1
    add_block(I1, J1, K1, True)
    add_block(I1, J1, K2, True)
    add_block(I1, J1, K3, True)
    add_block(I2, J1, K1, False)
    add_block(I2, J1, K2, False)
    add_block(I2, J1, K3, False)
    
    # Blocks for J2 (split into J2a, J2b)
    add_block(I1, J2a, K1, True)
    add_block(I1, J2b, K1, False)
    add_block(I1, J2a, K2, True)
    add_block(I1, J2b, K2, False)
    add_block(I1, J2a, K3, True)
    add_block(I1, J2b, K3, False)
    
    add_block(I2, J2a, K1, False)
    add_block(I2, J2b, K1, False)
    add_block(I2, J2a, K2, False)
    add_block(I2, J2b, K2, False)
    add_block(I2, J2a, K3, False)
    add_block(I2, J2b, K3, False)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _construct_block_decomposition(n, m, p)
    
    result = {"U": U, "V": V, "W": W}
    
    if _verify_fast(U, V, W, n, m, p):
        return result
    else:
        # Fallback to schoolbook if something went wrong
        return _schoolbook(n, m, p)
=======
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _winograd_333():
    # Winograd's F(3,3) algorithm for 3x3 matrix multiplication
    # Rank 23, with half-integer entries
    # U: 9x23, V: 9x23, W: 9x23
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Winograd F(3,3) - 23 multiplications
    # Based on Winograd's algorithm for 3x3 matrix multiply
    
    # The 23 bilinear forms for Winograd F(3,3)
    # M1: a00*c00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M2: a00*c01
    U[0, 1] = 1
    V[1, 1] = 1
    W[3, 1] = 1
    
    # M3: a00*c02
    U[0, 2] = 1
    V[2, 2] = 1
    W[6, 2] = 1
    
    # M4: a01*c00
    U[1, 3] = 1
    V[0, 3] = 1
    W[1, 3] = 1
    
    # M5: a01*c01
    U[1, 4] = 1
    V[1, 4] = 1
    W[4, 4] = 1
    
    # M6: a01*c02
    U[1, 5] = 1
    V[2, 5] = 1
    W[7, 5] = 1
    
    # M7: a02*c00
    U[2, 6] = 1
    V[0, 6] = 1
    W[2, 6] = 1
    
    # M8: a02*c01
    U[2, 7] = 1
    V[1, 7] = 1
    W[5, 7] = 1
    
    # M9: a02*c02
    U[2, 8] = 1
    V[2, 8] = 1
    W[8, 8] = 1
    
    # M10: (a00+a01)*c10
    U[0, 9] = 1; U[1, 9] = 1
    V[3, 9] = 1
    W[0, 9] = 1; W[1, 9] = 1
    
    # M11: (a00+a01)*c11
    U[0, 10] = 1; U[1, 10] = 1
    V[4, 10] = 1
    W[3, 10] = 1; W[4, 10] = 1
    
    # M12: (a00+a01)*c12
    U[0, 11] = 1; U[1, 11] = 1
    V[5, 11] = 1
    W[6, 11] = 1; W[7, 11] = 1
    
    # M13: (a01+a02)*c10
    U[1, 12] = 1; U[2, 12] = 1
    V[3, 12] = 1
    W[1, 12] = 1; W[2, 12] = 1
    
    # M14: (a01+a02)*c11
    U[1, 13] = 1; U[2, 13] = 1
    V[4, 13] = 1
    W[4, 13] = 1; W[5, 13] = 1
    
    # M15: (a01+a02)*c12
    U[1, 14] = 1; U[2, 14] = 1
    V[5, 14] = 1
    W[7, 14] = 1; W[8, 14] = 1
    
    # M16: (a00+a02)*c20
    U[0, 15] = 1; U[2, 15] = 1
    V[6, 15] = 1
    W[0, 15] = 1; W[2, 15] = 1
    
    # M17: (a00+a02)*c21
    U[0, 16] = 1; U[2, 16] = 1
    V[7, 16] = 1
    W[3, 16] = 1; W[5, 16] = 1
    
    # M18: (a00+a02)*c22
    U[0, 17] = 1; U[2, 17] = 1
    V[8, 17] = 1
    W[6, 17] = 1; W[8, 17] = 1
    
    # M19: (a10+a11)*c00
    U[3, 18] = 1; U[4, 18] = 1
    V[0, 18] = 1
    W[0, 18] = 1; W[3, 18] = 1
    
    # M20: (a10+a11)*c01
    U[3, 19] = 1; U[4, 19] = 1
    V[1, 19] = 1
    W[1, 19] = 1; W[4, 19] = 1
    
    # M21: (a10+a11)*c02
    U[3, 20] = 1; U[4, 20] = 1
    V[2, 20] = 1
    W[2, 20] = 1; W[5, 20] = 1
    
    # M22: (a11+a12)*c00
    U[4, 21] = 1; U[5, 21] = 1
    V[0, 21] = 1
    W[1, 21] = 1; W[4, 21] = 1
    
    # M23: (a11+a12)*c01
    U[4, 22] = 1; U[5, 22] = 1
    V[1, 22] = 1
    W[4, 22] = 1; W[7, 22] = 1
    
    # This is NOT correct - Winograd F(3,3) is more complex.
    # Let me use the actual Winograd F(3,3) with 23 multiplications.
    # The proper Winograd F(3,3) uses specific combinations.
    
    # Actually, let me just return a schoolbook 3x3x3 for now since
    # implementing proper Winograd F(3,3) with correct half-integers is complex.
    # The standard Winograd F(3,3) has rank 23 but requires careful construction.
    
    return None  # Not using this for now

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _get_block_factors(n_b, m_b, p_b):
    """Get the best known factor matrices for a block of size n_b x m_b x p_b"""
    if n_b == 2 and m_b == 2 and p_b == 2:
        return _strassen_222(), 7
    else:
        return _schoolbook_block(n_b, m_b, p_b), n_b * m_b * p_b

def _construct_block_decomposition(n, m, p, partition):
    """
    Construct block decomposition given a partition.
    partition: (i_parts, j_parts, k_parts) where each is a list of lists of indices
    """
    i_parts, j_parts, k_parts = partition
    
    # Calculate total rank
    total_rank = 0
    for i_p in i_parts:
        for j_p in j_parts:
            for k_p in k_parts:
                _, r_b = _get_block_factors(len(i_p), len(j_p), len(k_p))
                total_rank += r_b
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    rank_offset = 0
    
    for i_p in i_parts:
        for j_p in j_parts:
            for k_p in k_parts:
                n_b = len(i_p)
                m_b = len(j_p)
                p_b = len(k_p)
                
                Ub, Vb, Wb, r_b = _get_block_factors(n_b, m_b, p_b)
                
                for r in range(r_b):
                    for i_local, i_global in enumerate(i_p):
                        for j_local, j_global in enumerate(j_p):
                            U[i_global*m + j_global, rank_offset + r] += Ub[i_local*m_b + j_local, r]
                    
                    for j_local, j_global in enumerate(j_p):
                        for k_local, k_global in enumerate(k_p):
                            V[j_global*p + k_global, rank_offset + r] += Vb[j_local*p_b + k_local, r]
                    
                    for k_local, k_global in enumerate(k_p):
                        for i_local, i_global in enumerate(i_p):
                            W[k_global*n + i_global, rank_offset + r] += Wb[k_local*n_b + i_local, r]
                
                rank_offset += r_b
    
    return U.tolist(), V.tolist(), W.tolist(), total_rank

def _generate_partitions(n, m, p):
    """Generate all possible partitions of indices into contiguous blocks of size 1 or 2"""
    def _contiguous_partitions(dim):
        """Generate all ways to partition [0, dim) into contiguous blocks of size 1 or 2"""
        if dim == 0:
            return [[]]
        if dim == 1:
            return [[[0]]]
        if dim == 2:
            return [[[0, 1]], [[0], [1]]]
        
        results = []
        # Start with block [0]
        for rest in _contiguous_partitions(dim - 1):
            results.append([[0]] + rest)
        # Start with block [0, 1] if dim >= 2
        if dim >= 2:
            for rest in _contiguous_partitions(dim - 2):
                results.append([[0, 1]] + rest)
        return results
    
    i_parts = _contiguous_partitions(n)
    j_parts = _contiguous_partitions(m)
    k_parts = _contiguous_partitions(p)
    
    partitions = []
    for ip in i_parts:
        for jp in j_parts:
            for kp in k_parts:
                partitions.append((ip, jp, kp))
    
    return partitions

def _estimate_rank(partition, n, m, p):
    """Estimate the rank of a partition without constructing it"""
    i_parts, j_parts, k_parts = partition
    total_rank = 0
    for i_p in i_parts:
        for j_p in j_parts:
            for k_p in k_parts:
                n_b, m_b, p_b = len(i_p), len(j_p), len(k_p)
                if n_b == 2 and m_b == 2 and p_b == 2:
                    total_rank += 7
                else:
                    total_rank += n_b * m_b * p_b
    return total_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Generate all possible partitions
    partitions = _generate_partitions(n, m, p)
    
    # Sort by estimated rank
    ranked_partitions = sorted(partitions, key=lambda pt: _estimate_rank(pt, n, m, p))
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as upper bound
    
    # Try partitions in order of estimated rank
    for partition in ranked_partitions:
        if _remaining() < 2.0:
            break
        
        est_rank = _estimate_rank(partition, n, m, p)
        if est_rank >= best_rank:
            break
        
        U, V, W, actual_rank = _construct_block_decomposition(n, m, p, partition)
        
        if actual_rank < best_rank and _verify_fast(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = actual_rank
            break  # Found a valid decomposition with this rank
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
>>>>>>> REPLACE
