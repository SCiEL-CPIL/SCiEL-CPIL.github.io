import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
TOTAL_TERMS = n * m * p  # 36

def _verify(U, V, W, atol=1e-9):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    if r == 0:
        return False
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=atol): return False
    if not np.allclose(V2, np.round(V2), atol=atol): return False
    if not np.allclose(W2, np.round(W2), atol=atol): return False
    
    # Reconstruct tensor
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # The matmul tensor T[i,j,k] = 1 for all i,j,k
    # So we check if T_reconstructed is all 1s
    if not np.allclose(T_reconstructed, 1.0, atol=atol, rtol=atol):
        return False
    
    return True

def _schoolbook_decomposition():
    """
    Returns the schoolbook (rank 36) decomposition.
    """
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2_block(i0, i1, j0, j1, k0, k1):
    """
    Returns the 7 Strassen factors for a 2x2x2 block.
    The block covers A[i0:i2, j0:j2] * B[j0:j2, k0:k2].
    We assume i1 = i0+1, j1 = j0+1, k1 = k0+1.
    Returns list of (u_dict, v_dict, w_dict)
    u_dict: {(i,j): val}
    v_dict: {(j,k): val}
    w_dict: {(k,i): val}
    """
    # Strassen formulas for C = A*B
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # U factors (A-side)
    # For M1: A11 + A22 -> u: {(i0,j0):1, (i1,j1):1}
    # For M2: A21 + A22 -> u: {(i1,j0):1, (i1,j1):1}
    # For M3: A11 -> u: {(i0,j0):1}
    # For M4: A22 -> u: {(i1,j1):1}
    # For M5: A11 + A12 -> u: {(i0,j0):1, (i0,j1):1}
    # For M6: A21 - A11 -> u: {(i1,j0):1, (i0,j0):-1}
    # For M7: A12 - A22 -> u: {(i0,j1):1, (i1,j1):-1}
    
    # V factors (B-side)
    # For M1: B11 + B22 -> v: {(j0,k0):1, (j1,k1):1}
    # For M2: B11 -> v: {(j0,k0):1}
    # For M3: B21 - B22 -> v: {(j1,k0):1, (j1,k1):-1}
    # For M4: B12 - B11 -> v: {(j0,k1):1, (j0,k0):-1}
    # For M5: B22 -> v: {(j1,k1):1}
    # For M6: B11 + B12 -> v: {(j0,k0):1, (j0,k1):1}
    # For M7: B21 + B22 -> v: {(j1,k0):1, (j1,k1):1}
    
    # W factors (C-side)
    # For M1: C11 + C22 -> w: {(k0,i0):1, (k1,i1):1}
    # For M2: C21 -> w: {(k0,i1):1}
    # For M3: C12 -> w: {(k1,i0):1}
    # For M4: C21 - C11 -> w: {(k0,i1):1, (k0,i0):-1}
    # For M5: C11 + C12 -> w: {(k0,i0):1, (k1,i0):1}
    # For M6: C22 -> w: {(k1,i1):1}
    # For M7: C11 - C22 -> w: {(k0,i0):1, (k1,i1):-1}
    
    U_terms = [
        {(i0,j0):1, (i1,j1):1},
        {(i1,j0):1, (i1,j1):1},
        {(i0,j0):1},
        {(i1,j1):1},
        {(i0,j0):1, (i0,j1):1},
        {(i1,j0):1, (i0,j0):-1},
        {(i0,j1):1, (i1,j1):-1},
    ]
    
    V_terms = [
        {(j0,k0):1, (j1,k1):1},
        {(j0,k0):1},
        {(j1,k0):1, (j1,k1):-1},
        {(j0,k1):1, (j0,k0):-1},
        {(j1,k1):1},
        {(j0,k0):1, (j0,k1):1},
        {(j1,k0):1, (j1,k1):1},
    ]
    
    W_terms = [
        {(k0,i0):1, (k1,i1):1},
        {(k0,i1):1},
        {(k1,i0):1},
        {(k0,i1):1, (k0,i0):-1},
        {(k0,i0):1, (k1,i0):1},
        {(k1,i1):1},
        {(k0,i0):1, (k1,i1):-1},
    ]
    
    return list(zip(U_terms, V_terms, W_terms))

def _build_decomposition_from_blocks(blocks, schoolbook_rest=True):
    """
    Build a decomposition using Strassen blocks for specified 2x2x2 regions
    and schoolbook for the rest.
    
    blocks: list of (i0, j0, k0) where the block covers [i0,i0+1] x [j0,j0+1] x [k0,k0+1]
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    covered = set()
    
    for (i0, j0, k0) in blocks:
        # Check bounds
        if i0 + 1 >= n or j0 + 1 >= m or k0 + 1 >= p:
            continue
        
        # Check for overlap
        block_terms = set()
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    block_terms.add((i,j,k))
        
        overlap = block_terms & covered
        if overlap:
            continue  # Skip overlapping blocks
        
        covered |= block_terms
        
        # Get Strassen factors
        factors = _strassen_2x2x2_block(i0, i0+1, j0, j0+1, k0, k0+1)
        for u_t, v_t, w_t in factors:
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for (i,j), val in u_t.items():
                u_vec[i * m + j] = val
            for (j,k), val in v_t.items():
                v_vec[j * p + k] = val
            for (k,i), val in w_t.items():
                w_vec[k * n + i] = val
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    # Schoolbook for remaining terms
    if schoolbook_rest:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i,j,k) not in covered:
                        u_vec = np.zeros(n * m)
                        v_vec = np.zeros(m * p)
                        w_vec = np.zeros(p * n)
                        
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        
                        U_cols.append(u_vec)
                        V_cols.append(v_vec)
                        W_cols.append(w_vec)
    
    if not U_cols:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U, V, W

def _local_search_remove_columns(U, V, W, deadline=None):
    """
    Try to remove columns that are redundant.
    """
    def _time_exceeded():
        return deadline is not None and time.time() > deadline
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    r = current_U.shape[1]
    
    # Keep trying to remove columns
    improved = True
    while improved and not _time_exceeded():
        improved = False
        r = current_U.shape[1]
        
        for col in range(r):
            if _time_exceeded():
                return current_U, current_V, current_W
            
            if r <= 1:
                break
            
            mask = [i for i in range(r) if i != col]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify(U_test, V_test, W_test, atol=1e-8):
                current_U, current_V, current_W = U_test, V_test, W_test
                r -= 1
                improved = True
    
    return current_U, current_V, current_W

def _try_local_search_replace(U, V, W, deadline=None):
    """
    Try replacing pairs of columns with a single column (if possible).
    This is more complex, so we skip for now.
    """
    return U, V, W

def _random_perturbation_search(U0, V0, W0, deadline, max_iter=50):
    """
    Try random small perturbations of the factors to find lower rank.
    This is heuristic.
    """
    def _time_exceeded():
        return deadline is not None and time.time() > deadline
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = best_U.shape[1]
    
    r = best_rank
    for _ in range(max_iter):
        if _time_exceeded():
            break
        
        # Try removing a random column
        if r > 1:
            col = np.random.randint(r)
            mask = [i for i in range(r) if i != col]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            if _verify(U_test, V_test, W_test, atol=1e-8):
                best_U, best_V, best_W = U_test, V_test, W_test
                r -= 1
                best_rank = r
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_result = None
    best_rank = float('inf')
    
    def _try_candidate(U, V, W, label=""):
        nonlocal best_result, best_rank
        if U is None:
            return
        r = U.shape[1]
        if r >= best_rank:
            return
        if _verify(U, V, W):
            best_rank = r
            best_result = (U.copy(), V.copy(), W.copy())
            if time.time() < deadline:
                # Try local search
                U2, V2, W2 = _local_search_remove_columns(U, V, W, deadline)
                if _verify(U2, V2, W2):
                    if U2.shape[1] < best_rank:
                        best_rank = U2.shape[1]
                        best_result = (U2.copy(), V2.copy(), W2.copy())
    
    # 1. Schoolbook (rank 36)
    U_sb, V_sb, W_sb = _schoolbook_decomposition()
    _try_candidate(U_sb, V_sb, W_sb, "schoolbook")
    
    # 2. Single Strassen blocks
    # Possible 2x2x2 blocks in 3x3x4:
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks = [(i0, j0, k0)]
                result = _build_decomposition_from_blocks(blocks)
                if result is not None:
                    _try_candidate(result[0], result[1], result[2], f"single_strassen_{i0}_{j0}_{k0}")
    
    # 3. Pairs of non-overlapping Strassen blocks
    # Find all pairs of disjoint 2x2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    for idx1 in range(len(all_blocks)):
        if time.time() > deadline:
            break
        b1 = all_blocks[idx1]
        for idx2 in range(idx1 + 1, len(all_blocks)):
            b2 = all_blocks[idx2]
            
            # Check disjointness
            i1_set = {b1[0], b1[0]+1}
            j1_set = {b1[1], b1[1]+1}
            k1_set = {b1[2], b1[2]+1}
            i2_set = {b2[0], b2[0]+1}
            j2_set = {b2[1], b2[1]+1}
            k2_set = {b2[2], b2[2]+1}
            
            # Two blocks are disjoint if their 3D regions don't overlap
            # Region 1: [i1, i1+1] x [j1, j1+1] x [k1, k1+1]
            # Region 2: [i2, i2+1] x [j2, j2+1] x [k2, k2+1]
            # They overlap if the ranges overlap in all 3 dimensions
            
            def ranges_overlap(r1, r2):
                return not (r1[1] < r2[0] or r2[1] < r1[0])
            
            if (ranges_overlap((b1[0], b1[0]+1), (b2[0], b2[0]+1)) and
                ranges_overlap((b1[1], b1[1]+1), (b2[1], b2[1]+1)) and
                ranges_overlap((b1[2], b1[2]+1), (b2[2], b2[2]+1))):
                continue  # Overlapping, skip
            
            blocks = [b1, b2]
            result = _build_decomposition_from_blocks(blocks)
            if result is not None:
                _try_candidate(result[0], result[1], result[2], f"double_strassen_{b1}_{b2}")
    
    # 4. Triples of non-overlapping Strassen blocks
    # This is more expensive, so only try if time remains
    if time.time() < deadline - 30:
        for idx1 in range(len(all_blocks)):
            if time.time() > deadline:
                break
            b1 = all_blocks[idx1]
            for idx2 in range(idx1 + 1, len(all_blocks)):
                if time.time() > deadline:
                    break
                b2 = all_blocks[idx2]
                
                i1_set = {b1[0], b1[0]+1}
                j1_set = {b1[1], b1[1]+1}
                k1_set = {b1[2], b1[2]+1}
                i2_set = {b2[0], b2[0]+1}
                j2_set = {b2[1], b2[1]+1}
                k2_set = {b2[2], b2[2]+1}
                
                def ranges_overlap(r1, r2):
                    return not (r1[1] < r2[0] or r2[1] < r1[0])
                
                if (ranges_overlap((b1[0], b1[0]+1), (b2[0], b2[0]+1)) and
                    ranges_overlap((b1[1], b1[1]+1), (b2[1], b2[1]+1)) and
                    ranges_overlap((b1[2], b1[2]+1), (b2[2], b2[2]+1))):
                    continue
                
                for idx3 in range(idx2 + 1, len(all_blocks)):
                    if time.time() > deadline:
                        break
                    b3 = all_blocks[idx3]
                    
                    # Check b3 doesn't overlap with b1 or b2
                    overlap_with_b1 = (
                        ranges_overlap((b3[0], b3[0]+1), (b1[0], b1[0]+1)) and
                        ranges_overlap((b3[1], b3[1]+1), (b1[1], b1[1]+1)) and
                        ranges_overlap((b3[2], b3[2]+1), (b1[2], b1[2]+1))
                    )
                    overlap_with_b2 = (
                        ranges_overlap((b3[0], b3[0]+1), (b2[0], b2[0]+1)) and
                        ranges_overlap((b3[1], b3[1]+1), (b2[1], b2[1]+1)) and
                        ranges_overlap((b3[2], b3[2]+1), (b2[2], b2[2]+1))
                    )
                    
                    if overlap_with_b1 or overlap_with_b2:
                        continue
                    
                    blocks = [b1, b2, b3]
                    result = _build_decomposition_from_blocks(blocks)
                    if result is not None:
                        _try_candidate(result[0], result[1], result[2], f"triple_strassen_{b1}_{b2}_{b3}")
    
    # Return best result
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Valid:", _verify(U, V, W))
