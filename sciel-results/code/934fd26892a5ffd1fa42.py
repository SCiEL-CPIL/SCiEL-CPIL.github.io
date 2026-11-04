"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    def verify_decomp(U, V, W, n, m, p):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use known decompositions and try to find rank reductions
    # through linear dependencies among the tensor terms.
    
    # First, establish the schoolbook baseline
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try Strassen-based decomposition
    def get_strassen_222():
        U22 = np.zeros((4, 7))
        V22 = np.zeros((4, 7))
        W22 = np.zeros((4, 7))
        
        # M1 = (A11+A22)*(B11+B22) -> C11 + C22
        U22[0, 0] = 1; U22[3, 0] = 1
        V22[0, 0] = 1; V22[3, 0] = 1
        W22[0, 0] = 1; W22[3, 0] = 1
        
        # M2 = (A21+A22)*B11 -> C21
        U22[2, 1] = 1; U22[3, 1] = 1
        V22[0, 1] = 1
        W22[2, 1] = 1
        
        # M3 = A11*(B11-B21) -> C12 - C22
        U22[0, 2] = 1
        V22[0, 2] = 1; V22[2, 2] = -1
        W22[1, 2] = 1; W22[3, 2] = -1
        
        # M4 = A22*(B22-B12) -> -C11 + C12
        U22[3, 3] = 1
        V22[3, 3] = 1; V22[1, 3] = -1
        W22[0, 3] = -1; W22[1, 3] = 1
        
        # M5 = (A11+A12)*B22 -> C11 + C21
        U22[0, 4] = 1; U22[1, 4] = 1
        V22[3, 4] = 1
        W22[0, 4] = 1; W22[2, 4] = 1
        
        # M6 = (A21-A11)*(B11+B12) -> C22
        U22[2, 5] = 1; U22[0, 5] = -1
        V22[0, 5] = 1; V22[1, 5] = 1
        W22[3, 5] = 1
        
        # M7 = (A12-A22)*(B21+B22) -> C11 - C22
        U22[1, 6] = 1; U22[3, 6] = -1
        V22[2, 6] = 1; V22[3, 6] = 1
        W22[0, 6] = 1; W22[3, 6] = -1
        
        return U22, V22, W22
    
    # Find maximum number of non-overlapping 2x2x2 blocks
    i_starts = list(range(n - 1))
    j_starts = list(range(m - 1))
    k_starts = list(range(p - 1))
    
    all_blocks = [(i, j, k) for i in i_starts for j in j_starts for k in k_starts]
    
    def blocks_overlap(b1, b2):
        i1s, i1e = b1[0], b1[0]+1
        i2s, i2e = b2[0], b2[0]+1
        j1s, j1e = b1[1], b1[1]+1
        j2s, j2e = b2[1], b2[1]+1
        k1s, k1e = b1[2], b1[2]+1
        k2s, k2e = b2[2], b2[2]+1
        if i1s <= i2e and i2s <= i1e:
            if j1s <= j2e and j2s <= j1e:
                if k1s <= k2e and k2s <= k1e:
                    return True
        return False
    
    best_count = 0
    best_block_set = []
    
    num_blocks = len(all_blocks)
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for a in range(num_blocks):
        for b in range(a+1, num_blocks):
            if blocks_overlap(all_blocks[a], all_blocks[b]):
                conflict[a][b] = True
                conflict[b][a] = True
    
    block_indices = list(range(num_blocks))
    block_indices.sort(key=lambda idx: sum(conflict[idx]))
    
    def find_max_blocks(idx, current, used):
        nonlocal best_count, best_block_set
        if len(current) + (num_blocks - idx) <= best_count:
            return
        if idx >= num_blocks:
            if len(current) > best_count:
                best_count = len(current)
                best_block_set = current[:]
            return
        find_max_blocks(idx + 1, current, used)
        bi = block_indices[idx]
        if bi not in used:
            can_use = True
            for ub in used:
                if conflict[bi][ub]:
                    can_use = False
                    break
            if can_use:
                current.append(bi)
                used.add(bi)
                find_max_blocks(idx + 1, current, used)
                used.remove(bi)
                current.pop()
    
    find_max_blocks(0, [], set())
    
    best_config = [all_blocks[idx] for idx in best_block_set]
    num_blocks_found = len(best_config)
    
    strassen_rank_per_block = 7
    total_strassen_rank = num_blocks_found * strassen_rank_per_block
    
    covered = set()
    for (i0, j0, k0) in best_config:
        for i in [i0, i0+1]:
            for j in [j0, j0+1]:
                for k in [k0, k0+1]:
                    covered.add((i, j, k))
    
    num_covered = len(covered)
    num_schoolbook = n * m * p - num_covered
    total_rank = total_strassen_rank + num_schoolbook
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    for (i0, j0, k0) in best_config:
        U22, V22, W22 = get_strassen_222()
        for r in range(strassen_rank_per_block):
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    if U22[local_idx, r] != 0:
                        i = i0 + li
                        j = j0 + lj
                        U[i * m + j, col + r] = U22[local_idx, r]
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    if V22[local_idx, r] != 0:
                        j = j0 + lj
                        k = k0 + lk
                        V[j * p + k, col + r] = V22[local_idx, r]
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    if W22[local_idx, r] != 0:
                        k = k0 + lk
                        i = i0 + li
                        W[k * n + i, col + r] = W22[local_idx, r]
        col += strassen_rank_per_block
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    if verify_decomp(U, V, W, n, m, p) and total_rank < best_rank:
        best_U, best_V, best_W = U, V, W
        best_rank = total_rank
    
    # Now try to reduce rank further using linear algebra
    # Check for linear dependencies in the tensor decomposition
    # If we can find a linear combination of rank-1 terms that cancels out,
    # we can potentially reduce the rank.
    
    # Attempt: Look for 2x2 subproblems that can be solved with rank 3 instead of 4
    # (using the fact that 2x2x2 has rank 7, but maybe some substructures have lower rank)
    
    # For now, try to identify if any subset of terms can be replaced by fewer terms
    # This requires solving a system of equations
    
    # Try rank reduction via Gaussian elimination on the tensor
    # Build the tensor as a 3D array and check for decomposability
    
    # Convert best decomposition to tensor form
    def decomp_to_tensor(U, V, W, n, m, p):
        """Convert factor matrices to the multiplication tensor."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T[i, j, k] = s
        return T
    
    # Try to find a lower-rank decomposition by searching over possible
    # support patterns using the subspace method
    
    # For 4x5x6, known lower bound is around 60-70 based on asymptotic results
    # Let's try to find a decomposition with rank < best_rank using a
    # randomized approach with verification
    
    # Use the fact that we can try random linear combinations of the existing
    # rank-1 terms and check if they still span the same tensor space
    
    # Try: For each pair of rank-1 terms, check if they can be merged
    # This is a heuristic approach
    
    # Alternative: Use the LLL algorithm or similar to find short vectors
    # in the lattice of valid decompositions
    
    # For now, return the best found so far
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to improve by checking if we can use a different block configuration
    # or if there's a more efficient decomposition pattern
    
    # Check if the Strassen-based decomposition is optimal
    # by trying all possible configurations of non-overlapping blocks
    
    # The current approach already does this via branch and bound
    
    # Try a different strategy: use 3x3x3 or other block sizes
    # For 3x3x3, the best known rank is 23 (vs 27 schoolbook)
    
    # But for our dimensions, 2x2x2 blocks are the most natural
    
    # Let's try to see if we can find a decomposition with rank < best_rank
    # by using a more sophisticated search
    
    # For now, return what we have
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
