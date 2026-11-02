"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Generate all possible Strassen blocks
    all_blocks = []
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                all_blocks.append((i, j, k))
    
    num_blocks = len(all_blocks)
    
    # Precompute disjointness matrix
    # disjoint[a, b] is True if block a and block b are disjoint
    disjoint = np.zeros((num_blocks, num_blocks), dtype=bool)
    for a in range(num_blocks):
        ia, ja, ka = all_blocks[a]
        for b in range(num_blocks):
            ib, jb, kb = all_blocks[b]
            # Check if any dimension is disjoint
            i_disjoint = (ia + 1 < ib) or (ib + 1 < ia)
            j_disjoint = (ja + 1 < jb) or (jb + 1 < ja)
            k_disjoint = (ka + 1 < kb) or (kb + 1 < ka)
            if i_disjoint or j_disjoint or k_disjoint:
                disjoint[a, b] = True
    
    # Find the maximum set of pairwise disjoint blocks using backtracking
    best_set = []
    
    def backtrack(start, current_set):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        # Pruning: if we can't beat the best set even with all remaining blocks
        remaining = num_blocks - start
        if len(current_set) + remaining <= len(best_set):
            return
        
        for i in range(start, num_blocks):
            if time.time() > deadline - 10:
                return
            # Check if block i is disjoint from all blocks in current_set
            is_disjoint = True
            for j in current_set:
                if not disjoint[j, i]:
                    is_disjoint = False
                    break
            if is_disjoint:
                current_set.append(i)
                backtrack(i + 1, current_set)
                current_set.pop()
    
    backtrack(0, [])
    
    max_blocks_idx = best_set
    max_blocks = [all_blocks[i] for i in max_blocks_idx]
    
    def build_solution(blocks_to_use):
        """Build U, V, W from the given list of Strassen blocks."""
        # Determine which (i, j, k) terms are covered by Strassen blocks
        covered = set()
        for (i0, j0, k0) in blocks_to_use:
            for i in (i0, i0 + 1):
                for j in (j0, j0 + 1):
                    for k in (k0, k0 + 1):
                        covered.add((i, j, k))
        
        # Terms not covered are handled by schoolbook
        schoolbook_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
        
        total_rank = len(blocks_to_use) * 7 + len(schoolbook_terms)
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        for (i0, j0, k0) in blocks_to_use:
            # Strassen M1
            c = col
            U[i0 * m + j0, c] += 1
            U[(i0 + 1) * m + (j0 + 1), c] += 1
            V[j0 * p + k0, c] += 1
            V[(j0 + 1) * p + (k0 + 1), c] += 1
            W[k0 * n + i0, c] += 1
            W[(k0 + 1) * n + (i0 + 1), c] += 1
            col += 1
            
            # Strassen M2
            c = col
            U[(i0 + 1) * m + j0, c] += 1
            U[(i0 + 1) * m + (j0 + 1), c] += 1
            V[j0 * p + k0, c] += 1
            W[k0 * n + (i0 + 1), c] += 1
            W[(k0 + 1) * n + (i0 + 1), c] += -1
            col += 1
            
            # Strassen M3
            c = col
            U[i0 * m + j0, c] += 1
            V[j0 * p + (k0 + 1), c] += 1
            V[(j0 + 1) * p + (k0 + 1), c] += -1
            W[(k0 + 1) * n + i0, c] += 1
            W[(k0 + 1) * n + (i0 + 1), c] += 1
            col += 1
            
            # Strassen M4
            c = col
            U[(i0 + 1) * m + (j0 + 1), c] += 1
            V[(j0 + 1) * p + k0, c] += 1
            V[j0 * p + k0, c] += -1
            W[k0 * n + i0, c] += 1
            W[k0 * n + (i0 + 1), c] += 1
            col += 1
            
            # Strassen M5
            c = col
            U[i0 * m + j0, c] += 1
            U[i0 * m + (j0 + 1), c] += 1
            V[(j0 + 1) * p + (k0 + 1), c] += 1
            W[k0 * n + i0, c] += -1
            W[(k0 + 1) * n + i0, c] += 1
            col += 1
            
            # Strassen M6
            c = col
            U[(i0 + 1) * m + j0, c] += 1
            U[i0 * m + j0, c] += -1
            V[j0 * p + k0, c] += 1
            V[j0 * p + (k0 + 1), c] += 1
            W[(k0 + 1) * n + (i0 + 1), c] += 1
            col += 1
            
            # Strassen M7
            c = col
            U[i0 * m + (j0 + 1), c] += 1
            U[(i0 + 1) * m + (j0 + 1), c] += -1
            V[(j0 + 1) * p + k0, c] += 1
            V[(j0 + 1) * p + (k0 + 1), c] += 1
            W[k0 * n + i0, c] += 1
            col += 1
            
        # Schoolbook terms
        for (i, j, k) in schoolbook_terms:
            c = col
            U[i * m + j, c] += 1
            V[j * p + k, c] += 1
            W[k * n + i, c] += 1
            col += 1
            
        return U, V, W

    U, V, W = build_solution(max_blocks)
    
    # Validate the solution
    def validate(U, V, W):
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)
        
    if not validate(U, V, W):
        # Fallback to pure schoolbook if validation fails (should not happen)
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
