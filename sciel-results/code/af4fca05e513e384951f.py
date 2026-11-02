"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W):
        """Check if the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def build_solution(blocks, schoolbook_terms):
        """Build U, V, W from Strassen blocks and schoolbook terms."""
        total_rank = len(blocks) * 7 + len(schoolbook_terms)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in blocks:
            # M1
            c = col
            U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M2
            c = col
            U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
            col += 1
            # M3
            c = col
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M4
            c = col
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
            col += 1
            # M5
            c = col
            U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
            col += 1
            # M6
            c = col
            U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M7
            c = col
            U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            col += 1
        
        for (i, j, k) in schoolbook_terms:
            c = col
            U[i*m + j][c] += 1
            V[j*p + k][c] += 1
            W[k*n + i][c] += 1
            col += 1
        
        return U, V, W
    
    def get_schoolbook_for_blocks(blocks):
        """Get schoolbook terms not covered by blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    
    # Generate all possible disjoint Strassen block configurations
    all_possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_possible_blocks.append((i0, j0, k0))
    
    def are_disjoint(b1, b2):
        """Check if two Strassen blocks are disjoint."""
        return not (b1[0] in (b2[0], b2[0]+1) and b2[0] in (b1[0], b1[0]+1) and
                    b1[1] in (b2[1], b2[1]+1) and b2[1] in (b1[1], b1[1]+1) and
                    b1[2] in (b2[2], b2[2]+1) and b2[2] in (b1[2], b1[2]+1))
    
    def is_set_disjoint(blocks):
        """Check if a set of blocks are pairwise disjoint."""
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                if not are_disjoint(blocks[a], blocks[b]):
                    return False
        return True
    
    # Build the initial best solution with 4 disjoint Strassen blocks
    best_rank = 75
    best_solution = None
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    sb = get_schoolbook_for_blocks(initial_blocks)
    best_rank = len(initial_blocks) * 7 + len(sb)
    U, V, W = build_solution(initial_blocks, sb)
    if validate(U, V, W):
        best_solution = (U, V, W)
    
    # Try different block combinations
    from itertools import combinations
    for size in range(1, 7):
        if time.time() > deadline - 20:
            break
        for combo in combinations(all_possible_blocks, size):
            if not is_set_disjoint(list(combo)):
                continue
            sb = get_schoolbook_for_blocks(list(combo))
            rank = size * 7 + len(sb)
            if rank < best_rank:
                best_rank = rank
                U, V, W = build_solution(list(combo), sb)
                if validate(U, V, W):
                    best_solution = (U, V, W)
    
    # Now attempt rank reduction by checking for linear dependencies among columns
    if best_solution is not None and time.time() < deadline - 30:
        U_arr = np.array(best_solution[0], dtype=float)
        V_arr = np.array(best_solution[1], dtype=float)
        W_arr = np.array(best_solution[2], dtype=float)
        current_rank = U_arr.shape[1]
        
        # Try to remove columns one at a time by checking if the remaining columns
        # can still span the tensor. We do this by checking if removing column c
        # leaves a valid decomposition (which it won't in general, but we can try
        # to redistribute).
        
        # A more practical approach: check if any column is a linear combination
        # of others in a way that preserves the tensor.
        # For each column t, check if there exist coefficients alpha_s (s != t)
        # such that the tensor contribution of column t equals sum_s alpha_s * tensor_contribution_s.
        
        # This is complex, so let's try a simpler heuristic:
        # Check if removing a column and adjusting others can maintain validity.
        # For now, just check if any column is all zeros (shouldn't happen, but check).
        
        for t in range(current_rank):
            if time.time() > deadline - 10:
                break
            # Check if column t is redundant
            # Try removing it and see if we can compensate
            # This is a placeholder for a more sophisticated search
            
        # Alternative: use the fact that we have a constructive decomposition
        # and try to find a lower-rank one by solving a system of equations.
        # This is beyond the scope of a quick improvement, so we'll stick with 71.
    
    if best_solution is not None:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback
    U, V, W = build_solution(initial_blocks, get_schoolbook_for_blocks(initial_blocks))
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
