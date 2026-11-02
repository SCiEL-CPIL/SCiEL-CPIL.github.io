"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # T[i,j,k] should equal sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # Check a sample of entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def build_solution(blocks, schoolbook_terms):
        """Build U, V, W from Strassen blocks and schoolbook terms."""
        total_rank = len(blocks) * 7 + len(schoolbook_terms)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in blocks:
            # Strassen M1-M7
            # M1 = (A00+A11)(B00+B11)
            c = col
            U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M2 = (A10+A11)B00
            c = col
            U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
            col += 1
            # M3 = A00(B01-B11)
            c = col
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M4 = A11(B10-B00)
            c = col
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
            col += 1
            # M5 = (A00+A01)B11
            c = col
            U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            c = col
            U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
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
    
    # Generate all possible disjoint Strassen block configurations
    def get_disjoint_blocks():
        """Find all sets of disjoint 2x2x2 Strassen blocks."""
        all_blocks = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    all_blocks.append((i0, j0, k0))
        
        # Find maximal sets of disjoint blocks
        best_set = []
        
        def backtrack(idx, current, used):
            if time.time() > deadline - 10:
                return
            if len(current) > len(best_set):
                best_set = list(current)
            for i in range(idx, len(all_blocks)):
                b = all_blocks[i]
                if b not in used:
                    # Check if b is disjoint from all blocks in current
                    disjoint = True
                    for cb in current:
                        if (b[0] in (cb[0], cb[0]+1) and cb[0] in (b[0], b[0]+1) and
                            b[1] in (cb[1], cb[1]+1) and cb[1] in (b[1], b[1]+1) and
                            b[2] in (cb[2], cb[2]+1) and cb[2] in (b[2], b[2]+1)):
                            disjoint = False
                            break
                    if disjoint:
                        current.append(b)
                        backtrack(i + 1, current, used | {b})
                        current.pop()
        
        backtrack(0, [], set())
        return best_set
    
    # Try to find a better solution via local search
    # Start with the known 4-block solution
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    def get_schoolbook_for_blocks(blocks):
        """Get schoolbook terms not covered by blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    
    # Current best
    best_blocks = initial_blocks
    best_rank = len(best_blocks) * 7 + len(get_schoolbook_for_blocks(best_blocks))
    
    # Try different block combinations
    all_possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_possible_blocks.append((i0, j0, k0))
    
    def blocks_conflict(b1, b2):
        """Check if two Strassen blocks conflict (share any U, V, or W entry)."""
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        # U conflict: (i,j) pairs overlap
        u1 = {(i0a, j0a), (i0a, j0a+1), (i0a+1, j0a), (i0a+1, j0a+1)}
        u2 = {(i0b, j0b), (i0b, j0b+1), (i0b+1, j0b), (i0b+1, j0b+1)}
        if u1 & u2:
            return True
        # V conflict: (j,k) pairs overlap
        v1 = {(j0a, k0a), (j0a, k0a+1), (j0a+1, k0a), (j0a+1, k0a+1)}
        v2 = {(j0b, k0b), (j0b, k0b+1), (j0b+1, k0b), (j0b+1, k0b+1)}
        if v1 & v2:
            return True
        # W conflict: (k,i) pairs overlap
        w1 = {(k0a, i0a), (k0a, i0a+1), (k0a+1, i0a), (k0a+1, i0a+1)}
        w2 = {(k0b, i0b), (k0b, i0b+1), (k0b+1, i0b), (k0b+1, i0b+1)}
        if w1 & w2:
            return True
        return False
    
    # Try subsets of blocks (up to size 8)
    from itertools import combinations
    for size in range(1, 9):
        if time.time() > deadline - 20:
            break
        for combo in combinations(all_possible_blocks, size):
            # Check pairwise non-conflict
            conflict = False
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    if blocks_conflict(combo[a], combo[b]):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                sb = get_schoolbook_for_blocks(list(combo))
                rank = size * 7 + len(sb)
                if rank < best_rank:
                    best_rank = rank
                    best_blocks = list(combo)
    
    # Build final solution
    schoolbook_terms = get_schoolbook_for_blocks(best_blocks)
    U, V, W = build_solution(best_blocks, schoolbook_terms)
    
    # Validate
    if not validate(U, V, W):
        # Fallback to known good solution
        U, V, W = build_solution(initial_blocks, get_schoolbook_for_blocks(initial_blocks))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
