import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 140  # safety margin before 150s limit

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def verify(U_cols, V_cols, W_cols):
        """Verify that the decomposition exactly satisfies the tensor constraint."""
        rank = len(U_cols)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U_cols[r][a] * V_cols[r][b] * W_cols[r][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_remove_terms(U_cols, V_cols, W_cols, max_remove=2):
        """Try to remove up to max_remove terms by solving a linear system."""
        rank = len(U_cols)
        if rank <= 7:
            return None

        # Try removing pairs first, then single terms
        removal_sizes = list(range(min(max_remove, rank - 7), 0, -1))
        
        for size in removal_sizes:
            if time.time() > deadline - 10:
                return None
            
            # For small ranks, try all combinations; for larger, sample
            if rank - size <= 20:
                candidates = list(combinations(range(rank), size))
            else:
                # Sample random combinations
                candidates = []
                for _ in range(80):
                    candidates.append(tuple(np.random.choice(rank, size, replace=False)))
                # Also try consecutive groups
                for start in range(0, rank - size + 1, max(1, (rank - size) // 15)):
                    candidates.append(tuple(range(start, start + size)))

            for combo in candidates:
                if time.time() > deadline - 10:
                    return None
                
                keep = [r for r in range(rank) if r not in combo]
                keep_set = set(keep)
                
                # Build linear system: for each (i,k,j), sum_{r in keep} d_r * U_r[a]*V_r[b]*W_r[c] = target
                # where target = sum_{r in combo} U_r[a]*V_r[b]*W_r[c]
                equations = []
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            a = i * m + k
                            b = k * p + j
                            c = j * n + i
                            target = 0.0
                            for r in combo:
                                target += U_cols[r][a] * V_cols[r][b] * W_cols[r][c]
                            if abs(target) > 1e-12:
                                row = np.zeros(len(keep))
                                for idx, r in enumerate(keep):
                                    row[idx] = U_cols[r][a] * V_cols[r][b] * W_cols[r][c]
                                equations.append((row, target))
                
                if not equations:
                    continue
                
                A_sys = np.array([eq[0] for eq in equations])
                b_sys = np.array([eq[1] for eq in equations])
                
                try:
                    d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-12)
                    if np.allclose(A_sys @ d, b_sys, atol=1e-10):
                        # Valid removal found
                        new_U = [U_cols[r] for r in keep]
                        new_V = [V_cols[r] for r in keep]
                        new_W = [W_cols[r] for r in keep]
                        if verify(new_U, new_V, new_W):
                            return (new_U, new_V, new_W)
                except:
                    pass
        return None

    def try_merge_terms(U_cols, V_cols, W_cols):
        """Try to merge two terms into one by finding a rank-1 replacement."""
        rank = len(U_cols)
        if rank <= 2:
            return None
        
        # Try pairs of terms
        for r1 in range(rank):
            if time.time() > deadline - 10:
                return None
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 15:
                    return None
                
                keep = [r for r in range(rank) if r != r1 and r != r2]
                
                # The contribution of terms r1 and r2 to each tensor entry must be
                # reproducible by a single rank-1 term plus adjustments to kept terms.
                # Actually, we want to replace r1 and r2 with one new term.
                # This is harder. Instead, try: can we absorb r2 into r1?
                # i.e., find alpha such that r1 + alpha*r2 has the same effect
                # as r1 alone, and r2 can be removed.
                
                # Simpler: try to remove r2 and adjust all other terms
                keep_with_r1 = [r for r in range(rank) if r != r2]
                equations = []
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            a = i * m + k
                            b = k * p + j
                            c = j * n + i
                            target = U_cols[r2][a] * V_cols[r2][b] * W_cols[r2][c]
                            if abs(target) > 1e-12:
                                row = np.zeros(len(keep_with_r1))
                                for idx, r in enumerate(keep_with_r1):
                                    row[idx] = U_cols[r][a] * V_cols[r][b] * W_cols[r][c]
                                equations.append((row, target))
                
                if not equations:
                    continue
                
                A_sys = np.array([eq[0] for eq in equations])
                b_sys = np.array([eq[1] for eq in equations])
                
                try:
                    d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-12)
                    if np.allclose(A_sys @ d, b_sys, atol=1e-10):
                        new_U = [U_cols[r] for r in keep_with_r1]
                        new_V = [V_cols[r] for r in keep_with_r1]
                        new_W = [W_cols[r] for r in keep_with_r1]
                        if verify(new_U, new_V, new_W):
                            return (new_U, new_V, new_W)
                except:
                    pass
        return None

    # Build initial decomposition using 4x2 Strassen for first 4 columns
    # Strassen for 4x2 * 2x2: use 4x2 matrix product via 2x2 Strassen on top/bottom
    # Actually, let's use a cleaner approach:
    # For A (2x3) * B (3x9), split B into 3 blocks of 3 columns each.
    # For each 3-col block: A[0:2, 0:2] * B[0:2, block] + A[0:2, 2] * B[2, block]
    
    # Use Strassen for 2x2 * 2x3 (3 columns at a time)
    # 2x2 * 2x3 can be done with 7 multiplications (Strassen-style)
    # Let's just use schoolbook for now but try to be smarter
    
    # Strategy: For columns 0-3, use a 2x2x4 block product
    # 2x2 * 2x4: can use 7*2 = 14 mults or better
    # Actually 2x2 * 2x4 = two 2x2 * 2x2 = 7+7 = 14 mults
    # Or use 4x2 * 2x4 via Strassen: 7 mults for 4x4 but we only need 2x4 output
    
    # Let me just use the existing approach but with better blocks
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8] using Strassen for 2-column blocks (4 blocks)
    for block_start in range(0, 8, 2):
        if time.time() > deadline:
            break
        k0 = block_start
        k1 = block_start + 1

        # Strassen 7 terms for 2x2 * 2x2
        add_term({0: 1, 4: 1}, {k0: 1, p+k1: 1}, {k0*n: 1, k1*n+1: 1})
        add_term({3: 1, 4: 1}, {k0: 1}, {k0*n+1: 1, k1*n+1: -1})
        add_term({0: 1}, {k1: 1, p+k1: -1}, {k1*n: 1, k1*n+1: 1})
        add_term({4: 1}, {p+k0: 1, k0: -1}, {k0*n: 1, k0*n+1: 1})
        add_term({0: 1, 1: 1}, {p+k1: 1}, {k0*n: -1, k1*n: 1})
        add_term({3: 1, 0: -1}, {k0: 1, k1: 1}, {k1*n+1: 1})
        add_term({1: 1, 4: -1}, {p+k0: 1, p+k1: 1}, {k0*n: 1})

    # Single column k=8: schoolbook 2x2 x 2x1
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})

    # Part 2: A[0:2, 2] * B[2, 0:9] - schoolbook
    for i in range(n):
        if time.time() > deadline:
            break
        for k in range(p):
            add_term({i*m+2: 1}, {2*p+k: 1}, {k*n+i: 1})

    # Verify initial decomposition
    if not verify(U_cols, V_cols, W_cols):
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    add_term({i*m+k: 1}, {k*p+j: 1}, {j*n+i: 1})

    # Iteratively try to reduce rank
    for iteration in range(50):
        if time.time() > deadline - 15:
            break
        
        # Try removal first
        result = try_remove_terms(U_cols, V_cols, W_cols, max_remove=2)
        if result is not None:
            U_cols, V_cols, W_cols = result
            continue
        
        # Try single term removal
        result = try_remove_terms(U_cols, V_cols, W_cols, max_remove=1)
        if result is not None:
            U_cols, V_cols, W_cols = result
            continue
        
        # Try merging
        result = try_merge_terms(U_cols, V_cols, W_cols)
        if result is not None:
            U_cols, V_cols, W_cols = result
            continue
        
        # No improvement possible
        break

    rank = len(U_cols)

    # Final verification
    if not verify(U_cols, V_cols, W_cols):
        # Emergency fallback: schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    add_term({i*m+k: 1}, {k*p+j: 1}, {j*n+i: 1})
        rank = len(U_cols)

    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]

    return {"U": U, "V": V, "W": W}
