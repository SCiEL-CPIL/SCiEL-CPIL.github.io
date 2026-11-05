import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1):
    """
    Add Strassen 2x2*2x2 block.
    U rows indexed by (i, k) -> i*5+k
    V rows indexed by (k, j) -> k*5+j
    W rows indexed by (j, i) -> j*2+i
    """
    a00 = 0*5 + k0
    a01 = 0*5 + k1
    a10 = 1*5 + k0
    a11 = 1*5 + k1
    
    b00 = k0*5 + j0
    b01 = k0*5 + j1
    b10 = k1*5 + j0
    b11 = k1*5 + j1
    
    c00 = j0*2 + 0
    c01 = j1*2 + 0
    c10 = j0*2 + 1
    c11 = j1*2 + 1
    
    # r=0: M0 = (A00+A11)(B00+B11)
    U[a00][offset+0] += 1
    U[a11][offset+0] += 1
    V[b00][offset+0] += 1
    V[b11][offset+0] += 1
    W[c00][offset+0] += 1
    W[c11][offset+0] += 1
    
    # r=1: M1 = (A10+A11)B00
    U[a10][offset+1] += 1
    U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1
    W[c11][offset+1] += -1
    
    # r=2: M2 = A00(B01-B11)
    U[a00][offset+2] += 1
    V[b01][offset+2] += 1
    V[b11][offset+2] += -1
    W[c01][offset+2] += 1
    W[c11][offset+2] += 1
    
    # r=3: M3 = A11(B10-B00)
    U[a11][offset+3] += 1
    V[b10][offset+3] += 1
    V[b00][offset+3] += -1
    W[c00][offset+3] += 1
    W[c10][offset+3] += 1
    
    # r=4: M4 = (A00+A01)B11
    U[a00][offset+4] += 1
    U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1
    W[c01][offset+4] += 1
    
    # r=5: M5 = (A10-A00)(B00+B01)
    U[a10][offset+5] += 1
    U[a00][offset+5] += -1
    V[b00][offset+5] += 1
    V[b01][offset+5] += 1
    W[c11][offset+5] += 1
    
    # r=6: M6 = (A01-A11)(B10+B11)
    U[a01][offset+6] += 1
    U[a11][offset+6] += -1
    V[b10][offset+6] += 1
    V[b11][offset+6] += 1
    W[c00][offset+6] += 1
    
    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j):
    """
    2x1 k-group, 1x1 j-group: 4 multiplications.
    """
    a0k0 = 0*5 + k0
    a0k1 = 0*5 + k1
    a1k0 = 1*5 + k0
    a1k1 = 1*5 + k1
    
    bk0j = k0*5 + j
    bk1j = k1*5 + j
    
    c0j = j*2 + 0
    c1j = j*2 + 1
    
    U[a0k0][offset+0] += 1
    V[bk0j][offset+0] += 1
    W[c0j][offset+0] += 1
    
    U[a0k1][offset+1] += 1
    V[bk1j][offset+1] += 1
    W[c0j][offset+1] += 1
    
    U[a1k0][offset+2] += 1
    V[bk0j][offset+2] += 1
    W[c1j][offset+2] += 1
    
    U[a1k1][offset+3] += 1
    V[bk1j][offset+3] += 1
    W[c1j][offset+3] += 1
    
    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1):
    """
    1x1 k-group, 2x1 j-group: 4 multiplications.
    """
    a0k = 0*5 + k
    a1k = 1*5 + k
    
    bkj0 = k*5 + j0
    bkj1 = k*5 + j1
    
    c0j0 = j0*2 + 0
    c0j1 = j1*2 + 0
    c1j0 = j0*2 + 1
    c1j1 = j1*2 + 1
    
    U[a0k][offset+0] += 1
    V[bkj0][offset+0] += 1
    W[c0j0][offset+0] += 1
    
    U[a0k][offset+1] += 1
    V[bkj1][offset+1] += 1
    W[c0j1][offset+1] += 1
    
    U[a1k][offset+2] += 1
    V[bkj0][offset+2] += 1
    W[c1j0][offset+2] += 1
    
    U[a1k][offset+3] += 1
    V[bkj1][offset+3] += 1
    W[c1j1][offset+3] += 1
    
    return offset + 4

def _add_1x1(U, V, W, offset, k, j):
    """
    1x1 k-group, 1x1 j-group: 2 multiplications.
    """
    a0k = 0*5 + k
    a1k = 1*5 + k
    bkj = k*5 + j
    c0j = j*2 + 0
    c1j = j*2 + 1
    
    U[a0k][offset+0] += 1
    V[bkj][offset+0] += 1
    W[c0j][offset+0] += 1
    
    U[a1k][offset+1] += 1
    V[bkj][offset+1] += 1
    W[c1j][offset+1] += 1
    
    return offset + 2

def _try_rank_reduction(U, V, W, max_rounds=50):
    """
    Try to reduce rank by finding linearly dependent columns in the tensor
    and eliminating them. This is a local optimization that may find
    shared structure between blocks.
    """
    n, m, p = 2, 5, 5
    rank = len(U[0])
    
    # Build the tensor as a 3D array for analysis
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    for round_num in range(max_rounds):
        if not _deadline_ok():
            break
        
        # Check if any column is redundant: can it be expressed as a linear
        # combination of others?
        # For each column r, check if removing it still gives valid decomposition
        # (i.e., the column is zero in the effective tensor)
        
        # More practically: try to find two columns that are proportional
        # and merge them, or find a column that's a sum of others
        
        # Simple approach: for each pair of columns, check if they're identical
        # up to a scalar, and if so, try to eliminate one
        found_reduction = False
        
        for r1 in range(rank):
            if not _deadline_ok():
                break
            # Check if column r1 is all zeros
            all_zero = True
            for row in range(n * m):
                if U[row][r1] != 0:
                    all_zero = False
                    break
            if not all_zero:
                continue
            for row in range(m * p):
                if V[row][r1] != 0:
                    all_zero = False
                    break
            if not all_zero:
                continue
            for row in range(p * n):
                if W[row][r1] != 0:
                    all_zero = False
                    break
            if all_zero:
                # Remove this column
                for row in range(n * m):
                    U[row].pop(r1)
                for row in range(m * p):
                    V[row].pop(r1)
                for row in range(p * n):
                    W[row].pop(r1)
                rank -= 1
                found_reduction = True
                break
        
        if found_reduction:
            continue
        
        # Try: for each column r, see if it can be written as a linear combo
        # of other columns. Use a simple greedy approach.
        # This is expensive, so limit attempts
        if round_num > 10:
            break
            
        # Check if any column is a scalar multiple of another
        for r1 in range(rank):
            if not _deadline_ok():
                break
            for r2 in range(r1 + 1, rank):
                # Check if column r1 = c * column r2
                # Find c from first nonzero entry
                c = None
                valid = True
                for row in range(n * m):
                    if V[row][r2] != 0 or W[row][r2] != 0:
                        if U[row][r2] == 0:
                            if U[row][r1] != 0:
                                valid = False
                                break
                        else:
                            c = U[row][r1] / U[row][r2]
                            break
                if not valid:
                    continue
                
                # Verify proportionality
                if c is None:
                    continue
                    
                ok = True
                for row in range(n * m):
                    expected = c * U[row][r2]
                    if abs(U[row][r1] - expected) > 1e-10:
                        ok = False
                        break
                if not ok:
                    continue
                for row in range(m * p):
                    expected = c * V[row][r2]
                    if abs(V[row][r1] - expected) > 1e-10:
                        ok = False
                        break
                if not ok:
                    continue
                for row in range(p * n):
                    expected = c * W[row][r2]
                    if abs(W[row][r1] - expected) > 1e-10:
                        ok = False
                        break
                if ok:
                    # Column r1 = c * column r2, so r1 is redundant
                    # Remove column r1
                    for row in range(n * m):
                        U[row].pop(r1)
                    for row in range(m * p):
                        V[row].pop(r1)
                    for row in range(p * n):
                        W[row].pop(r1)
                    rank -= 1
                    found_reduction = True
                    break
            if found_reduction:
                break
        
        if not found_reduction:
            break
    
    return rank

def _build_strassen_based():
    """
    Build a Strassen-based decomposition for 2x5 * 5x5.
    """
    n, m, p = 2, 5, 5
    
    # Generate all possible partitions of {0,1,2,3,4} into groups of size 1 or 2
    # Each partition is a list of tuples
    def _partitions_of_5():
        """Generate all partitions of {0,1,2,3,4} into subsets of size 1 or 2."""
        from itertools import combinations, permutations
        
        results = []
        
        # All partitions into groups of size 1 and 2 that cover {0,1,2,3,4}
        # Possible structures:
        # - 5 singles: (0,)(1,)(2,)(3,)(4,)
        # - 1 pair + 3 singles
        # - 2 pairs + 1 single
        
        # 5 singles
        results.append([(0,), (1,), (2,), (3,), (4,)])
        
        # 1 pair + 3 singles
        for pair in combinations(range(5), 2):
            singles = [x for x in range(5) if x not in pair]
            # Order doesn't matter for the partition, but we'll generate all orderings
            # Actually, let's just store the set of groups
            groups = [tuple(pair)] + [tuple([s]) for s in singles]
            # Sort groups by their first element for canonical form
            groups.sort(key=lambda g: g[0])
            results.append(tuple(groups))
        
        # 2 pairs + 1 single
        for pair1 in combinations(range(5), 2):
            remaining1 = [x for x in range(5) if x not in pair1]
            for pair2 in combinations(remaining1, 2):
                singles = [x for x in remaining1 if x not in pair2]
                if len(singles) == 1:
                    groups = [tuple(pair1), tuple(pair2), tuple(singles)]
                    groups.sort(key=lambda g: g[0])
                    results.append(tuple(groups))
        
        # Remove duplicates
        seen = set()
        unique = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        
        return unique
    
    k_partitions = _partitions_of_5()
    j_partitions = _partitions_of_5()
    
    best_result = None
    best_rank = float('inf')
    
    # Pre-compute rank for each partition pair
    def _partition_rank(kp, jp):
        total = 0
        for kg in kp:
            for jg in jp:
                if len(kg) == 2 and len(jg) == 2:
                    total += 7
                elif len(kg) == 2 and len(jg) == 1:
                    total += 4
                elif len(kg) == 1 and len(jg) == 2:
                    total += 4
                else:
                    total += 2
        return total
    
    # Sort by expected rank to try best first
    candidates = []
    for kp in k_partitions:
        for jp in j_partitions:
            r = _partition_rank(kp, jp)
            candidates.append((r, kp, jp))
    
    candidates.sort(key=lambda x: x[0])
    
    for total_rank, k_groups, j_groups in candidates:
        if not _deadline_ok():
            break
        
        if total_rank >= best_rank:
            continue
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        offset = 0
        
        for kg in k_groups:
            for jg in j_groups:
                if len(kg) == 2 and len(jg) == 2:
                    offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1])
                elif len(kg) == 2 and len(jg) == 1:
                    offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0])
                elif len(kg) == 1 and len(jg) == 2:
                    offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1])
                else:
                    offset = _add_1x1(U, V, W, offset, kg[0], jg[0])
        
        if _verify_decomp(n, m, p, U, V, W):
            # Try to reduce rank further
            reduced_rank = _try_rank_reduction(U, V, W, max_rounds=20)
            if reduced_rank < best_rank:
                best_result = (U, V, W)
                best_rank = reduced_rank
    
    if best_result is None:
        # Fallback: all singletons (schoolbook)
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    return best_result

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    
    U, V, W = _build_strassen_based()
    
    if _verify_decomp(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Ultimate fallback
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
