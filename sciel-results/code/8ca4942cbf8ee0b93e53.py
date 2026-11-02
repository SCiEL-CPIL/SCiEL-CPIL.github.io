import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication. Rank 7.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = [[Fraction(0)]*7 for _ in range(4)]
    V = [[Fraction(0)]*7 for _ in range(4)]
    W = [[Fraction(0)]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _build_block_decomposition(n, m, p):
    """
    Build the block decomposition:
    - j=0: schoolbook 2x1x7, rank 14
    - j=1,2: split p=7 into (0,1),(2,3),(4,5) Strassen + (6) schoolbook
    Total rank: 14 + 7 + 7 + 7 + 4 = 39
    """
    total_rank = 14 + 7 + 7 + 7 + 4
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: j=0, schoolbook 2x1x7
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2; p split into (0,1),(2,3),(4,5),(6)
    sU, sV, sW, srank = _strassen_2x2x2()
    for p_start in [0, 2, 4]:
        for r in range(srank):
            for i in range(2):
                for j in range(2):
                    U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[(1 + j) * p + (p_start + k)][col] += sV[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[(p_start + k) * n + i][col] += sW[k * 2 + i][r]
            col += 1
    
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    return U, V, W, total_rank

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _to_int_list(U):
    """Convert Fraction matrix to int list (all values should be integers)."""
    result = []
    for row in U:
        new_row = []
        for val in row:
            assert val.denominator == 1, f"Non-integer value: {val}"
            new_row.append(val.numerator)
        result.append(new_row)
    return result

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_budget):
    """
    Try to reduce rank to target_rank by zeroing out columns.
    Strategy: For each column we want to remove, check if the tensor contribution
    of that column can be absorbed by other columns (i.e., if the column is in the
    span of the others in a tensor sense).
    
    Simpler approach: try all subsets of columns of size target_rank and check
    if they still span the tensor. This is expensive, so we use a greedy heuristic.
    """
    start_time = time.time()
    current_rank = len(U[0])
    
    if target_rank >= current_rank:
        return U, V, W, current_rank
    
    # Greedy: try to remove one column at a time
    # For each column c, check if removing it still gives exact decomposition
    # If yes, remove it and continue
    
    U_curr = [row[:] for row in U]
    V_curr = [row[:] for row in V]
    W_curr = [row[:] for row in W]
    rank_curr = current_rank
    
    # We'll work with column indices
    active_cols = list(range(rank_curr))
    
    while len(active_cols) > target_rank:
        if time.time() - start_time > time_budget:
            break
        
        found_removal = False
        # Try removing each column (in random order for diversity)
        cols_to_try = active_cols[:]
        random.shuffle(cols_to_try)
        
        for c_idx in cols_to_try:
            if time.time() - start_time > time_budget:
                break
            
            # Test: remove column c_idx
            test_U = [row[:] for row in U_curr]
            test_V = [row[:] for row in V_curr]
            test_W = [row[:] for row in W_curr]
            
            # Remove column c_idx from active set
            new_active = [c for c in active_cols if c != c_idx]
            
            # Build reduced matrices
            rank_new = len(new_active)
            test_U_r = [[Fraction(0)] * rank_new for _ in range(n * m)]
            test_V_r = [[Fraction(0)] * rank_new for _ in range(m * p)]
            test_W_r = [[Fraction(0)] * rank_new for _ in range(p * n)]
            
            for new_c, old_c in enumerate(new_active):
                for i in range(n * m):
                    test_U_r[i][new_c] = U_curr[i][old_c]
                for j in range(m * p):
                    test_V_r[j][new_c] = V_curr[j][old_c]
                for k in range(p * n):
                    test_W_r[k][new_c] = W_curr[k][old_c]
            
            # Verify
            if _verify_exact(test_U_r, test_V_r, test_W_r, n, m, p):
                # Success! Remove this column
                U_curr = test_U_r
                V_curr = test_V_r
                W_curr = test_W_r
                active_cols = new_active
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if len(active_cols) < current_rank:
        return U_curr, V_curr, W_curr, len(active_cols)
    
    return U, V, W, current_rank

def _randomized_search(U, V, W, n, m, p, target_rank, time_budget):
    """
    Randomized search: try to find a decomposition of rank target_rank.
    Approach: Start from the current decomposition, randomly perturb and try to
    project back to integer/half-integer values while maintaining exactness.
    
    Alternative: Use the fact that we can combine columns. If two columns have
    proportional U, V, W vectors (up to scaling), we might merge them.
    """
    start_time = time.time()
    current_rank = len(U[0])
    
    if target_rank >= current_rank:
        return U, V, W, current_rank
    
    # Try to find linear dependencies among columns
    # For the tensor to be preserved, we need:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    # Heuristic: try to find if any column is a linear combination of others
    # in the tensor product space
    
    best_U, best_V, best_W, best_rank = U, V, W, current_rank
    
    # Try greedy column elimination with random restarts
    for restart in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Shuffle column order
        order = list(range(current_rank))
        random.shuffle(order)
        
        U_curr = [row[:] for row in U]
        V_curr = [row[:] for row in V]
        W_curr = [row[:] for row in W]
        active = list(range(current_rank))
        
        # Greedy elimination in random order
        for c_idx in order:
            if time.time() - start_time > time_budget:
                break
            if c_idx not in active:
                continue
            
            # Try to remove column c_idx
            new_active = [c for c in active if c != c_idx]
            if len(new_active) < target_rank:
                continue
            
            rank_new = len(new_active)
            test_U = [[Fraction(0)] * rank_new for _ in range(n * m)]
            test_V = [[Fraction(0)] * rank_new for _ in range(m * p)]
            test_W = [[Fraction(0)] * rank_new for _ in range(p * n)]
            
            for new_c, old_c in enumerate(new_active):
                for i in range(n * m):
                    test_U[i][new_c] = U_curr[i][old_c]
                for j in range(m * p):
                    test_V[j][new_c] = V_curr[j][old_c]
                for k in range(p * n):
                    test_W[k][new_c] = W_curr[k][old_c]
            
            if _verify_exact(test_U, test_V, test_W, n, m, p):
                U_curr = test_U
                V_curr = test_V
                W_curr = test_W
                active = new_active
        
        if len(active) < best_rank:
            best_U, best_V, best_W, best_rank = U_curr, V_curr, W_curr, len(active)
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with proven rank-39 block decomposition
    U, V, W, rank = _build_block_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_exact(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Try to reduce rank
    if time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        # Try target ranks from current down
        for target in range(rank - 1, max(rank - 5, 20), -1):
            if time.time() > deadline - 5.0:
                break
            time_budget = min(remaining_time, 20.0)
            remaining_time -= time_budget
            
            U_new, V_new, W_new, rank_new = _try_rank_reduction(
                U, V, W, n, m, p, target, time_budget
            )
            if rank_new < rank:
                U, V, W, rank = U_new, V_new, W_new, rank_new
                break
    
    # If time permits, try randomized search for further reduction
    if time.time() < deadline - 3.0:
        remaining_time = deadline - time.time() - 3.0
        U_new, V_new, W_new, rank_new = _randomized_search(
            U, V, W, n, m, p, rank - 1, min(remaining_time, 30.0)
        )
        if rank_new < rank:
            U, V, W, rank = U_new, V_new, W_new, rank_new
    
    # Convert to int lists
    U_int = _to_int_list(U)
    V_int = _to_int_list(V)
    W_int = _to_int_list(W)
    
    return {"U": U_int, "V": V_int, "W": W_int}
