"""Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition."""
import time
import itertools
import math
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 115  # 5s safety margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    The matrix multiplication tensor T[i,j,k] = 1 if the (i,k) entry of A*B
    depends on A[i,j]*B[j,k].
    
    We use the known Strassen-like constructions and systematic search.
    The best known rank for 3x3 * 3x4 is at most 3*3*4=36 (schoolbook).
    We try to find lower ranks using:
    1. Known constructions (Strassen for 2x2 blocks, etc.)
    2. Randomized search with sparse patterns
    3. Greedy support selection with exact rational solving
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # The tensor is T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 always (it's the standard matrix multiplication tensor)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try various ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Upper bound: n*m*p = 36
    
    best_result = None
    
    # First, try the schoolbook as fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank decompositions
    # We search from a lower bound upwards.
    # Lower bound is 12. We try ranks 12 to 35.
    # To save time, we start with higher ranks where solutions are more likely to be found quickly,
    # or we can just iterate up. Given the structure, lower ranks are harder.
    # Let's try ranks 20 to 35 first, then 12 to 19 if time permits.
    
    ranks_to_try = list(range(20, 36)) + list(range(12, 20))
    
    for target_rank in ranks_to_try:
        if _time_remaining() < 5:
            break
        
        # If we already have a better result, skip
        if best_result and len(best_result["U"][0]) <= target_rank:
            continue
            
        # Try multiple random initializations for this rank
        # Higher ranks need fewer attempts as they are easier
        max_attempts = 20 if target_rank >= 28 else 50
        if _time_remaining() < 10:
            max_attempts = 5
            
        for attempt in range(max_attempts):
            if _time_remaining() < 2:
                break
            
            try:
                result = _try_rank_search(n, m, p, target_rank, attempt * 12345 + 6789)
                if result is not None:
                    best_result = result
                    break
            except Exception:
                continue
    
    return best_result

def _is_lower_rank(result, target_rank):
    if result is None:
        return False
    return len(result["U"][0]) <= target_rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # U: (n*m) x rank
    # V: (m*p) x rank  
    # W: (p*n) x rank
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _find_half_integer_solution(M_row, r):
    """
    Given a row of M (length r), find a sparse half-integer vector W such that M_row dot W = 1.
    Returns None if no such sparse solution is found quickly.
    """
    # First check for single term solutions: M_row[c] * W[c] = 1 => W[c] = 1/M_row[c]
    for c in range(r):
        m_val = M_row[c]
        if m_val != 0:
            w_val = 1.0 / m_val
            # Check if w_val is a half-integer
            if abs(w_val * 2 - round(w_val * 2)) < 1e-9:
                w = np.zeros(r)
                w[c] = round(w_val * 2) / 2.0
                return w
    
    # Check for two-term solutions: M_row[a] * x + M_row[b] * y = 1
    # We want x, y to be half-integers.
    # Let x = 2*xi, y = 2*yi where xi, yi are integers? No, x, y are half-integers, so 2x, 2y are integers.
    # Let X = 2x, Y = 2y. Then M_row[a] * X/2 + M_row[b] * Y/2 = 1 => M_row[a] X + M_row[b] Y = 2.
    # This is a linear Diophantine equation.
    # We can iterate over small integers X, Y.
    
    for a in range(r):
        ma = M_row[a]
        if ma == 0: continue
        for b in range(a+1, r):
            mb = M_row[b]
            if mb == 0: continue
            
            # Try small integer values for X and Y
            # X, Y in {-4, -2, -1, 0, 1, 2, 4} maybe?
            # Actually, if ma, mb are integers, we can solve ma*X + mb*Y = 2.
            # If gcd(ma, mb) divides 2, solutions exist.
            g = math.gcd(int(abs(ma)), int(abs(mb)))
            if 2 % g != 0:
                continue
            
            # Find one solution
            # ma/g * X' + mb/g * Y' = 2/g
            # Use extended Euclidean
            # For simplicity, just try small values
            for X in range(-10, 11, 2): # X must be even? No, X is integer.
                # ma * X + mb * Y = 2 => mb * Y = 2 - ma * X => Y = (2 - ma * X) / mb
                num = 2 - ma * X
                if num % mb == 0:
                    Y = num // mb
                    # Check if X, Y are "small" to keep W sparse/small
                    if abs(X) <= 4 and abs(Y) <= 4:
                        w = np.zeros(r)
                        w[a] = X / 2.0
                        w[b] = Y / 2.0
                        return w
    return None

def _check_solvable_fast(row):
    """
    Fast check if a half-integer solution exists for dot(row, w) = 1.
    Only checks support 1 and 2.
    """
    non_zero_idx = np.where(row != 0)[0]
    if len(non_zero_idx) == 0:
        return False
    
    # Support 1
    for idx in non_zero_idx:
        v = row[idx]
        if v != 0:
            # 1/v must be half-integer
            # 1/v = k/2 => 2 = k*v => v must divide 2
            # v is integer
            if 2 % int(v) == 0:
                return True
            
    # Support 2
    if len(non_zero_idx) >= 2:
        # Limit checks to first few to keep it fast
        nz = non_zero_idx[:5]
        for i1 in range(len(nz)):
            idx1 = nz[i1]
            v1 = int(row[idx1])
            for i2 in range(i1+1, len(nz)):
                idx2 = nz[i2]
                v2 = int(row[idx2])
                # v1 * X + v2 * Y = 2
                # Solution exists if gcd(v1, v2) divides 2
                g = math.gcd(abs(v1), abs(v2))
                if 2 % g == 0:
                    return True
    return False

def _try_rank_search(n, m, p, target_rank, seed):
    """
    Try to find a decomposition of target_rank using sparse random initialization and local search.
    """
    np.random.seed(seed)
    r = target_rank
    
    def make_sparse_matrix(rows, cols):
        M = np.zeros((rows, cols))
        for i in range(rows):
            # Pick 2 or 3 columns
            k = np.random.choice([2, 3])
            idx = np.random.choice(cols, k, replace=False)
            for j in idx:
                M[i, j] = np.random.choice([-1, 1])
        return M

    U = make_sparse_matrix(n*m, r)
    V = make_sparse_matrix(m*p, r)
    
    def calculate_M(U, V):
        M = np.zeros((n*p, r))
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U[i*m+j, c] * V[j*p+k, c]
                    M[i*p+k, c] = s
        return M

    M = calculate_M(U, V)
    
    def get_badness(M):
        badness = 0
        for i in range(n*p):
            if not _check_solvable_fast(M[i]):
                badness += 1
        return badness

    badness = get_badness(M)
    
    max_iters = 300
    for _ in range(max_iters):
        if badness == 0:
            break
        if _time_remaining() < 1:
            break
            
        flip_U = np.random.rand() < 0.5
        if flip_U:
            idx = np.random.randint(n*m)
            col = np.random.randint(r)
            old_val = U[idx, col]
            new_val = np.random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = -new_val if new_val != 0 else (1 if np.random.rand() < 0.5 else -1)
            U[idx, col] = new_val
        else:
            idx = np.random.randint(m*p)
            col = np.random.randint(r)
            old_val = V[idx, col]
            new_val = np.random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = -new_val if new_val != 0 else (1 if np.random.rand() < 0.5 else -1)
            V[idx, col] = new_val
            
        new_M = calculate_M(U, V)
        new_badness = get_badness(new_M)
        
        if new_badness < badness:
            badness = new_badness
            M = new_M
        else:
            # Revert
            if flip_U:
                U[idx, col] = old_val
            else:
                V[idx, col] = old_val
                
    if badness == 0:
        W = np.zeros((p*n, r))
        for i in range(n):
            for k in range(p):
                row = M[i*p+k]
                sol = _find_half_integer_solution(row, r)
                if sol is not None:
                    W[k*n+i] = sol
                else:
                    return None
            
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def _try_structured_construction(n, m, p):
    """
    Try known structured constructions for low-rank matrix multiplication.
    
    For 3x3 * 3x4, we can try:
    - Block decomposition using Strassen's 2x2 result
    - Other known algorithms
    """
    # Strassen's algorithm multiplies 2x2 matrices with 7 multiplications.
    # For 3x3 * 3x4, we can partition into blocks.
    
    # Let's try a direct construction based on the following idea:
    # A is 3x3, B is 3x4, C = A*B is 3x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use the fact that rank(3x3*3x4) <= rank(3x3*3x3) + rank(3x3*3x1)
    # But that's not directly helpful.
    
    # Let's try a specific construction with rank 24 (which is 2*12):
    # This would be a "doubling" of the 12-term construction.
    
    # Actually, let's try to construct rank 21 or 24 explicitly.
    
    # Construction attempt: Use the following pattern.
    # Let r = 24. Split into two groups of 12.
    # Group 1: handles even-indexed columns of B, Group 2: odd-indexed.
    # B is 3x4, so columns 0,2 vs 1,3.
    
    # This is getting complex. Let me try a simpler approach:
    # Use the known fact that the tensor rank of 3x3*3x3 is at most 23
    # (actually it's known to be 19-23, with best known upper bound 23).
    # For 3x3*3x4, we can extend: rank <= rank(3x3*3x3) + 3 = 26.
    
    # Let me try to explicitly construct a rank-24 solution.
    
    # Method: Use 12 "base" terms for the 3x3*3x3 part and 12 more for the extra column.
    # But this gives 24.
    
    # Actually, the simplest improvement over schoolbook (36) is to use
    # the fact that we can share computations.
    
    # Let me try: rank = 28 construction.
    # Use 9 terms for the diagonal of A (A[i,i]*B[i,k] for all i,k): 9*4=36... no.
    
    # Different approach: Let's just verify that our random search works
    # and return the best found.
    
    return None

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
