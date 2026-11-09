import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Falls back to the schoolbook decomposition (rank 60) if no lower-rank solution is found
    within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12
    schoolbook_rank = n * m * p  # 60

    def build_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        U: (nm x rank), V: (mp x rank), W: (pn x rank)
        For each (i, j, k), the sum over r of U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should be 1.
        """
        # Convert to Fraction for exact arithmetic
        U_f = [[Fraction(x) for x in row] for row in U]
        V_f = [[Fraction(x) for x in row] for row in V]
        W_f = [[Fraction(x) for x in row] for row in W]

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    ui = U_f[i * m + j]
                    vj = V_f[j * p + k]
                    wk = W_f[k * n + i]
                    for r in range(rank):
                        total += ui[r] * vj[r] * wk[r]
                    if total != 1:
                        return False
        return True

    def try_search(rank, time_budget):
        """
        Try to find a valid decomposition of the given rank using random search.
        """
        if time.time() > deadline:
            return None

        # For small ranks, we can try random integer entries in a small range
        # The search space is huge, so this is mostly a lottery ticket
        # We'll try a few different strategies
        
        # Strategy 1: Random integers in {-1, 0, 1}
        max_attempts = 5000
        attempts = 0
        while attempts < max_attempts and time.time() < deadline - 5:
            U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(nm)]
            V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(mp)]
            W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(pn)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
            attempts += 1
        
        # Strategy 2: Try to reuse schoolbook structure with some zeroing
        # This is unlikely to work but worth a try
        if time.time() < deadline - 5:
            U_base = [[0] * rank for _ in range(nm)]
            V_base = [[0] * rank for _ in range(mp)]
            W_base = [[0] * rank for _ in range(pn)]
            
            # Map schoolbook indices to potentially fewer columns
            # This is a heuristic: try to merge some columns
            col_map = {}
            col_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        key = (i, j, k)
                        if key not in col_map:
                            col_map[key] = col_idx
                            col_idx += 1
                        c = col_map[key]
                        if c < rank:
                            U_base[i * m + j][c] = 1
                            V_base[j * p + k][c] = 1
                            W_base[k * n + i][c] = 1
            
            if verify(U_base, V_base, W_base, rank):
                return {"U": U_base, "V": V_base, "W": W_base}

        return None

    # Try ranks from 60 down to some lower bound
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But for matrix multiplication, the rank is at least n*m*p / max(n,m,p) by Strassen-like bounds
    # For 2x5x6, the best known is likely still 60 or close to it
    # Let's try a few ranks: 60, 59, 58, ..., down to maybe 50
    
    # First, ensure we have a valid fallback
    fallback = build_schoolbook()
    
    # Try to find a lower rank
    best_result = fallback
    best_rank = schoolbook_rank
    
    # Try ranks from 59 down to 55 (a small range to keep runtime manageable)
    for rank in range(schoolbook_rank - 1, 54, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_search(rank, 10)
        if result is not None:
            best_result = result
            best_rank = rank
            # If we found one, we could stop, but let's see if we can go lower
            # For now, break to save time
            break
    
    return best_result
# EVOLVE-BLOCK-END
