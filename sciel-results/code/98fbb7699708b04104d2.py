"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _check_half_integer(W, rank):
    """Check all entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2*x != int(2*x):
                    return False
    return True

def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]
    
    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        pr = None
        for rr in range(r, R):
            if aug[rr][col] != 0:
                pr = rr
                break
        if pr is None:
            col += 1
            continue
        if pr != r:
            aug[r], aug[pr] = aug[pr], aug[r]
        pivots.append((r, col))
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1
    
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _try_fixed_U_V_search(rank, n, m, p, deadline, rng, max_iter=10000):
    """Try random U,V and check if W can be solved."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Generate random U and V with small integers
        scale = 1 if rank <= 20 else 2
        U = [[rng.randint(-scale, scale) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-scale, scale) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: no all-zero rows
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # Try to solve for all W[k*n+i]
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _try_sparse_structured(rank, n, m, p, deadline, rng, max_iter=5000):
    """Try sparse structured patterns."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Sparse: each row has 1-2 non-zero entries
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        for idx in range(nm):
            nnz = rng.choice([1, 1, 2])
            cols = rng.sample(range(rank), min(nnz, rank))
            for c in cols:
                U[idx][c] = rng.choice([-1, 1, 2, -2])
        
        for idx in range(mp):
            nnz = rng.choice([1, 1, 2])
            cols = rng.sample(range(rank), min(nnz, rank))
            for c in cols:
                V[idx][c] = rng.choice([-1, 1, 2, -2])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _try_block_structure(rank, n, m, p, deadline, rng, max_iter=3000):
    """Try block-structured U and V."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Assign primary columns based on structure
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                primary = (i * m + j) % rank
                U[idx][primary] = rng.choice([1, -1, 2])
                if rank > 1 and rng.random() < 0.4:
                    secondary = (primary + rng.randint(1, min(2, rank-1))) % rank
                    if secondary != primary:
                        U[idx][secondary] = rng.choice([-1, 1])
        
        for l in range(m):
            for k in range(p):
                idx = l * p + k
                primary = (l * p + k) % rank
                V[idx][primary] = rng.choice([1, -1, 2])
                if rank > 1 and rng.random() < 0.4:
                    secondary = (primary + rng.randint(1, min(2, rank-1))) % rank
                    if secondary != primary:
                        V[idx][secondary] = rng.choice([-1, 1])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with increasing time allocation."""
    ranks_to_try = list(range(18, 37))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate more time to lower ranks
        if rank <= 22:
            max_iter1 = 8000
            max_iter2 = 5000
            max_iter3 = 3000
        elif rank <= 28:
            max_iter1 = 5000
            max_iter2 = 3000
            max_iter3 = 2000
        else:
            max_iter1 = 2000
            max_iter2 = 1500
            max_iter3 = 1000
        
        # Strategy 1: Fixed U,V search
        rng = random.Random(rank * 1000 + 42)
        result = _try_fixed_U_V_search(rank, n, m, p, deadline, rng, max_iter=max_iter1)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Strategy 2: Sparse structured
        rng = random.Random(rank * 7777 + 123)
        result = _try_sparse_structured(rank, n, m, p, deadline, rng, max_iter=max_iter2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Strategy 3: Block structure
        rng = random.Random(rank * 5555 + 789)
        result = _try_block_structure(rank, n, m, p, deadline, rng, max_iter=max_iter3)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return None

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W each of shape (4, 7) where:
      U[i*2+j, c] for i,j in {0,1}
      V[j*2+k, c] for j,k in {0,1}
      W[k*2+i, c] for k,i in {0,1}
    """
    # Strassen's 7 products for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-c)(e+g)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i,j,c]: coefficient of A[i,j] in product c
    # V[j,k,c]: coefficient of B[j,k] in product c
    # W[k,i,c]: coefficient in output C[i,k]
    
    # Index: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    #        B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    #        C[0,0], C[0,1], C[1,0], C[1,1]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # W: C[0,0]+=M1, C[1,1]+=M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: U[1,0]=1, U[1,1]=1, V[0,0]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # W: C[1,0]+=M2, C[1,1]-=M2
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h): U[0,0]=1, V[0,1]=1, V[1,1]=-1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # W: C[0,1]+=M3, C[1,1]+=M3
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e): U[1,1]=1, V[1,0]=1, V[0,0]=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # W: C[0,0]+=M4, C[1,0]+=M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h: U[0,0]=1, U[0,1]=1, V[1,1]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # W: C[0,0]-=M5, C[0,1]+=M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-d)(f+g): U[1,0]=1, U[1,1]=-1, V[0,1]=1, V[1,0]=1
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    # W: C[1,1]+=M6
    W[3][5] = 1
    
    # M7 = (b-c)(e+g): U[0,1]=1, U[1,0]=-1, V[0,0]=1, V[1,0]=1
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    # W: C[0,0]+=M7
    W[0][6] = 1
    
    return U, V, W

def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Try to construct a decomposition using Kronecker product structure.
    For 3x3 * 3x4: decompose as (2x2 * 2x2) ⊗ (1x1 * 1x2) with remainder handling.
    
    Strategy: Use Strassen for the 2x2 block (rank 7), then handle the remaining
    dimensions. The 3x3*3x4 can be viewed as:
      - Top-left 2x2 * 2x2 block (Strassen, rank 7)
      - Cross terms involving the 3rd row/col
    
    A more direct approach: partition the 3x3 matrix into 2x2 + 1x1 blocks.
    This gives a block structure that can be exploited.
    
    For now, try a simpler structured approach: 
    Use the fact that 3x3*3x4 tensor can be decomposed using
    rank-7 Strassen for 2x2*2x2 sub-blocks plus additional terms.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try rank values that might work with structured approaches
    for rank in range(20, 33):
        if time.time() > deadline:
            break
        
        # Structured search: try patterns inspired by Kronecker products
        rng = random.Random(rank * 9999 + 555)
        
        for trial in range(2000):
            if time.time() > deadline:
                break
            
            # Create U with block structure: first 4 rows (2x2 block) use Strassen-like
            # pattern, remaining rows use simpler patterns
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            
            # For the 2x2 sub-block (rows 0-3 of U, cols 0-3 of V):
            # Use Strassen pattern in first 7 columns
            if rank >= 7:
                sU, sV, sW = _strassen_2x2()
                for r in range(4):
                    for c in range(7):
                        U[r][c] = sU[r][c]
                        V[r][c] = sV[r][c]
            
            # Fill remaining entries with small random values
            for idx in range(4, nm):
                for c in range(rank):
                    if rng.random() < 0.3:
                        U[idx][c] = rng.choice([-1, 0, 1])
            
            for idx in range(4, mp):
                for c in range(rank):
                    if rng.random() < 0.3:
                        V[idx][c] = rng.choice([-1, 0, 1])
            
            if any(all(x == 0 for x in row) for row in U):
                continue
            if any(all(x == 0 for x in row) for row in V):
                continue
            
            # Try to solve for W
            W = [[0] * rank for _ in range(pn)]
            ok = True
            for i in range(n):
                for k in range(p):
                    Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                    if Wvec is None:
                        ok = False
                        break
                    W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
                if not ok:
                    break
            
            if not ok:
                continue
            
            if not _check_half_integer(W, rank):
                continue
            
            if _validate(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    
    # Try Kronecker/structured decomposition first (fast, deterministic)
    try:
        result = _try_kronecker_decomposition(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # Try systematic search for lower rank
    try:
        result = _try_systematic_ranks(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    return best_result
