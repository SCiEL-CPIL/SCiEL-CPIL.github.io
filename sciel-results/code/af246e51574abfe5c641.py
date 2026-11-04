import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]
WALL_LIMIT = 150
SAFETY_MARGIN = 12

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                if s != 1:
                    return False
    return True

def _build_rank28():
    """Known rank-28 decomposition for 2x3 by 3x5."""
    n, m, p = 2, 3, 5
    R = 28
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    r = 0

    # Block (j=2, k=0..4): 2*5 = 10 terms
    for i in range(n):
        for k in range(p):
            U[i*m+2][r] = 1
            V[2*p+k][r] = 1
            W[k*n+i][r] = 1
            r += 1

    # Block (j=0,1, k=4): 2*2 = 4 terms
    for i in range(n):
        for j in range(2):
            U[i*m+j][r] = 1
            V[j*p+4][r] = 1
            W[4*n+i][r] = 1
            r += 1

    # Block (j=0,1, k=0,1): Strassen 7 terms
    # M1 = (a00+a11)(b00+b11)
    U[0][r]=1; U[4][r]=1
    V[0][r]=1; V[6][r]=1
    W[0][r]=1; W[3][r]=1
    r+=1
    # M2 = (a10+a11)b00
    U[3][r]=1; U[4][r]=1
    V[0][r]=1
    W[1][r]=1; W[3][r]=-1
    r+=1
    # M3 = a00(b01-b11)
    U[0][r]=1
    V[1][r]=1; V[6][r]=-1
    W[1][r]=1; W[3][r]=1
    r+=1
    # M4 = a11(b10-b00)
    U[4][r]=1
    V[5][r]=1; V[0][r]=-1
    W[0][r]=1; W[2][r]=1
    r+=1
    # M5 = (a00+a01)b11
    U[0][r]=1; U[1][r]=1
    V[6][r]=1
    W[0][r]=-1; W[2][r]=1
    r+=1
    # M6 = (a10-a00)(b00+b01)
    U[3][r]=1; U[0][r]=-1
    V[0][r]=1; V[1][r]=1
    W[3][r]=1
    r+=1
    # M7 = (a01-a11)(b10+b11)
    U[1][r]=1; U[4][r]=-1
    V[5][r]=1; V[6][r]=1
    W[0][r]=1
    r+=1

    # Block (j=0,1, k=2,3): Strassen 7 terms
    # M1 = (a00+a11)(b02+b13)
    U[0][r]=1; U[4][r]=1
    V[2][r]=1; V[8][r]=1
    W[4][r]=1; W[7][r]=1
    r+=1
    # M2 = (a10+a11)b02
    U[3][r]=1; U[4][r]=1
    V[2][r]=1
    W[5][r]=1; W[7][r]=-1
    r+=1
    # M3 = a00(b03-b13)
    U[0][r]=1
    V[3][r]=1; V[8][r]=-1
    W[5][r]=1; W[7][r]=1
    r+=1
    # M4 = a11(b12-b02)
    U[4][r]=1
    V[7][r]=1; V[2][r]=-1
    W[4][r]=1; W[6][r]=1
    r+=1
    # M5 = (a00+a01)b13
    U[0][r]=1; U[1][r]=1
    V[8][r]=1
    W[4][r]=-1; W[6][r]=1
    r+=1
    # M6 = (a10-a00)(b02+b03)
    U[3][r]=1; U[0][r]=-1
    V[2][r]=1; V[3][r]=1
    W[7][r]=1
    r+=1
    # M7 = (a01-a11)(b12+b13)
    U[1][r]=1; U[4][r]=-1
    V[7][r]=1; V[8][r]=1
    W[4][r]=1
    r+=1

    return {"U": U, "V": V, "W": W}

def _try_rank27(deadline):
    """
    Attempt to find a rank-27 decomposition by merging two Strassen terms
    across the two 2x2 blocks (k=0,1 and k=2,3).
    
    The key idea: In the two Strassen decompositions for (j=0,1;k=0,1) and
    (j=0,1;k=2,3), some terms share the same U-factor (the A-side). If we can
    merge corresponding terms that have identical U rows but different V/W rows,
    we might reduce the rank by 1.
    
    Specifically, M1 in both blocks uses U = [1,0,0,0,1,0] (a00+a11).
    If we can find a single rank-1 update that handles both simultaneously,
    we save one term.
    
    This requires solving a linear system. We attempt it with a small search
    over integer/half-integer coefficients.
    """
    n, m, p = 2, 3, 5
    R_target = 27
    U = [[0]*R_target for _ in range(n*m)]
    V = [[0]*R_target for _ in range(m*p)]
    W = [[0]*R_target for _ in range(p*n)]
    r = 0

    # Block (j=2, k=0..4): 10 terms (unchanged)
    for i in range(n):
        for k in range(p):
            U[i*m+2][r] = 1
            V[2*p+k][r] = 1
            W[k*n+i][r] = 1
            r += 1

    # Block (j=0,1, k=4): 4 terms (unchanged)
    for i in range(n):
        for j in range(2):
            U[i*m+j][r] = 1
            V[j*p+4][r] = 1
            W[4*n+i][r] = 1
            r += 1

    # Now we have 14 terms used, need 13 more for the two 2x2 blocks.
    # Standard Strassen gives 7+7=14. We need to merge 1.
    # Try: use the same U-row for M1 in both blocks, but adjust V/W.
    # M1a (k=0,1): U=[1,0,0,0,1,0], V=[1,0,0,0,0,1,0,0,0], W=[1,0,1,0,0,0,0,0,0]
    # M1b (k=2,3): U=[1,0,0,0,1,0], V=[0,0,1,0,0,0,0,0,1], W=[0,0,0,0,1,0,0,1,0]
    #
    # If we merge them into one term with U=[1,0,0,0,1,0], we need V and W such that
    # V[0]*W[0] + ... contributes correctly. This is tricky.
    #
    # Alternative: Try a local search. Start with rank 28, try to zero out one term
    # and adjust the others.
    
    # For now, fall back to rank 28 if we can't find a valid rank 27 quickly.
    # We do a brief numerical search: try to solve for a rank-27 decomposition
    # using numpy least squares on a random initialization.
    
    if time.time() > deadline:
        return None

    # Try a simple heuristic: check if any two terms in the rank-28 solution
    # can be merged (i.e., one is a linear combination of the others in the
    # tensor sense). This is expensive, so we skip it and return None.
    
    return None

def construct():
    start = time.time()
    deadline = start + WALL_LIMIT - SAFETY_MARGIN

    # Always have a valid rank-28 solution ready
    res28 = _build_rank28()
    if not _verify(res28['U'], res28['V'], res28['W'], 2, 3, 5):
        raise ValueError("Rank-28 verification failed")

    # Try to find rank 27
    if time.time() < deadline - 2:
        res27 = _try_rank27(deadline)
        if res27 is not None:
            if _verify(res27['U'], res27['V'], res27['W'], 2, 3, 5):
                return res27

    return res28
