"""Matrix multiplication 3x5 by 5x6 — rank-minimal exact tensor decomposition.

Strategy:
  Split the inner dimension (5) into 2+2+1 and p=6 into 2+2+2.
  Use Strassen rank-7 for all 2x2x2 blocks.
  For 1-wide slices (m=1 or p=1 or n=1), use schoolbook (which is optimal for those shapes).
  
  Blocks:
    n=3 -> 2+1
    m=5 -> 2+2+1
    p=6 -> 2+2+2
    
  2x2x2 blocks: 7 each. Count: (2 in n) * (2 in m) * (3 in p) = 12 blocks * 7 = 84
  2x1x2 blocks (n=2, m=1, p=2): schoolbook = 2*1*2 = 4. Count: (1 in n) * (1 in m) * (3 in p) = 3 * 4 = 12
  1x2x2 blocks (n=1, m=2, p=2): schoolbook = 1*2*2 = 4. Count: (1 in n) * (2 in m) * (3 in p) = 6 * 4 = 24
  1x1x2 blocks (n=1, m=1, p=2): schoolbook = 1*1*2 = 2. Count: (1 in n) * (1 in m) * (3 in p) = 3 * 2 = 6
  2x2x1 blocks (n=2, m=2, p=1): schoolbook = 2*2*1 = 4. Count: (2 in n) * (2 in m) * (0 in p)... wait p is all 2s.
  
  Let me recount:
  n splits: [2, 1]
  m splits: [2, 2, 1]
  p splits: [2, 2, 2]
  
  For each combination (ni, mj, pk):
    - If ni=2, mj=2, pk=2: Strassen rank-7 -> 7 terms
    - Otherwise: schoolbook rank = ni*mj*pk
    
  Combinations:
    ni=2:
      mj=2, pk=2: 7 (3 p-blocks) -> 21
      mj=2, pk=2: 7 (already counted)
      Actually: for ni=2:
        mj=2: pk=2,2,2 -> 3 blocks * 7 = 21
        mj=2: pk=2,2,2 -> 3 blocks * 7 = 21
        mj=1: pk=2,2,2 -> 3 blocks * (2*1*2=4) = 12
      Total for ni=2: 21+21+12 = 54
    ni=1:
      mj=2: pk=2,2,2 -> 3 blocks * (1*2*2=4) = 12
      mj=2: pk=2,2,2 -> 3 blocks * (1*2*2=4) = 12
      mj=1: pk=2,2,2 -> 3 blocks * (1*1*2=2) = 6
      Total for ni=1: 12+12+6 = 30
    
  Grand total: 54 + 30 = 84
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return 7 terms for 2x2x2 matrix multiplication (Strassen).
    
    Each term is (u, v, w) where u, v, w are length-4 vectors
    indexed as (row, col) -> row*2+col in row-major order.
    
    Convention:
      U[i*2+k, c] selects A[i,k]
      V[k*2+j, c] selects B[k,j]  
      W[j*2+i, c] accumulates into C[i,j]
      
    Strassen's 7 products for 2x2:
      A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
      
      m1 = (a+d)(e+h)
      m2 = (c+d)e
      m3 = a(f-h)
      m4 = d(g-e)
      m5 = (a+b)h
      m6 = (c-a)(e+f)
      m7 = (b-d)(g+h)
      
      p = m1 + m4 - m5 + m7
      q = m3 + m5
      r = m2 + m4
      s = m1 - m2 + m3 + m6
    """
    # Index: (row, col) -> row*2 + col
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    terms = []
    
    # m1 = (a+d)(e+h): U has a=1 at 0, d=1 at 3; V has e=1 at 0, h=1 at 3
    u = [1,0,0,1]; v = [1,0,0,1]; w = [1,0,0,1]  # p+=m1, s+=m1
    terms.append((u, v, w))
    
    # m2 = (c+d)e: U has c=1 at 2, d=1 at 3; V has e=1 at 0
    u = [0,0,1,1]; v = [1,0,0,0]; w = [0,0,1,-1]  # r+=m2, s-=m2
    terms.append((u, v, w))
    
    # m3 = a(f-h): U has a=1 at 0; V has f=1 at 1, h=-1 at 3
    u = [1,0,0,0]; v = [0,1,0,-1]; w = [0,1,0,1]  # q+=m3, s+=m3
    terms.append((u, v, w))
    
    # m4 = d(g-e): U has d=1 at 3; V has g=1 at 2, e=-1 at 0
    u = [0,0,0,1]; v = [-1,0,1,0]; w = [1,0,1,0]  # p+=m4, r+=m4
    terms.append((u, v, w))
    
    # m5 = (a+b)h: U has a=1 at 0, b=1 at 1; V has h=1 at 3
    u = [1,1,0,0]; v = [0,0,0,1]; w = [-1,1,0,0]  # p-=m5, q+=m5
    terms.append((u, v, w))
    
    # m6 = (c-a)(e+f): U has c=1 at 2, a=-1 at 0; V has e=1 at 0, f=1 at 1
    u = [-1,0,1,0]; v = [1,1,0,0]; w = [0,0,0,1]  # s+=m6
    terms.append((u, v, w))
    
    # m7 = (b-d)(g+h): U has b=1 at 1, d=-1 at 3; V has g=1 at 2, h=1 at 3
    u = [0,1,0,-1]; v = [0,0,1,1]; w = [1,0,0,0]  # p+=m7
    terms.append((u, v, w))
    
    return terms

def _verify_222(terms):
    """Verify 2x2x2 Strassen terms."""
    n = 2
    for i in range(n):
        for k in range(n):
            for j in range(n):
                for ip in range(n):
                    for jp in range(n):
                        expected = 1 if (i == ip and j == jp) else 0
                        actual = 0
                        for (u, v, w) in terms:
                            actual += u[i*2+k] * v[k*2+j] * w[jp*2+ip]
                        if actual != expected:
                            return False
    return True

def _schoolbook_block(ni, mj, pk):
    """Return schoolbook terms for ni x mj x pk block."""
    terms = []
    for i in range(ni):
        for k in range(mj):
            for j in range(pk):
                u = [0]*(ni*mj)
                v = [0]*(mj*pk)
                w = [0]*(pk*ni)
                u[i*mj+k] = 1
                v[k*pk+j] = 1
                w[j*ni+i] = 1
                terms.append((u, v, w))
    return terms

def _embed_block(U, V, W, terms, i_off, k_off, j_off, ni, mj, pk, col_start):
    """Embed block terms into global matrices.
    
    i_off: row offset in A/C
    k_off: col offset in A / row offset in B
    j_off: col offset in B/C
    """
    col = col_start
    for (u, v, w) in terms:
        for idx in range(len(u)):
            if u[idx] != 0:
                ii = idx // mj
                kk = idx % mj
                U[(i_off+ii)*5 + (k_off+kk)][col] += u[idx]
        for idx in range(len(v)):
            if v[idx] != 0:
                kk = idx // pk
                jj = idx % pk
                V[(k_off+kk)*6 + (j_off+jj)][col] += v[idx]
        for idx in range(len(w)):
            if w[idx] != 0:
                jj = idx // ni
                ii = idx % ni
                W[(j_off+jj)*3 + (i_off+ii)][col] += w[idx]
        col += 1
    return col

def construct():
    """
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose
    the 3x5 @ 5x6 matrix multiplication tensor.
    
    U: (n*m, rank) = (15, rank)
    V: (m*p, rank) = (30, rank)
    W: (p*n, rank) = (18, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    
    # Verify Strassen
    strassen_terms = _strassen_222()
    if not _verify_222(strassen_terms):
        # Fallback: schoolbook 8 for 2x2x2
        strassen_terms = _schoolbook_block(2, 2, 2)
    
    # Splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Compute total rank
    total_rank = 0
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    i_off = 0
    for ni in n_splits:
        k_off = 0
        for mj in m_splits:
            j_off = 0
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    terms = strassen_terms
                else:
                    terms = _schoolbook_block(ni, mj, pk)
                
                col = _embed_block(U, V, W, terms, i_off, k_off, j_off, ni, mj, pk, col)
                
                j_off += pk
            
            k_off += mj
        
        i_off += ni
    
    # Verify full decomposition
    valid = True
    for i in range(n):
        for k in range(m):
            for j in range(p):
                for ip in range(n):
                    for jp in range(p):
                        expected = 1 if (i == ip and j == jp) else 0
                        actual = 0
                        for c in range(total_rank):
                            actual += U[i*m+k][c] * V[k*p+j][c] * W[jp*n+ip][c]
                        if actual != expected:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to full schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m+k][col] = 1
                    V[k*p+j][col] = 1
                    W[j*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
