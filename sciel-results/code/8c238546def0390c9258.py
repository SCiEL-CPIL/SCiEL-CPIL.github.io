import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 6
    
    def build_decomposition(factors):
        """factors is a list of (u_dict, v_dict, w_dict)"""
        rank = len(factors)
        U = [[Fraction(0)] * rank for _ in range(n*m)]
        V = [[Fraction(0)] * rank for _ in range(m*p)]
        W = [[Fraction(0)] * rank for _ in range(p*n)]
        for r, (ud, vd, wd) in enumerate(factors):
            for idx, val in ud.items():
                U[idx][r] += val
            for idx, val in vd.items():
                V[idx][r] += val
            for idx, val in wd.items():
                W[idx][r] += val
        return U, V, W
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if s != 1:
                        return False
        return True
    
    def schoolbook():
        factors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    factors.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
        return factors
    
    def strassen_block(i_off, j_off, k_off):
        """Return 7 factors for 2x2x2 block at offsets"""
        def ui(ii, jj): return (i_off+ii)*5 + (j_off+jj)
        def vi(jj, kk): return (j_off+jj)*6 + (k_off+kk)
        def wi(kk, ii): return (k_off+kk)*2 + (i_off+ii)
        
        return [
            ({ui(0,0):1, ui(1,1):1}, {vi(0,0):1, vi(1,1):1}, {wi(0,0):1, wi(1,1):1}),
            ({ui(1,0):1, ui(1,1):1}, {vi(0,0):1}, {wi(0,1):1}),
            ({ui(0,0):1}, {vi(1,0):1, vi(1,1):-1}, {wi(1,0):1}),
            ({ui(1,1):1}, {vi(0,1):1, vi(1,0):-1}, {wi(0,1):1}),
            ({ui(0,0):1, ui(0,1):1}, {vi(1,1):1}, {wi(0,0):-1}),
            ({ui(1,0):1, ui(0,0):-1}, {vi(0,0):1, vi(1,0):1}, {wi(1,1):1}),
            ({ui(0,1):1, ui(1,1):-1}, {vi(0,1):1, vi(1,1):1}, {wi(0,0):1}),
        ]
    
    # Try rank-48 decomposition
    factors = []
    # 6 Strassen blocks for 2x2x2 sub-blocks
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            factors.extend(strassen_block(0, j_off, k_off))
    # Schoolbook for j=4, k in 0..5
    for i in range(2):
        for k in range(6):
            factors.append(({i*5+4: 1}, {4*6+k: 1}, {k*2+i: 1}))
    
    U, V, W = build_decomposition(factors)
    
    if verify(U, V, W):
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    # Fallback to schoolbook
    factors = schoolbook()
    U, V, W = build_decomposition(factors)
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
