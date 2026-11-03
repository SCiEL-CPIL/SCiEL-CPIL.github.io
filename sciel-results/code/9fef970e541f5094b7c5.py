import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Local Strassen factors for 2x2 * 2x2 -> 2x2
    # Indices: 0=a00, 1=a01, 2=a10, 3=a11 (and similarly for b, c)
    # 7 terms, each with (u_indices, v_indices, w_indices) and coefficients
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11) -> c00+1, c11+1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (a10+a11)b00 -> c10+1, c11-1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3 = a00(b01-b11) -> c01+1, c11+1
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # M4 = a11(b10-b00) -> c00+1, c10+1
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5 = (a00+a01)b11 -> c00-1, c01+1
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6 = (a10-a00)(b00+b01) -> c11+1
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (a01-a11)(b10+b11) -> c00+1
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    total_rank = 56  # 2 halves * 4 column blocks * 7 Strassen terms
    
    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]   # 16 x 56
    
    col = 0
    for half in range(2):  # 0: A[:,0:2] & B[0:2,:], 1: A[:,2:4] & B[2:4,:]
        for jj in range(p // 2):  # 4 column blocks of size 2
            for (u_terms, v_terms, w_terms) in strassen_terms:
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                # U: A is 2x4, row-major. A[i, l] at index i*m + l
                # For half h, columns are 2h and 2h+1
                # local 0=a00=(0,2h), 1=a01=(0,2h+1), 2=a10=(1,2h), 3=a11=(1,2h+1)
                for (li, coeff) in u_terms:
                    i = li // 2
                    l_local = li % 2
                    l_global = 2 * half + l_local
                    gu = i * m + l_global
                    U[gu][col] += coeff
                
                # V: B is 4x8, row-major. B[l, k] at index l*p + k
                # For half h, rows are 2h and 2h+1. For col block jj, cols are 2jj and 2jj+1
                # local 0=b00=(2h,2jj), 1=b01=(2h,2jj+1), 2=b10=(2h+1,2jj), 3=b11=(2h+1,2jj+1)
                for (li, coeff) in v_terms:
                    l_local_row = li // 2
                    k_local = li % 2
                    l_global = 2 * half + l_local_row
                    k_global = 2 * jj + k_local
                    gv = l_global * p + k_global
                    V[gv][col] += coeff
                
                # W: C is 2x8, output convention W[k*n + i, r]
                # For col block jj, cols are 2jj and 2jj+1
                # local 0=c00=(0,2jj), 1=c01=(0,2jj+1), 2=c10=(1,2jj), 3=c11=(1,2jj+1)
                for (li, coeff) in w_terms:
                    i = li // 2
                    k_local = li % 2
                    k_global = 2 * jj + k_local
                    gw = k_global * n + i
                    W[gw][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
