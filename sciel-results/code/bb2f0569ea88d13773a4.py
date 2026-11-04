"""Matrix multiplication 4x4x6 with recursive Strassen decomposition (rank 77)."""
import time

DIMENSIONS = [4, 4, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

# Strassen coefficients for 2x2 x 2x2
# A blocks: 0=A00, 1=A01, 2=A10, 3=A11
# B blocks: 0=B00, 1=B01, 2=B10, 3=B11
_ALPHA = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: (A00+A11)
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: (A10+A11)
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: (A00+A01)
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: (A10-A00)
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: (A01-A11)
]
_BETA = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: (B00+B11)
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: (B01-B11)
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: (B10-B00)
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: (B00+B01)
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: (B10+B11)
]
# Reconstruct: for each Strassen term s, list of (out_block_row, out_block_col, coeff)
_RECON = [
    [(0, 0, 1), (1, 1, 1)],        # M1 -> C00+1, C11+1
    [(1, 0, 1), (1, 1, -1)],       # M2 -> C10+1, C11-1
    [(0, 1, 1), (1, 1, 1)],        # M3 -> C01+1, C11+1
    [(0, 0, 1), (1, 0, 1)],        # M4 -> C00+1, C10+1
    [(0, 0, -1), (0, 1, 1)],       # M5 -> C00-1, C01+1
    [(1, 1, 1)],                   # M6 -> C11+1
    [(0, 0, 1)],                   # M7 -> C00+1
]


def construct():
    """
    Rank-77 decomposition of 4x4x6 matrix multiplication tensor.
    
    Strategy: Recursive Strassen.
    - Split B into B1 (4x4) and B2 (4x2).
    - A*B1: 7 top-level Strassen terms, each a 2x2x2x2 product with inner Strassen (7) = 49
    - A*B2: 7 top-level Strassen terms, each a 2x2x2x1 product with schoolbook (4) = 28
    Total: 77
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    where n=4, m=4, p=6
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    R = 77
    U = [[0] * R for _ in range(n * m)]   # (16, 77)
    V = [[0] * R for _ in range(m * p)]   # (24, 77)
    W = [[0] * R for _ in range(p * n)]   # (24, 77)
    
    col = 0
    
    # Part 1: A * B1 where B1 is 4x4 (columns 0-3 of B)
    # Top-level Strassen on 2x2 blocks of A (2x2 each) and B1 (2x2 each)
    for s in range(7):
        for si in range(7):
            # U: A entries. Top-level block (ar, br) with inner (xr, xc)
            # Global A index: (ar*2+xr) * 4 + (br*2+xc)
            for ar in range(2):
                for br in range(2):
                    block_a = ar * 2 + br
                    if _ALPHA[s].get(block_a, 0) == 0:
                        continue
                    a_coeff_top = _ALPHA[s][block_a]
                    for xr in range(2):
                        for xc in range(2):
                            inner_a = xr * 2 + xc
                            if _ALPHA[si].get(inner_a, 0) == 0:
                                continue
                            a_coeff_inner = _ALPHA[si][inner_a]
                            u_idx = (ar * 2 + xr) * 4 + (br * 2 + xc)
                            U[u_idx][col] += a_coeff_top * a_coeff_inner
            
            # V: B1 entries. Top-level block (ac, bc) with inner (yr, yc)
            # B1 column offset: 0 (columns 0-3 of B)
            # Global B index: (ac*2+yr) * 6 + (bc*3+yc)  [bc in {0,1}, yc in {0,1}]
            for ac in range(2):
                for bc in range(2):
                    block_b = ac * 2 + bc
                    if _BETA[s].get(block_b, 0) == 0:
                        continue
                    b_coeff_top = _BETA[s][block_b]
                    for yr in range(2):
                        for yc in range(2):
                            inner_b = yr * 2 + yc
                            if _BETA[si].get(inner_b, 0) == 0:
                                continue
                            b_coeff_inner = _BETA[si][inner_b]
                            v_idx = (ac * 2 + yr) * 6 + (bc * 3 + yc)
                            V[v_idx][col] += b_coeff_top * b_coeff_inner
            
            # W: C entries. Top-level output block (a_out, b_out) with inner (za, zb)
            # Global C index: (b_out*3 + zb) * 4 + (a_out*2 + za)
            for a_out, b_out, gamma_top in _RECON[s]:
                for za, zb, gamma_inner in _RECON[si]:
                    w_idx = (b_out * 3 + zb) * 4 + (a_out * 2 + za)
                    W[w_idx][col] += gamma_top * gamma_inner
            
            col += 1
            if col >= R:
                break
        if col >= R:
            break
    
    # Part 2: A * B2 where B2 is 4x2 (columns 4-5 of B)
    # Top-level Strassen on 2x2 blocks of A (2x2 each) and B2 (2x1 each)
    for s in range(7):
        for u_local in range(2):
            for v_local in range(2):
                # U: A entries. Top-level block (ar, br)
                for ar in range(2):
                    for br in range(2):
                        block_a = ar * 2 + br
                        if _ALPHA[s].get(block_a, 0) == 0:
                            continue
                        a_coeff = _ALPHA[s][block_a]
                        for xr in range(2):
                            for xc in range(2):
                                u_idx = (ar * 2 + xr) * 4 + (br * 2 + xc)
                                U[u_idx][col] += a_coeff
                
                # V: B2 entries. Top-level block (ac, bc) where bc in {0,1}
                # B2 column offset: 3 (columns 3-4 of B, i.e., global cols 4-5)
                # Global B index: (ac*2+yr) * 6 + (bc*3 + 3)  [v_local is the inner col]
                for ac in range(2):
                    for bc in range(2):
                        block_b = ac * 2 + bc
                        if _BETA[s].get(block_b, 0) == 0:
                            continue
                        b_coeff = _BETA[s][block_b]
                        for yr in range(2):
                            v_idx = (ac * 2 + yr) * 6 + (bc * 3 + 3)
                            V[v_idx][col] += b_coeff
                
                # W: C entries. Top-level output block (a_out, b_out)
                # Global C index: (b_out*3 + 3) * 4 + (a_out*2 + u_local)
                for a_out, b_out, gamma in _RECON[s]:
                    w_idx = (b_out * 3 + 3) * 4 + (a_out * 2 + u_local)
                    W[w_idx][col] += gamma
                
                col += 1
                if col >= R:
                    break
            if col >= R:
                break
        if col >= R:
            break
    
    return {"U": U, "V": V, "W": W}
