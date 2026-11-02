import time
import sys

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _get_winograd_2x2x2():
    """
    Returns 7 rank-1 terms for 2x2x2 matmul.
    U, V, W are lists of 7 vectors.
    U vectors have length 4 (indices i*2+j).
    V vectors have length 4 (indices j*2+k).
    W vectors have length 4 (indices k*2+i).
    """
    Uterms = []
    Vterms = []
    Wterms = []
    
    # Standard Winograd F^T (2,2,2) rank 7 terms
    # p1=(a+d)(e+h)
    Uterms.append([1, 0, 0, 1]) # a+d
    Vterms.append([1, 0, 0, 1]) # e+h
    Wterms.append([1, 0, 0, 1]) # C00, C11
    
    # p2=(c+d)e
    Uterms.append([0, 0, 1, 1]) # c+d
    Vterms.append([1, 0, 0, 0]) # e
    Wterms.append([0, 0, 1, -1]) # C10 - C11
    
    # p3=a(f-h)
    Uterms.append([1, 0, 0, 0]) # a
    Vterms.append([0, 1, 0, -1]) # f-h
    Wterms.append([0, 1, 0, 0]) # C01
    
    # p4=d(g-e)
    Uterms.append([0, 0, 0, 1]) # d
    Vterms.append([-1, 0, 1, 0]) # g-e
    Wterms.append([-1, 1, 1, 0]) # -C00 + C01 + C10
    
    # p5=(a+b)g
    Uterms.append([1, 1, 0, 0]) # a+b
    Vterms.append([0, 0, 1, 0]) # g
    Wterms.append([1, 1, 0, 0]) # C00 + C01
    
    # p6=(c-a)(e+f)
    Uterms.append([-1, 0, 1, 0]) # c-a
    Vterms.append([1, 1, 0, 0]) # e+f
    Wterms.append([0, 0, -1, 1]) # -C10 + C11
    
    # p7=(b-d)(h+f)
    Uterms.append([0, 1, 0, -1]) # b-d
    Vterms.append([0, 1, 0, 1]) # h+f
    Wterms.append([-1, 1, 1, 0]) # -C00 + C01 + C10
    
    return Uterms, Vterms, Wterms

def _schoolbook_2x1x2_block(n, m, p, i_base, j_base, k_base):
    """
    Generates schoolbook terms for a 2x1x2 block (2 i's, 1 j, 2 k's).
    Actually, we just need to add single terms for each (i,j,k).
    """
    terms_U = []
    terms_V = []
    terms_W = []
    for i in range(2):
        for k in range(2):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            gi = i_base + i
            gj = j_base
            gk = k_base + k
            u[gi * m + gj] = 1
            v[gj * p + gk] = 1
            w[gk * n + gi] = 1
            terms_U.append(u)
            terms_V.append(v)
            terms_W.append(w)
    return terms_U, terms_V, terms_W

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 12 blocks * 7 + 24 schoolbook = 84 + 24 = 108
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    U_win, V_win, W_win = _get_winograd_2x2x2()
    
    # j blocks: 0-1, 2-3
    for jb in range(2):
        j_base = jb * 2
        # i blocks: 0-1, 2-3
        for ib in range(2):
            i_base = ib * 2
            # k blocks: 0-1, 2-3, 4-5
            for kb in range(3):
                k_base = kb * 2
                
                for r in range(7):
                    u_local = U_win[r]
                    v_local = V_win[r]
                    w_local = W_win[r]
                    
                    # Map local to global for U
                    for li in range(2):
                        for lj in range(2):
                            gi = i_base + li
                            gj = j_base + lj
                            U[gi * m + gj][col] = u_local[li * 2 + lj]
                            
                    # Map local to global for V
                    for lj in range(2):
                        for lk in range(2):
                            gj = j_base + lj
                            gk = k_base + lk
                            V[gj * p + gk][col] = v_local[lj * 2 + lk]
                            
                    # Map local to global for W
                    for lk in range(2):
                        for li in range(2):
                            gk = k_base + lk
                            gi = i_base + li
                            W[gk * n + gi][col] = w_local[lk * 2 + li]
                            
                    col += 1
                    
    # j = 4 slice: 4x1x6
    # Use schoolbook for each (i, k)
    j_base = 4
    for i in range(n):
        for k in range(p):
            U[i * m + j_base][col] = 1
            V[j_base * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
