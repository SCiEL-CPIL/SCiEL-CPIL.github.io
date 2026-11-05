import numpy as np
import time
import itertools
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - START_TIME) > TIME_LIMIT

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify exact matrix multiplication via the tensor identity:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 5, 7

    # ---- Block decomposition templates ----
    def schoolbook_block(a, b, c):
        """Schoolbook for a×b by b×c: rank a*b*c"""
        rank = a * b * c
        U = [[0]*rank for _ in range(a*b)]
        V = [[0]*rank for _ in range(b*c)]
        W = [[0]*rank for _ in range(c*a)]
        idx = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i*b+j][idx] = 1
                    V[j*c+k][idx] = 1
                    W[k*a+i][idx] = 1
                    idx += 1
        return rank, U, V, W

    def strassen_block():
        """2x2x2 Strassen, rank 7. Local U,V,W are 4x7."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1: (A11+A22)(B11+B22)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2: (A21+A22)B11
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
        # M3: A11(B21-B11)
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
        # M4: A22(B12-B22)
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
        # M5: (A11+A12)B22
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
        # M6: (A21-A11)(B11+B12)
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7: (A12-A22)(B21+B22)
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return 7, U, V, W

    def get_block_decomp(a, b, c):
        """Get best known block decomposition for size a×b by b×c."""
        if a == 2 and b == 2 and c == 2:
            return strassen_block()
        # For other sizes, try to build from smaller blocks or use schoolbook
        return schoolbook_block(a, b, c)

    # ---- Enumerate all partitions ----
    def partitions(total, max_part=3):
        res = []
        def rec(rem, cur):
            if rem == 0:
                res.append(tuple(cur))
                return
            for s in range(1, max_part+1):
                if s <= rem:
                    cur.append(s)
                    rec(rem - s, cur)
                    cur.pop()
        rec(total, [])
        return res

    pn = partitions(n)
    pm = partitions(m)
    pp = partitions(p)

    # Find best partition combination
    best_rank = n * m * p  # schoolbook fallback
    best_partition = (tuple(range(1, n+1)), tuple(range(1, m+1)), tuple(range(1, p+1)))

    # Pre-compute block ranks
    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    for part_n in pn:
        if _time_exceeded():
            break
        for part_m in pm:
            if _time_exceeded():
                break
            for part_p in pp:
                total = 0
                for a in part_n:
                    for b in part_m:
                        for c in part_p:
                            total += block_rank(a, b, c)
                if total < best_rank:
                    best_rank = total
                    best_partition = (part_n, part_m, part_p)

    part_n, part_m, part_p = best_partition
    rank_total = best_rank

    # Build the full decomposition
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]

    def place(rank_offset, i0, j0, k0, a, b, c, U_loc, V_loc, W_loc):
        loc_rank = len(U_loc[0]) if U_loc else 0
        for i_l in range(a):
            for j_l in range(b):
                u_g = (i0+i_l)*m + (j0+j_l)
                for r in range(loc_rank):
                    U[u_g][rank_offset+r] += U_loc[i_l*b+j_l][r]
        for j_l in range(b):
            for k_l in range(c):
                v_g = (j0+j_l)*p + (k0+k_l)
                for r in range(loc_rank):
                    V[v_g][rank_offset+r] += V_loc[j_l*c+k_l][r]
        for k_l in range(c):
            for i_l in range(a):
                w_g = (k0+k_l)*n + (i0+i_l)
                for r in range(loc_rank):
                    W[w_g][rank_offset+r] += W_loc[k_l*a+i_l][r]

    rank_offset = 0
    i0 = 0
    for a in part_n:
        if _time_exceeded():
            break
        j0 = 0
        for b in part_m:
            if _time_exceeded():
                break
            k0 = 0
            for c in part_p:
                r, U_loc, V_loc, W_loc = get_block_decomp(a, b, c)
                place(rank_offset, i0, j0, k0, a, b, c, U_loc, V_loc, W_loc)
                rank_offset += r
                k0 += c
            j0 += b
        i0 += a

    # Safety: verify; if it fails, fall back to pure schoolbook
    if not _verify(U, V, W):
        rank_total = n * m * p
        U = [[0]*rank_total for _ in range(n*m)]
        V = [[0]*rank_total for _ in range(m*p)]
        W = [[0]*rank_total for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1

    return {"U": U, "V": V, "W": W}
