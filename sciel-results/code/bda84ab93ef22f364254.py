"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Strassen 2x2x2 rank-7 terms: (U_dict, V_dict, W_dict)
# U: (i,j) for A 2x2, V: (j,k) for B 2x2, W: (k,i) for C 2x2
STRASSEN_222 = [
    ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
    ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
    ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
    ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
    ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
    ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
    ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
]

# Strassen 2x2x1 rank-4 terms: (U_dict, V_dict, W_dict)
# U: (i,j) for A 2x2, V: (j,k) for B 2x1, W: (k,i) for C 2x1
STRASSEN_221 = [
    ({(0,0):1,(1,1):1}, {(0,0):1,(1,0):1}, {(0,0):1,(1,0):1}),
    ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
    ({(0,0):1}, {(0,1):1}, {(0,1):1}),
    ({(1,1):1}, {(1,1):1}, {(0,1):1}),
]

# 4x4x4 Strassen: 7 products, each (A_block, B_block, [C_blocks])
# A blocks: 00=(0,0), 01=(0,2), 10=(2,0), 11=(2,2)
# B blocks: same
# C blocks: same
# M1=(A00+A11)(B00+B11)->C00,C11 ; M2=(A10+A11)B00->C10,-C11
# M3=A00(B01-B11)->C01,C11 ; M4=A11(B10-B00)->C00,C10
# M5=(A00+A01)B11->-C00,C01 ; M6=(A10-A00)(B00+B01)->C11
# M7=(A01-A11)(B10+B11)->C00
def _expand_444():
    terms = []
    def add_prod(a_off, b_off, w_offs):
        for u,v,w in STRASSEN_222:
            gu,gv,gw = {},{},{}
            for (li,lj),cu in u.items():
                key=(a_off[0]+li, a_off[1]+lj)
                gu[key]=gu.get(key,0)+cu
            for (lj,lk),cv in v.items():
                key=(b_off[0]+lj, b_off[1]+lk)
                gv[key]=gv.get(key,0)+cv
            for (lk,li),cw in w.items():
                for (wr,wc,cc) in w_offs:
                    key=(wc+lk, wr+li)  # W index (k,i)
                    gw[key]=gw.get(key,0)+cw*cc
            terms.append((gu,gv,gw))
    # M1
    add_prod((0,0),(0,0),[(0,0,1),(2,2,1)])
    add_prod((0,0),(2,2),[(0,0,1),(2,2,1)])
    add_prod((2,2),(0,0),[(0,0,1),(2,2,1)])
    add_prod((2,2),(2,2),[(0,0,1),(2,2,1)])
    # M2
    add_prod((2,0),(0,0),[(2,0,1),(2,2,-1)])
    add_prod((2,2),(0,0),[(2,0,1),(2,2,-1)])
    # M3
    add_prod((0,0),(0,2),[(0,2,1),(2,2,1)])
    add_prod((0,0),(2,2),[(0,2,-1),(2,2,-1)])
    # M4
    add_prod((2,2),(2,0),[(0,0,1),(2,0,1)])
    add_prod((2,2),(0,0),[(0,0,-1),(2,0,-1)])
    # M5
    add_prod((0,0),(2,2),[(0,0,-1),(0,2,1)])
    add_prod((0,2),(2,2),[(0,0,-1),(0,2,1)])
    # M6
    add_prod((2,0),(0,0),[(2,2,1)])
    add_prod((2,0),(0,2),[(2,2,1)])
    add_prod((0,0),(0,0),[(2,2,-1)])
    add_prod((0,0),(0,2),[(2,2,-1)])
    # M7
    add_prod((0,2),(2,0),[(0,0,1)])
    add_prod((0,2),(2,2),[(0,0,1)])
    add_prod((2,2),(2,0),[(0,0,-1)])
    add_prod((2,2),(2,2),[(0,0,-1)])
    return terms

# 4x4x2 Strassen: 7 products, each (A_block, B_block, [C_blocks])
# B blocks: 0=(0,0) 2x1, 1=(2,0) 2x1
# C blocks: 0=(0,0) 2x1, 1=(2,0) 2x1
# M1=(A00+A11)(B0+B1)->C0,C1 ; M2=(A10+A11)B0->C1
# M3=A00 B1->C1 ; M4=A11 B0->C0
# M5=(A00+A01)B1->-C0,C1 ; M6=(A10-A00)(B0+B1)->C1
# M7=(A01-A11)(B0+B1)->C0
def _expand_442():
    terms = []
    def add_prod(a_off, b_off, w_offs):
        for u,v,w in STRASSEN_221:
            gu,gv,gw = {},{},{}
            for (li,lj),cu in u.items():
                key=(a_off[0]+li, a_off[1]+lj)
                gu[key]=gu.get(key,0)+cu
            for (lj,lk),cv in v.items():
                key=(b_off[0]+lj, b_off[1]+lk)
                gv[key]=gv.get(key,0)+cv
            for (lk,li),cw in w.items():
                for (wr,wc,cc) in w_offs:
                    key=(wc+lk, wr+li)
                    gw[key]=gw.get(key,0)+cw*cc
            terms.append((gu,gv,gw))
    # M1
    add_prod((0,0),(0,0),[(0,0,1),(2,0,1)])
    add_prod((0,0),(2,0),[(0,0,1),(2,0,1)])
    add_prod((2,2),(0,0),[(0,0,1),(2,0,1)])
    add_prod((2,2),(2,0),[(0,0,1),(2,0,1)])
    # M2
    add_prod((2,0),(0,0),[(2,0,1)])
    add_prod((2,2),(0,0),[(2,0,1)])
    # M3
    add_prod((0,0),(2,0),[(2,0,1)])
    # M4
    add_prod((2,2),(0,0),[(0,0,1)])
    # M5
    add_prod((0,0),(2,0),[(0,0,-1),(2,0,1)])
    add_prod((0,2),(2,0),[(0,0,-1),(2,0,1)])
    # M6
    add_prod((2,0),(0,0),[(2,0,1)])
    add_prod((2,0),(2,0),[(2,0,1)])
    add_prod((0,0),(0,0),[(2,0,-1)])
    add_prod((0,0),(2,0),[(2,0,-1)])
    # M7
    add_prod((0,2),(0,0),[(0,0,1)])
    add_prod((0,2),(2,0),[(0,0,1)])
    add_prod((2,2),(0,0),[(0,0,-1)])
    add_prod((2,2),(2,0),[(0,0,-1)])
    return terms

def _build_decomposition(n, m, p):
    """Build rank-77 decomposition: 4x4x4 (49) + 4x4x2 (28)."""
    terms = _expand_444() + _expand_442()
    rank = len(terms)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for r,(gu,gv,gw) in enumerate(terms):
        for (i,j),c in gu.items():
            U[i*m+j, r] = c
        for (j,k),c in gv.items():
            V[j*p+k, r] = c
        for (k,i),c in gw.items():
            W[k*n+i, r] = c
    return U, V, W

def _verify(U, V, W, n, m, p, seed=42):
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_exp = A @ B
    C_com = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        C_com += np.einsum('ij,jk,ki->ik', u, v, w)
    return np.allclose(C_com, C_exp, atol=1e-9)

def _remove_redundant(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    keep = list(range(rank))
    for i in range(len(keep)):
        if i >= len(keep):
            break
        ri = keep[i]
        u_i, v_i, w_i = U[:,ri], V[:,ri], W[:,ri]
        for j in range(i+1, len(keep)):
            rj = keep[j]
            u_j, v_j, w_j = U[:,rj], V[:,rj], W[:,rj]
            # find scale c from a nonzero pair
            mask = np.abs(u_i) > tol
            if not np.any(mask):
                mask = np.abs(v_i) > tol
                src_i, src_j = v_i, v_j
            else:
                src_i, src_j = u_i, u_j
            if not np.any(mask):
                mask = np.abs(w_i) > tol
                src_i, src_j = w_i, w_j
            if not np.any(mask):
                if np.allclose(src_j, 0, atol=tol):
                    keep.remove(rj); j -= 1
                continue
            c = src_j[mask] / src_i[mask]
            if not np.allclose(c, c[0], rtol=1e-7, atol=tol):
                continue
            c = c[0]
            if not np.allclose(u_j, c*u_i, rtol=1e-7, atol=tol):
                continue
            if not np.allclose(v_j, c*v_i, rtol=1e-7, atol=tol):
                continue
            if not np.allclose(w_j, c*w_i, rtol=1e-7, atol=tol):
                continue
            keep.remove(rj); j -= 1
    if len(keep) < rank:
        return U[:,keep], V[:,keep], W[:,keep], True
    return U, V, W, False

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _build_decomposition(n, m, p)
    if not _verify(U, V, W, n, m, p):
        # fallback schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    U, V, W, _ = _remove_redundant(U, V, W, n, m, p)
    if not _verify(U, V, W, n, m, p):
        U, V, W = _build_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
