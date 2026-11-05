import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    """
    Construct a rank-117 exact tensor decomposition for 5x5 matrix multiplication
    using a 2+2+1 block partition with Strassen's algorithm on 2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    _start = time.time()
    
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    for a_idx in range(3):
        for b_idx in range(3):
            for c_idx in range(3):
                sa = parts[a_idx]
                sb = parts[b_idx]
                sc = parts[c_idx]
                oa = offsets[a_idx]
                ob = offsets[b_idx]
                oc = offsets[c_idx]
                
                if sa == 2 and sb == 2 and sc == 2:
                    # Strassen 2x2x2, 7 terms
                    # M1 = (A00+A11)(B00+B11)
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+0)*5+(ob+0)] = 1; u[(oa+1)*5+(ob+1)] = 1
                    v[(ob+0)*5+(oc+0)] = 1; v[(ob+1)*5+(oc+1)] = 1
                    w[(oc+0)*5+(oa+0)] = 1; w[(oc+1)*5+(oa+1)] = 1
                    add_col(u, v, w)
                    
                    # M2 = (A10+A11)B00
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+1)*5+(ob+0)] = 1; u[(oa+1)*5+(ob+1)] = 1
                    v[(ob+0)*5+(oc+0)] = 1
                    w[(oc+0)*5+(oa+1)] = 1; w[(oc+1)*5+(oa+1)] = -1
                    add_col(u, v, w)
                    
                    # M3 = A00(B01-B11)
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+0)*5+(ob+0)] = 1
                    v[(ob+0)*5+(oc+1)] = 1; v[(ob+1)*5+(oc+1)] = -1
                    w[(oc+1)*5+(oa+0)] = 1; w[(oc+1)*5+(oa+1)] = 1
                    add_col(u, v, w)
                    
                    # M4 = A11(B10-B00)
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+1)*5+(ob+1)] = 1
                    v[(ob+1)*5+(oc+0)] = 1; v[(ob+0)*5+(oc+0)] = -1
                    w[(oc+0)*5+(oa+0)] = 1; w[(oc+0)*5+(oa+1)] = 1
                    add_col(u, v, w)
                    
                    # M5 = (A00+A01)B11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+0)*5+(ob+0)] = 1; u[(oa+0)*5+(ob+1)] = 1
                    v[(ob+1)*5+(oc+1)] = 1
                    w[(oc+0)*5+(oa+0)] = -1; w[(oc+1)*5+(oa+0)] = 1
                    add_col(u, v, w)
                    
                    # M6 = (A10-A00)(B00+B01)
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+1)*5+(ob+0)] = 1; u[(oa+0)*5+(ob+0)] = -1
                    v[(ob+0)*5+(oc+0)] = 1; v[(ob+0)*5+(oc+1)] = 1
                    w[(oc+1)*5+(oa+1)] = 1
                    add_col(u, v, w)
                    
                    # M7 = (A01-A11)(B10+B11)
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(oa+0)*5+(ob+1)] = 1; u[(oa+1)*5+(ob+1)] = -1
                    v[(ob+1)*5+(oc+0)] = 1; v[(ob+1)*5+(oc+1)] = 1
                    w[(oc+0)*5+(oa+0)] = 1
                    add_col(u, v, w)
                    
                else:
                    # Schoolbook
                    for ui in range(sa):
                        for vi in range(sb):
                            for wi in range(sc):
                                gi = oa + ui
                                gj = ob + vi
                                gk = oc + wi
                                u = [0]*25; v = [0]*25; w = [0]*25
                                u[gi*5+gj] = 1
                                v[gj*5+gk] = 1
                                w[gk*5+gi] = 1
                                add_col(u, v, w)
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for ci in range(rank):
        if time.time() - _start > _DEADLINE:
            # Should not happen for this fixed-size construction, but guard anyway
            pass
        for ri in range(25):
            U[ri][ci] = U_cols[ci][ri]
            V[ri][ci] = V_cols[ci][ri]
            W[ri][ci] = W_cols[ci][ri]
    
    return {"U": U, "V": V, "W": W}
