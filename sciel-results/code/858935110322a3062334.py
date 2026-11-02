"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 130.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def naive_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_blocks(left_b_idx, right_b_idx, W_idx_map, start_r):
        """
        Build the 7 rank-1 terms for one 2x2 * 2x2 block.
        left_b_idx: [0, 1] or [2, 3] (column indices in B)
        right_b_idx: [4, 5] or [6, 7] (column indices in B, i.e., 4 + [0,1] or 4 + [2,3])
        W_idx_map: mapping from (output_row, output_col) to W index
                   where output_row is 0 or 1, output_col is 0 or 1
        Returns list of 7 tuples: (U_contribs, V_contribs, W_contribs)
        where each is a dict {idx: val}
        """
        # Strassen's 7 products for C = A*B where A, B are 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # C[0,0] = M1 + M4 - M5 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M7
        # C[1,1] = M1 - M2 + M3 + M6
        
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B indices for this block: left_b_idx[0], left_b_idx[1], right_b_idx[0], right_b_idx[1]
        # B local: (0,0)=left_b_idx[0], (0,1)=left_b_idx[1], (1,0)=right_b_idx[0], (1,1)=right_b_idx[1]
        
        b00, b01, b10, b11 = left_b_idx[0], left_b_idx[1], right_b_idx[0], right_b_idx[1]
        
        strassen = []
        
        # M1
        u = {0: 1, 3: 1}
        v = {b00: 1, b11: 1}
        w = {W_idx_map[(0,0)]: 1, W_idx_map[(1,1)]: 1}
        strassen.append((u, v, w))
        
        # M2
        u = {2: 1, 3: 1}
        v = {b00: 1}
        w = {W_idx_map[(1,0)]: 1, W_idx_map[(1,1)]: -1}
        strassen.append((u, v, w))
        
        # M3
        u = {0: 1}
        v = {b01: 1, b11: -1}
        w = {W_idx_map[(0,1)]: 1, W_idx_map[(1,1)]: 1}
        strassen.append((u, v, w))
        
        # M4
        u = {3: 1}
        v = {b10: 1, b00: -1}
        w = {W_idx_map[(0,0)]: 1, W_idx_map[(1,0)]: 1}
        strassen.append((u, v, w))
        
        # M5
        u = {0: 1, 1: 1}
        v = {b11: 1}
        w = {W_idx_map[(0,0)]: -1, W_idx_map[(0,1)]: 1}
        strassen.append((u, v, w))
        
        # M6
        u = {2: 1, 0: -1}
        v = {b00: 1, b01: 1}
        w = {W_idx_map[(1,1)]: 1}
        strassen.append((u, v, w))
        
        # M7
        u = {1: 1, 3: -1}
        v = {b10: 1, b11: 1}
        w = {W_idx_map[(0,0)]: 1, W_idx_map[(1,0)]: -1}
        strassen.append((u, v, w))
        
        return strassen

    def try_rank_13():
        """
        Attempt to find a rank-13 decomposition by sharing one rank-1 term
        between the two Strassen-based 2x2*2x2 blocks.
        We enumerate all possible single-term sharing patterns.
        """
        R = 13
        
        # W index mapping: W[k*n + r] where k is column of C (0-3), r is row of C (0-1)
        # C is 2x4: rows 0,1; cols 0,1,2,3
        # W index = k*n + r = 2*k + r
        W_idx = {}
        for k in range(4):
            for r in range(2):
                W_idx[(r, k)] = k * n + r  # k*n + r
        
        # Left block: cols 0,1 of C => W indices (0,0)=0, (0,1)=2, (1,0)=1, (1,1)=3
        # Right block: cols 2,3 of C => W indices (0,2)=4, (0,3)=6, (1,2)=5, (1,3)=7
        
        left_W_map = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        right_W_map = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        # Left B columns: 0,1 ; Right B columns: 2,3
        left_b = [0, 1]
        right_b = [2, 3]
        left_br = [4, 5]
        right_br = [6, 7]
        
        left_strassen = build_strassen_blocks(left_b, left_br, left_W_map, 0)
        right_strassen = build_strassen_blocks(right_b, right_br, right_W_map, 7)
        
        # Now try all 7x7 = 49 possible sharing patterns:
        # Pick term i from left (i in 0..6) and term j from right (j in 0..6)
        # to merge into one shared term
        # For sharing to work, the U, V, W contributions must be compatible:
        # i.e., U_left_i + U_right_j, V_left_i + V_right_j, W_left_i + W_right_j
        # But actually, the rank-1 term is a product u * v * w, so we can't simply
        # add them. We need u_left * v_left * w_left = u_right * v_right * w_right
        # as rank-1 tensors, which means the products must be scalar multiples.
        
        # For a valid merge, we need:
        # (u_left, v_left, w_left) and (u_right, v_right, w_right) to define the same
        # rank-1 tensor, i.e., u_left = c * u_right, v_left = d * v_right, w_left = (1/cd) * w_right
        # for some nonzero scalars c, d.
        
        best = None
        for i in range(7):
            for j in range(7):
                ul, vl, wl = left_strassen[i]
                ur, vr, wr = right_strassen[j]
                
                # Check if they can be merged: the sets of nonzero indices must be compatible
                # For simplicity, check if one is a scalar multiple of the other in each component
                # This requires the support to be the same or one to be a subset
                u_keys_l = set(ul.keys())
                u_keys_r = set(ur.keys())
                v_keys_l = set(vl.keys())
                v_keys_r = set(vr.keys())
                w_keys_l = set(wl.keys())
                w_keys_r = set(wr.keys())
                
                if u_keys_l != u_keys_r or v_keys_l != v_keys_r or w_keys_l != w_keys_r:
                    continue
                
                # Find scaling factors
                c = None
                d = None
                valid = True
                for key in u_keys_l:
                    if ur[key] == 0:
                        valid = False
                        break
                    ratio = ul[key] / ur[key]
                    if c is None:
                        c = ratio
                    elif abs(c - ratio) > 1e-10:
                        valid = False
                        break
                if not valid:
                    continue
                if c == 0:
                    continue
                    
                for key in v_keys_l:
                    if vr[key] == 0:
                        valid = False
                        break
                    ratio = vl[key] / vr[key]
                    if d is None:
                        d = ratio
                    elif abs(d - ratio) > 1e-10:
                        valid = False
                        break
                if not valid:
                    continue
                if d == 0:
                    continue
                
                e = 1.0 / (c * d)
                for key in w_keys_l:
                    if wr[key] == 0:
                        valid = False
                        break
                    ratio = wl[key] / wr[key]
                    if abs(e - ratio) > 1e-10:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try to build the rank-13 decomposition
                # Remove term i from left and term j from right
                # Add merged term
                merged_u = {k: ul[k] / c for k in u_keys_l}
                merged_v = {k: vr[k] for k in v_keys_r}
                merged_w = {k: wr[k] / e for k in w_keys_r}
                
                U = [[0]*R for _ in range(4)]
                V = [[0]*R for _ in range(8)]
                W = [[0]*R for _ in range(8)]
                
                r_idx = 0
                for ii in range(7):
                    if ii == i:
                        continue
                    u, v, w = left_strassen[ii]
                    for idx, val in u.items():
                        U[idx][r_idx] += val
                    for idx, val in v.items():
                        V[idx][r_idx] += val
                    for idx, val in w.items():
                        W[idx][r_idx] += val
                    r_idx += 1
                
                for jj in range(7):
                    if jj == j:
                        continue
                    u, v, w = right_strassen[jj]
                    for idx, val in u.items():
                        U[idx][r_idx] += val
                    for idx, val in v.items():
                        V[idx][r_idx] += val
                    for idx, val in w.items():
                        W[idx][r_idx] += val
                    r_idx += 1
                
                # Add merged term
                for idx, val in merged_u.items():
                    U[idx][r_idx] += val
                for idx, val in merged_v.items():
                    V[idx][r_idx] += val
                for idx, val in merged_w.items():
                    W[idx][r_idx] += val
                
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None

    def strassen_rect():
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        left_W_map = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        right_W_map = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        left_strassen = build_strassen_blocks([0,1], [4,5], left_W_map, 0)
        right_strassen = build_strassen_blocks([2,3], [6,7], right_W_map, 7)
        
        r = 0
        for u, v, w in left_strassen:
            for idx, val in u.items():
                U[idx][r] += val
            for idx, val in v.items():
                V[idx][r] += val
            for idx, val in w.items():
                W[idx][r] += val
            r += 1
        for u, v, w in right_strassen:
            for idx, val in u.items():
                U[idx][r] += val
            for idx, val in v.items():
                V[idx][r] += val
            for idx, val in w.items():
                W[idx][r] += val
            r += 1
        
        if verify(U, V, W):
            return {"U": U, "V": W}  # typo fix
        return None
    
    def strassen_rect_fixed():
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        left_W_map = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        right_W_map = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        left_strassen = build_strassen_blocks([0,1], [4,5], left_W_map, 0)
        right_strassen = build_strassen_blocks([2,3], [6,7], right_W_map, 7)
        
        r = 0
        for u, v, w in left_strassen:
            for idx, val in u.items():
                U[idx][r] += val
            for idx, val in v.items():
                V[idx][r] += val
            for idx, val in w.items():
                W[idx][r] += val
            r += 1
        for u, v, w in right_strassen:
            for idx, val in u.items():
                U[idx][r] += val
            for idx, val in v.items():
                V[idx][r] += val
            for idx, val in w.items():
                W[idx][r] += val
            r += 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    # Try rank-13 first
    result = try_rank_13()
    if result is not None:
        return result
    
    # Fall back to rank-14
    result = strassen_rect_fixed()
    if result is not None:
        return result
    
    return naive_decomp()
# EVOLVE-BLOCK-END
