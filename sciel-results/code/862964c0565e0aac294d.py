"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a sophisticated approach: start from a good initial decomposition,
then iteratively reduce rank by finding linearly dependent terms and merging pairs.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_ok(margin=5.0):
        return (time.time() - start_time) < deadline - margin
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if not time_ok():
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
    def add_strassen_2x2x2(r1, c1, k1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Build initial decomposition
    # Use [2,3] partition
    blocks = [(0, 2), (2, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if not time_ok():
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2x2(r1, c1, k1)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if not time_ok():
                break
        if not time_ok():
            break
    
    def try_merge_pairs():
        """Try to merge pairs of terms into a single rank-1 term."""
        if not time_ok(10):
            return False
        
        for i in range(len(terms)):
            if not time_ok(10):
                break
            for j in range(i + 1, len(terms)):
                if not time_ok(10):
                    break
                
                U1, V1, W1 = terms[i]
                U2, V2, W2 = terms[j]
                
                # Build combined tensor
                tensor = {}
                for u_idx, u_val in U1.items():
                    for v_idx, v_val in V1.items():
                        for w_idx, w_val in W1.items():
                            key = (u_idx, v_idx, w_idx)
                            tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
                for u_idx, u_val in U2.items():
                    for v_idx, v_val in V2.items():
                        for w_idx, w_val in W2.items():
                            key = (u_idx, v_idx, w_idx)
                            tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
                
                tensor = {k: v for k, v in tensor.items() if v != 0}
                if not tensor:
                    continue
                
                u_indices = sorted(set(k[0] for k in tensor.keys()))
                v_indices = sorted(set(k[1] for k in tensor.keys()))
                w_indices = sorted(set(k[2] for k in tensor.keys()))
                
                ref_val = None
                ref_i = ref_j = ref_k = None
                for (i, j, k), val in tensor.items():
                    if val != 0:
                        ref_val = val
                        ref_i, ref_j, ref_k = i, j, k
                        break
                
                if ref_val is None:
                    continue
                
                a = {ref_i: 1.0}
                b = {ref_j: 1.0}
                c = {ref_k: 1.0}
                
                for i in u_indices:
                    if i != ref_i:
                        val = tensor.get((i, ref_j, ref_k), 0)
                        a[i] = val / ref_val
                
                for j in v_indices:
                    if j != ref_j:
                        val = tensor.get((ref_i, j, ref_k), 0)
                        b[j] = val / ref_val
                
                for k in w_indices:
                    if k != ref_k:
                        val = tensor.get((ref_i, ref_j, k), 0)
                        c[k] = val / ref_val
                
                is_rank1 = True
                for (i, j, k), val in tensor.items():
                    expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
                    if abs(val - expected) > 1e-9:
                        is_rank1 = False
                        break
                
                if is_rank1:
                    a_frac = {k: Fraction(v).limit_denominator(10000) for k, v in a.items()}
                    b_frac = {k: Fraction(v).limit_denominator(10000) for k, v in b.items()}
                    c_frac = {k: Fraction(v).limit_denominator(10000) for k, v in c.items()}
                    
                    all_fracs = list(a_frac.values()) + list(b_frac.values()) + list(c_frac.values())
                    if not all_fracs:
                        continue
                    
                    lcm = 1
                    for f in all_fracs:
                        d = f.denominator
                        lcm = lcm * d // gcd(lcm, d)
                    
                    a_int = {k: int(f * lcm) for k, f in a_frac.items()}
                    b_int = {k: int(f * lcm) for k, f in b_frac.items()}
                    c_int = {k: int(f * lcm) for k, f in c_frac.items()}
                    
                    valid = True
                    for (i, j, k), val in tensor.items():
                        expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
                        if abs(val * lcm - expected) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_int, b_int, c_int))
                        return True
        
        return False
    
    def try_remove_dependent():
        """Try to remove a term that is linearly dependent on others."""
        if not time_ok(10):
            return False
        
        rank = len(terms)
        if rank <= 49:
            return False
        
        # Build matrix of term vectors
        all_indices = set()
        for U_dict, V_dict, W_dict in terms:
            for u_idx in U_dict:
                for v_idx in V_dict:
                    for w_idx in W_dict:
                        all_indices.add((u_idx, v_idx, w_idx))
        
        index_list = sorted(all_indices)
        index_to_pos = {idx: pos for pos, idx in enumerate(index_list)}
        dim = len(index_list)
        
        M = np.zeros((rank, dim), dtype=np.float64)
        for i, (U_dict, V_dict, W_dict) in enumerate(terms):
            for u_idx, u_val in U_dict.items():
                for v_idx, v_val in V_dict.items():
                    for w_idx, w_val in W_dict.items():
                        key = (u_idx, v_idx, w_idx)
                        if key in index_to_pos:
                            M[i, index_to_pos[key]] = u_val * v_val * w_val
        
        # Try removing each term and checking if it's in the span
        for i in range(rank - 1, -1, -1):
            if not time_ok(10):
                break
            
            remaining = [j for j in range(rank) if j != i]
            if len(remaining) < 49:
                continue
            
            # Use a subset for speed
            if len(remaining) > 60:
                subset = remaining[:60]
            else:
                subset = remaining
            
            M_sub = M[subset, :]
            try:
                x, residuals, rank_sub, _ = np.linalg.lstsq(M_sub, M[i], rcond=1e-10)
                residual = M_sub @ x - M[i]
                res_norm = np.linalg.norm(residual)
                i_norm = np.linalg.norm(M[i])
                
                if i_norm > 0 and res_norm / i_norm < 1e-8:
                    terms.pop(i)
                    return True
            except:
                continue
        
        return False
    
    # Iterative rank reduction
    for _ in range(2000):
        if not time_ok(10):
            break
        if len(terms) <= 49:
            break
        
        made_progress = False
        if try_merge_pairs():
            made_progress = True
        if not made_progress and try_remove_dependent():
            made_progress = True
        
        if not made_progress:
            # Try a few more passes with different orderings
            for _ in range(5):
                if not time_ok(10):
                    break
                if try_merge_pairs():
                    break
                if try_remove_dependent():
                    break
            else:
                break
    
    rank = len(terms)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
