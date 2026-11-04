import time
import numpy as np
from fractions import Fraction
from math import gcd

_start_time = time.time()
_TIME_LIMIT = 110

DIMENSIONS = [5, 5, 5]

def construct():
    global _start_time, _TIME_LIMIT
    if time.time() - _start_time > _TIME_LIMIT:
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105.0  # 5s safety margin before 110s hard limit
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        if time.time() > deadline:
            return
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))
    
    def add_strassen_2x2x2(r1, c1, k1):
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
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
        
        # M1 = (A00+A11)(B00+B11)
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        # M2 = (A10+A11)B00
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        # M3 = A00(B01-B11)
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        # M4 = A11(B10-B00)
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        # M5 = (A00+A01)B11
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        # M6 = (A10-A00)(B00+B01)
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        # M7 = (A01-A11)(B10+B11)
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Partition: [2, 2, 1]
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    add_strassen_2x2x2(i_start, j_start, k_start)
                else:
                    add_schoolbook_block(i_start, i_start + ni, j_start, j_start + mj, k_start, k_start + pk)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert term to a flat 3D tensor representation."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return tensor
    
    def is_rank1(combined):
        """Check if a sparse 3D tensor is rank-1. Returns (a_dict, b_dict, c_dict) or None."""
        if time.time() > deadline:
            return None
        
        combined = {k: v for k, v in combined.items() if v != 0}
        if not combined:
            return None
        
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        # Find a nonzero reference element
        ref_val = None
        ref_i = ref_j = ref_k = None
        for (i, j, k), val in combined.items():
            if val != 0:
                ref_val = val
                ref_i, ref_j, ref_k = i, j, k
                break
        
        if ref_val is None:
            return None
        
        # Compute a[i], b[j], c[k]
        a = {ref_i: 1.0}
        b = {ref_j: 1.0}
        c = {ref_k: 1.0}
        
        for i in u_indices:
            if i != ref_i:
                val = combined.get((i, ref_j, ref_k), 0)
                a[i] = val / ref_val
        
        for j in v_indices:
            if j != ref_j:
                val = combined.get((ref_i, j, ref_k), 0)
                b[j] = val / ref_val
        
        for k in w_indices:
            if k != ref_k:
                val = combined.get((ref_i, ref_j, k), 0)
                c[k] = val / ref_val
        
        # Verify: tensor[i,j,k] should equal a[i]*b[j]*c[k]*ref_val
        for (i, j, k), val in combined.items():
            expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
            if abs(val - expected) > 1e-9:
                return None
        
        # Scale to get integer/half-integer entries
        a_frac = {k: Fraction(v).limit_denominator(10000) for k, v in a.items()}
        b_frac = {k: Fraction(v).limit_denominator(10000) for k, v in b.items()}
        c_frac = {k: Fraction(v).limit_denominator(10000) for k, v in c.items()}
        
        all_fracs = list(a_frac.values()) + list(b_frac.values()) + list(c_frac.values())
        if not all_fracs:
            return None
        
        lcm = 1
        for f in all_fracs:
            d = f.denominator
            lcm = lcm * d // gcd(lcm, d)
        
        a_int = {k: int(f * lcm) for k, f in a_frac.items()}
        b_int = {k: int(f * lcm) for k, f in b_frac.items()}
        c_int = {k: int(f * lcm) for k, f in c_frac.items()}
        
        # Verify integer version exactly
        for (i, j, k), val in combined.items():
            expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
            if abs(val * lcm - expected) > 1e-9:
                return None
        
        return (a_int, b_int, c_int)
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor. Returns (a_dict, b_dict, c_dict) or None."""
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        return is_rank1(combined)
    
    # Precompute tensor representations for faster merging
    term_tensors = [term_to_tensor(*t) for t in terms]
    
    def term_sparsity(t):
        return len(t[0]) * len(t[1]) * len(t[2])
    
    def terms_overlap(t1, t2):
        """Check if two terms share any U, V, or W indices."""
        u_overlap = set(t1[0].keys()) & set(t2[0].keys())
        v_overlap = set(t1[1].keys()) & set(t2[1].keys())
        w_overlap = set(t1[2].keys()) & set(t2[2].keys())
        return len(u_overlap) + len(v_overlap) + len(w_overlap)
    
    # Iterative pair-merge: keep merging until no more merges are found
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        merged = False
        # Sort by sparsity, then by overlap potential
        order = sorted(range(len(terms)), key=lambda i: term_sparsity(terms[i]))
        
        for idx_i in range(len(order)):
            if time.time() > deadline - 8:
                break
            i = order[idx_i]
            if i >= len(terms):
                continue
            # Find best merge partner: prefer terms with more overlap
            best_j = -1
            best_overlap = 0
            for idx_j in range(len(order)):
                j = order[idx_j]
                if j <= i or j >= len(terms):
                    continue
                ov = terms_overlap(terms[i], terms[j])
                if ov > best_overlap:
                    best_overlap = ov
                    best_j = j
                if ov > 0:
                    # Check this candidate immediately if it has overlap
                    result = can_merge(terms[i], terms[j])
                    if result is not None:
                        a_new, b_new, c_new = result
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_new, b_new, c_new))
                        term_tensors.pop(j)
                        term_tensors.pop(i)
                        term_tensors.append(term_to_tensor(a_new, b_new, c_new))
                        merged = True
                        break
            if merged:
                break
    
    # Additional passes with different ordering strategies
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        merged = False
        # Try random-ish ordering by shuffling based on term index
        import random
        rng = random.Random(42 + len(terms))
        indices = list(range(len(terms)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 3:
                break
            if i >= len(terms):
                continue
            for j in indices:
                if j <= i or j >= len(terms):
                    continue
                if terms_overlap(terms[i], terms[j]) == 0:
                    continue
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    a_new, b_new, c_new = result
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    term_tensors.pop(j)
                    term_tensors.pop(i)
                    term_tensors.append(term_to_tensor(a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
                break
    
    rank = len(terms)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
