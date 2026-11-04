import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
            
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
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
        
    def add_laderman_3x3x3(r1, c1, k1):
        """Add Laderman terms for a 3x3x3 block starting at (r1,c1,k1)."""
        # Laderman's algorithm for 3x3 matrix multiplication
        # Reference: Laderman, "A 3x3 matrix multiplication in 23 multiplications"
        # Indices: A[r1+r][c1+c], B[c1+c][k1+k], C[k1+k][r1+r]
        
        def idx_a(r, c): return (r1 + r) * m + (c1 + c)
        def idx_b(r, c): return (c1 + r) * p + (k1 + c)
        def idx_c(r, c): return (k1 + r) * n + (r1 + c)
        
        # 23 terms
        # M1 = A00 * B00 -> C00
        add_term({idx_a(0,0): 1}, {idx_b(0,0): 1}, {idx_c(0,0): 1})
        # M2 = A11 * B11 -> C11
        add_term({idx_a(1,1): 1}, {idx_b(1,1): 1}, {idx_c(1,1): 1})
        # M3 = A22 * B22 -> C22
        add_term({idx_a(2,2): 1}, {idx_b(2,2): 1}, {idx_c(2,2): 1})
        
        # M4 = (A00+A11) * (B00+B11) -> C00+C11 (partially)
        # This is not quite right, Laderman is specific.
        # Let's use the standard 23-term list.
        # I will use a known list of 23 bilinear forms for 3x3.
        
        # Term 1: A00 B00 -> C00
        # Term 2: A11 B11 -> C11
        # Term 3: A22 B22 -> C22
        
        # Term 4: (A00+A11) (B00+B11) -> C00, C11, C01, C10
        # This creates cross terms. We need to subtract them or handle them.
        # It is safer to use the explicit list.
        
        # Explicit Laderman 3x3 (23 terms):
        # 1. A00 B00 -> C00
        # 2. A11 B11 -> C11
        # 3. A22 B22 -> C22
        # 4. (A00+A11) (B00+B11) -> C00+C11+C01+C10 (needs correction)
        # Actually, let's just use schoolbook for 3x3 if I can't recall the exact 23 terms perfectly.
        # Schoolbook 3x3 is 27.
        # 27 vs 23. Difference is small.
        # But 23 is better.
        
        # I will use a simplified version: Schoolbook for 3x3 (27 terms) to ensure correctness, 
        # and rely on the merge to reduce it.
        for r in range(3):
            for c in range(3):
                for k in range(3):
                    add_term({idx_a(r,c): 1}, {idx_b(c,k): 1}, {idx_c(k,r): 1})
                    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    
    # Partition: [2, 3]
    # Indices: 0,1 (size 2), 2 (size 3)
    blocks = [(0, 2), (2, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2x2(r1, c1, k1)
                elif ar == 3 and ac == 3 and bc == 3:
                    add_laderman_3x3x3(r1, c1, k1)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    def term_to_tensor(U_dict, V_dict, W_dict):
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return tensor
        
    def can_merge(term1, term2):
        if time.time() > deadline:
            return None
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            return None
        
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        if len(u_indices) <= 1 and len(v_indices) <= 1 and len(w_indices) <= 1:
            if len(u_indices) == 1 and len(v_indices) == 1 and len(w_indices) == 1:
                val = combined[(u_indices[0], v_indices[0], w_indices[0])]
                if abs(val) < 1e-12:
                    return None
                return (1, {u_indices[0]: int(val)}, {v_indices[0]: 1}, {w_indices[0]: 1})
            return None
        
        ref_i, ref_j, ref_k = None, None, None
        ref_val = None
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    val = combined.get((i, j, k), 0)
                    if abs(val) > 1e-12:
                        ref_i, ref_j, ref_k = i, j, k
                        ref_val = val
                        break
                if ref_val is not None:
                    break
            if ref_val is not None:
                break
        
        if ref_val is None:
            return None
        
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
        
        for (i, j, k), val in combined.items():
            expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
            if abs(val - expected) > 1e-9:
                return None
        
        a_frac = {k: Fraction(v).limit_denominator(1000) for k, v in a.items()}
        b_frac = {k: Fraction(v).limit_denominator(1000) for k, v in b.items()}
        c_frac = {k: Fraction(v).limit_denominator(1000) for k, v in c.items()}
        
        denoms = [d.numerator for d in a_frac.values()] + \
                 [d.numerator for d in b_frac.values()] + \
                 [d.numerator for d in c_frac.values()]
        lcm = 1
        for d in denoms:
            lcm = lcm * d // gcd(lcm, d)
        
        a_int = {k: int(v * lcm) for k, v in a_frac.items()}
        b_int = {k: int(v * lcm) for k, v in b_frac.items()}
        c_int = {k: int(v * lcm) for k, v in c_frac.items()}
        
        for (i, j, k), val in combined.items():
            expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
            if abs(val * lcm - expected) > 1e-9:
                return None
        
        return (lcm, a_int, b_int, c_int)
        
    # Greedy merge
    for pass_num in range(20):
        if time.time() > deadline - 10:
            break
        merged = False
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    _, a_new, b_new, c_new = result
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
            
    rank = len(terms)
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
