"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Initial rank ~89, then greedy local search attempts to reduce rank.
"""
import time
import numpy as np

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
    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Partition: [4, 1]
    blocks = [(0, 4), (4, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 4 and ac == 4 and bc == 4:
                    # Decompose 4x4x4 into four 2x2x2 Strassen blocks
                    for ri in range(0, 4, 2):
                        for ci in range(0, 4, 2):
                            for ki in range(0, 4, 2):
                                add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Greedy local search: try to merge pairs of terms
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert term to a flat 3D tensor representation."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return tensor
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor."""
        if time.time() > deadline:
            return None
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Check if combined is rank-1: all 2x2 minors must be zero
        # Flatten to a 3D array and check
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        if len(u_indices) <= 1 or len(v_indices) <= 1 or len(w_indices) <= 1:
            # Trivially rank 1 if one dimension is 1
            # But need to verify it's actually nonzero and consistent
            if len(u_indices) == 1 and len(v_indices) == 1 and len(w_indices) == 1:
                return (combined[u_indices[0], v_indices[0], w_indices[0]], 
                        {u_indices[0]: 1}, {v_indices[0]: 1}, {w_indices[0]: 1})
            # More complex check needed
            pass
        
        # For rank-1 check: tensor[i,j,k] = a[i]*b[j]*c[k]
        # Pick a reference element
        ref_i, ref_j, ref_k = u_indices[0], v_indices[0], w_indices[0]
        ref_val = combined.get((ref_i, ref_j, ref_k), 0)
        if ref_val == 0:
            return None
        
        # Compute a[i] = tensor[i, ref_j, ref_k] / ref_val
        # Compute b[j] = tensor[ref_i, j, ref_k] / ref_val  
        # Compute c[k] = tensor[ref_i, ref_j, k] / ref_val
        a = {}
        b = {}
        c = {}
        a[ref_i] = 1.0
        b[ref_j] = 1.0
        c[ref_k] = 1.0
        
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
        # Find gcd-like scaling
        all_vals = list(a.values()) + list(b.values()) + list(c.values())
        # Convert to fractions for exact arithmetic
        from fractions import Fraction
        a_frac = {k: Fraction(v).limit_denominator(1000) for k, v in a.items()}
        b_frac = {k: Fraction(v).limit_denominator(1000) for k, v in b.items()}
        c_frac = {k: Fraction(v).limit_denominator(1000) for k, v in c.items()}
        
        # Find common denominator
        denoms = [d.numerator for d in a_frac.values()] + \
                 [d.numerator for d in b_frac.values()] + \
                 [d.numerator for d in c_frac.values()]
        if not denoms:
            return None
        from math import gcd
        lcm = 1
        for d in denoms:
            lcm = lcm * d // gcd(lcm, d)
        
        # Scale to integers
        a_int = {k: int(v * lcm) for k, v in a_frac.items()}
        b_int = {k: int(v * lcm) for k, v in b_frac.items()}
        c_int = {k: int(v * lcm) for k, v in c_frac.items()}
        
        # Verify integer version
        for (i, j, k), val in combined.items():
            expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
            if abs(val * lcm - expected) > 1e-9:
                return None
        
        return (lcm, a_int, b_int, c_int)
    
    # Try merging pairs
    merged = True
    while merged and time.time() < deadline - 10:
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
                    # Remove terms i and j, add new term
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
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
