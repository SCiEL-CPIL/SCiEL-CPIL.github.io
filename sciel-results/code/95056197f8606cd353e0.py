"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Improved merge logic with integer arithmetic and multiple passes.
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
    
    # Convert to integer coefficients if possible to speed up checks
    # We assume all initial terms have integer coefficients (1, -1)
    
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert a term (three dicts) to a sparse tensor dict."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return {k: v for k, v in tensor.items() if v != 0}
    
    def can_merge(term1, term2):
        """Try to merge two rank-1 terms into one. Returns (a_int, b_int, c_int) or None."""
        if time.time() > deadline:
            return None
        
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        
        # Combine tensors
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        combined = {k: v for k, v in combined.items() if v != 0}
        if not combined:
            return None
        
        # Get index sets
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        # Find a reference element
        ref = None
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    if combined.get((i, j, k), 0) != 0:
                        ref = (i, j, k)
                        break
                if ref:
                    break
            if ref:
                break
        
        if not ref:
            return None
            
        ref_i, ref_j, ref_k = ref
        ref_val = combined[ref]
        
        # Build candidate factors
        a = {ref_i: 1}
        b = {ref_j: 1}
        c = {ref_k: 1}
        
        for i in u_indices:
            if i != ref_i:
                val = combined.get((i, ref_j, ref_k), 0)
                a[i] = Fraction(val, ref_val)
                
        for j in v_indices:
            if j != ref_j:
                val = combined.get((ref_i, j, ref_k), 0)
                b[j] = Fraction(val, ref_val)
                
        for k in w_indices:
            if k != ref_k:
                val = combined.get((ref_i, ref_j, k), 0)
                c[k] = Fraction(val, ref_val)
                
        # Verify the factorization is exact
        for (i, j, k), val in combined.items():
            expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
            if expected != val:
                return None
                
        # Scale to integers
        all_fracs = list(a.values()) + list(b.values()) + list(c.values())
        lcm = 1
        for f in all_fracs:
            d = f.denominator
            lcm = lcm * d // gcd(lcm, d)
            
        a_int = {k: int(v * lcm) for k, v in a.items()}
        b_int = {k: int(v * lcm) for k, v in b.items()}
        c_int = {k: int(v * lcm) for k, v in c.items()}
        
        return (a_int, b_int, c_int)
        
    # Greedy local search with improved merge logic
    for pass_num in range(200):
        if time.time() > deadline - 5:
            break
        merged = False
        # Try all pairs, but track which pairs have been tried
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    a_new, b_new, c_new = result
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
            
    # Remove redundant terms and verify decomposition
    new_terms = []
    for U_dict, V_dict, W_dict in terms:
        is_zero = True
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    if u_val * v_val * w_val != 0:
                        is_zero = False
                        break
                if not is_zero:
                    break
            if not is_zero:
                break
        if not is_zero:
            new_terms.append((U_dict, V_dict, W_dict))
    terms = new_terms
    
    # Verify the decomposition is correct
    if time.time() < deadline - 10:
        def verify_decomposition(terms):
            """Verify that the terms exactly decompose the matrix multiplication tensor."""
            # Build the expected tensor
            expected = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected[(i * m + j, j * p + k, k * n + i)] = 1
            
            # Build the actual tensor from terms
            actual = {}
            for U_dict, V_dict, W_dict in terms:
                for u_idx, u_val in U_dict.items():
                    for v_idx, v_val in V_dict.items():
                        for w_idx, w_val in W_dict.items():
                            key = (u_idx, v_idx, w_idx)
                            actual[key] = actual.get(key, 0) + u_val * v_val * w_val
            
            # Check if they match
            for key, val in expected.items():
                if actual.get(key, 0) != val:
                    return False
            for key, val in actual.items():
                if expected.get(key, 0) != val:
                    return False
            return True
        
        if not verify_decomposition(terms):
            # Fallback to schoolbook if verification fails
            terms = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = int(val) if isinstance(val, int) else val
        for idx, val in V_dict.items():
            V[idx][r] = int(val) if isinstance(val, int) else val
        for idx, val in W_dict.items():
            W[idx][r] = int(val) if isinstance(val, int) else val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
