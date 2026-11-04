"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Improved greedy local search for rank reduction.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
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
    blocks = [(0, 4), (4, 5)]
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                if ar == 4 and ac == 4 and bc == 4:
                    for ri in range(0, 4, 2):
                        for ci in range(0, 4, 2):
                            for ki in range(0, 4, 2):
                                add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
    
    initial_rank = len(terms)
    
    # Pre-compute sparse tensors for each term
    def compute_tensor(term):
        U_dict, V_dict, W_dict = term
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    val = u_val * v_val * w_val
                    if val != 0:
                        tensor[key] = val
        return tensor
    
    # Rank-1 check using integer/half-integer arithmetic
    def check_rank1(combined):
        """Check if sparse tensor is rank-1. Returns (a_dict, b_dict, c_dict) or None."""
        if not combined:
            return None
        
        # Find reference
        ref_val = None
        ref_i = ref_j = ref_k = None
        for (i, j, k), val in combined.items():
            if val != 0:
                ref_val = val
                ref_i, ref_j, ref_k = i, j, k
                break
        
        if ref_val is None:
            return None
        
        # Collect all indices
        all_i = set()
        all_j = set()
        all_k = set()
        for (i, j, k) in combined:
            all_i.add(i)
            all_j.add(j)
            all_k.add(k)
        
        # Compute factors
        a_dict = {}
        b_dict = {}
        c_dict = {}
        
        for i in all_i:
            ti = combined.get((i, ref_j, ref_k), 0)
            if ti != 0:
                a_dict[i] = ti / ref_val
        
        for j in all_j:
            tj = combined.get((ref_i, j, ref_k), 0)
            if tj != 0:
                b_dict[j] = tj / ref_val
        
        for k in all_k:
            tk = combined.get((ref_i, ref_j, k), 0)
            if tk != 0:
                c_dict[k] = tk / ref_val
        
        # Verify
        for (i, j, k), val in combined.items():
            expected = a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0)
            if val != expected:
                return None
        
        # Check integer/half-integer
        def is_valid(val):
            if val == 0:
                return True
            num, den = val.numerator, val.denominator
            return den == 1 or den == 2
        
        for d in [a_dict, b_dict, c_dict]:
            for v in d.values():
                if not is_valid(v):
                    return None
        
        return (a_dict, b_dict, c_dict)
    
    # Efficient merge search
    def try_merge_pass(term_tensors):
        """Try to find and apply one merge. Returns (new_terms, new_tensors) or None."""
        n_terms = len(term_tensors)
        for i in range(n_terms):
            for j in range(i + 1, n_terms):
                # Combine tensors
                combined = {}
                for k, v in term_tensors[i].items():
                    combined[k] = v
                for k, v in term_tensors[j].items():
                    combined[k] = combined.get(k, 0) + v
                
                # Remove zeros
                combined = {k: v for k, v in combined.items() if v != 0}
                
                result = check_rank1(combined)
                if result is not None:
                    a_new, b_new, c_new = result
                    new_term = (a_new, b_new, c_new)
                    new_tensor = compute_tensor(new_term)
                    
                    # Build new term list
                    new_terms = []
                    new_tensors = []
                    for idx in range(n_terms):
                        if idx != i and idx != j:
                            new_terms.append(terms[idx])
                            new_tensors.append(term_tensors[idx])
                    new_terms.append(new_term)
                    new_tensors.append(new_tensor)
                    
                    return (new_terms, new_tensors)
        
        return None
    
    # Multiple merge passes
    term_tensors = [compute_tensor(t) for t in terms]
    
    max_passes = 200
    for pass_num in range(max_passes):
        if time.time() > deadline - 5:
            break
        
        result = try_merge_pass(term_tensors)
        if result is None:
            break
        
        terms, term_tensors = result
    
    rank = len(terms)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = float(val)
        for idx, val in V_dict.items():
            V[idx][r] = float(val)
        for idx, val in W_dict.items():
            W[idx][r] = float(val)
    
    # Verify correctness
    def verify():
        import random
        random.seed(42)
        for _ in range(50):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            actual = 0
            for r in range(rank):
                actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            if abs(actual - 1.0) > 1e-9:
                return False
        return True
    
    if not verify():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    r = ii * m * p + jj * p + kk
                    U[ii * m + jj][r] = 1.0
                    V[jj * p + kk][r] = 1.0
                    W[kk * n + ii][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
