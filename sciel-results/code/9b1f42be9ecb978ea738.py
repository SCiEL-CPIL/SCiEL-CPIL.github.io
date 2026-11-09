"""
5x5x5 matrix multiplication tensor decomposition.
Uses a 4+1 block structure with recursive Strassen for the 4x4x4 diagonal block.
Rank: 110.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the 5x5x5 matrix multiplication tensor.
    Decomposition strategy:
    - Split indices into I0={0,1,2,3} and I1={4}.
    - Block (I0, I0, I0) uses Strassen 4x4x4 (rank 49).
    - Other blocks use schoolbook multiplication.
    Total rank: 49 + 61 = 110.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Store columns as dictionaries to allow easy addition and checking
    # Key: index in the flattened matrix (e.g., i*m+j for U)
    # Value: coefficient
    cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        Adds a rank-1 term to the decomposition.
        If the term is zero, it is skipped.
        If the term is a scalar multiple of an existing term, it is skipped.
        """
        # Merge entries
        u_merged = {}
        for k, v in u_dict.items():
            u_merged[k] = u_merged.get(k, 0) + v
        v_merged = {}
        for k, v in v_dict.items():
            v_merged[k] = v_merged.get(k, 0) + v
        w_merged = {}
        for k, v in w_dict.items():
            w_merged[k] = w_merged.get(k, 0) + v
            
        # Remove zeros
        u_merged = {k: v for k, v in u_merged.items() if v != 0}
        v_merged = {k: v for k, v in v_merged.items() if v != 0}
        w_merged = {k: v for k, v in w_merged.items() if v != 0}
        
        # Skip if any component is empty (zero vector)
        if not u_merged or not v_merged or not w_merged:
            return
            
        # Check for linear dependency (scalar multiple) with existing terms
        # This is an optimization to keep the rank minimal if redundant terms are generated
        for existing_u, existing_v, existing_w in cols:
            if u_merged.keys() != existing_u.keys() or v_merged.keys() != existing_v.keys() or w_merged.keys() != existing_w.keys():
                continue
            
            # Check if u_merged is a scalar multiple of existing_u
            scale = None
            is_multiple = True
            for k in u_merged:
                if existing_u[k] == 0:
                    if u_merged[k] != 0:
                        is_multiple = False
                        break
                    continue
                s = u_merged[k] / existing_u[k]
                if scale is None:
                    scale = s
                elif abs(s - scale) > 1e-12:
                    is_multiple = False
                    break
            
            if is_multiple and scale is not None:
                # Verify V
                for k in v_merged:
                    if existing_v[k] == 0:
                        if v_merged[k] != 0:
                            is_multiple = False
                            break
                        continue
                    s = v_merged[k] / existing_v[k]
                    if abs(s - scale) > 1e-12:
                        is_multiple = False
                        break
                # Verify W
                if is_multiple:
                    for k in w_merged:
                        if existing_w[k] == 0:
                            if w_merged[k] != 0:
                                is_multiple = False
                                break
                            continue
                        s = w_merged[k] / existing_w[k]
                        if abs(s - scale) > 1e-12:
                            is_multiple = False
                            break
                
                if is_multiple and scale is not None and abs(scale) > 1e-12:
                    return # Redundant
        
        cols.append((u_merged, v_merged, w_merged))
        
    # Define index groups
    group_4 = [0, 1, 2, 3]
    group_1 = [4]
    
    def add_schoolbook(i_set, j_set, k_set):
        """Adds schoolbook terms for the given index sets."""
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    add_term(
                        {i * m + j: 1.0},
                        {j * p + k: 1.0},
                        {k * n + i: 1.0}
                    )
                    
    # Strassen 2x2x2 base terms
    # Format: (u_dict, v_dict, w_dict) with keys (local_i, local_j)
    STRASSEN_2x2 = [
        ( {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1} ),
        ( {(1,0):1, (1,1):1}, {(0,0):1},         {(1,0):1, (1,1):-1}),
        ( {(0,0):1},         {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1} ),
        ( {(1,1):1},         {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1} ),
        ( {(0,0):1, (0,1):1}, {(1,1):1},         {(0,1):1, (0,0):-1}),
        ( {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1} ),
        ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1} ),
    ]
    
    def add_strassen_4x4(offset_i, offset_j, offset_k):
        """
        Adds Strassen 4x4x4 terms.
        This is composed of 7 macro-terms, each using the 2x2 Strassen logic.
        """
        # Local indices for the 4x4 block: 0,1 are in the first half, 2,3 in the second
        # But Strassen 4x4 is usually built from 7 Strassen 2x2 calls.
        # Let's implement the macro structure directly.
        
        # Macro terms for 4x4 Strassen
        # Q1 = (A00+A11)(B00+B11)
        # Q2 = (A10+A11)B00
        # Q3 = A00(B01-B11)
        # Q4 = A11(B10-A00)
        # Q5 = (A00+A01)B11
        # Q6 = (A10-A00)(B00+B01)
        # Q7 = (A01-A11)(B10+B11)
        
        # We map these to local 2x2 indices (0,1) for the sub-blocks
        # Sub-block (0,0) of 4x4 is global (0,0)
        # Sub-block (1,1) of 4x4 is global (2,3)
        
        macro_defs = [
            # (u_macro, v_macro, w_macro) where values are in {0,1} for quadrants
            # Quadrants: (0,0), (0,1), (1,0), (1,1)
            ( {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1} ), # Q1
            ( {(1,0):1, (1,1):1}, {(0,0):1},         {(1,0):1, (1,1):-1}), # Q2
            ( {(0,0):1},         {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1} ), # Q3
            ( {(1,1):1},         {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1} ), # Q4
            ( {(0,0):1, (0,1):1}, {(1,1):1},         {(0,1):1, (0,0):-1}), # Q5
            ( {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1} ),         # Q6
            ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1} ),         # Q7
        ]
        
        for u_m, v_m, w_m in macro_defs:
            # For each macro term, expand into 7 Strassen 2x2 terms
            for u_s, v_s, w_s in STRASSEN_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Expand U
                for (li, lj), val_u_s in u_s.items():
                    for (mi, mj), val_u_m in u_m.items():
                        val = val_u_s * val_u_m
                        if val != 0:
                            # Map local indices to global
                            # li, lj are in {0,1} for 2x2 Strassen
                            # mi, mj are in {0,1} for 4x4 quadrants
                            # Global row: offset + mi*2 + li
                            # Global col: offset + mj*2 + lj
                            gi = offset_i + mi * 2 + li
                            gj = offset_j + mj * 2 + lj
                            u_dict[gi * m + gj] = u_dict.get(gi * m + gj, 0) + val
                
                # Expand V
                for (lj, lk), val_v_s in v_s.items():
                    for (mj, mk), val_v_m in v_m.items():
                        val = val_v_s * val_v_m
                        if val != 0:
                            gj = offset_j + mj * 2 + lj
                            gk = offset_k + mk * 2 + lk
                            v_dict[gj * p + gk] = v_dict.get(gj * p + gk, 0) + val
                            
                # Expand W
                for (li, lk), val_w_s in w_s.items():
                    for (mi, mk), val_w_m in w_m.items():
                        val = val_w_s * val_w_m
                        if val != 0:
                            gi = offset_i + mi * 2 + li
                            gk = offset_k + mk * 2 + lk
                            w_dict[gk * n + gi] = w_dict.get(gk * n + gi, 0) + val
                            
                add_term(u_dict, v_dict, w_dict)
                
    # Build the decomposition
    # Block (0,1,2,3) x (0,1,2,3) x (0,1,2,3)
    add_strassen_4x4(0, 0, 0)
    
    # Off-diagonal blocks: Schoolbook
    # I0={0,1,2,3}, I1={4}
    # Blocks:
    # (I0, I0, I1)
    add_schoolbook(group_4, group_4, group_1)
    # (I0, I1, I0)
    add_schoolbook(group_4, group_1, group_4)
    # (I1, I0, I0)
    add_schoolbook(group_1, group_4, group_4)
    # (I0, I1, I1)
    add_schoolbook(group_4, group_1, group_1)
    # (I1, I0, I1)
    add_schoolbook(group_1, group_4, group_1)
    # (I1, I1, I0)
    add_schoolbook(group_1, group_1, group_4)
    # (I0, I1, I0) - Wait, did I miss some?
    # The blocks are defined by (I_a, J_b, K_c)
    # We have done:
    # (4,4,4)
    # (4,4,1)
    # (4,1,4)
    # (1,4,4)
    # (4,1,1)
    # (1,4,1)
    # (1,1,4)
    # Missing: (1,1,1)
    add_schoolbook(group_1, group_1, group_1)
    
    rank = len(cols)
    
    # Construct numpy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_d, v_d, w_d) in enumerate(cols):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
