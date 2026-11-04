import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # We'll build a rank-80 decomposition using block structure
    # M_{3,4,7} = M_{3,2,7} + M_{3,2,7} (split the middle dimension 4 = 2+2)
    # Each M_{3,2,7} = M_{3,2,3} + M_{3,2,3} + M_{3,2,1} (split last dim 7 = 3+3+1)
    # Each M_{3,2,3} = M_{3,2,2} + M_{3,2,1} (split last dim 3 = 2+1)
    # Each M_{3,2,2} = M_{2,2,2} + M_{1,2,2} (split first dim 3 = 2+1)
    # M_{2,2,2} uses Strassen (rank 7)
    # M_{1,2,2} uses schoolbook (rank 4)
    # M_{3,2,1} uses schoolbook (rank 6)
    
    # Total rank: 2 * (2*(7+4) + 6) = 2 * (22 + 6) = 2 * 28 = 56? 
    # Wait, let me recalculate:
    # M_{3,2,2} = M_{2,2,2}(rank 7) + M_{1,2,2}(rank 4) = rank 11
    # M_{3,2,3} = M_{3,2,2}(rank 11) + M_{3,2,1}(rank 6) = rank 17
    # M_{3,2,7} = M_{3,2,3}(rank 17) + M_{3,2,3}(rank 17) + M_{3,2,1}(rank 6) = rank 40
    # M_{3,4,7} = M_{3,2,7}(rank 40) + M_{3,2,7}(rank 40) = rank 80
    
    # Now I need to construct the actual factor matrices.
    # The tensor T has indices: T[i*m+j, j*p+k, k*n+i] = 1 for the multiplication tensor
    # i in [0,n), j in [0,m), k in [0,p)
    
    # I'll build this by defining the Strassen decomposition for 2x2x2 and then
    # assembling the blocks.
    
    # Let's define helper functions to build sub-decompositions
    
    def build_strassen_222():
        """
        Build the Strassen decomposition for M_{2,2,2}.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 multiplication.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's algorithm for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # In tensor form: U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
        # where i,j,k in {0,1}
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # P1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # U indices: i*2+j
        # A[0,0] is at U[0*2+0] = U[0]
        # A[1,1] is at U[1*2+1] = U[3]
        # B[0,0] is at V[0*2+0] = V[0]
        # B[1,1] is at V[1*2+1] = V[3]
        # W[0*2+0] = W[0] for C[0,0]
        # W[0*2+1] = W[1] for C[0,1]
        # W[1*2+0] = W[2] for C[1,0]
        # W[1*2+1] = W[3] for C[1,1]
        
        # For P1: U[0]=1, U[3]=1; V[0]=1, V[3]=1
        # C00 gets +1, C01 gets +1, C10 gets +1, C11 gets +1
        # Wait, P1 = (a+d)(e+h) contributes to all four output entries
        # C00 += P1, C01 += P1, C10 += P1, C11 += P1? No...
        # Actually P1 = (a+d)(e+h) is added to C00 and C11, and subtracted from C01 and C10?
        # No, let me re-check:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So P1 contributes: +1 to C00, 0 to C01, 0 to C10, +1 to C11
        # P2 contributes: 0 to C00, 0 to C01, +1 to C10, -1 to C11
        # P3 contributes: 0 to C00, +1 to C01, 0 to C10, +1 to C11
        # P4 contributes: +1 to C00, 0 to C01, +1 to C10, 0 to C11
        # P5 contributes: -1 to C00, +1 to C01, 0 to C10, 0 to C11
        # P6 contributes: 0 to C00, 0 to C01, 0 to C10, +1 to C11
        # P7 contributes: +1 to C00, 0 to C01, 0 to C10, 0 to C11
        
        # P1 = (a+d)(e+h)
        # a = A[0,0], d = A[1,1]
        # e = B[0,0], h = B[1,1]
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # P2 = (c+d)e
        # c = A[1,0], d = A[1,1]
        # e = B[0,0]
        U[2, 1] = 1  # c = A[1,0] at index 1*2+0=2
        U[3, 1] = 1  # d = A[1,1] at index 1*2+1=3
        V[0, 1] = 1  # e = B[0,0] at index 0*2+0=0
        W[2, 1] = 1  # C10
        W[3, 1] = -1 # C11
        
        # P3 = a(f-h)
        # a = A[0,0]
        # f = B[0,1], h = B[1,1]
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f = B[0,1] at index 0*2+1=1
        V[3, 2] = -1 # h = B[1,1] at index 1*2+1=3
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # P4 = d(g-e)
        # d = A[1,1]
        # g = B[1,0], e = B[0,0]
        U[3, 3] = 1  # d
        V[2, 3] = 1  # g = B[1,0] at index 1*2+0=2
        V[0, 3] = -1 # e = B[0,0] at index 0*2+0=0
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10
        
        # P5 = (a+b)h
        # a = A[0,0], b = A[0,1]
        # h = B[1,1]
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b = A[0,1] at index 0*2+1=1
        V[3, 4] = 1  # h = B[1,1] at index 1*2+1=3
        W[0, 4] = -1 # C00
        W[1, 4] = 1  # C01
        
        # P6 = (c-a)(e+f)
        # c = A[1,0], a = A[0,0]
        # e = B[0,0], f = B[0,1]
        U[2, 5] = 1  # c
        U[0, 5] = -1 # a
        V[0, 5] = 1  # e
        V[1, 5] = 1  # f
        W[3, 5] = 1  # C11
        
        # P7 = (b-d)(g+h)
        # b = A[0,1], d = A[1,1]
        # g = B[1,0], h = B[1,1]
        U[1, 6] = 1  # b
        U[3, 6] = -1 # d
        V[2, 6] = 1  # g
        V[3, 6] = 1  # h
        W[0, 6] = 1  # C00
        
        return U, V, W
    
    def build_schoolbook(n_sub, m_sub, p_sub):
        """
        Build schoolbook decomposition for M_{n_sub, m_sub, p_sub}.
        Returns U (n_sub*m_sub, rank), V (m_sub*p_sub, rank), W (p_sub*n_sub, rank)
        """
        rank = n_sub * m_sub * p_sub
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j, col] = 1
                    V[j * p_sub + k, col] = 1
                    W[k * n_sub + i, col] = 1
                    col += 1
        return U, V, W
    
    def combine_decomps(decomp1, decomp2, offset1, offset2):
        """
        Combine two decompositions.
        decomp1 = (U1, V1, W1) for some sub-problem
        decomp2 = (U2, V2, W2) for another sub-problem
        offset1, offset2 specify how to map indices.
        This is a placeholder - we'll handle the specific cases directly.
        """
        pass
    
    # For the full M_{3,4,7}, I need to carefully map the indices.
    # Let me take a different approach: directly construct the full decomposition
    # by enumerating all the Strassen and schoolbook components and mapping them
    # to the global index space.
    
    # Global dimensions: n=3, m=4, p=7
    # U: (12, R), V: (28, R), W: (21, R)
    
    # I'll build this step by step, tracking the column offset in the rank dimension.
    
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_component(U_sub, V_sub, W_sub, i_map, j_map, k_map, i_offset, j_offset, k_offset):
        """
        Add a sub-decomposition to the global decomposition.
        U_sub: (n_s*m_s, r) for sub-problem with dims (n_s, m_s, p_s)
        V_sub: (m_s*p_s, r)
        W_sub: (p_s*n_s, r)
        i_map: function mapping sub i to global i
        j_map: function mapping sub j to global j  
        k_map: function mapping sub k to global k
        i_offset, j_offset, k_offset: base offsets
        """
        n_s = U_sub.shape[0] // max(m_s, 1) if m_s > 0 else 0
        # This approach is getting complicated. Let me just directly build it.
        pass
    
    # Let me just directly enumerate all the rank-1 terms.
    # I'll create a list of (U_idx, V_idx, W_idx, coeff) tuples.
    
    terms = []
    
    # Process the two halves of the m dimension (4 = 2+2)
    # Half 0: j in [0, 2)
    # Half 1: j in [2, 4)
    
    for half in range(2):
        j_base = half * 2  # 0 or 2
        
        # For each half, we have M_{3,2,7}
        # Split k dimension: 7 = 3 + 3 + 1
        # k ranges: [0,3), [3,6), [6,7)
        
        for k_block_idx in range(3):
            if k_block_idx < 2:
                k_base = k_block_idx * 3  # 0 or 3
                k_size = 3
            else:
                k_base = 6
                k_size = 1
            
            if k_size == 3:
                # M_{3,2,3} = M_{3,2,2} + M_{3,2,1}
                # Split k: 3 = 2 + 1
                for k_sub_idx in range(2):
                    if k_sub_idx == 0:
                        k_sub_base = k_base
                        k_sub_size = 2
                    else:
                        k_sub_base = k_base + 2
                        k_sub_size = 1
                    
                    if k_sub_size == 2:
                        # M_{3,2,2} = M_{2,2,2} + M_{1,2,2}
                        # Split i: 3 = 2 + 1
                        for i_sub_idx in range(2):
                            if i_sub_idx == 0:
                                i_sub_base = 0
                                i_sub_size = 2
                            else:
                                i_sub_base = 2
                                i_sub_size = 1
                            
                            if i_sub_size == 2:
                                # M_{2,2,2} - use Strassen
                                U_s, V_s, W_s = build_strassen_222()
                                for r in range(7):
                                    for i_s in range(2):
                                        for j_s in range(2):
                                            for k_s in range(2):
                                                # Check if this term contributes
                                                u_val = U_s[i_s * 2 + j_s, r]
                                                v_val = V_s[j_s * 2 + k_s, r]
                                                w_val = W_s[k_s * 2 + i_s, r]
                                                if u_val != 0 and v_val != 0 and w_val != 0:
                                                    i_g = i_sub_base + i_s
                                                    j_g = j_base + j_s
                                                    k_g = k_sub_base + k_s
                                                    coeff = u_val * v_val * w_val
                                                    if coeff != 0:
                                                        terms.append((i_g * 4 + j_g, j_g * 7 + k_g, k_g * 3 + i_g, coeff))
                            else:
                                # M_{1,2,2} - schoolbook
                                # i is fixed at i_sub_base (=2)
                                for j_s in range(2):
                                    for k_s in range(2):
                                        i_g = i_sub_base
                                        j_g = j_base + j_s
                                        k_g = k_sub_base + k_s
                                        terms.append((i_g * 4 + j_g, j_g * 7 + k_g, k_g * 3 + i_g, 1))
                    else:
                        # M_{3,2,1} - schoolbook
                        for i_s in range(3):
                            for j_s in range(2):
                                i_g = i_s
                                j_g = j_base + j_s
                                k_g = k_sub_base
                                terms.append((i_g * 4 + j_g, j_g * 7 + k_g, k_g * 3 + i_g, 1))
            else:
                # M_{3,2,1} - schoolbook
                for i_s in range(3):
                    for j_s in range(2):
                        i_g = i_s
                        j_g = j_base + j_s
                        k_g = k_base
                        terms.append((i_g * 4 + j_g, j_g * 7 + k_g, k_g * 3 + i_g, 1))
    
    # Now I need to collect all unique (U_idx, V_idx, W_idx) combinations and their coefficients.
    # For a rank-1 decomposition, each term should have exactly one nonzero U, one nonzero V, one nonzero W.
    # But the Strassen terms might have multiple nonzeros in U or V for a single rank-1 component.
    # Wait, no - in the tensor decomposition, each rank-1 term has one U index, one V index, one W index.
    # The Strassen decomposition already gives us that: each of the 7 rank-1 terms has
    # specific U[i,j], V[j,k], W[k,i] entries.
    
    # Actually, I think I'm overcomplicating this. The terms list has (U_idx, V_idx, W_idx, coeff)
    # where each is a single index. Let me verify my Strassen construction is correct.
    
    # In build_strassen_222, for P1:
    # U[0,0]=1, U[3,0]=1 means both U indices 0 and 3 are 1 for rank component 0.
    # This is NOT a rank-1 term in the usual sense - it's a rank-1 term where the U vector
    # has two nonzeros. That's fine for the tensor decomposition.
    
    # So the terms should be grouped by rank component, not by individual (U_idx, V_idx, W_idx) tuples.
    # Let me restructure.
    
    # Actually, let me just build the U, V, W matrices directly by accumulating contributions.
    
    # First, let me figure out the total rank.
    # From the analysis: total rank = 80
    
    # But wait, I need to count more carefully. Let me just build it and see.
    
    # Let me restart with a cleaner approach: directly construct U, V, W matrices.
    
    R = 80  # expected total rank
    U = np.zeros((12, R), dtype=int)
    V = np.zeros((28, R), dtype=int)
    W = np.zeros((21, R), dtype=int)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff, col_ref):
        U[u_idx, col_ref] += coeff
        V[v_idx, col_ref] += coeff
        W[w_idx, col_ref] += coeff
    
    # This approach won't work because each rank-1 component can have multiple nonzeros
    # in U, V, W simultaneously.
    
    # Let me take yet another approach: just enumerate all the rank-1 components explicitly.
    
    # I'll create a list of rank-1 components, where each component is a dict with
    # 'u': {idx: val}, 'v': {idx: val}, 'w': {idx: val}
    
    components = []
    
    for half in range(2):
        j_base = half * 2
        
        for k_block_idx in range(3):
            if k_block_idx < 2:
                k_base = k_block_idx * 3
                k_size = 3
            else:
                k_base = 6
                k_size = 1
            
            if k_size == 3:
                for k_sub_idx in range(2):
                    if k_sub_idx == 0:
                        k_sub_base = k_base
                        k_sub_size = 2
                    else:
                        k_sub_base = k_base + 2
                        k_sub_size = 1
                    
                    if k_sub_size == 2:
                        for i_sub_idx in range(2):
                            if i_sub_idx == 0:
                                i_sub_base = 0
                                i_sub_size = 2
                            else:
                                i_sub_base = 2
                                i_sub_size = 1
                            
                            if i_sub_size == 2:
                                # Strassen M_{2,2,2}
                                U_s, V_s, W_s = build_strassen_222()
                                for r in range(7):
                                    u_dict = {}
                                    v_dict = {}
                                    w_dict = {}
                                    for i_s in range(2):
                                        for j_s in range(2):
                                            u_val = U_s[i_s * 2 + j_s, r]
                                            if u_val != 0:
                                                i_g = i_sub_base + i_s
                                                j_g = j_base + j_s
                                                u_idx = i_g * 4 + j_g
                                                u_dict[u_idx] = u_dict.get(u_idx, 0) + u_val
                                    for j_s in range(2):
                                        for k_s in range(2):
                                            v_val = V_s[j_s * 2 + k_s, r]
                                            if v_val != 0:
                                                j_g = j_base + j_s
                                                k_g = k_sub_base + k_s
                                                v_idx = j_g * 7 + k_g
                                                v_dict[v_idx] = v_dict.get(v_idx, 0) + v_val
                                    for k_s in range(2):
                                        for i_s in range(2):
                                            w_val = W_s[k_s * 2 + i_s, r]
                                            if w_val != 0:
                                                i_g = i_sub_base + i_s
                                                k_g = k_sub_base + k_s
                                                w_idx = k_g * 3 + i_g
                                                w_dict[w_idx] = w_dict.get(w_idx, 0) + w_val
                                    if u_dict and v_dict and w_dict:
                                        components.append({'u': u_dict, 'v': v_dict, 'w': w_dict})
                            else:
                                # M_{1,2,2} schoolbook: i fixed at 2
                                for j_s in range(2):
                                    for k_s in range(2):
                                        i_g = i_sub_base
                                        j_g = j_base + j_s
                                        k_g = k_sub_base + k_s
                                        u_idx = i_g * 4 + j_g
                                        v_idx = j_g * 7 + k_g
                                        w_idx = k_g * 3 + i_g
                                        components.append({'u': {u_idx: 1}, 'v': {v_idx: 1}, 'w': {w_idx: 1}})
                    else:
                        # M_{3,2,1} schoolbook
                        for i_s in range(3):
                            for j_s in range(2):
                                i_g = i_s
                                j_g = j_base + j_s
                                k_g = k_sub_base
                                u_idx = i_g * 4 + j_g
                                v_idx = j_g * 7 + k_g
                                w_idx = k_g * 3 + i_g
                                components.append({'u': {u_idx: 1}, 'v': {v_idx: 1}, 'w': {w_idx: 1}})
            else:
                # M_{3,2,1} schoolbook
                for i_s in range(3):
                    for j_s in range(2):
                        i_g = i_s
                        j_g = j_base + j_s
                        k_g = k_base
                        u_idx = i_g * 4 + j_g
                        v_idx = j_g * 7 + k_g
                        w_idx = k_g * 3 + i_g
                        components.append({'u': {u_idx: 1}, 'v': {v_idx: 1}, 'w': {w_idx: 1}})
    
    # Now build U, V, W from components
    R = len(components)
    U = np.zeros((12, R), dtype=int)
    V = np.zeros((28, R), dtype=int)
    W = np.zeros((21, R), dtype=int)
    
    for r, comp in enumerate(components):
        for idx, val in comp['u'].items():
            U[idx, r] = val
        for idx, val in comp['v'].items():
            V[idx, r] = val
        for idx, val in comp['w'].items():
            W[idx, r] = val
    
    # Verify the decomposition
    def verify(U, V, W):
        n, m, p = 3, 4, 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if val != 1:
                        return False, (i, j, k, val)
        return True, None
    
    valid, err = verify(U, V, W)
    
    if not valid:
        # Fall back to schoolbook
        R = 84
        U = np.zeros((12, R), dtype=int)
        V = np.zeros((28, R), dtype=int)
        W = np.zeros((21, R), dtype=int)
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(7):
                    U[i * 4 + j, col] = 1
                    V[j * 7 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
    else:
        # Use the computed rank
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
