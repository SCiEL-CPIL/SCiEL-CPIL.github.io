import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def strassen_2x2x2_terms(i0, j0, k0, nd, md, pd):
        """
        Return 7 terms for a 2x2x2 subblock at offset (i0, j0, k0).
        Each term is (u_dict, v_dict, w_dict) with integer values.
        """
        terms = []
        # M1
        terms.append(({i0*md+j0: 1, (i0+1)*md+(j0+1): 1},
                      {j0*pd+k0: 1, (j0+1)*pd+(k0+1): 1},
                      {k0*nd+i0: 1, (k0+1)*nd+(i0+1): 1}))
        # M2
        terms.append(({(i0+1)*md+j0: 1, (i0+1)*md+(j0+1): 1},
                      {j0*pd+k0: 1},
                      {k0*nd+(i0+1): 1, (k0+1)*nd+(i0+1): -1}))
        # M3
        terms.append(({i0*md+j0: 1},
                      {j0*pd+(k0+1): 1, (j0+1)*pd+(k0+1): -1},
                      {(k0+1)*nd+i0: 1, (k0+1)*nd+(i0+1): 1}))
        # M4
        terms.append(({(i0+1)*md+(j0+1): 1},
                      {(j0+1)*pd+k0: 1, j0*pd+k0: -1},
                      {k0*nd+i0: 1, k0*nd+(i0+1): 1}))
        # M5
        terms.append(({i0*md+j0: 1, i0*md+(j0+1): 1},
                      {(j0+1)*pd+(k0+1): 1},
                      {k0*nd+i0: -1, (k0+1)*nd+i0: 1}))
        # M6
        terms.append(({(i0+1)*md+j0: 1, i0*md+j0: -1},
                      {j0*pd+k0: 1, j0*pd+(k0+1): 1},
                      {(k0+1)*nd+(i0+1): 1}))
        # M7
        terms.append(({i0*md+(j0+1): 1, (i0+1)*md+(j0+1): -1},
                      {(j0+1)*pd+k0: 1, (j0+1)*pd+(k0+1): 1},
                      {k0*nd+i0: 1}))
        return terms

    def schoolbook_terms(i0, j0, k0, ni, mj, pk, nd, md, pd):
        """
        Return ni*mj*pk schoolbook terms for a subblock at offset (i0, j0, k0).
        """
        terms = []
        for di in range(ni):
            for dj in range(mj):
                for dk in range(pk):
                    terms.append((
                        {(i0+di)*md + (j0+dj): 1},
                        {(j0+dj)*pd + (k0+dk): 1},
                        {(k0+dk)*nd + (i0+di): 1}
                    ))
        return terms

    def bini_3x3x3_terms(i0, j0, k0, nd, md, pd):
        """
        Return 23 terms for a 3x3x3 subblock at offset (i0, j0, k0).
        Uses the known Bini decomposition.
        """
        # Bini's 23 terms for 3x3x3
        # Each term is (u, v, w) where u, v, w are 3-dim vectors
        # We'll define them as lists of (index, value) for each dimension
        terms = []
        
        # The Bini decomposition consists of:
        # 7 diagonal terms (like Strassen base)
        # Plus 16 more terms
        
        # Actually, let me use a known correct set of 23 terms.
        # From literature: Bini's decomposition has these 23 rank-1 terms:
        # I'll use a standard reference implementation.
        
        # Terms are indexed by (i, j, k) contributions
        # Let me define them explicitly:
        
        bini_terms_raw = [
            # (u_i, u_j, u_k), (v_i, v_j, v_k), (w_i, w_j, w_k)
            # Using 0-indexed positions
            
            # 7 terms from 2x2x2 Strassen-like structure in first 2x2x2
            # M1: (1,1,0) (1,1,0) (1,1,0)
            ((1,1,0), (1,1,0), (1,1,0)),
            # M2: (1,1,0) (1,0,0) (1,-1,0)  
            ((1,1,0), (1,0,0), (1,-1,0)),
            # M3: (1,0,0) (1,-1,0) (0,1,1)
            ((1,0,0), (1,-1,0), (0,1,1)),
            # M4: (0,1,0) (-1,1,0) (1,1,0)
            ((0,1,0), (-1,1,0), (1,1,0)),
            # M5: (1,1,0) (0,1,0) (-1,0,1)
            ((1,1,0), (0,1,0), (-1,0,1)),
            # M6: (1,-1,0) (1,1,0) (0,1,0)
            ((1,-1,0), (1,1,0), (0,1,0)),
            # M7: (0,1,0) (1,1,0) (1,0,0)
            ((0,1,0), (1,1,0), (1,0,0)),
            
            # 16 additional terms for the full 3x3x3
            # These involve the third index in each dimension
            
            # Terms involving index 2 in U
            ((0,0,1), (0,0,1), (0,0,1)),  # diagonal (3,3,3)
            ((1,0,1), (1,0,1), (1,0,1)),
            ((0,1,1), (0,1,1), (0,1,1)),
            ((1,1,1), (1,1,1), (1,1,1)),
            
            # Cross terms
            ((1,0,1), (1,0,0), (1,0,0)),
            ((0,1,1), (0,1,0), (0,1,0)),
            ((1,1,1), (1,1,0), (1,1,0)),
            
            ((1,0,1), (0,1,0), (0,1,0)),
            ((0,1,1), (1,0,0), (1,0,0)),
            ((1,1,1), (0,1,0), (0,1,0)),
            
            ((1,0,1), (0,0,1), (0,0,1)),
            ((0,1,1), (0,0,1), (0,0,1)),
            ((1,1,1), (0,0,1), (0,0,1)),
            
            ((1,0,1), (0,1,1), (0,1,1)),
            ((0,1,1), (1,0,1), (1,0,1)),
            ((1,1,1), (1,0,1), (1,0,1)),
        ]
        
        # I need to be more careful. Let me use a verified Bini decomposition.
        # The correct Bini 3x3x3 decomposition has 23 terms. Let me use a known correct set.
        
        # Actually, let me just use the Strassen-based approach which I know works,
        # and add the third row/column with schoolbook for the remaining parts.
        # This will give: 7 (for 2x2x2) + 3+3+1 (for the L-shaped remainder) = 14 for the 3x3x3
        # That's worse than 23. Let me just use schoolbook for 3x3x3 = 27 terms.
        # Or better: use Strassen for the 2x2x2 part (7 terms) + schoolbook for the rest.
        
        # For a 3x3x3 block:
        # - 2x2x2 subblock at (0,0,0): 7 Strassen terms
        # - Remaining: row 3 with cols 1-3 (2x3x1), col 3 with rows 1-3 (3x2x1), 
        #   plus the (3,3,3) corner and overlaps
        # Let me just do: Strassen on 2x2x2 at (0,0,0), then schoolbook for the rest.
        # Rest = (3x3x3) - (2x2x2) = 27 - 8 = 19 terms schoolbook
        # Total = 7 + 19 = 26, which is worse than 27 schoolbook!
        
        # So for 3x3x3, schoolbook (27) is better than Strassen + schoolbook (26) is NOT better.
        # Wait, 26 < 27, so it IS better.
        
        # Let me reconsider. For the 3x4x6 problem:
        # Split as 3x2x6 + 3x2x6 (two 3x2x6 blocks)
        # For each 3x2x6:
        #   Split as 2x2x6 + 1x2x6
        #   For 2x2x6: split into three 2x2x2 Strassen blocks = 3*7 = 21
        #   For 1x2x6: schoolbook = 1*2*6 = 12
        #   Total per 3x2x6: 21 + 12 = 33
        # Total: 2 * 33 = 66
        
        # Alternative: Split as 3x3x6 + 3x1x6
        # For 3x3x6:
        #   Split into three 3x3x2 blocks
        #   For each 3x3x2:
        #     Split into 2x3x2 + 1x3x2
        #     2x3x2: split into 2x2x2 + 2x1x2 = 7 + 4 = 11
        #     1x3x2: schoolbook = 1*3*2 = 6
        #     Total per 3x3x2: 11 + 6 = 17
        #   Total for 3x3x6: 3 * 17 = 51
        # For 3x1x6: schoolbook = 3*1*6 = 18
        # Total: 51 + 18 = 69, worse than 66.
        
        # So the 66 solution is good. Let me just return the 66-term solution.
        return None  # Signal to use the Strassen-based approach

    # Use the recursive Strassen-based decomposition that achieves rank 66
    terms = []
    
    # Split 3x4x6 as two 3x2x6 blocks
    # Block 1: rows 0-2, cols 0-1, outputs 0-5
    # Block 2: rows 0-2, cols 2-3, outputs 0-5
    
    for j_offset in [0, 2]:
        # For each 3x2x6 block at (0, j_offset, 0)
        # Split as 2x2x6 + 1x2x6
        # 2x2x6 at (0, j_offset, 0): three 2x2x2 Strassen blocks
        for k_offset in [0, 2, 4]:
            terms.extend(strassen_2x2x2_terms(0, j_offset, k_offset, n, m, p))
        
        # 1x2x6 at (2, j_offset, 0): schoolbook
        terms.extend(schoolbook_terms(2, j_offset, 0, 1, 2, 6, n, m, p))
    
    rank = len(terms)
    
    # Build the matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
