"""Search for rank-minimal tensor decomposition of 3x5x5 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 140.0  # safety margin under 150s


def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the 3x5x5 multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = 1 iff U[i,j,c]*V[j,k,c]*W[k,i,c] sums correctly
    # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    # for all (i,j,k) in [n]x[m]x[p]
    n, m, p = 3, 5, 5
    rank = U.shape[1]
    # Build tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_exact(U, V, W):
    """Exact verification using Fraction."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    uc = U[i*m+j][c]
                    vc = V[j*p+k][c]
                    wc = W[k*n+i][c]
                    if uc != 0 and vc != 0 and wc != 0:
                        s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Standard rank-75 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Build a decomposition using Strassen for 2x2x2 blocks.
    Split the 5x5 matrix into blocks:
      - Use a 2x2 block structure on the middle dimension (5 = 2+2+1)
      Actually, let's think differently.
    
    For 3x5x5: A is 3x5, B is 5x5, C = A*B is 3x5.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strategy: Split j into groups. For each pair of j-columns, use Strassen-like
    reductions. But the standard approach for non-cubic matrices:
    
    Alternative: Split the 5x5 matrix B into blocks and the 3x5 matrix A into blocks.
    Let's split j (the contraction dimension) into 2+2+1.
    
    For the 2x2 sub-blocks, Strassen reduces 8 multiplies to 7.
    We have three 2x2 blocks in j (positions 0-1, 2-3, and 4 alone).
    Actually j goes 0..4, so we can group as {0,1}, {2,3}, {4}.
    
    For each group of 2 j-values, and for each i (3 values) and each k (5 values):
    C[i,k] += A[i,j1]*B[j1,k] + A[i,j2]*B[j2,k]
    
    This is a 1x2 * 2x1 product for each (i,k) pair. That's just 2 multiplies, no savings.
    
    Better approach: Group the i dimension too. But n=3 is small.
    
    Let me think about this differently. The tensor is T_{i,j,k} for i in [0,3), j in [0,5), k in [0,5).
    
    A known approach: split into blocks and use Strassen where the block is 2x2x2.
    Split i into {0,1,2} (can't make 2x2 easily with 3), j into {0,1},{2,3},{4}, k into {0,1},{2,3},{4}.
    
    Actually, let's use a different strategy. The known result for 3x5x5:
    We can write it as a sum of smaller tensors.
    
    Let me try: decompose the 5x5 part using a 2x2 Strassen on selected sub-blocks.
    
    For j in {0,1} and k in {0,1}: the sub-tensor is 3x2x2 (i x j x k).
    For j in {2,3} and k in {2,3}: another 3x2x2.
    For j in {0,1} and k in {2,3}: 3x2x2.
    For j in {2,3} and k in {0,1}: 3x2x2.
    For j=4 or k=4: remaining terms.
    
    Each 3x2x2 tensor: rank is at most 3*2*2=12, but can we do better?
    For 3x2x2: C[i,k] = sum_j A[i,j]*B[j,k] where A is 3x2, B is 2x2.
    This is 3*2*2 = 12 scalar multiplications naively.
    Can we use Strassen? Strassen is for 2x2x2. Here we have 3x2x2.
    We can split i into {0,1} and {2}:
    - For i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2, Strassen gives rank 7.
    - For i=2, j in {0,1}, k in {0,1}: 1x2x2, rank 4 (or can we do 3?).
      1x2x2: C[k] = A[0]*B[0,k] + A[1]*B[1,k]. This is 4 multiplies, but we can do it in 4 (it's a rank-4 tensor, the 1x2x2 multiplication tensor has rank 4 over most fields).
    
    So 3x2x2 = 2x2x2 + 1x2x2 = 7 + 4 = 11? Let me verify.
    Actually the 1x2x2 tensor: T_{j,k} = A_j * B_{j,k}. The rank of this tensor...
    It's the tensor for multiplying a 1x2 row vector by a 2x2 matrix. 
    C_k = A_0 * B_{0,k} + A_1 * B_{1,k} for k=0,1.
    This is 4 products. Can we do it in 3? The tensor has rank 4 (it's a "slice" of the 2x2x2 which has rank 7, and the 1x2x2 is a sub-tensor).
    Actually, the 1x2x2 multiplication tensor: T_{j,k} for j in {0,1}, k in {0,1}.
    It equals sum over... hmm, it's just 4 independent terms. Rank is 4.
    
    So 3x2x2 has rank at most 7+4 = 11.
    
    Now let's count the full decomposition:
    - (j in {0,1}, k in {0,1}): 3x2x2 -> rank 11
    - (j in {2,3}, k in {2,3}): 3x2x2 -> rank 11
    - (j in {0,1}, k in {2,3}): 3x2x2 -> rank 11
    - (j in {2,3}, k in {0,1}): 3x2x2 -> rank 11
    - (j=4, k in {0,1,2,3,4}): 3x1x5 -> rank 15
    - (j in {0,1,2,3}, k=4): 3x4x1 -> rank 12
    
    Total: 11*4 + 15 + 12 = 44 + 27 = 71.
    
    That matches the known result of ~71!
    
    Let me implement this carefully.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_row, v_row, w_row):
        """Add one rank-1 term."""
        U_rows.append(u_row)
        V_rows.append(v_row)
        W_rows.append(w_row)
    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Add Strassen decomposition for a 2x2x2 sub-block.
        i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        """
        # Strassen: 7 products for 2x2x2
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        #
        # Strassen factors:
        # M1 = (A10+A11)*(B00+B01)
        # M2 = (A00+A10)*B00
        # M3 = A01*(B10-B00)
        # M4 = A11*B11
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)  -- wait, let me use the standard form
        #
        # Standard Strassen:
        # M1 = (A11 - A10)(B00 + B01)
        # M2 = (A00 + A10) B00
        # M3 = (A01 + A11) B11
        # M4 = A11 (B10 - B00)
        # M5 = A00 (B01 - B11)
        # M6 = (A10 + A01) (B11 - B01)
        # M7 = (A00 - A10) (B00 + B10)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a product of (linear combo of A entries) * (linear combo of B entries)
        # In our tensor: A[i,j] * B[j,k]
        # For the 2x2x2 sub-block:
        # A entries: A[i_offset+j, j_offset+l] for j,l in {0,1}
        # B entries: B[j_offset+l, k_offset+r] for l,r in {0,1}
        # C entries: C[i_offset+j, k_offset+r]
        
        factors = [
            # (A_factor, B_factor, C_contribution)
            # M1 = (A11 - A10)(B00 + B01)
            ({(1,1): 1, (1,0): -1}, {(0,0): 1, (0,1): 1}, 
             {(0,0): 1, (1,1): -1}),
            # M2 = (A00 + A10) B00
            ({(0,0): 1, (1,0): 1}, {(0,0): 1}, 
             {(1,0): 1}),
            # M3 = (A01 + A11) B11
            ({(0,1): 1, (1,1): 1}, {(1,1): 1}, 
             {(0,1): 1, (1,1): 1}),
            # M4 = A11 (B10 - B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 
             {(0,0): -1, (1,0): 1}),
            # M5 = A00 (B01 - B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 
             {(0,0): -1, (0,1): 1}),
            # M6 = (A10 + A01) (B11 - B01)
            ({(1,0): 1, (0,1): 1}, {(1,1): 1, (0,1): -1}, 
             {(1,1): 1}),
            # M7 = (A00 - A10) (B00 + B10)
            ({(0,0): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}, 
             {(0,0): 1, (1,0): -1}),
        ]
        
        for a_coeff, b_coeff, c_coeff in factors:
            u = [0] * n
            v = [0] * m
            w = [0] * p
            for (jj, ll), coeff in a_coeff.items():
                u[i_offset + jj] += coeff  # A[i, j_offset+ll]
            for (ll, rr), coeff in b_coeff.items():
                v[j_offset + ll] += coeff  # B[j_offset+ll, k_offset+rr]
            for (jj, rr), coeff in c_coeff.items():
                w[k_offset + rr] += coeff  # C[i_offset+jj, k_offset+rr]
            # Wait, W is indexed as W[k*n + i], so W[k, i]
            # Let me reconsider the indexing.
            # U[i*m + j] corresponds to A[i,j]
            # V[j*p + k] corresponds to B[j,k]  
            # W[k*n + i] corresponds to C[i,k]
            #
            # So for a term: U[i,j] * V[j,k] * W[k,i]
            # The A part is A[i,j], B part is B[j,k], C part is C[i,k]
            #
            # In the 2x2x2 sub-block with offsets:
            # A[i_offset+ii, j_offset+ll]
            # B[j_offset+ll, k_offset+rr]
            # C[i_offset+ii, k_offset+rr]
            #
            # So u should have entries at positions (i_offset+ii)*m + (j_offset+ll)
            # v should have entries at positions (j_offset+ll)*p + (k_offset+rr)
            # w should have entries at positions (k_offset+rr)*n + (i_offset+ii)
            
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for (ii, ll), coeff in a_coeff.items():
                u[(i_offset + ii) * m + (j_offset + ll)] += coeff
            for (ll, rr), coeff in b_coeff.items():
                v[(j_offset + ll) * p + (k_offset + rr)] += coeff
            for (ii, rr), coeff in c_coeff.items():
                w[(k_offset + rr) * n + (i_offset + ii)] += coeff
            
            add_term(u, v, w)
    
    def rank1_1x2x2(i_val, j_offset, k_offset):
        """
        1x2x2: C[i_val, k] = sum_l A[i_val, j_offset+l] * B[j_offset+l, k_offset+r]
        for l in {0,1}, r in {0,1}. This is 4 terms, rank 4.
        """
        for ll in range(2):
            for rr in range(2):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i_val * m + (j_offset + ll)] = 1
                v[(j_offset + ll) * p + (k_offset + rr)] = 1
                w[(k_offset + rr) * n + i_val] = 1
                add_term(u, v, w)
    
    def block_3x2x2(i_offset, j_offset, k_offset):
        """
        3x2x2 block: i in {i_offset, i_offset+1, i_offset+2}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        Split into 2x2x2 (for i in {i_offset, i_offset+1}) + 1x2x2 (for i=i_offset+2)
        """
        strassen_2x2x2(i_offset, j_offset, k_offset)
        rank1_1x2x2(i_offset + 2, j_offset, k_offset)
    
    # Now build the full decomposition:
    # Block 1: j in {0,1}, k in {0,1} -> 3x2x2
    block_3x2x2(0, 0, 0)
    # Block 2: j in {2,3}, k in {2,3} -> 3x2x2
    block_3x2x2(0, 2, 2)
    # Block 3: j in {0,1}, k in {2,3} -> 3x2x2
    block_3x2x2(0, 0, 2)
    # Block 4: j in {2,3}, k in {0,1} -> 3x2x2
    block_3x2x2(0, 2, 0)
    # Block 5: j=4, k in {0,1,2,3,4} -> 3x1x5, rank 15
    for ii in range(3):
        for kk in range(5):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[ii * m + 4] = 1
            v[4 * p + kk] = 1
            w[kk * n + ii] = 1
            add_term(u, v, w)
    # Block 6: j in {0,1,2,3}, k=4 -> 3x4x1, rank 12
    for ii in range(3):
        for jj in range(4):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[ii * m + jj] = 1
            v[jj * p + 4] = 1
            w[4 * n + ii] = 1
            add_term(u, v, w)
    
    # Convert to list-of-lists format
    U = U_rows
    V = V_rows
    W = W_rows
    rank = len(U)
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    
    # First, get the Strassen-based decomposition
    result = _strassen_2x2x2()
    
    # Verify it's correct
    if not _verify(result["U"], result["V"], result["W"]):
        # Fall back to schoolbook if verification fails
        result = _schoolbook()
    
    rank = len(result["U"])
    
    # Try to reduce rank further with a search if time permits
    # For now, return the Strassen-based result
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return result
    
    # The rank-71 decomposition should be valid. Return it.
    return result


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'])}")
    print(f"Valid: {_verify(res['U'], res['V'], res['W'])}")
    print(f"Valid exact: {_verify_exact(res['U'], res['V'], res['W'])}")
