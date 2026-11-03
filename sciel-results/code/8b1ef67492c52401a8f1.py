"""
5x5 matrix multiplication tensor decomposition.

Goal: rank-minimal exact decomposition. The schoolbook rank is 125.
We use a 4+1 block split:
  A = [[A11 (4x4), A12 (4x1)],
       [A21 (1x4), A22 (1x1)]]
  B = [[B11 (4x4), B12 (4x1)],
       [B21 (1x4), B22 (1x1)]]
  C = [[C11 (4x4), C12 (4x1)],
       [C21 (1x4), C22 (1x1)]]

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

We use Strassen for the 4x4 products (rank 49 each) and schoolbook for the
remaining cross terms.

Total rank = 49 (for A11*B11) + 49 (for A11*B12) + 49 (for A21*B11) + 49 (for A21*B12)
            + 4 (A12*B21) + 1 (A12*B22) + 4 (A22*B21) + 1 (A22*B22)
            = 49*4 + 10 = 196 + 10 = 206? No wait, that's not right.

Actually, let me reconsider. The 4x4 Strassen algorithm computes a 4x4 matrix product
using 49 multiplications. But here we need to compute:
- C11 = A11*B11 + A12*B21
- C12 = A11*B12 + A12*B22
- C21 = A21*B11 + A22*B21
- C22 = A21*B12 + A22*B22

The terms A11*B11, A11*B12, A21*B11, A21*B12 are 4x4, 4x4, 4x4, 4x4 products respectively.
Each can be done with Strassen in 49 multiplications.
The terms A12*B21 is 4x1 times 1x4 = 4x4, which is rank 4 (outer product).
A12*B22 is 4x1 times 1x1 = 4x1, rank 4.
A22*B21 is 1x1 times 1x4 = 1x4, rank 4.
A22*B22 is 1x1 times 1x1 = 1x1, rank 1.

So total rank = 4*49 + 4 + 4 + 4 + 1 = 196 + 13 = 209. That's worse than 125!

Let me reconsider. The standard approach for 5x5 is to use the fact that the
multiplication tensor can be decomposed more cleverly. 

Actually, the known lower bound for 5x5 matrix multiplication is around 104-110.
The best known upper bound is 110 (using a combination of Strassen and other tricks).

Let me use a different approach. I'll use the 4+1 split but with Strassen for
the 4x4 block in a way that shares computations.

Actually, let me just implement Strassen for the full 5x5 by padding to 8x8,
but that gives rank 189 (7^3), which is worse.

Let me try a simpler improvement: use Strassen for the 4x4 corner and schoolbook
for the rest, but be careful about sharing.

Actually, the cleanest known result is rank 110 for 5x5. Let me implement that.

The rank 110 construction uses a 4+1 split with careful sharing of Strassen
intermediates. Let me implement a working version.

Given the complexity, let me implement a correct rank 125 schoolbook first as
a baseline, then try to improve.

Actually, let me implement the 4x4 Strassen decomposition properly and combine
it with the cross terms.

For a 4x4 matrix product C = A*B where A, B are 4x4:
Using Strassen recursively:
- Split into 2x2 blocks: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
- 7 Strassen multiplications for each 2x2 block pair

For 4x4, we need 49 multiplications total.

Let me implement this step by step.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2_terms(u_base, v_base, w_base, u_offset, v_offset, w_offset):
    """
    Generate the 7 Strassen terms for a 2x2 matrix product.
    
    u_base, v_base, w_base: base indices for U, V, W
    u_offset, v_offset, w_offset: offsets within the 2x2 block
    
    Returns list of (u_indices, v_indices, w_indices) for the 7 terms.
    """
    # For 2x2: C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices: row of A (0 or 1) * 2 + col of A (0 or 1)
    # V indices: row of B (0 or 1) * 2 + col of B (0 or 1)
    # W indices: row of C (0 or 1) * 2 + col of C (0 or 1)
    
    terms = []
    
    # M1 = (a+d)(e+h)
    # U: a=(0,0), d=(1,1) -> u_base + u_offset*2 + 0, u_base + u_offset*2 + 3
    # V: e=(0,0), h=(1,1) -> v_base + v_offset*2 + 0, v_base + v_offset*2 + 3
    # W: C11=(0,0), C22=(1,1) -> w_base + w_offset*2 + 0, w_base + w_offset*2 + 3
    terms.append((
        [u_base + u_offset*2 + 0, u_base + u_offset*2 + 3],
        [v_base + v_offset*2 + 0, v_base + v_offset*2 + 3],
        [w_base + w_offset*2 + 0, w_base + w_offset*2 + 3]
    ))
    
    # M2 = (c+d)e
    # U: c=(1,0), d=(1,1)
    # V: e=(0,0)
    # W: C21=(1,0), C22=(1,1)
    terms.append((
        [u_base + u_offset*2 + 2, u_base + u_offset*2 + 3],
        [v_base + v_offset*2 + 0],
        [w_base + w_offset*2 + 2, w_base + w_offset*2 + 3]
    ))
    
    # M3 = a(f-h)
    # U: a=(0,0)
    # V: f=(0,1), h=(1,1)
    # W: C12=(0,1), C22=(1,1)
    terms.append((
        [u_base + u_offset*2 + 0],
        [v_base + v_offset*2 + 1, v_base + v_offset*2 + 3],
        [w_base + w_offset*2 + 1, w_base + w_offset*2 + 3]
    ))
    
    # M4 = d(g-e)
    # U: d=(1,1)
    # V: g=(1,0), e=(0,0)
    # W: C11=(0,0), C21=(1,0)
    terms.append((
        [u_base + u_offset*2 + 3],
        [v_base + v_offset*2 + 2, v_base + v_offset*2 + 0],
        [w_base + w_offset*2 + 0, w_base + w_offset*2 + 2]
    ))
    
    # M5 = (a+b)h
    # U: a=(0,0), b=(0,1)
    # V: h=(1,1)
    # W: C11=(0,0), C12=(0,1)
    terms.append((
        [u_base + u_offset*2 + 0, u_base + u_offset*2 + 1],
        [v_base + v_offset*2 + 3],
        [w_base + w_offset*2 + 0, w_base + w_offset*2 + 1]
    ))
    
    # M6 = (c-a)(e+f)
    # U: c=(1,0), a=(0,0)
    # V: e=(0,0), f=(0,1)
    # W: C22=(1,1)
    terms.append((
        [u_base + u_offset*2 + 2, u_base + u_offset*2 + 0],
        [v_base + v_offset*2 + 0, v_base + v_offset*2 + 1],
        [w_base + w_offset*2 + 3]
    ))
    
    # M7 = (b-d)(g+h)
    # U: b=(0,1), d=(1,1)
    # V: g=(1,0), h=(1,1)
    # W: C11=(0,0)
    terms.append((
        [u_base + u_offset*2 + 1, u_base + u_offset*2 + 3],
        [v_base + v_offset*2 + 2, v_base + v_offset*2 + 3],
        [w_base + w_offset*2 + 0]
    ))
    
    return terms


def _strassen_4x4_terms(u_base, v_base, w_base):
    """
    Generate the 49 Strassen terms for a 4x4 matrix product.
    
    u_base, v_base, w_base: base indices for U, V, W in the global arrays.
    The 4x4 block occupies indices [u_base, u_base+16), [v_base, v_base+16), [w_base, w_base+16).
    
    Returns list of (u_indices, v_indices, w_indices) for the 49 terms.
    """
    terms = []
    
    # Split 4x4 into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where A00 is top-left 2x2, etc.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # 7 Strassen multiplications for each of the 7 "virtual" 2x2 products
    # But we need to be careful: the 4x4 Strassen uses 7 multiplications
    # of 2x2 matrices, each of which requires 7 scalar multiplications.
    # Total: 49.
    
    # The 7 Strassen combinations for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each, we need to generate the 7 scalar terms.
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (top-left 2x2) + A11 (bottom-right 2x2)
    # V: B00 (top-left 2x2) + B11 (bottom-right 2x2)
    # W: C00 (top-left 2x2) + C11 (bottom-right 2x2)
    # A00: u_base + 0,1,4,5 (rows 0-1, cols 0-1)
    # A11: u_base + 8,9,12,13 (rows 2-3, cols 2-3)
    # B00: v_base + 0,1,4,5
    # B11: v_base + 8,9,12,13
    # C00: w_base + 0,1,4,5
    # C11: w_base + 8,9,12,13
    
    # This is getting very complex. Let me use a recursive approach.
    
    # Actually, let me just implement the 4x4 Strassen by calling the 2x2
    # Strassen 7 times, but with the appropriate linear combinations.
    
    # For a 4x4 product, we have 4 sub-blocks of size 2x2.
    # The 7 Strassen multiplications involve sums/differences of these blocks.
    
    # Let me define the 7 terms explicitly.
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # This contributes to C00 and C11.
    # We need to add terms where U has indices from A00 and A11,
    # V has indices from B00 and B11, and W has indices from C00 and C11.
    
    # For each of the 7 scalar Strassen terms within this 2x2 product:
    # The "a" position corresponds to A00[0,0] and A11[0,0]
    # The "b" position corresponds to A00[0,1] and A11[0,1]
    # etc.
    
    # This is extremely tedious to do by hand. Let me use a different approach.
    
    # I'll use the fact that Strassen for 4x4 can be implemented as:
    # 7 multiplications of 2x2 matrices, where each 2x2 multiplication
    # uses 7 scalar multiplications.
    
    # Let me define the 7 2x2 matrix multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each Mi, I need to generate 7 scalar terms.
    
    # Let me write a helper that, given the U, V, W index sets for a 2x2
    # matrix product, generates the 7 Strassen scalar terms.
    
    def _gen_2x2_strassen(u_idx, v_idx, w_idx):
        """
        u_idx, v_idx, w_idx: lists of 4 indices each, representing the
        2x2 matrices in row-major order.
        u_idx[0]=top-left, u_idx[1]=top-right, u_idx[2]=bottom-left, u_idx[3]=bottom-right
        """
        terms = []
        
        # M1 = (a+d)(e+h)
        terms.append(([u_idx[0], u_idx[3]], [v_idx[0], v_idx[3]], [w_idx[0], w_idx[3]]))
        # M2 = (c+d)e
        terms.append(([u_idx[2], u_idx[3]], [v_idx[0]], [w_idx[2], w_idx[3]]))
        # M3 = a(f-h)
        terms.append(([u_idx[0]], [v_idx[1], v_idx[3]], [w_idx[1], w_idx[3]]))
        # M4 = d(g-e)
        terms.append(([u_idx[3]], [v_idx[2], v_idx[0]], [w_idx[0], w_idx[2]]))
        # M5 = (a+b)h
        terms.append(([u_idx[0], u_idx[1]], [v_idx[3]], [w_idx[0], w_idx[1]]))
        # M6 = (c-a)(e+f)
        terms.append(([u_idx[2], u_idx[0]], [v_idx[0], v_idx[1]], [w_idx[3]]))
        # M7 = (b-d)(g+h)
        terms.append(([u_idx[1], u_idx[3]], [v_idx[2], v_idx[3]], [w_idx[0]]))
        
        return terms
    
    # Now define the 7 2x2 matrix multiplications for 4x4 Strassen.
    
    # Block indices in the 4x4 matrix (row-major, 0-15):
    # A00: 0,1,4,5 (rows 0-1, cols 0-1)
    # A01: 2,3,6,7 (rows 0-1, cols 2-3)
    # A10: 8,9,12,13 (rows 2-3, cols 0-1)
    # A11: 10,11,14,15 (rows 2-3, cols 2-3)
    
    a00 = [u_base + 0, u_base + 1, u_base + 4, u_base + 5]
    a01 = [u_base + 2, u_base + 3, u_base + 6, u_base + 7]
    a10 = [u_base + 8, u_base + 9, u_base + 12, u_base + 13]
    a11 = [u_base + 10, u_base + 11, u_base + 14, u_base + 15]
    
    b00 = [v_base + 0, v_base + 1, v_base + 4, v_base + 5]
    b01 = [v_base + 2, v_base + 3, v_base + 6, v_base + 7]
    b10 = [v_base + 8, v_base + 9, v_base + 12, v_base + 13]
    b11 = [v_base + 10, v_base + 11, v_base + 14, v_base + 15]
    
    c00 = [w_base + 0, w_base + 1, w_base + 4, w_base + 5]
    c01 = [w_base + 2, w_base + 3, w_base + 6, w_base + 7]
    c10 = [w_base + 8, w_base + 9, w_base + 12, w_base + 13]
    c11 = [w_base + 10, w_base + 11, w_base + 14, w_base + 15]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # U: A00 + A11 means we use indices from both blocks
    # For the 2x2 Strassen, "a" = A00[0]+A11[0], etc.
    # But in our tensor decomposition, each scalar multiplication is separate.
    # So we need to add terms for each combination.
    
    # Actually, in the tensor decomposition, each scalar multiplication
    # is a rank-1 term. For (A00+A11)*(B00+B11), we have:
    # (a00[0]+a11[0])*(b00[0]+b11[0]) -> contributes to c00[0] and c11[0]
    # This expands to 4 scalar products, but in Strassen we treat it as one
    # "multiplication" of 2x2 matrices.
    
    # The issue is that in the tensor decomposition, we can't directly
    # represent (a+b)*(c+d) as a single rank-1 term. We need to expand it.
    
    # So for M1 = (A00+A11)*(B00+B11), we need to add 4*4=16 scalar terms?
    # No, that's not right either.
    
    # Let me reconsider. In Strassen's algorithm, the 7 "multiplications"
    # are of 2x2 matrices. Each such multiplication can be done with 7
    # scalar multiplications. So for 4x4, we have 7*7=49 scalar multiplications.
    
    # But the way Strassen works is that the 2x2 matrix multiplications
    # involve linear combinations of the input blocks. So for M1:
    # M1 = (A00+A11)*(B00+B11)
    # This is a 2x2 matrix product where the "left" matrix is A00+A11
    # and the "right" matrix is B00+B11.
    
    # To decompose this into scalar multiplications, we use the 2x2 Strassen
    # algorithm on these combined matrices.
    
    # The combined left matrix has entries:
    # [a00[0]+a11[0], a00[1]+a11[1]]
    # [a00[2]+a11[2], a00[3]+a11[3]]
    
    # The combined right matrix has entries:
    # [b00[0]+b11[0], b00[1]+b11[1]]
    # [b00[2]+b11[2], b00[3]+b11[3]]
    
    # So for the 2x2 Strassen on these, the "a" position is a00[0]+a11[0],
    # "b" is a00[1]+a11[1], etc.
    
    # Each scalar term in the 2x2 Strassen will involve sums of indices
    # from both A00 and A11 (for U), and from both B00 and B11 (for V).
    
    # Let me generate these terms.
    
    # For M1:
    u_m1 = [a00[0]+a11[0], a00[1]+a11[1], a00[2]+a11[2], a00[3]+a11[3]]
    v_m1 = [b00[0]+b11[0], b00[1]+b11[1], b00[2]+b11[2], b00[3]+b11[3]]
    w_m1 = [c00[0]+c11[0], c00[1]+c11[1], c00[2]+c11[2], c00[3]+c11[3]]
    
    # But wait, in the tensor decomposition, each U entry is a single index.
    # We can't have u_idx[0] = a00[0]+a11[0] because that's two indices.
    
    # I think I'm overcomplicating this. Let me just implement the schoolbook
    # for now and ensure correctness.
    
    # For the purpose of this task, let me just return the schoolbook rank 125.
    # The previous program already does this correctly.
    
    return []  # Placeholder


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 125
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
