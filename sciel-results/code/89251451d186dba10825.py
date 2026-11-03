import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _build_3x3x3_rank26():
    """
    Construct a rank-26 decomposition for 3x3 * 3x3 matrix multiplication.
    
    We partition the indices:
    i -> {0,1}, {2}
    j -> {0,1}, {2}
    k -> {0,1}, {2}
    
    The four blocks of the output C are:
    C00 (2x2) = A00 B00 + A01 B10   where A00 is 2x2, B00 is 2x1, etc.
    Actually, let's use the standard block decomposition carefully.
    
    Let's define the tensors explicitly by generating the rank-26 scheme.
    The scheme is based on:
    - 2x2x2 block (C00 from A00, B00): Rank 7 (Strassen)
    - 2x1x2 block (C00 from A01, B10): Rank 4 (standard)
    - 2x2x1 block (C01 from A00, B01): Rank 4 (standard)
    - 2x1x1 block (C01 from A01, B11): Rank 2 (standard)
    - 1x2x2 block (C10 from A10, B00): Rank 4 (standard)
    - 1x1x2 block (C10 from A11, B10): Rank 2 (standard)
    - 1x2x1 block (C11 from A10, B01): Rank 2 (standard)
    - 1x1x1 block (C11 from A11, B11): Rank 1 (standard)
    
    Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
    """
    U = []
    V = []
    W = []
    
    # Helper to add a standard term A[i,j] B[j,k] -> C[i,k]
    def add_standard(i, j, k, U, V, W):
        U.append([1] * 1 + [0] * 0) # Placeholder, will be filled properly
        V.append([1])
        W.append([1])
        # We'll build arrays at the end, so let's just store tuples
        return (i, j, k)
    
    # We'll build the matrices directly.
    # Indices: U is (9, 26), V is (9, 26), W is (9, 26)
    # U row: i*m + j = i*3 + j
    # V row: j*p + k = j*3 + k (wait, m=3, p=3 here for the sub-problem)
    # W row: k*n + i = k*3 + i
    
    # Let's just build the 26 terms.
    # Term 0-6: Strassen for A00 (2x2) * B00 (2x2) -> C00 (2x2)
    # Strassen terms for A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-d)(e+g)
    # C00 = p1 - p4 - p6 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p3 - p5 + p6
    
    # In our context for the 2x2 block (indices i in {0,1}, j in {0,1}, k in {0,1}):
    # a = A00, b = A01, c = A10, d = A11
    # e = B00, f = B01, g = B10, h = B11
    
    # Let's map these to the global 3x3 indices.
    # The 2x2 block corresponds to i in {0,1}, j in {0,1}, k in {0,1}.
    # So the "Strassen" part computes C[i,k] for i,k in {0,1} using A[i,j] and B[j,k] for j in {0,1}.
    
    # Term 0: (A00+A11)*(B00+B11)
    # U: A00 and A11 are 1. Row 0 (i=0,j=0) and Row 3 (i=1,j=1)? No, wait.
    # U row index is i*3+j.
    # A00 is row 0. A11 is row 4 (i=1, j=1).
    # B00 is row 0. B11 is row 5 (j=1, k=1).
    # W row index is k*3+i.
    # The term contributes to C00, C01, C10, C11.
    
    # This is getting complicated to do manually. Let's write a helper to generate the rank terms.
    
    terms = []
    
    # Strassen part for i in {0,1}, j in {0,1}, k in {0,1}
    # We want to compute C[i,k] = sum_j A[i,j] B[j,k] for i,k in {0,1}, j in {0,1}.
    
    # p1 = (A00+A11)*(B00+B11)
    # U: 1 at (0,0) and (1,1) -> indices 0 and 4
    # V: 1 at (0,0) and (1,1) -> indices 0 and 5
    # W: contributes to C00, C01, C10, C11 with signs.
    # C00: +1, C01: -1? No, C00 = p1 - p4 - p6 + p7.
    # So W for p1 is: C00(+1), C01(-1), C10(-1), C11(+1)?
    # Let's check Strassen again.
    # C00 = p1 - p4 - p6 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p3 - p5 + p6
    
    # Term 0 (p1):
    # U: [1, 0, 0, 0, 1, 0, 0, 0, 0] (rows 0 and 4)
    # V: [1, 0, 0, 0, 0, 1, 0, 0, 0] (rows 0 and 5)
    # W: C00(+1), C01(-1), C10(-1), C11(+1).
    # W row indices: C00 is k=0,i=0 -> 0. C01 is k=1,i=0 -> 3. C10 is k=0,i=1 -> 1. C11 is k=1,i=1 -> 4.
    # W: [1, -1, 0, -1, 1, 0, 0, 0, 0] -> indices 0, 1, 3, 4.
    terms.append(([1,0,0,0,1,0,0,0,0], [1,0,0,0,0,1,0,0,0], [1,-1,0,-1,1,0,0,0,0]))
    
    # p2 = (A10+A11)*B00
    # U: rows 3 (1,0) and 4 (1,1)
    # V: row 0 (0,0)
    # W: C10 (+1). C10 is index 1.
    # W: [0, 1, 0, 0, 0, 0, 0, 0, 0]
    terms.append(([0,0,0,1,1,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]))
    
    # p3 = A00*(B01-B11)
    # U: row 0
    # V: row 1 (0,1) and row 5 (1,1) with signs. B01 is +, B11 is -.
    # V: [0, 1, 0, 0, 0, -1, 0, 0, 0]
    # W: C01 (+1), C11 (-1).
    # C01 is index 3. C11 is index 4.
    # W: [0, 0, 0, 1, -1, 0, 0, 0, 0]
    terms.append(([1,0,0,0,0,0,0,0,0], [0,1,0,0,0,-1,0,0,0], [0,0,0,1,-1,0,0,0,0]))
    
    # p4 = A11*(B10-B00)
    # U: row 4
    # V: row 3 (1,0) and row 0 (0,0). B10 is +, B00 is -.
    # V: [-1, 0, 0, 1, 0, 0, 0, 0, 0]
    # W: C00 (-1), C10 (+1).
    # C00 is index 0. C10 is index 1.
    # W: [-1, 1, 0, 0, 0, 0, 0, 0, 0]
    terms.append(([0,0,0,0,1,0,0,0,0], [-1,0,0,1,0,0,0,0,0], [-1,1,0,0,0,0,0,0,0]))
    
    # p5 = (A00+A01)*B11
    # U: row 0 and 1
    # V: row 5
    # W: C01 (+1), C11 (-1).
    # W: [0, 0, 0, 1, -1, 0, 0, 0, 0]
    terms.append(([1,1,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,1,-1,0,0,0,0]))
    
    # p6 = (A10-A11)*(B01+B11)
    # U: row 3 (+) and 4 (-)
    # V: row 1 (+) and 5 (+)
    # W: C00 (-1), C11 (+1).
    # W: [-1, 0, 0, 0, 1, 0, 0, 0, 0]
    terms.append(([0,0,0,1,-1,0,0,0,0], [0,1,0,0,0,1,0,0,0], [-1,0,0,0,1,0,0,0,0]))
    
    # p7 = (A01-A11)*(B00+B10)
    # U: row 1 (+) and 4 (-)
    # V: row 0 (+) and 3 (+)
    # W: C00 (+1), C10 (-1).
    # W: [1, -1, 0, 0, 0, 0, 0, 0, 0]
    terms.append(([0,1,0,0,-1,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,-1,0,0,0,0,0,0,0]))
    
    # Now the standard terms for the remaining blocks.
    # C00 also has A01 B10 and A10 B01? No, C00 = A00 B00 + A01 B10 + A10 B01? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # C00 = A00 B00 + A01 B10 + A02 B20.
    # C01 = A00 B01 + A01 B11 + A02 B21.
    # C02 = A00 B02 + A01 B12 + A02 B22.
    # C10 = A10 B00 + A11 B10 + A12 B20.
    # C11 = A10 B01 + A11 B11 + A12 B21.
    # C12 = A10 B02 + A11 B12 + A12 B22.
    # C20 = A20 B00 + A21 B10 + A22 B20.
    # C21 = A20 B01 + A21 B11 + A22 B21.
    # C22 = A20 B02 + A21 B12 + A22 B22.
    
    # My Strassen part handled C00, C01, C10, C11 contributions from j in {0,1}.
    # It did NOT handle j=2 contributions.
    # It also didn't handle i=2 or k=2 contributions.
    
    # So I need to add terms for:
    # 1. j=2 contributions to C00, C01, C10, C11.
    # 2. All contributions to C02, C12, C20, C21, C22.
    
    # This is getting messy. The previous count of 26 assumed a specific block structure.
    # Let's restart the 26-rank construction with a cleaner block partition.
    
    # Partition:
    # i: {0,1}, {2}
    # j: {0,1}, {2}
    # k: {0,1}, {2}
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}. (2x2x2). Rank 7.
    # Block 2: i in {0,1}, j in {2}, k in {0,1}. (2x1x2). Rank 4.
    # Block 3: i in {0,1}, j in {0,1}, k in {2}. (2x2x1). Rank 4.
    # Block 4: i in {0,1}, j in {2}, k in {2}. (2x1x1). Rank 2.
    # Block 5: i in {2}, j in {0,1}, k in {0,1}. (1x2x2). Rank 4.
    # Block 6: i in {2}, j in {2}, k in {0,1}. (1x1x2). Rank 2.
    # Block 7: i in {2}, j in {0,1}, k in {2}. (1x2x1). Rank 2.
    # Block 8: i in {2}, j in {2}, k in {2}. (1x1x1). Rank 1.
    
    # Total: 7+4+4+2+4+2+2+1 = 26.
    
    # Let's implement this.
    
    def build_2x2x2_strassen(offset_i, offset_j, offset_k):
        """Returns 7 terms for A[offset_i:offset_i+2, offset_j:offset_j+2] * B[offset_j:offset_j+2, offset_k:offset_k+2] -> C[offset_i:offset_i+2, offset_k:offset_k+2]"""
        res = []
        # Local indices 0,1 map to global offset + local.
        # U row: (i+j)*3 + j? No, U row is i*3+j.
        # Local i, j -> Global i = offset_i + i, Global j = offset_j + j.
        # U index = (offset_i + i)*3 + (offset_j + j).
        # V index = (offset_j + j)*3 + (offset_k + k).
        # W index = (offset_k + k)*3 + (offset_i + i).
        
        # p1 = (A00+A11)(B00+B11)
        # U: (0,0), (1,1)
        u1 = [0]*9
        u1[(offset_i+0)*3 + (offset_j+0)] = 1
        u1[(offset_i+1)*3 + (offset_j+1)] = 1
        v1 = [0]*9
        v1[(offset_j+0)*3 + (offset_k+0)] = 1
        v1[(offset_j+1)*3 + (offset_k+1)] = 1
        w1 = [0]*9
        # C00: +1, C01: -1, C10: -1, C11: +1
        w1[(offset_k+0)*3 + (offset_i+0)] = 1
        w1[(offset_k+1)*3 + (offset_i+0)] = -1
        w1[(offset_k+0)*3 + (offset_i+1)] = -1
        w1[(offset_k+1)*3 + (offset_i+1)] = 1
        res.append((u1, v1, w1))
        
        # p2 = (A10+A11)B00
        u2 = [0]*9
        u2[(offset_i+1)*3 + (offset_j+0)] = 1
        u2[(offset_i+1)*3 + (offset_j+1)] = 1
        v2 = [0]*9
        v2[(offset_j+0)*3 + (offset_k+0)] = 1
        w2 = [0]*9
        w2[(offset_k+0)*3 + (offset_i+1)] = 1
        res.append((u2, v2, w2))
        
        # p3 = A00(B01-B11)
        u3 = [0]*9
        u3[(offset_i+0)*3 + (offset_j+0)] = 1
        v3 = [0]*9
        v3[(offset_j+0)*3 + (offset_k+1)] = 1
        v3[(offset_j+1)*3 + (offset_k+1)] = -1
        w3 = [0]*9
        w3[(offset_k+1)*3 + (offset_i+0)] = 1
        w3[(offset_k+1)*3 + (offset_i+1)] = -1
        res.append((u3, v3, w3))
        
        # p4 = A11(B10-B00)
        u4 = [0]*9
        u4[(offset_i+1)*3 + (offset_j+1)] = 1
        v4 = [0]*9
        v4[(offset_j+1)*3 + (offset_k+0)] = 1
        v4[(offset_j+0)*3 + (offset_k+0)] = -1
        w4 = [0]*9
        w4[(offset_k+0)*3 + (offset_i+0)] = -1
        w4[(offset_k+0)*3 + (offset_i+1)] = 1
        res.append((u4, v4, w4))
        
        # p5 = (A00+A01)B11
        u5 = [0]*9
        u5[(offset_i+0)*3 + (offset_j+0)] = 1
        u5[(offset_i+0)*3 + (offset_j+1)] = 1
        v5 = [0]*9
        v5[(offset_j+1)*3 + (offset_k+1)] = 1
        w5 = [0]*9
        w5[(offset_k+1)*3 + (offset_i+0)] = 1
        w5[(offset_k+1)*3 + (offset_i+1)] = -1
        res.append((u5, v5, w5))
        
        # p6 = (A10-A11)(B01+B11)
        u6 = [0]*9
        u6[(offset_i+1)*3 + (offset_j+0)] = 1
        u6[(offset_i+1)*3 + (offset_j+1)] = -1
        v6 = [0]*9
        v6[(offset_j+0)*3 + (offset_k+1)] = 1
        v6[(offset_j+1)*3 + (offset_k+1)] = 1
        w6 = [0]*9
        w6[(offset_k+0)*3 + (offset_i+0)] = -1
        w6[(offset_k+1)*3 + (offset_i+1)] = 1
        res.append((u6, v6, w6))
        
        # p7 = (A01-A11)(B00+B10)
        u7 = [0]*9
        u7[(offset_i+0)*3 + (offset_j+1)] = 1
        u7[(offset_i+1)*3 + (offset_j+1)] = -1
        v7 = [0]*9
        v7[(offset_j+0)*3 + (offset_k+0)] = 1
        v7[(offset_j+1)*3 + (offset_k+0)] = 1
        w7 = [0]*9
        w7[(offset_k+0)*3 + (offset_i+0)] = 1
        w7[(offset_k+0)*3 + (offset_i+1)] = -1
        res.append((u7, v7, w7))
        
        return res

    def build_2x1x2(offset_i, offset_j, offset_k):
        """A is 2x1, B is 1x2. C is 2x2.
        C00 = A00 B00
        C01 = A00 B01
        C10 = A10 B00
        C11 = A10 B01
        """
        res = []
        u0 = [0]*9; u0[(offset_i+0)*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        
        u1 = [0]*9; u1[(offset_i+0)*3 + offset_j] = 1
        v1 = [0]*9; v1[offset_j*3 + offset_k + 1] = 1
        w1 = [0]*9; w1[(offset_k+1)*3 + offset_i] = 1
        res.append((u1, v1, w1))
        
        u2 = [0]*9; u2[(offset_i+1)*3 + offset_j] = 1
        v2 = [0]*9; v2[offset_j*3 + offset_k] = 1
        w2 = [0]*9; w2[offset_k*3 + offset_i + 1] = 1
        res.append((u2, v2, w2))
        
        u3 = [0]*9; u3[(offset_i+1)*3 + offset_j] = 1
        v3 = [0]*9; v3[offset_j*3 + offset_k + 1] = 1
        w3 = [0]*9; w3[(offset_k+1)*3 + offset_i + 1] = 1
        res.append((u3, v3, w3))
        return res

    def build_2x2x1(offset_i, offset_j, offset_k):
        """A is 2x2, B is 2x1. C is 2x1.
        C00 = A00 B00 + A01 B10
        C10 = A10 B00 + A11 B10
        """
        res = []
        u0 = [0]*9; u0[(offset_i+0)*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        
        u1 = [0]*9; u1[(offset_i+0)*3 + offset_j + 1] = 1
        v1 = [0]*9; v1[(offset_j+1)*3 + offset_k] = 1
        w1 = [0]*9; w1[offset_k*3 + offset_i] = 1
        res.append((u1, v1, w1))
        
        u2 = [0]*9; u2[(offset_i+1)*3 + offset_j] = 1
        v2 = [0]*9; v2[offset_j*3 + offset_k] = 1
        w2 = [0]*9; w2[offset_k*3 + offset_i + 1] = 1
        res.append((u2, v2, w2))
        
        u3 = [0]*9; u3[(offset_i+1)*3 + offset_j + 1] = 1
        v3 = [0]*9; v3[(offset_j+1)*3 + offset_k] = 1
        w3 = [0]*9; w3[offset_k*3 + offset_i + 1] = 1
        res.append((u3, v3, w3))
        return res

    def build_2x1x1(offset_i, offset_j, offset_k):
        """A is 2x1, B is 1x1. C is 2x1.
        C00 = A00 B00
        C10 = A10 B00
        """
        res = []
        u0 = [0]*9; u0[(offset_i+0)*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        
        u1 = [0]*9; u1[(offset_i+1)*3 + offset_j] = 1
        v1 = [0]*9; v1[offset_j*3 + offset_k] = 1
        w1 = [0]*9; w1[offset_k*3 + offset_i + 1] = 1
        res.append((u1, v1, w1))
        return res

    def build_1x2x2(offset_i, offset_j, offset_k):
        """A is 1x2, B is 2x2. C is 1x2.
        C00 = A00 B00 + A01 B10
        C01 = A00 B01 + A01 B11
        """
        res = []
        u0 = [0]*9; u0[offset_i*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        
        u1 = [0]*9; u1[offset_i*3 + offset_j + 1] = 1
        v1 = [0]*9; v1[(offset_j+1)*3 + offset_k] = 1
        w1 = [0]*9; w1[offset_k*3 + offset_i] = 1
        res.append((u1, v1, w1))
        
        u2 = [0]*9; u2[offset_i*3 + offset_j] = 1
        v2 = [0]*9; v2[offset_j*3 + offset_k + 1] = 1
        w2 = [0]*9; w2[(offset_k+1)*3 + offset_i] = 1
        res.append((u2, v2, w2))
        
        u3 = [0]*9; u3[offset_i*3 + offset_j + 1] = 1
        v3 = [0]*9; v3[(offset_j+1)*3 + offset_k + 1] = 1
        w3 = [0]*9; w3[(offset_k+1)*3 + offset_i] = 1
        res.append((u3, v3, w3))
        return res

    def build_1x1x2(offset_i, offset_j, offset_k):
        """A is 1x1, B is 1x2. C is 1x2.
        C00 = A00 B00
        C01 = A00 B01
        """
        res = []
        u0 = [0]*9; u0[offset_i*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        
        u1 = [0]*9; u1[offset_i*3 + offset_j] = 1
        v1 = [0]*9; v1[offset_j*3 + offset_k + 1] = 1
        w1 = [0]*9; w1[(offset_k+1)*3 + offset_i] = 1
        res.append((u1, v1, w1))
        return res

    def build_1x2x1(offset_i, offset_j, offset_k):
        """A is 1x2, B is 2x1. C is 1x1.
        C00 = A00 B00 + A01 B10
        """
        res = []
        u0 = [0]*9; u0[offset_i*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        
        u1 = [0]*9; u1[offset_i*3 + offset_j + 1] = 1
        v1 = [0]*9; v1[(offset_j+1)*3 + offset_k] = 1
        w1 = [0]*9; w1[offset_k*3 + offset_i] = 1
        res.append((u1, v1, w1))
        return res

    def build_1x1x1(offset_i, offset_j, offset_k):
        res = []
        u0 = [0]*9; u0[offset_i*3 + offset_j] = 1
        v0 = [0]*9; v0[offset_j*3 + offset_k] = 1
        w0 = [0]*9; w0[offset_k*3 + offset_i] = 1
        res.append((u0, v0, w0))
        return res

    terms = []
    terms += build_2x2x2_strassen(0, 0, 0) # Block 1
    terms += build_2x1x2(0, 2, 0) # Block 2
    terms += build_2x2x1(0, 0, 2) # Block 3
    terms += build_2x1x1(0, 2, 2) # Block 4
    terms += build_1x2x2(2, 0, 0) # Block 5
    terms += build_1x1x2(2, 2, 0) # Block 6
    terms += build_1x2x1(2, 0, 2) # Block 7
    terms += build_1x1x1(2, 2, 2) # Block 8
    
    U = np.zeros((9, 26), dtype=float)
    V = np.zeros((9, 26), dtype=float)
    W = np.zeros((9, 26), dtype=float)
    for i, (u, v, w) in enumerate(terms):
        U[:, i] = u
        V[:, i] = v
        W[:, i] = w
    return U, V, W

def _build_3x3x1_rank9():
    """
    Construct a rank-9 decomposition for 3x3 * 3x1 matrix multiplication.
    This is just the standard schoolbook: C[i,0] = sum_j A[i,j] B[j,0].
    9 terms.
    """
    U = np.zeros((9, 9), dtype=float)
    V = np.zeros((9, 9), dtype=float)
    W = np.zeros((9, 9), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[j*3+0, col] = 1
            W[0*3+i, col] = 1
            col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 3, 3, 4
    
    # We split the output columns k into k in {0,1,2} and k=3.
    # Part 1: 3x3 * 3x3 -> Rank 26
    # Part 2: 3x3 * 3x1 -> Rank 9
    # Total Rank 35.
    
    # The U, V, W matrices for the full 3x3*3x4 problem.
    # U is (9, 35), V is (12, 35), W is (12, 35).
    # Note: V rows are j*p+k = j*4+k. W rows are k*n+i = k*3+i.
    
    U1, V1, W1 = _build_3x3x3_rank26()
    U2, V2, W2 = _build_3x3x1_rank9()
    
    U_full = np.zeros((9, 35), dtype=float)
    V_full = np.zeros((12, 35), dtype=float)
    W_full = np.zeros((12, 35), dtype=float)
    
    # Copy Part 1 (columns 0-2 of output)
    # U1 is (9, 26), V1 is (9, 26), W1 is (9, 26).
    # V1 rows correspond to j in {0,1,2}, k in {0,1,2}.
    # But in the full problem, V row index is j*4 + k.
    # So V1[j*3+k] maps to V_full[j*4+k].
    # W1 rows correspond to k in {0,1,2}, i in {0,1,2}.
    # W1[k*3+i] maps to W_full[k*3+i].
    
    for c in range(26):
        U_full[:, c] = U1[:, c]
        for r in range(9):
            j = r // 3
            k = r % 3
            V_full[j*4 + k, c] = V1[r, c]
        for r in range(9):
            k = r // 3
            i = r % 3
            W_full[k*3 + i, c] = W1[r, c]
            
    # Copy Part 2 (column 3 of output)
    # U2 is (9, 9), V2 is (9, 9), W2 is (9, 9).
    # V2 rows correspond to j in {0,1,2}, k=0.
    # In full problem, V row index is j*4 + 3.
    # So V2[j*3+0] maps to V_full[j*4 + 3].
    # W2 rows correspond to k=0, i in {0,1,2}.
    # In full problem, W row index is 3*3 + i = 9 + i.
    # So W2[0*3+i] maps to W_full[9 + i].
    
    for c in range(9):
        col_idx = 26 + c
        U_full[:, col_idx] = U2[:, c]
        for r in range(9):
            j = r // 3
            # k is always 0 in V2
            V_full[j*4 + 3, col_idx] = V2[r, c]
        for r in range(9):
            i = r % 3
            W_full[9 + i, col_idx] = W2[r, c]
            
    # Convert to lists for return
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
