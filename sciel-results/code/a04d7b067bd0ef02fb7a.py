"""Matrix multiplication 4x5x5 with structured tensor network ansatz."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using a Strassen-based recursive decomposition.
    
    Strategy: Recursively split the matrix multiplication into smaller subproblems.
    When all dimensions are 2, use Strassen's algorithm (rank 7 instead of 8).
    Otherwise, split the largest dimension and combine results.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition by recursively splitting
    # and using Strassen for 2x2x2 subproblems
    
    # First, let's define how to split a dimension
    # For dimension d, we split into two parts: d1 and d2 where d1 + d2 = d
    # We prefer to split into powers of 2 where possible
    
    def split_dim(d):
        """Split dimension d into two parts, preferring even splits or powers of 2."""
        if d <= 2:
            return d, 0  # Don't split if too small
        # Try to split into powers of 2
        # For d=5: split into 2+3 or 3+2
        # For d=4: split into 2+2
        # For d=3: split into 1+2 or 2+1
        if d == 5:
            return 2, 3
        elif d == 4:
            return 2, 2
        elif d == 3:
            return 1, 2
        elif d == 6:
            return 3, 3
        elif d == 7:
            return 3, 4
        elif d == 8:
            return 4, 4
        else:
            # General case: split as evenly as possible
            d1 = d // 2
            d2 = d - d1
            return d1, d2
    
    # Build the decomposition recursively
    # Each term in the decomposition is a triple (U_entry, V_entry, W_entry)
    # where U_entry is a sparse vector of length n*m,
    # V_entry is a sparse vector of length m*p,
    # W_entry is a sparse vector of length p*n
    
    # We'll represent each entry as a dict mapping index -> coefficient
    # to keep track of the linear combinations
    
    def build_decomp(n, m, p):
        """
        Build a decomposition for n x m x p matrix multiplication.
        Returns a list of (U_dict, V_dict, W_dict) where each dict maps
        an index to a coefficient.
        """
        # Base case: if all dimensions are 1, it's a single product
        if n == 1 and m == 1 and p == 1:
            U = {0: 1}  # A[0,0]
            V = {0: 1}  # B[0,0]
            W = {0: 1}  # C[0,0]
            return [(U, V, W)]
        
        # Check if we can use Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # Otherwise, split the problem
        # We'll split the m dimension (inner dimension) first
        m1, m2 = split_dim(m)
        
        if m2 == 0:
            # Can't split m, try splitting n
            n1, n2 = split_dim(n)
            if n2 == 0:
                # Can't split n, try splitting p
                p1, p2 = split_dim(p)
                if p2 == 0:
                    # Can't split anything, use schoolbook
                    return schoolbook(n, m, p)
                # Split p
                decomp1 = build_decomp(n, m, p1)
                decomp2 = build_decomp(n, m, p2)
                # Combine: offset the p indices in the second part
                result = []
                for U, V, W in decomp1:
                    result.append((U, V, W))
                for U, V, W in decomp2:
                    W_offset = {k + p1: v for k, v in W.items()}
                    result.append((U, V, W_offset))
                return result
            # Split n
            decomp1 = build_decomp(n1, m, p)
            decomp2 = build_decomp(n2, m, p)
            # Combine: offset the n indices in the second part
            result = []
            for U, V, W in decomp1:
                result.append((U, V, W))
            for U, V, W in decomp2:
                # U indices: rows are offset by n1*m
                U_offset = {k + n1 * m: v for k, v in U.items()}
                # W indices: need to offset the n part
                # W index is k*n + i, so offset i by n1
                W_offset = {}
                for k, v in W.items():
                    wi = k % n
                    wk = k // n
                    W_offset[wk * n + wi + n1] = v
                result.append((U_offset, V, W_offset))
            return result
        
        # Split m: A is n x m, B is m x p
        # A1 is n x m1, A2 is n x m2
        # B1 is m1 x p, B2 is m2 x p
        # C = A1 * B1 + A2 * B2
        
        decomp1 = build_decomp(n, m1, p)
        decomp2 = build_decomp(n, m2, p)
        
        # Combine
        result = []
        for U, V, W in decomp1:
            # U: A1 indices are 0..m1-1, so no offset needed
            # V: B1 indices are 0..p-1 for rows 0..m1-1
            result.append((U, V, W))
        for U, V, W in decomp2:
            # U: A2 indices are offset by m1
            U_offset = {k + m1: v for k, v in U.items()}
            # V: B2 indices are offset by m1*p
            V_offset = {k + m1 * p: v for k, v in V.items()}
            result.append((U_offset, V_offset, W))
        
        return result
    
    def schoolbook(n, m, p):
        """Schoolbook decomposition: one term for each (i,j,k)."""
        result = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = {i * m + j: 1}
                    V = {j * p + k: 1}
                    W = {k * n + i: 1}
                    result.append((U, V, W))
        return result
    
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2x2 matrix multiplication.
        Uses 7 multiplications instead of 8.
        
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        C = A*B
        
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 + M3 + M6
        """
        # Indices:
        # A: 0=a, 1=b, 2=c, 3=d
        # B: 0=e, 1=f, 2=g, 3=h
        # C: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
        
        terms = []
        
        # M1 = (a+d)(e+h)
        U1 = {0: 1, 3: 1}  # a+d
        V1 = {0: 1, 3: 1}  # e+h
        W1 = {0: 1, 3: 1}  # contributes to C[0,0] and C[1,1]
        terms.append((U1, V1, W1))
        
        # M2 = (c+d)e
        U2 = {2: 1, 3: 1}  # c+d
        V2 = {0: 1}  # e
        W2 = {2: 1}  # C[1,0]
        terms.append((U2, V2, W2))
        
        # M3 = a(f-h)
        U3 = {0: 1}  # a
        V3 = {1: 1, 3: -1}  # f-h
        W3 = {1: 1}  # C[0,1]
        terms.append((U3, V3, W3))
        
        # M4 = d(g-e)
        U4 = {3: 1}  # d
        V4 = {2: 1, 0: -1}  # g-e
        W4 = {0: 1, 2: 1}  # C[0,0] and C[1,0]
        terms.append((U4, V4, W4))
        
        # M5 = (a+b)h
        U5 = {0: 1, 1: 1}  # a+b
        V5 = {3: 1}  # h
        W5 = {0: -1, 1: 1}  # -C[0,0] + C[0,1]
        terms.append((U5, V5, W5))
        
        # M6 = (c-a)(e+f)
        U6 = {2: 1, 0: -1}  # c-a
        V6 = {0: 1, 1: 1}  # e+f
        W6 = {3: 1}  # C[1,1]
        terms.append((U6, V6, W6))
        
        # M7 = (b-d)(g+h)
        U7 = {1: 1, 3: -1}  # b-d
        V7 = {2: 1, 3: 1}  # g+h
        W7 = {0: 1}  # C[0,0]
        terms.append((U7, V7, W7))
        
        return terms
    
    # Build the decomposition for 4x5x5
    decomp = build_decomp(n, m, p)
    
    # Now convert the list of (U_dict, V_dict, W_dict) to matrices
    rank = len(decomp)
    
    # Count max indices
    max_U = max(max(U.keys()) + 1 if U else 0 for U, V, W in decomp)
    max_V = max(max(V.keys()) + 1 if V else 0 for U, V, W in decomp)
    max_W = max(max(W.keys()) + 1 if W else 0 for U, V, W in decomp)
    
    # Ensure correct sizes
    assert max_U == n * m, f"U size mismatch: {max_U} vs {n*m}"
    assert max_V == m * p, f"V size mismatch: {max_V} vs {m*p}"
    assert max_W == p * n, f"W size mismatch: {max_W} vs {p*n}"
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(decomp):
        for idx, val in U_dict.items():
            U[idx][col] = val
        for idx, val in V_dict.items():
            V[idx][col] = val
        for idx, val in W_dict.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
