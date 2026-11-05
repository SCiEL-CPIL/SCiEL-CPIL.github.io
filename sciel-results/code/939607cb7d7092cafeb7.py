import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Cache for decompositions
    # Key: (n, m, p)
    # Value: list of (u, v, w) tuples, where u, v, w are numpy arrays of 0/1/0.5/-0.5 etc.
    decomp_cache = {}
    
    def get_decomp(n, m, p):
        if (n, m, p) in decomp_cache:
            return decomp_cache[(n, m, p)]
            
        # Base cases
        if n == 1 and m == 1 and p == 1:
            # Rank 1
            u = np.array([1])
            v = np.array([1])
            w = np.array([1])
            res = [(u, v, w)]
            decomp_cache[(n, m, p)] = res
            return res
            
        if n == 2 and m == 2 and p == 2:
            # Strassen
            # C = AB
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B22)
            # M4 = A22(B12 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to express this as sum of rank-1 tensors u v w
            # Each Mk is a product of linear combinations.
            # This is tricky to represent directly as (u, v, w) for the whole 2x2x2 block.
            # Instead, we can just use schoolbook for 2x2x2 if we can't easily generate the 7 vectors.
            # But Strassen is better.
            # Let's generate the 7 vectors.
            
            # Indices: i in {0,1}, j in {0,1}, k in {0,1}
            # u is length 2 (i), v is length 2 (j), w is length 2 (k)
            
            # M1: (A00+A11)(B00+B11)
            # A_part: [1, 1] (for i=0,1? No, A00 is i=0,j=0; A11 is i=1,j=1)
            # This requires careful mapping.
            
            # It is easier to just use schoolbook for now if Strassen generation is complex.
            # Schoolbook rank is 8. Strassen is 7.
            # Let's implement Strassen vectors explicitly.
            
            vectors = []
            
            # M1
            # A: A00 + A11 -> u1=[1,0], v1=[1,0]? No.
            # A00 is at (0,0). A11 is at (1,1).
            # So A_part for M1 is A00 + A11.
            # This is not a rank-1 tensor in A alone.
            # Strassen works by pre-computing M's.
            # The decomposition of the tensor T is sum of 7 rank-1 tensors.
            # T = sum M_k (u_k \otimes v_k \otimes w_k) ?
            # No, T = sum_{k=1}^7 (U_k \otimes V_k \otimes W_k)
            # where U_k, V_k, W_k are vectors.
            
            # Reference:
            # u1 = [1, 0], v1 = [1, 0], w1 = [1, 0] ? No.
            
            # Let's use a known list of Strassen vectors for 2x2x2.
            # From literature:
            # 1. u=[1,0], v=[1,0], w=[1,0] (A00 B00 C00)
            # This is just schoolbook.
            
            # Actually, I will use schoolbook for 2x2x2 to be safe, as rank 8 is close to 7.
            # Or I can hardcode the 7 vectors.
            
            # Let's try to find the 7 vectors.
            # This is a standard problem.
            # I will use schoolbook for 2x2x2 for now to ensure correctness.
            # Rank 8.
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u = np.zeros(2); u[i] = 1
                        v = np.zeros(2); v[j] = 1
                        w = np.zeros(2); w[k] = 1
                        res.append((u, v, w))
            
            decomp_cache[(n, m, p)] = res
            return res

        # Recursive case
        best_rank = float('inf')
        best_split = None
        best_decomps = None
        
        # Try all splits
        for n1 in range(1, n):
            n2 = n - n1
            for m1 in range(1, m):
                m2 = m - m1
                for p1 in range(1, p):
                    p2 = p - p1
                    
                    # Get decompositions for 8 blocks
                    d111 = get_decomp(n1, m1, p1)
                    d112 = get_decomp(n1, m1, p2)
                    d121 = get_decomp(n1, m2, p1)
                    d122 = get_decomp(n1, m2, p2)
                    d211 = get_decomp(n2, m1, p1)
                    d212 = get_decomp(n2, m1, p2)
                    d221 = get_decomp(n2, m2, p1)
                    d222 = get_decomp(n2, m2, p2)
                    
                    total_rank = len(d111) + len(d112) + len(d121) + len(d122) + len(d211) + len(d212) + len(d221) + len(d222)
                    
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_split = (n1, n2, m1, m2, p1, p2)
                        best_decomps = (d111, d112, d121, d122, d211, d212, d221, d222)
                        
        if best_split:
            n1, n2, m1, m2, p1, p2 = best_split
            d111, d112, d121, d122, d211, d212, d221, d222 = best_decomps
            
            res = []
            
            # Expand d111 (n1, m1, p1) to (n, m, p)
            # u is length n1 -> pad to n (first n1)
            # v is length m1 -> pad to m (first m1)
            # w is length p1 -> pad to p (first p1)
            for u, v, w in d111:
                res.append((np.pad(u, (0, n2)), np.pad(v, (0, m2)), np.pad(w, (0, p2))))
                
            for u, v, w in d112:
                res.append((np.pad(u, (0, n2)), np.pad(v, (0, m2)), np.pad(w, (p1, 0))))
                
            for u, v, w in d121:
                res.append((np.pad(u, (0, n2)), np.pad(v, (m1, 0)), np.pad(w, (0, p2))))
                
            for u, v, w in d122:
                res.append((np.pad(u, (0, n2)), np.pad(v, (m1, 0)), np.pad(w, (p1, 0))))
                
            for u, v, w in d211:
                res.append((np.pad(u, (n1, 0)), np.pad(v, (0, m2)), np.pad(w, (0, p2))))
                
            for u, v, w in d212:
                res.append((np.pad(u, (n1, 0)), np.pad(v, (0, m2)), np.pad(w, (p1, 0))))
                
            for u, v, w in d221:
                res.append((np.pad(u, (n1, 0)), np.pad(v, (m1, 0)), np.pad(w, (0, p2))))
                
            for u, v, w in d222:
                res.append((np.pad(u, (n1, 0)), np.pad(v, (m1, 0)), np.pad(w, (p1, 0))))
            
            decomp_cache[(n, m, p)] = res
            return res
        else:
            # Fallback to schoolbook
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = np.zeros(n); u[i] = 1
                        v = np.zeros(m); v[j] = 1
                        w = np.zeros(p); w[k] = 1
                        res.append((u, v, w))
            decomp_cache[(n, m, p)] = res
            return res

    # Get the decomposition
    decomps = get_decomp(n, m, p)
    rank = len(decomps)
    
    # Construct U, V, W
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u, v, w) in enumerate(decomps):
        # u is length n, v is length m, w is length p
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        for i in range(n):
            for j in range(m):
                if u[i] != 0 and v[j] != 0:
                    U[i*m + j, c] = u[i] * v[j]
                    
        for j in range(m):
            for k in range(p):
                if v[j] != 0 and w[k] != 0:
                    V[j*p + k, c] = v[j] * w[k]
                    
        for k in range(p):
            for i in range(n):
                if w[k] != 0 and u[i] != 0:
                    W[k*n + i, c] = w[k] * u[i]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
