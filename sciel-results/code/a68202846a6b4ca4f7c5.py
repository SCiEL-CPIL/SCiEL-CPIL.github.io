"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _decomp_2x2x2():
    """Return rank-7 decomposition of 2x2x2 matrix multiplication."""
    # Strassen's 7-term decomposition for 2x2x2
    # C = A * B where A, B are 2x2
    # Terms: (u, v, w) where C += w * (u @ A) * (B @ v) essentially
    # More precisely: C[i,k] = sum_c W[k*n+i, c] * (sum_a U[i*m+a, c] * A[i,a]) * (sum_b V[a*p+b, c] * B[a,b])
    # For n=m=p=2, indices are 0,1
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    u = [1, 1]  # A00, A11
    v = [1, 1]  # B00, B11
    w = [1, 1]  # C00, C11
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    u = [0, 1, 1]  # A10, A11 -- wait, u should be length m=2
    # Let me redo this more carefully.
    # u is a vector of length m (rows of A to sum), v is vector of length p (columns of B to sum)
    # Actually for the tensor decomposition:
    # T[i,a,k,b] = sum_c U[i,a,c] * V[a,b,c] * W[k,i,c]
    # For 2x2x2: T is 2*2*2 = 8 entries, but we want rank 7.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    terms = []
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1, C01 -= M1, C10 -= M1
    terms.append(([1,1], [1,1], [1,-1,-1,1]))  # w order: C00,C01,C10,C11 -> but W is indexed [k*n+i]
    # W[k*n+i, c]: for n=2, W[0]=C00 (k=0,i=0), W[1]=C01 (k=0,i=1), W[2]=C10 (k=1,i=0), W[3]=C11 (k=1,i=1)
    # M1 contributes +1 to C00, -1 to C01? No.
    # Let me use the standard formula:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So:
    # M1: C00+=1, C11+=1, C01-=1? No, C01 doesn't get M1 directly in standard Strassen.
    # Let me just use the known correct 7-term decomposition.
    
    terms = []
    # (u, v, w) where u has length m=2, v has length p=2, w has length p*n=4
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1, 1], [1, 1], [1, 0, 0, 1]))  # C00+=1, C11+=1
    # M2 = (A10+A11)B00
    terms.append(([0, 1, 1], [1, 0], [0, 0, 1, 1]))  # wait, u length should be 2
    terms = []
    # Let me be very careful. u length = m = 2, v length = p = 2, w length = p*n = 4
    
    # M1 = (A00+A11)(B00+B11): u=[1,1], v=[1,1], contributes to C00(+1), C11(+1)
    terms.append(([1,1], [1,1], [1,0,0,1]))
    # M2 = (A10+A11)B00: u=[0,1,1] -- NO, u must have length 2. A10 is index 1, A11 is index 1? 
    # A is 2x2: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    # u[i] for i in 0..1: u[0] multiplies A[i,0], u[1] multiplies A[i,1]? 
    # Actually U[i*m+a, c] where i is row of A, a is column of A. So u is per (i,a) pair.
    # For m=2, U has 2*2=4 entries. u_vec = U.flatten() would be length 4.
    # Similarly V has m*p = 4 entries, w has p*n = 4 entries.
    
    terms = []
    # Let me define: U_flat length n*m=4, V_flat length m*p=4, W_flat length p*n=4
    # M1 = (A00+A11)(B00+B11)
    # A00 = A[0,0], A11 = A[1,1]
    # B00 = B[0,0], B11 = B[1,1]
    # U: A[0,0]=1, A[1,1]=1 -> U_flat = [1,0,0,1] (indices: 00,01,10,11)
    # V: B[0,0]=1, B[1,1]=1 -> V_flat = [1,0,0,1] (indices: 00,01,10,11)
    # W: C00 += 1, C11 += 1 -> W_flat = [1,0,0,1] (indices: k*n+i: 00,01,10,11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)B00
    # U: A[1,0]=1, A[1,1]=1 -> U_flat = [0,0,1,1]
    # V: B[0,0]=1 -> V_flat = [1,0,0,0]
    # W: C10 += 1, C11 += 1 -> W_flat = [0,0,1,1]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,1]))
    
    # M3 = A00(B10-B11)
    # U: A[0,0]=1 -> U_flat = [1,0,0,0]
    # V: B[1,0]=1, B[1,1]=-1 -> V_flat = [0,0,1,-1]
    # W: C01 += 1, C11 += 1 -> W_flat = [0,1,0,1]
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    
    # M4 = A11(B01-B00)
    # U: A[1,1]=1 -> U_flat = [0,0,0,1]
    # V: B[0,1]=1, B[0,0]=-1 -> V_flat = [-1,1,0,0]
    # W: C00 += 1, C10 += 1 -> W_flat = [1,0,1,0]
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    
    # M5 = (A00+A01)B11
    # U: A[0,0]=1, A[0,1]=1 -> U_flat = [1,1,0,0]
    # V: B[1,1]=1 -> V_flat = [0,0,0,1]
    # W: C01 += 1, C11 += 1 -> W_flat = [0,1,0,1]
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,1]))
    
    # M6 = (A10-A00)(B00+B01)
    # U: A[1,0]=1, A[0,0]=-1 -> U_flat = [-1,0,1,0]
    # V: B[0,0]=1, B[0,1]=1 -> V_flat = [1,1,0,0]
    # W: C00 += 1, C10 += 1 -> W_flat = [1,0,1,0]
    terms.append(([-1,0,1,0], [1,1,0,0], [1,0,1,0]))
    
    # M7 = (A01-A11)(B10+B11)
    # U: A[0,1]=1, A[1,1]=-1 -> U_flat = [0,1,0,-1]
    # V: B[1,0]=1, B[1,1]=1 -> V_flat = [0,0,1,1]
    # W: C01 += 1, C11 += 1 -> W_flat = [0,1,0,1]
    terms.append(([0,1,0,-1], [0,0,1,1], [0,1,0,1]))
    
    return terms

def _embed_term(term, i0, j0, k0, n, m, p):
    """Embed a term from a sub-problem into the full problem.
    term = (u, v, w) for sub-problem of size (n_sub, m_sub, p_sub)
    i0, j0, k0 are offsets in the full matrix.
    """
    n_sub = int(np.sqrt(len(w)))  # This is wrong, w has length p_sub * n_sub
    # Better: pass dimensions
    pass

def _build_decomp(n, m, p):
    """Build a low-rank decomposition for n x m x p matrix multiplication.
    Returns list of (u, v, w) tuples.
    """
    if n == 2 and m == 2 and p == 2:
        return _decomp_2x2x2()
    
    # If any dimension is 1, use rank 1 terms
    if n == 1 and m == 1 and p == 1:
        return [([1], [1], [1])]
    
    # For general case, try to split
    # If m is even, split m
    if m % 2 == 0:
        m1, m2 = m // 2, m - m // 2
        terms1 = _build_decomp(n, m1, p)
        terms2 = _build_decomp(n, m2, p)
        # Embed terms1 in first m1 columns of A, first m1 rows of B
        all_terms = []
        for u, v, w in terms1:
            u_full = list(u) + [0] * (n * m2)
            v_full = list(v) + [0] * (m2 * p)
            w_full = list(w)
            all_terms.append((u_full, v_full, w_full))
        for u, v, w in terms2:
            u_full = [0] * (n * m1) + list(u)
            v_full = [0] * (m1 * p) + list(v)
            w_full = list(w)
            all_terms.append((u_full, v_full, w_full))
        return all_terms
    
    # If n is even, split n
    if n % 2 == 0:
        n1, n2 = n // 2, n - n // 2
        terms1 = _build_decomp(n1, m, p)
        terms2 = _build_decomp(n2, m, p)
        all_terms = []
        for u, v, w in terms1:
            u_full = list(u) + [0] * (n2 * m)
            v_full = list(v)
            # w has length p*n, split into first n1 and last n2
            w_full = list(w[:p*n1]) + [0] * (p*n2)
            all_terms.append((u_full, v_full, w_full))
        for u, v, w in terms2:
            u_full = [0] * (n1 * m) + list(u)
            v_full = list(v)
            w_full = [0] * (p*n1) + list(w[p*n1:])
            all_terms.append((u_full, v_full, w_full))
        return all_terms
    
    # If p is even, split p
    if p % 2 == 0:
        p1, p2 = p // 2, p - p // 2
        terms1 = _build_decomp(n, m, p1)
        terms2 = _build_decomp(n, m, p2)
        all_terms = []
        for u, v, w in terms1:
            u_full = list(u)
            v_full = list(v) + [0] * (m*p2)
            # w has length p*n, split into first p1 and last p2
            w_full = list(w[:p1*n]) + [0] * (p2*n)
            all_terms.append((u_full, v_full, w_full))
        for u, v, w in terms2:
            u_full = list(u)
            v_full = [0] * (m*p1) + list(v)
            w_full = [0] * (p1*n) + list(w[p1*n:])
            all_terms.append((u_full, v_full, w_full))
        return all_terms
    
    # Fallback: schoolbook
    return _schoolbook_terms(n, m, p)

def _schoolbook_terms(n, m, p):
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))
    return terms

def construct():
    n, m, p = DIMENSIONS
    
    # Try to build a low-rank decomposition
    terms = _build_decomp(n, m, p)
    
    # Build U, V, W matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = np.array(u)
        V[:, c] = np.array(v)
        W[:, c] = np.array(w)
    
    # Verify correctness
    def verify(U, V, W, n, m, p, max_samples=100):
        np.random.seed(42)
        for _ in range(max_samples):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_actual = np.zeros((n, p))
            for c in range(rank):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                C_actual += w.T * np.sum(u * A, axis=1, keepdims=True) * np.sum(B * v, axis=0, keepdims=True).T
            # More careful computation:
            C_actual = np.zeros((n, p))
            for c in range(rank):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                # M_c = (sum_a U[i,a,c] * A[i,a]) * (sum_b V[a,b,c] * B[a,b])
                # But this isn't quite right. Let me use the tensor formula directly.
                pass
            # Use direct tensor contraction
            C_actual = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += W[k * n + i, c]
                        # sum over a, b: U[i*m+a, c] * A[i,a] * V[a*p+b, c] * B[a,b]
                        # This is expensive. Let me just check with small samples.
                    C_actual[i, k] = val
            if not np.allclose(C_actual, C_expected, atol=1e-9):
                return False
        return True
    
    # Simpler verification: construct the full tensor and check
    def verify_simple(U, V, W, n, m, p):
        # Check that for all i, a, k, b: T[i,a,k,b] = delta[i,k] * 1 = sum_c U[i,a,c]*V[a,b,c]*W[k,i,c]
        # T is the matrix multiplication tensor: T[i,a,k,b] = 1 if a==b and k==i? No.
        # C[i,k] = sum_a A[i,a] * B[a,k]
        # So the tensor T has T[i,a,k,b] = 1 if a==b and i==k? No, that's not right either.
        # The tensor is: T[i,a,k,b] = 1 if a==b and i==k is wrong.
        # Actually: C[i,k] = sum_a A[i,a] B[a,k], so the tensor T[i,a,k,b] = 1 if a==b and... 
        # No, T is a 4-index tensor where T[i,a,k,b] = 1 if a==b and i==k is not correct.
        # The correct tensor: T[i,a,k,b] = 1 if a==b and... no, it's just that we sum over a.
        # T[i,a,k,b] = 1 if a==b is not right. Let me think again.
        # C[i,k] = sum_a A[i,a] * B[a,k]
        # So the bilinear form is: sum_{a} A[i,a] B[a,k]
        # The tensor T has entries T[i,a,k,b] = 1 if a==b and... no, T[i,a,k,b] = 1 if a==b is not correct.
        # Actually, the tensor is T[i,a,k,b] = 1 if a==b and i==k? No.
        # The tensor for matrix mult: T[i,a,k,b] = 1 if a==b, summed over a gives C[i,k] when k is free.
        # Wait, no. T[i,a,k,b] should be such that sum_{a,b} A[i,a] T[i,a,k,b] B[b,k] = C[i,k].
        # So T[i,a,k,b] = 1 if a==b, otherwise 0. And we sum over a and b.
        # Yes! T[i,a,k,b] = delta[a,b].
        
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    for b in range(m):
                        expected = 1.0 if a == b else 0.0
                        # But wait, C[i,k] = sum_a A[i,a] B[a,k], so we need sum_a A[i,a] B[a,k]
                        # This means T[i,a,k,b] = 1 if a==b, and we sum over a,b.
                        # But then sum_{a,b} A[i,a] T[i,a,k,b] B[b,k] = sum_{a,b} A[i,a] delta[a,b] B[b,k] = sum_a A[i,a] B[a,k]. Correct!
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i * m + a, c] * V[a * p + b, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-9:
                            return False
        return True
    
    if not verify_simple(U, V, W, n, m, p):
        # Fall back to schoolbook
        terms = _schoolbook_terms(n, m, p)
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (u, v, w) in enumerate(terms):
            U[:, c] = np.array(u)
            V[:, c] = np.array(v)
            W[:, c] = np.array(w)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
