import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Build the matrix multiplication tensor T of shape (9, 18, 18)
    # T[u, v, w] = 1 if and only if there exist i, j, k such that
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def verify(U, V, W):
        """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T"""
        R = U.shape[1]
        recon = np.zeros((nm, mp, pn), dtype=float)
        for c in range(R):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(recon, T, atol=1e-10)
    
    def build_schoolbook():
        R = 54
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W
    
    # Try to build a rank-48 decomposition using Laderman's rank-24 algorithm
    # for 3x3x3, applied to two blocks.
    # 
    # Laderman's algorithm for 3x3 matrix multiplication uses 24 products.
    # The key insight: if we have a rank-R decomposition for 3x3x3, we can
    # get a rank-2R decomposition for 3x3x6 by treating the 6 columns as
    # two independent 3-column blocks.
    #
    # For 3x3x3: T3[i,j,k] = 1 iff u=i*3+j, v=j*3+k, w=k*3+i
    # For 3x3x6: T6[i,j,k] = 1 iff u=i*3+j, v=j*6+k, w=k*3+i
    #   Block 1 (k=0,1,2): v is in [0,9), w is in [0,9)
    #   Block 2 (k=3,4,5): v is in [9,18), w is in [9,18)
    #
    # So we can reuse the same U for both blocks, and split V and W.
    
    # Let's try to find a valid rank-24 algorithm for 3x3.
    # I'll use a known construction. Let me try the one from Laderman.
    # 
    # Actually, let me try a different approach: use a verified rank-24 
    # algorithm. I'll construct it carefully.
    #
    # One known rank-24 algorithm for 3x3 (from various sources):
    # Let me use the formulation where we have 24 triples (a_form, b_form, c_form)
    # such that sum over all triples of a_form * b_form * c_form = (A*B)_{ik}
    
    # I'll try to use a systematic approach. Let me define the 24 products.
    # Based on Laderman's paper, the 24 products can be organized as follows:
    
    # For 3x3, let A = [a_ij], B = [b_jk], C = [c_ik]
    # The 24 products M_t = (sum_i alpha_t(i) * a_i) * (sum_jk beta_t(j,k) * b_jk) * (sum_ik gamma_t(i,k) * c_ik)
    
    # This is complex. Let me try a different strategy:
    # Use the fact that rank(3x3x6) <= 2 * rank(3x3x3).
    # If I can find ANY valid rank-24 decomposition for 3x3, I'm done.
    
    # Let me try to construct one using a known method.
    # I'll use the "Laderman" algorithm with the following 24 triples.
    # Each triple is (U_vec, V_vec, W_vec) where these are vectors of length 9.
    
    # Actually, let me just try to solve this more directly.
    # I'll use a random search with verification for rank 24 on 3x3.
    # If it fails quickly, I'll fall back to schoolbook.
    
    def try_laderman_24():
        """Try to construct a rank-24 decomposition for 3x3x3."""
        R = 24
        U3 = np.zeros((9, R), dtype=float)
        V3 = np.zeros((9, R), dtype=float)
        W3 = np.zeros((9, R), dtype=float)
        
        # I'll use a known set of 24 products for Laderman's algorithm.
        # The following is based on the Laderman algorithm as described in
        # various references. Each product t has forms for A, B, and C.
        
        # Let me use a simpler approach: I'll try to verify a specific set.
        # If it doesn't work, return None.
        
        # Product 1: (a11+a22+a33)(b11+b22+b33)(c11+c22+c33) - no, this is wrong.
        # The C forms are linear combinations of c_ik, not products.
        
        # Let me use a different known rank-24 algorithm.
        # I'll use the one from the "Tensor rank" literature.
        
        # Actually, I realize I should just try to find a valid decomposition
        # by using a smarter construction. Let me use the fact that:
        # rank(3x3x3) <= 24 is known, so such a decomposition exists.
        # But I need to actually construct it.
        
        # Let me try a random search with a time limit.
        start = time.time()
        max_time = 30  # seconds
        
        for attempt in range(100):
            if time.time() - start > max_time:
                break
            
            np.random.seed(attempt)
            U3 = np.random.randn(9, R)
            V3 = np.random.randn(9, R)
            W3 = np.random.randn(9, R)
            
            # Check if this is a valid decomposition for 3x3x3
            T3 = np.zeros((9, 9, 9), dtype=float)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        T3[i*3+j, j*3+k, k*3+i] = 1.0
            
            recon = np.zeros((9, 9, 9), dtype=float)
            for c in range(R):
                recon += np.einsum('i,j,k->ijk', U3[:, c], V3[:, c], W3[:, c])
            
            if np.allclose(recon, T3, atol=1e-10):
                return U3, V3, W3
        
        return None
    
    # Try the random search for rank 24
    result = try_laderman_24()
    
    if result is not None:
        U3, V3, W3 = result
        # Build the rank-48 decomposition for 3x3x6
        R = 48
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        
        # First block (k=0,1,2)
        U[:, 0:24] = U3
        V[0:9, 0:24] = V3
        V[9:18, 0:24] = 0
        W[0:9, 0:24] = W3
        W[9:18, 0:24] = 0
        
        # Second block (k=3,4,5)
        U[:, 24:48] = U3
        V[0:9, 24:48] = 0
        V[9:18, 24:48] = V3
        W[0:9, 24:48] = 0
        W[9:18, 24:48] = W3
        
        # Verify
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
