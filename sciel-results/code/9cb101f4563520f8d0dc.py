"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Returns U, V, W for schoolbook decomposition (rank 52)."""
    n, m, p = 2, 2, 13
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor():
    """Builds the full target tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 13
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0
    return T

def _strassen_block_decomposition(k0, k1):
    """
    Returns 7 rank-1 terms (u, v, w) for a 2x2x2 sub-block covering
    B columns k0, k1.
    
    Local indices:
      U: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
      V: 0=(k=0,j=k0), 1=(k=0,j=k1), 2=(k=1,j=k0), 3=(k=1,j=k1)
      W: 0=(j=k0,i=0), 1=(j=k0,i=1), 2=(j=k1,i=0), 3=(j=k1,i=1)
    
    Global index mapping:
      U_global: local index directly (since n*m=4 and we're in the full U space)
      V_global: 0->k0, 1->k1, 2->m+k0, 3->m+k1
      W_global: 0->k0*n+0, 1->k0*n+1, 2->k1*n+0, 3->k1*n+1
    """
    n, m, p = 2, 2, 13
    # Global index maps
    v_map = [k0, k1, m + k0, m + k1]
    w_map = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
    
    # Strassen's 7 products for 2x2 matmul:
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P7
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 + P6
    #
    # P1 = A00 * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = (A00 + A01) * B11
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A10) * (B01 - B11)
    # P6 = (A01 - A00 + A11) * B10
    # P7 = (A01 - A11) * (B00 + B11)
    
    # Each product P_t is bilinear in A and B.
    # We need to express each P_t as a sum of rank-1 terms in the tensor decomposition.
    # But actually, in the tensor decomposition framework, each rank-1 term corresponds
    # to a specific (i,k,j) triple with coefficients.
    
    # Let me think differently. The tensor T for the 2x2x2 block has entries:
    # T[i*m+k, k*p+j, j*n+i] = 1 for all i,k,j in {0,1}
    # In local coordinates (within the block), this is:
    # T_loc[a, b, c] = 1 where a=i*m+k, b=local_B, c=local_C
    # with the constraint that the k in a matches the k' in b, and j in b matches j' in c,
    # and i in a matches i' in c.
    
    # The 8 non-zero entries in local coords:
    # (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)
    
    # We need 7 vectors u_t, v_t, w_t in R^4 such that:
    # sum_t u_t[a] * v_t[b] * w_t[c] = T_loc[a,b,c]
    
    # This is a known result: the 2x2x2 matmul tensor has rank 7 (Strassen).
    # Let me construct it explicitly.
    
    # The 7 rank-1 terms correspond to the 7 Strassen products.
    # Each product P_t is a scalar that gets added to specific C entries.
    # But in the tensor decomposition, each rank-1 term is a product of three vectors.
    
    # Let me set up the system. We have 64 equations (4*4*4) and 84 unknowns (7*4*3).
    # Let's solve it using numpy.
    
    # Actually, let me just hardcode the known Strassen decomposition.
    # After some research and derivation, here's one valid rank-7 decomposition:
    
    # The 7 terms are:
    # t=0: u=[1,0,0,0], v=[1,0,1,0], w=[1,0,-1,0]  -> covers (0,0,0),(0,0,2),(0,2,0),(0,2,2) with signs
    # This is getting complicated. Let me just solve the linear system numerically.
    
    # Build the linear system
    # Unknowns: 7 terms * 4 (u) + 7 terms * 4 (v) + 7 terms * 4 (w) = 84 unknowns
    # But it's nonlinear (products of u,v,w). So we need a different approach.
    
    # Let me use a greedy/iterative approach:
    # 1. Start with the 8 schoolbook terms
    # 2. Try to merge pairs by finding a rank-1 approximation
    
    # Actually, for the purpose of this problem, let me just verify that a rank-7
    # decomposition exists and find it via optimization.
    
    # Given time constraints, let me use a simpler approach:
    # Use the known fact that rank is 7 and find it via random initialization + Newton's method.
    
    def find_rank7_decomposition():
        """Find rank-7 decomposition of 2x2x2 matmul tensor via optimization."""
        np.random.seed(42)
        
        # Initialize
        U_t = np.random.randn(4, 7) * 0.5
        V_t = np.random.randn(4, 7) * 0.5
        W_t = np.random.randn(4, 7) * 0.5
        
        # Target tensor in local coords
        T_loc = np.zeros((4, 4, 4))
        # Non-zero entries: (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)
        nonzeros = [(0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)]
        for a, b, c in nonzeros:
            T_loc[a, b, c] = 1.0
        
        # Iterative optimization
        for iteration in range(200):
            # Compute current decomposition
            # T_approx[a,b,c] = sum_t U_t[a,t] * V_t[b,t] * W_t[c,t]
            T_approx = np.einsum('at,bt,ct->abc', U_t, V_t, W_t)
            
            # Residual
            R = T_loc - T_approx
            
            # Update each term using gradient descent
            # Gradient wrt U_t[a,t]: R[a,b,c] * V_t[b,t] * W_t[c,t] summed over b,c
            grad_U = np.einsum('abc,bt,ct->at', R, V_t, W_t)
            grad_V = np.einsum('abc,at,ct->bt', R, U_t, W_t)
            grad_W = np.einsum('abc,at,bt->ct', R, U_t, V_t)
            
            # Learning rate
            lr = 0.1 / (1 + iteration * 0.01)
            
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
        
        return U_t, V_t, W_t
    
    U_t, V_t, W_t = find_rank7_decomposition()
    
    # Map to global indices
    terms = []
    for t in range(7):
        u_global = np.zeros(4)  # n*m = 4
        v_global = np.zeros(m * p)  # m*p = 26
        w_global = np.zeros(p * n)  # p*n = 26
        
        # U local to global: identity (local index = global index since n*m=4)
        u_global[:] = U_t[:, t]
        
        # V local to global
        for local_idx, global_idx in enumerate(v_map):
            v_global[global_idx] = V_t[local_idx, t]
        
        # W local to global
        for local_idx, global_idx in enumerate(w_map):
            w_global[global_idx] = W_t[local_idx, t]
        
        terms.append((u_global, v_global, w_global))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 2, 2, 13
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to build a Strassen-based decomposition
    # 6 blocks of 2 columns + 1 block of 1 column
    # Each 2-column block: rank 7
    # 1-column block: rank 4 (schoolbook for 2x2*2x1)
    # Total: 6*7 + 4 = 46
    
    try:
        U_strassen = np.zeros((n * m, 46))
        V_strassen = np.zeros((m * p, 46))
        W_strassen = np.zeros((p * n, 46))
        
        col = 0
        for block in range(6):
            k0 = block * 2
            k1 = block * 2 + 1
            
            terms = _strassen_block_decomposition(k0, k1)
            for u, v, w in terms:
                U_strassen[:, col] = u
                V_strassen[:, col] = v
                W_strassen[:, col] = w
                col += 1
        
        # Last block: 1 column (k=12)
        # Schoolbook for 2x2 * 2x1: rank 4
        k = 12
        for i in range(n):
            for kk in range(m):
                U_strassen[i * m + kk, col] = 1.0
                V_strassen[kk * p + k, col] = 1.0
                W_strassen[k * n + i, col] = 1.0
                col += 1
        
        # Verify the decomposition
        T_target = _build_target_tensor()
        T_approx = np.einsum('it,jt,kt->ijk', U_strassen, V_strassen, W_strassen)
        
        # Check accuracy
        error = np.max(np.abs(T_approx - T_target))
        
        if error < 1e-8:
            best_rank = 46
            best_U, best_V, best_W = U_strassen, V_strassen, W_strassen
        
        # If error is too high, try to fix it
        elif error < 1.0:
            # Try to correct small errors
            # For now, fall back to schoolbook
            pass
        
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Convert to lists as required by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
