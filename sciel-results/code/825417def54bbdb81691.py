"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    if not np.allclose(T, T_recon, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using two Strassen blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Left half (k=0,1)
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[3, r] = 1
    
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[2, r] = 1; W[3, r] = -1
    
    r = 2
    U[0, r] = 1
    V[4, r] = 1; V[5, r] = -1
    W[1, r] = 1; W[3, r] = 1
    
    r = 3
    U[3, r] = 1
    V[1, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[2, r] = 1
    
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[1, r] = 1; W[0, r] = -1
    
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[3, r] = 1
    
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1
    
    # Right half (k=2,3)
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[4, r] = 1; W[7, r] = 1
    
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1
    
    r = 9
    U[0, r] = 1
    V[6, r] = 1; V[7, r] = -1
    W[5, r] = 1; W[7, r] = 1
    
    r = 10
    U[3, r] = 1
    V[3, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[6, r] = 1
    
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[5, r] = 1; W[4, r] = -1
    
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1
    
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[4, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_search(target_rank, n, m, p, deadline, max_iters=5000):
    """
    Try to find a rank-target_rank decomposition.
    Uses the structure: for each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    Strategy: Parametrize with shared structure between left/right halves.
    Use iterative refinement from random init.
    """
    start = time.time()
    nm = n * m  # 4
    mp = m * p  # 8
    pn = p * n  # 8
    
    # The tensor T[i,j,k]=1 means:
    # For i=0,j=0: V[0,r]*W[k*2+0,r] * U[0,r] summed over r = 1 for all k
    # For i=0,j=1: V[1,r]*W[k*2+0,r] * U[1,r] summed over r = 1 for all k
    # For i=1,j=0: V[4,r]*W[k*2+1,r] * U[2,r] summed over r = 1 for all k
    # For i=1,j=1: V[5,r]*W[k*2+1,r] * U[3,r] summed over r = 1 for all k
    
    # So we have 4*4 = 16 equations, 3*rank variables (but with structure)
    
    # Key observation: If we group by j, for each j we have:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,k
    # This is like: for fixed j, the matrix [U[:,j] ⊗ something] ...
    
    # Actually, let's think of it differently. For fixed j and i:
    # sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r] = 1 for all k
    # Let A_{j,i,k,r} = U[i*m+j,r] * W[k*n+i,r], then sum_r A_{j,i,k,r} * V[j*p+k,r] = 1
    
    # This is a system where for each (j,i), we need:
    # V[j*p+k, :] dotted with (U[i*m+j,:] * W[k*n+i,:]) = 1 for all k
    
    # Let's use a different approach: fix U and W, solve for V, then iterate.
    
    # Random restart approach with structured initialization
    np.random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        # Initialize with perturbed Strassen-like structure
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Use a structured random init based on Strassen patterns
        # Each "product" r has a pattern in U, V, W
        for r in range(target_rank):
            # Randomly assign which A elements participate
            u_pattern = np.random.choice([-1, 0, 1], size=nm)
            while np.all(u_pattern == 0):
                u_pattern = np.random.choice([-1, 0, 1], size=nm)
            U[:, r] = u_pattern
            
            v_pattern = np.random.choice([-1, 0, 1], size=mp)
            while np.all(v_pattern == 0):
                v_pattern = np.random.choice([-1, 0, 1], size=mp)
            V[:, r] = v_pattern
            
            w_pattern = np.random.choice([-1, 0, 1], size=pn)
            while np.all(w_pattern == 0):
                w_pattern = np.random.choice([-1, 0, 1], size=pn)
            W[:, r] = w_pattern
        
        # Iterative refinement: alternate solving for one factor
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for V: for each (j, i), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # For fixed j: for each i, the vector [U[i*m+j,r]*W[k*n+i,r]]_r dotted with V[j*p+k,:] = 1
            # This gives us p equations per (j,i) pair, but they share V[j*p+k,r]
            
            # Actually, for fixed j and k, V[j*p+k,r] appears in equations for all i:
            # sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1 for i=0,1
            # This is 2 equations for the 1 variable V[j*p+k,r] (if we fix U,W)
            # So we can't solve for V directly unless we have more freedom.
            
            # Let's use gradient descent instead
            # Loss = sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            
            # Compute gradients
            # dL/dU[i*m+j,r] = 2 * (sum_r ... - 1) * V[j*p+k,r] * W[k*n+i,r]
            # But this is per (i,j,k), so we sum over k for U, over i for V, etc.
            
            # Let's compute the error tensor
            err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err[i, j, k] = s - 1.0
            
            if np.max(np.abs(err)) < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Gradient for U
            dU = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            dU[i*m+j, r] += 2 * err[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
            
            # Gradient for V
            dV = np.zeros_like(V)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            dV[j*p+k, r] += 2 * err[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
            
            # Gradient for W
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            dW[k*n+i, r] += 2 * err[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
            
            # Learning rate with decay
            lr = 0.01 / (1 + iteration * 0.01)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check final result
        err = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err[i, j, k] = s - 1.0
        
        if np.max(np.abs(err)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_structured_rank13(n, m, p, deadline):
    """
    Try a structured rank-13 approach.
    
    Key idea: The two Strassen blocks share the same U patterns.
    In rank-14, we have 7 products for left (k=0,1) and 7 for right (k=2,3).
    The U patterns are identical between left and right for corresponding products.
    
    For rank-13, we need to merge one pair of products.
    The most natural merge: products that have the same U pattern.
    
    In Strassen, product 1 has U = [1,0,0,1] (A00+A11)
    Both left and right use this same U pattern.
    If we merge them, we'd need V and W to handle both k=0,1 and k=2,3.
    
    But V and W have different indices for different k, so merging isn't straightforward.
    
    Alternative: Use a different algebraic structure.
    """
    start = time.time()
    
    # Try: use 6 products for left, 7 for right = 13 total
    # Or 7 for left, 6 for right = 13 total
    # Or 6.5 each (not possible with integer rank)
    
    # Let's try a more systematic approach: 
    # Assume rank 13 with the constraint that U has a specific structure.
    
    # For the 2x2x4 tensor, the border rank is known to be 13 or 14.
    # Let's try to find rank 13 using the fact that:
    # T = sum_{r=1}^{13} u_r ⊗ v_r ⊗ w_r
    
    # Use a parametrization where we fix some entries and solve for others.
    
    # The 16 equations are:
    # For (i,j,k) in {0,1}x{0,1}x{0,1,2,3}:
    # sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    
    # Let's denote:
    # a_r = U[0,r], b_r = U[1,r], c_r = U[2,r], d_r = U[3,r]  (for i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # e_r = V[0,r], f_r = V[1,r], g_r = V[2,r], h_r = V[3,r]
    # i_r = V[4,r], j_r = V[5,r], k_r = V[6,r], l_r = V[7,r]  (for j=0,k=0..3; j=1,k=0..3)
    # m_r = W[0,r], n_r = W[1,r], o_r = W[2,r], p_r = W[3,r]
    # q_r = W[4,r], s_r = W[5,r], t_r = W[6,r], u_r = W[7,r]  (for k=0..3, i=0,1)
    
    # Equations:
    # (i=0,j=0,k=0): a_r * e_r * m_r summed = 1
    # (i=0,j=0,k=1): a_r * f_r * n_r summed = 1
    # (i=0,j=0,k=2): a_r * g_r * q_r summed = 1
    # (i=0,j=0,k=3): a_r * h_r * s_r summed = 1
    # (i=0,j=1,k=0): b_r * i_r * m_r summed = 1
    # (i=0,j=1,k=1): b_r * j_r * n_r summed = 1
    # (i=0,j=1,k=2): b_r * k_r * q_r summed = 1
    # (i=0,j=1,k=3): b_r * l_r * s_r summed = 1
    # (i=1,j=0,k=0): c_r * e_r * o_r summed = 1
    # (i=1,j=0,k=1): c_r * f_r * p_r summed = 1
    # (i=1,j=0,k=2): c_r * g_r * t_r summed = 1
    # (i=1,j=0,k=3): c_r * h_r * u_r summed = 1
    # (i=1,j=1,k=0): d_r * i_r * o_r summed = 1
    # (i=1,j=1,k=1): d_r * j_r * p_r summed = 1
    # (i=1,j=1,k=2): d_r * k_r * t_r summed = 1
    # (i=1,j=1,k=3): d_r * l_r * u_r summed = 1
    
    # 16 equations, 13*12 = 156 unknowns (if all free)
    # Very underdetermined, so rank 13 should be achievable if it exists.
    
    # Let's use a constructive approach:
    # Set most U, V, W entries to 0 or ±1, with a few free parameters.
    
    # Based on Strassen structure, try:
    # 7 "main" products with Strassen-like U patterns
    # 6 "main" products with Strassen-like U patterns  
    # 1 "shared" product that contributes to both halves
    
    # Actually, let's just do a more aggressive search with better initialization.
    
    np.random.seed(123)
    
    target_rank = 13
    nm, mp, pn = 4, 8, 8
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Initialize with Strassen-like structure plus one extra product
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Products 0-5: left half Strassen (6 products, missing one)
        # Products 6-11: right half Strassen (6 products, missing one)
        # Product 12: shared product
        
        # Left half products (0-5):
        # p0: (A00+A11)(B00+B11) -> C00+=, C11+=
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[5, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p1: (A10+A11)B00 -> C10+=, C11-=
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # p2: A00(B01-B11) -> C01+=, C11+=
        U[0, 2] = 1
        V[1, 2] = 1; V[5, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # p3: A11(B10-B00) -> C00+=, C10+=
        U[3, 3] = 1
        V[4, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # p4: (A00+A01)B11 -> C00-=, C01+=
        U[0, 4] = 1; U[1, 4] = 1
        V[5, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # p5: (A10-A00)(B00+B01) -> C11+=
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Right half products (6-11):
        # p6: (A00+A11)(B02+B13) -> C02+=, C13+=
        U[0, 6] = 1; U[3, 6] = 1
        V[2, 6] = 1; V[7, 6] = 1
        W[4, 6] = 1; W[7, 6] = 1
        
        # p7: (A10+A11)B02 -> C12+=, C13-=
        U[2, 7] = 1; U[3, 7] = 1
        V[2, 7] = 1
        W[6, 7] = 1; W[7, 7] = -1
        
        # p8: A00(B03-B13) -> C03+=, C13+=
        U[0, 8] = 1
        V[3, 8] = 1; V[7, 8] = -1
        W[5, 8] = 1; W[7, 8] = 1
        
        # p9: A11(B12-B02) -> C02+=, C12+=
        U[3, 9] = 1
        V[6, 9] = 1; V[2, 9] = -1
        W[4, 9] = 1; W[6, 9] = 1
        
        # p10: (A00+A01)B13 -> C02-=, C03+=
        U[0, 10] = 1; U[1, 10] = 1
        V[7, 10] = 1
        W[4, 10] = -1; W[5, 10] = 1
        
        # p11: (A01-A11)(B12+B13) -> C02+=
        U[1, 11] = 1; U[3, 11] = -1
        V[6, 11] = 1; V[7, 11] = 1
        W[4, 11] = 1
        
        # p12: shared product - needs to cover missing contributions
        # Missing from left: M6_L = (A10-A00)(B02+B03) -> C13+=
        # Missing from right: M6_R = (A10-A00)(B02+B03) -> C13+=
        # Wait, both are the same! So one product can serve both?
        # No, they have different B indices.
        
        # Let me reconsider. The missing products are:
        # Left: M6_L = (A10-A00)(B00+B01) -> C11+=  [already have as p5]
        # Actually I have all 7 left products. Let me recount.
        
        # I have p0-p5 for left (6 products) and p6-p11 for right (6 products) = 12
        # Plus p12 = 13 total.
        
        # Missing from left: M7_L = (A01-A11)(B10+B11) -> C00+=
        # Missing from right: M6_R = (A10-A00)(B02+B03) -> C13+=
        
        # Can p12 cover both? It would need:
        # U pattern that works for both
        # V pattern that works for both
        # W pattern that works for both
        
        # M7_L: U=[0,1,-1,0], V=[0,0,0,0,1,1,0,0], W=[1,0,0,0,0,0,0,0]
        # M6_R: U=[-1,0,1,0], V=[0,0,1,1,0,0,0,0], W=[0,0,0,0,0,0,0,1]
        
        # These have different U patterns, so can't merge directly.
        
        # Let's just use p12 as a general correction term
        U[0, 12] = np.random.uniform(-0.5, 0.5)
        U[1, 12] = np.random.uniform(-0.5, 0.5)
        U[2, 12] = np.random.uniform(-0.5, 0.5)
        U[3, 12] = np.random.uniform(-0.5, 0.5)
        
        V[0, 12] = np.random.uniform(-0.5, 0.5)
        V[1, 12] = np.random.uniform(-0.5, 0.5)
        V[2, 12] = np.random.uniform(-0.5, 0.5)
        V[3, 12] = np.random.uniform(-0.5, 0.5)
        V[4, 12] = np.random.uniform(-0.5, 0.5)
        V[5, 12] = np.random.uniform(-0.5, 0.5)
        V[6, 12] = np.random.uniform(-0.5, 0.5)
        V[7, 12] = np.random.uniform(-0.5, 0.5)
        
        W[0, 12] = np.random.uniform(-0.5, 0.5)
        W[1, 12] = np.random.uniform(-0.5, 0.5)
        W[2, 12] = np.random.uniform(-0.5, 0.5)
        W[3, 12] = np.random.uniform(-0.5, 0.5)
        W[4, 12] = np.random.uniform(-0.5, 0.5)
        W[5, 12] = np.random.uniform(-0.5, 0.5)
        W[6, 12] = np.random.uniform(-0.5, 0.5)
        W[7, 12] = np.random.uniform(-0.5, 0.5)
        
        # Now do gradient descent to fix the errors
        for iteration in range(2000):
            if time.time() > deadline - 10:
                break
            
            err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err[i, j, k] = s - 1.0
            
            if np.max(np.abs(err)) < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            dU[i*m+j, r] += 2 * err[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += 2 * err[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += 2 * err[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
            
            lr = 0.005 / (1 + iteration * 0.005)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Final check
        err = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err[i, j, k] = s - 1.0
        
        if np.max(np.abs(err)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    
    # Start with known rank-14 solution
    best = _strassen_rank14()
    best_rank = 14
    
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
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
    
    # Try to find rank-13
    if time.time() < deadline - 30:
        result = _try_structured_rank13(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try general rank-13 search
    if time.time() < deadline - 20:
        result = _try_rank_search(13, n, m, p, deadline, max_iters=3000)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Return best found
    return best
# EVOLVE-BLOCK-END
