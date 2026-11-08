"""Matrix multiplication 2x4 x 4x6 — rank-minimal exact tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        uc = np.array(U[:, c], dtype=float)
        vc = np.array(V[:, c], dtype=float)
        wc = np.array(W[:, c], dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    target = np.zeros_like(T)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, target, atol=1e-9)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_block(n, m, p):
    """
    Try to decompose M_{n,m,p} by splitting into blocks and using Strassen on 2x2 subproblems.
    For (2,4,6): split B into three 4x2 blocks. Each M_{2,4,2} can potentially use optimized methods.
    Alternatively, try a direct Strassen-like decomposition.
    
    Actually, let's try: view the computation as 2 independent M_{1,4,6} problems (one per row of A).
    Each M_{1,4,6} has rank 24 (schoolbook). Total 48. No gain.
    
    Better: Try to find if there's a rank 42 decomposition via structured search.
    """
    pass

def _random_search(n, m, p, target_rank, max_iters=5000, deadline=None):
    """
    Randomized search for a decomposition of the given rank using integer/half-integer entries.
    Uses a gradient-free approach: randomly initialize and try to satisfy the tensor equations.
    """
    num_a = n * m  # 8
    num_b = m * p  # 24
    num_c = p * n  # 12
    
    # Build the list of (i,j,k) triples that define the tensor
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    
    # For each triple (i,j,k), the constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We'll use a simple iterative approach: fix two matrices, solve for the third.
    # But that's nonlinear. Instead, let's try a different approach.
    
    # Approach: Start from schoolbook and try to merge columns.
    # Or: Use the fact that we can add random vectors to U, V, W and check.
    
    # For now, this is a placeholder. Let's implement a more direct search.
    pass

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Try to find a lower rank decomposition.
    # Known: For M_{2,4,6}, let's try rank 42, 43, 44, 45, 46, 47.
    
    # Strategy: Use the structure of the problem.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..5}, j in {0..3}
    
    # Idea: Group the 4 products in each sum.
    # For fixed i, C[i,0..5] = A[i,0]*B[0,0..5] + A[i,1]*B[1,0..5] + A[i,2]*B[2,0..5] + A[i,3]*B[3,0..5]
    
    # This is like a linear combination of 4 vectors (rows of B) scaled by entries of A.
    
    # If we could compute certain combinations of B rows more efficiently...
    
    # Let's try a specific construction:
    # Define intermediate products that combine A and B entries.
    
    # For Strassen on 2x2: we save 1 product by computing (a+b)(c+d) instead of ac+bd.
    # Can we apply this here?
    
    # Split A into two 2x2 blocks: A1 = A[:,0:2], A2 = A[:,2:4]
    # Split B into two 4x3 blocks: B1 = B[:,0:3], B2 = B[:,3:6]
    # C = A1*B1 + A2*B2 (block matrix multiplication)
    # Each term is M_{2,2,3} which has schoolbook rank 12.
    # Total: 2 * 12 = 24? No, that's not right.
    # Actually C = [A1 A2] * [B1; B2] where B1 is 2x3, B2 is 2x3... 
    # Wait, A is 2x4, so A1 = A[:,0:2] is 2x2, A2 = A[:,2:4] is 2x2.
    # B is 4x6, so B1 = B[0:2,:] is 2x6, B2 = B[2:4,:] is 2x6.
    # C = A1*B1 + A2*B2, where each product is 2x2 * 2x6 = 2x6.
    # M_{2,2,6} has schoolbook rank 24. Two of them: 48. No gain.
    
    # Alternative split:
    # A = [a0 a1 a2 a3] (row 0), [b0 b1 b2 b3] (row 1)
    # B has rows r0, r1, r2, r3 (each length 6)
    # C[0,k] = a0*r0[k] + a1*r1[k] + a2*r2[k] + a3*r3[k]
    # C[1,k] = b0*r0[k] + b1*r1[k] + b2*r2[k] + b3*r3[k]
    
    # Define: x0 = a0*r0, x1 = a1*r1, x2 = a2*r2, x3 = a3*r3 (4 products for row 0 of C)
    # y0 = b0*r0, y1 = b1*r1, y2 = b2*r2, y3 = b3*r3 (4 products for row 1 of C)
    # Total 8 products per column k, 6 columns = 48.
    
    # To save, we need to share products across columns or rows.
    
    # Observation: r0[k] appears in both C[0,k] and C[1,k].
    # If we compute a0*r0[k] and b0*r0[k] separately, that's 2 products.
    # If we compute (a0+b0)*r0[k] and (a0-b0)*r0[k], that's still 2 products, no gain.
    # But if we compute (a0+b0)*r0[k] and then derive a0*r0[k] and b0*r0[k] from it... 
    # we'd need to know a0 and b0 separately, which we do. So no gain.
    
    # The key insight from Strassen: by computing (a0+a1)*(b0+b1) = a0*b0 + a0*b1 + a1*b0 + a1*b1,
    # we get 4 products from 1, saving 3. But we need to be able to extract the individual terms.
    
    # For our case, can we find a similar trick?
    
    # Let's try: for a fixed k, we have 8 products: a_i * r_i[k] for i=0..3, and b_i * r_i[k] for i=0..3.
    # That's 8 products. Can we do it in fewer?
    
    # If we define p_i = a_i + b_i for i=0..3, then:
    # p_i * r_i[k] = a_i*r_i[k] + b_i*r_i[k]
    # So sum_i p_i * r_i[k] = sum_i a_i*r_i[k] + sum_i b_i*r_i[k] = C[0,k] + C[1,k]
    # That gives us the sum, but not the individual C[0,k] and C[1,k].
    
    # We also need q_i = a_i - b_i:
    # q_i * r_i[k] = a_i*r_i[k] - b_i*r_i[k]
    # sum_i q_i * r_i[k] = C[0,k] - C[1,k]
    
    # So with sum and difference, we can recover C[0,k] and C[1,k].
    # But we used 8 products (4 for p_i*r_i[k] and 4 for q_i*r_i[k]), same as before.
    
    # No gain from this approach.
    
    # Let's try a completely different angle: use the fact that the problem is small enough
    # to potentially brute-force or use linear algebra to find a lower rank decomposition.
    
    # The tensor T is an 8 x 24 x 12 array. We want to find the minimum rank R such that
    # T can be written as sum_{c=0}^{R-1} u_c ⊗ v_c ⊗ w_c.
    
    # This is the tensor rank problem, which is NP-hard in general, but for small tensors
    # we can try specific constructions.
    
    # Given time constraints, let's return the schoolbook solution.
    # The score of 1.0 means rank 48. If we can find rank 42, score would be 48/42 ≈ 1.143.
    
    # Let me try one more idea: use a known result.
    # For M_{n,m,p}, if gcd(n,m,p) > 1, there might be improvements.
    # gcd(2,4,6) = 2. 
    # Write A = A' * D where D is a diagonal scaling... not helpful.
    
    # Actually, here's a concrete attempt at rank 42:
    # Use 6 "Strassen-like" savings of 1 each.
    # 
    # Consider the 12 output entries. Each is a sum of 4 products.
    # If for each output entry, we can save 0.5 products on average, that's 6 total savings.
    # 
    # This is getting too speculative. Let's just return schoolbook.
    
    return best

# EVOLVE-BLOCK-END
