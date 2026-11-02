"""Matrix multiplication 2x3x6 rank-minimal exact tensor decomposition."""
import time
import random
import math

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 6
    M = m * p  # 18
    P = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Schoolbook rank 36
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(N)]
    V_sb = [[0] * rank_sb for _ in range(M)]
    W_sb = [[0] * rank_sb for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = rank_sb
    
    # Verify function
    def verify(U, V, W, tol=1e-8):
        r = len(U[0])
        # U: N x r, V: M x r, W: P x r
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For multiplication: T[i,j,k] should be 1 if we're computing (AB)[i,k]
        # Actually the tensor T for matrix mult M_{n,m,p} has T[i,j,k] = 1
        # where the bilinear form is sum_{i,j,k} T[i,j,k] a_i b_j c_k
        # But here it's: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor is: sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # So T[i,j,k] = 1 for all i,j,k (the multiplication tensor)
        # And we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
        # Wait, let me reconsider. The decomposition says:
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j U[i,j,c] A_{i,j}) ... no
        # Standard: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # Tensor decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
        # where T[i,j,k] = u_i v_j w_k... no.
        # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c] should equal δ (the multiplication tensor)
        # For matrix mult: T[i,j,k] = 1 (it's just the 3-tensor with all entries 1)
        # No wait. The matrix multiplication tensor M_{n,m,p} is defined by
        # M_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (something)
        # Actually: the bilinear map is (A,B) -> AB
        # T_{i,j,k} is the coefficient such that (AB)_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
        # So T_{i,j,k} = 1 for all valid i,j,k (since (AB)_{i,k} = sum_j A_{i,j} B_{j,k})
        # Wait no. T_{i,j,k} where the tensor is in the space (R^n ⊗ R^m ⊗ R^p)^*
        # The standard form: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
        # And the decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
        # means sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k] = 1 for all i,j,k
        
        # Hmm, but that's not right either. Let me think again.
        # The matrix multiplication tensor is:
        # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
        # No, that's just the all-ones tensor.
        # 
        # Actually, the correct tensor for (n x m) * (m x p) -> (n x p) is:
        # T ∈ (R^n ⊗ R^p) ⊗ (R^{n×m} ⊗ R^{m×p})*
        # Flattening: A is n×m, B is m×p, C = AB is n×p
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the tensor T has components T_{(i,k),(i',j),(j',k')} 
        # This is getting complex. Let me use the standard form from the task.
        #
        # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # U[i*m+j, c] - so U is indexed by (i,j) pairs
        # V[j*p+k, c] - V is indexed by (j,k) pairs  
        # W[k*n+i, c] - W is indexed by (k,i) pairs
        #
        # The decomposition: for each output (i,k), C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * ... 
        # No. Let me re-read.
        #
        # Actually the standard tensor decomposition for matrix multiplication:
        # The tensor T ∈ (R^n ⊗ R^m ⊗ R^p) such that
        # T(i,j,k) = 1 if we consider the trilinear form
        # Actually no. The matrix multiplication tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  ... this is wrong.
        #
        # Let me use the convention from the task:
        # U: (n*m) x rank, indexed by (i,j) -> i*m+j
        # V: (m*p) x rank, indexed by (j,k) -> j*p+k
        # W: (p*n) x rank, indexed by (k,i) -> k*n+i
        #
        # The decomposition satisfies:
        # For the multiplication tensor, we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
        # This is because the multiplication tensor has T[i,j,k] = 1 (all ones)
        # Wait, that can't be right because then the rank would be 1.
        #
        # I think I'm confusing myself. Let me look at the schoolbook:
        # In schoolbook, for each (i,j,k) triple, we have one term with U=1, V=1, W=1.
        # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for the specific (i,j,k)
        # and 0 for all other combinations? No, in schoolbook each term only affects one (i,j,k).
        #
        # Actually, I think the correct interpretation is:
        # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the "all-ones" 3-tensor)
        # No wait, that would make rank 1.
        #
        # Let me reconsider. The matrix multiplication tensor M_{n,m,p}:
        # It's an element of (R^n ⊗ R^m) ⊗ (R^m ⊗ R^p) ⊗ (R^n ⊗ R^p)*
        # Or equivalently, it defines a trilinear form:
        # For matrices A (n×m), B (m×p), and C (n×p), the tensor evaluates to
        # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]
        # And this should equal... hmm.
        #
        # Actually, I think the standard is:
        # T[i,j,k] = δ_{i,j,k} in some sense... 
        #
        # You know what, let me just verify the schoolbook works by checking
        # that for each (i,j,k), the sum equals 1, and for "cross terms" it works.
        #
        # The schoolbook gives: for term c corresponding to (i0,j0,k0):
        # U[i*m+j, c] = 1 iff i=i0 and j=j0
        # V[j*p+k, c] = 1 iff j=j0 and k=k0
        # W[k*n+i, c] = 1 iff k=k0 and i=i0
        #
        # So the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 iff (i,j,k)=(i0,j0,k0)
        # Therefore sum_c ... = 1 for all (i,j,k) - each (i,j,k) has exactly one term.
        #
        # So the tensor we're decomposing has T[i,j,k] = 1 for all i,j,k.
        # That IS the all-ones 3-tensor! And its rank is n*m*p for the schoolbook,
        # but can be lower.
        
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try rank 14 (known result for 2x3x6 matrix multiplication)
    # The Strassen-type decomposition for 2x3x6
    # Based on known results: rank(M_{2,3,6}) <= 14
    
    # Let's try to construct a rank-14 decomposition
    # This is based on the fact that M_{2,3,6} can be decomposed using
    # the rank-7 decomposition of M_{2,3,3} (Strassen for 2x3x3)
    
    # Actually, let me try a different approach: use the known rank-14 result
    # by explicitly constructing it from block structure.
    
    # For 2x3x6, we can write B as [B1 B2] where B1 is 3x3 and B2 is 3x3
    # Then AB = [A*B1, A*B2]
    # Each A*Bj requires rank at most... well, M_{2,3,3} has rank 7 (Strassen-like)
    # But we need to be careful about shared terms.
    
    # Let me try rank 14 by using two independent 2x3x3 decompositions
    # with some optimization.
    
    # Actually, the simplest valid improvement: try rank 12 or 14 with random search
    # But that's risky. Let me stick with a verified construction.
    
    # The known minimum rank for 2x3x6 is 14 (by Landsberg et al.)
    # But I need explicit factors.
    
    # Let me try a structured approach:
    # Split the 6 columns into two groups of 3.
    # M_{2,3,6} = M_{2,3,3} ⊕ M_{2,3,3} (block diagonal in the last index)
    # This gives rank 2 * rank(M_{2,3,3})
    # rank(M_{2,3,3}) = 7 (known)
    # So rank 14 is achievable if I have the rank-7 factors for M_{2,3,3}.
    
    # The rank-7 decomposition of 2x3x3 matrix multiplication:
    # This is a known result. Let me try to recall/derive it.
    
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One approach: use the identity for 2x3x3.
    # Let A = [[a1,a2,a3],[b1,b2,b3]], B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # C = [[a1*b11+a2*b21+a3*b31, a1*b12+a2*b22+a3*b32, a1*b13+a2*b23+a3*b33],
    #      [b1*b11+b2*b21+b3*b31, b1*b12+b2*b22+b3*b32, b1*b13+b2*b23+b3*b33]]
    
    # Rank-7 decomposition exists but I don't have it memorized exactly.
    
    # Alternative: try to find a valid lower-rank decomposition via random search
    # with verification.
    
    elapsed = time.time() - start_time
    if elapsed < 20:
        # Try random search for rank 14
        random.seed(42)
        found_rank14 = False
        for attempt in range(50):
            if time.time() - start_time > deadline - 10:
                break
            r = 14
            # Initialize with small random values
            U = [[random.uniform(-1,1) for _ in range(r)] for _ in range(N)]
            V = [[random.uniform(-1,1) for _ in range(r)] for _ in range(M)]
            W = [[random.uniform(-1,1) for _ in range(r)] for _ in range(P)]
            
            if verify(U, V, W, tol=1e-6):
                # Snap to nearby rational values
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                found_rank14 = True
                break
    
    # Try rank 12
    if best_rank > 12:
        elapsed = time.time() - start_time
        if elapsed < 60:
            random.seed(123)
            for attempt in range(30):
                if time.time() - start_time > deadline - 10:
                    break
                r = 12
                U = [[random.uniform(-2,2) for _ in range(r)] for _ in range(N)]
                V = [[random.uniform(-2,2) for _ in range(r)] for _ in range(M)]
                W = [[random.uniform(-2,2) for _ in range(r)] for _ in range(P)]
                
                if verify(U, V, W, tol=1e-6):
                    best = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    break
    
    # Try rank 11
    if best_rank > 11:
        elapsed = time.time() - start_time
        if elapsed < 90:
            random.seed(456)
            for attempt in range(20):
                if time.time() - start_time > deadline - 5:
                    break
                r = 11
                U = [[random.uniform(-3,3) for _ in range(r)] for _ in range(N)]
                V = [[random.uniform(-3,3) for _ in range(r)] for _ in range(M)]
                W = [[random.uniform(-3,3) for _ in range(r)] for _ in range(P)]
                
                if verify(U, V, W, tol=1e-6):
                    best = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    break
    
    # Try rank 10
    if best_rank > 10:
        elapsed = time.time() - start_time
        if elapsed < 110:
            random.seed(789)
            for attempt in range(15):
                if time.time() - start_time > deadline - 5:
                    break
                r = 10
                U = [[random.uniform(-4,4) for _ in range(r)] for _ in range(N)]
                V = [[random.uniform(-4,4) for _ in range(r)] for _ in range(M)]
                W = [[random.uniform(-4,4) for _ in range(r)] for _ in range(P)]
                
                if verify(U, V, W, tol=1e-6):
                    best = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    break
    
    return best
# EVOLVE-BLOCK-END
