import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # Dimensions: U is (n*m, r) = (4, r), V is (m*p, r) = (6, r), W is (p*n, r) = (6, r)
    
    def build_target():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = build_target()
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose the tensor."""
        r = len(U[0])
        T = np.zeros((n*m, m*p, p*n))
        for col in range(r):
            u = np.array(U[:, col], dtype=float)
            v = np.array(V[:, col], dtype=float)
            w = np.array(W[:, col], dtype=float)
            T += np.einsum('i,j,k->ijk', u, v, w)
        return np.allclose(T, T_target, atol=1e-9)

    # Strategy: Try known explicit rank-10 decompositions.
    # The 2x2x3 matrix multiplication tensor has rank 10.
    # We construct it by combining Strassen's 2x2x2 (rank 7) with 3 more terms
    # for the third output column, but with sharing.
    
    # Actually, let's use a direct algebraic construction.
    # For M(2,2,3), we can write it as:
    # C = A * B where A is 2x2, B is 2x3
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[0,2] = A00*B02 + A01*B12
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    # C[1,2] = A10*B02 + A11*B12
    
    # Output index: W[k*n+i, r] -> for i=0: k=0->0, k=1->1, k=2->2; for i=1: k=0->3, k=1->4, k=2->5
    # So W indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    
    # Let's try to find a rank-10 decomposition by solving a reduced system.
    # Use the gauge freedom: we can permute and scale the rank-1 terms.
    
    # Direct construction approach: 
    # Known that rank(2,2,3) = 10. Let me try to construct it explicitly.
    
    # Approach: Use a parametric family and solve.
    # For rank r=10, we have 4*10 + 6*10 + 6*10 = 160 variables, 12 equations.
    # This is very underdetermined. Let's fix a structure.
    
    # Better: Use the fact that we can embed Strassen's algorithm.
    # Strassen for 2x2x2 uses 7 products. For 2x2x3, think of B as [B_left | B_right]
    # where B_left is 2x2 (cols 0,1) and B_right is 2x1 (col 2).
    
    # Actually, let me try a completely different structural approach:
    # Use the "tensor train" or "CP decomposition" with rational entries.
    # Set up the 12 equations and solve them using a greedy/structured approach.
    
    # Let me try to directly construct a rank-10 solution with entries in {0, ±1, ±1/2}.
    
    # I'll use a known construction from the literature or derive one.
    # The key insight: M(2,2,3) can be decomposed using 10 rank-1 tensors.
    
    # Let me try a systematic approach: assume entries are in {0, 1, -1, 0.5, -0.5}
    # and use a search over structured patterns.
    
    # Actually, the most reliable approach: use sympy to solve a reduced system.
    # Fix the first few terms to match the schoolbook pattern partially,
    # then solve for the remaining terms.
    
    # Let me try: start with 6 terms that handle the "diagonal" (i=j or specific patterns)
    # and 4 additional terms for the off-diagonal parts.
    
    # Alternative: Use the known result that rank(2,2,3) = 10 and construct
    # using a specific known decomposition.
    
    # Here's a concrete approach: 
    # 1. Use 6 terms for the "identity-like" part (like schoolbook but compressed)
    # 2. Use 4 more terms for the remaining structure
    
    # Let me try to construct it manually with the following structure:
    # Term 1-6: Handle the main multiplications
    # Term 7-10: Handle the cross terms
    
    # Actually, let me use a more principled approach:
    # Set up the system and use a SAT/SMT-like search over {0, ±1, ±1/2}
    
    def try_construct_rank10():
        """Try to construct a rank-10 decomposition with entries in {0, ±1, ±1/2}."""
        values = [0, 1, -1, 0.5, -0.5]
        
        # We need U (4x10), V (6x10), W (6x10)
        # 12 equations: for each (a,b,c) with T_target[a,b,c]=1, sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # For other (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # This is a 12-equation system in 160 variables.
        # Let's fix a structure to reduce the search space.
        
        # Fix the first 6 columns to be "diagonal" terms:
        # Col 0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contributes to T[0,0,0]
        # Col 1: U[0,1]=1, V[2,1]=1, W[2,1]=1 -> contributes to T[0,2,2]
        # etc.
        
        # Actually, let me use a different strategy: 
        # Use the fact that the problem is highly symmetric and try specific patterns.
        
        # Let me try a known decomposition from the literature.
        # For M(2,2,3), one rank-10 decomposition is:
        
        # I'll use a constructive approach: 
        # Let the 10 terms be defined by their (u, v, w) vectors.
        
        # Try: 6 terms for the "outer product" structure + 4 correction terms
        
        # Term structure:
        # t1: u=(1,0,0,0), v=(1,0,0,0,0,0), w=(1,0,0,0,0,0) -> T[0,0,0]+=1
        # t2: u=(1,0,0,0), v=(0,0,1,0,0,0), w=(0,1,0,0,0,0) -> T[0,2,1]+=1  (i=0,j=0,k=1)
        # t3: u=(1,0,0,0), v=(0,0,0,0,1,0), w=(0,0,1,0,0,0) -> T[0,4,2]+=1  (i=0,j=0,k=2)
        # t4: u=(0,1,0,0), v=(0,1,0,0,0,0), w=(0,0,0,1,0,0) -> T[1,1,3]+=1  (i=0,j=1,k=0)
        # t5: u=(0,1,0,0), v=(0,1,0,0,0,0), w=(0,0,0,0,1,0) -> T[1,1,4]+=1  (i=0,j=1,k=1)
        # t6: u=(0,1,0,0), v=(0,1,0,0,0,0), w=(0,0,0,0,0,1) -> T[1,1,5]+=1  (i=0,j=1,k=2)
        # t7: u=(0,0,1,0), v=(1,0,0,0,0,0), w=(1,0,0,0,0,0) -> T[2,0,0]+=1  (i=1,j=0,k=0)
        # t8: u=(0,0,1,0), v=(0,0,1,0,0,0), w=(0,1,0,0,0,0) -> T[2,2,1]+=1  (i=1,j=0,k=1)
        # t9: u=(0,0,1,0), v=(0,0,0,0,1,0), w=(0,0,1,0,0,0) -> T[2,4,2]+=1  (i=1,j=0,k=2)
        # t10: u=(0,0,0,1), v=(0,1,0,0,0,0), w=(0,0,0,1,0,0) -> T[3,1,3]+=1  (i=1,j=1,k=0)
        
        # Wait, this is just the schoolbook with 12 terms. I need to compress to 10.
        
        # Let me use a different approach: algebraic elimination.
        # Start with the 12-term schoolbook and try to merge pairs.
        
        # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if there exist
        # u',v',w' such that u1*v1*w1 + u2*v2*w2 = u'*v'*w'
        # This requires the two rank-1 tensors to be "compatible".
        
        # A simpler approach: use the known Strassen-like compression.
        
        # For 2x2x2, Strassen reduces 8 to 7. The key is:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For 2x2x3, we can extend this. Let me try to construct it.
        
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01, b02], [b10, b11, b12]]
        # C = A*B = [[c00, c01, c02], [c10, c11, c12]]
        
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c02 = a00*b02 + a01*b12
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # c12 = a10*b02 + a11*b12
        
        # Let me define 10 products:
        # P1 = (a00+a11)*(b00+b10)  -> helps with c00, c10
        # P2 = (a10+a11)*(b00)      -> helps with c10
        # P3 = a00*(b10-b11)        -> helps with c00, c01
        # P4 = a11*(b01-b00)        -> helps with c11, c10
        # P5 = (a00+a01)*(b11+b12)  -> helps with c01, c02
        # P6 = (a10-a00)*(b00+b01)  -> helps with c10, c11
        # P7 = (a01-a11)*(b10+b11)  -> helps with c01, c11
        # P8 = a00*b02               -> helps with c02
        # P9 = a01*b12               -> helps with c02
        # P10 = (a10+a11)*(b02)     -> helps with c12
        
        # Let me verify these:
        # c00 = a00*b00 + a01*b10
        #   From P1: a00*b00 + a00*b10 + a11*b00 + a11*b10
        #   From P2: a10*b00 + a11*b00
        #   From P3: a00*b10 - a00*b11
        #   From P6: a10*b00 + a10*b01 - a00*b00 - a00*b01
        #   From P7: a01*b10 + a01*b11 - a11*b10 - a11*b11
        #   Sum: a00*b00 + a00*b10 + a11*b00 + a11*b10 + a10*b00 + a11*b00 
        #         + a00*b10 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
        #         + a01*b10 + a01*b11 - a11*b10 - a11*b11
        #   This is getting messy. Let me just use a computational approach.
        
        # I'll use a different strategy: brute force search over a structured space.
        # Since the system is underdetermined, I'll fix most variables and solve for the rest.
        
        # Let me use numpy to solve a linear system for each rank-1 term,
        # given the other terms.
        
        # Actually, the cleanest approach: use the fact that we can solve for
        # the W matrix given U and V (or vice versa) as a linear system.
        
        # For fixed U (4x10) and V (6x10), the equations for W are:
        # For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is linear in W! For each c, we have a system in W[c,0..9].
        
        # Specifically, for fixed a,b: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
        # For each c, this gives 12 equations (one for each (a,b) pair that matters)
        # in 10 unknowns W[c,0..9]. Wait, no - for each c, the unknowns are W[c,0..9],
        # and the equations come from all (a,b) pairs.
        
        # Let me set this up properly. For a given c (0..5), we need W[c,0..9] such that
        # for all (a,b): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # This is a linear system: M_c * W[c,:] = t_c
        # where M_c is a (4*6) x 10 matrix and t_c is a vector of length 24.
        
        # But we only need it to hold for the 12 non-zero entries of T.
        # For the other 12 entries, T[a,b,c]=0, so we need sum_r U[a,r]*V[b,r]*W[c,r] = 0.
        
        # So for each c, we have 24 equations in 10 unknowns. This is overdetermined!
        # We need to find U,V such that these systems are consistent.
        
        # This is the right structural approach: 
        # 1. Choose U and V (40 + 60 = 100 variables)
        # 2. For each c, check if the linear system for W[c,:] is consistent
        # 3. If consistent, compute W[c,:] as the minimum-norm solution
        
        # The consistency condition is a polynomial constraint on U and V.
        
        # Let me implement this:
        
        def solve_for_W(U, V, c):
            """Given U (4x10) and V (6x10), solve for W[c,:] (length 10)."""
            # Build the linear system
            # For each (a,b): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            A_mat = np.zeros((24, 10))
            b_vec = np.zeros(24)
            for a in range(4):
                for b in range(6):
                    idx = a * 6 + b
                    A_mat[idx, :] = U[a, :] * V[b, :]
                    # Find the target value
                    # a = i*m + j = i*2 + j, so i = a//2, j = a%2
                    # b = j*p + k = j*3 + k, so j = b//3, k = b%3
                    i = a // 2
                    j_a = a % 2
                    j_b = b // 3
                    k = b % 3
                    if j_a == j_b:
                        # Check if this is a valid tensor entry
                        # T[i*m+j, j*p+k, k*n+i] = 1
                        # a = i*2+j, b = j*3+k, c = k*2+i
                        c_expected = k * 2 + i
                        if c_expected == c:
                            b_vec[idx] = 1.0
                        else:
                            b_vec[idx] = 0.0
                    else:
                        b_vec[idx] = 0.0
            
            # Solve the overdetermined system
            # Use least squares
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                # Check if the solution is exact
                residual = np.linalg.norm(A_mat @ sol - b_vec)
                if residual < 1e-9:
                    return sol
                else:
                    return None
            except:
                return None
        
        # Now search over U and V. Use random initialization and gradient-free search.
        # Or better: use the structure to constrain U and V.
        
        # Let me try a different tactic: use the known rank-10 decomposition
        # from the literature. I'll hardcode a known solution.
        
        # From Bini, De Falco, De Lorenzi (1992) or similar:
        # The rank of M(2,2,3) is 10. Here's a construction:
        
        # I'll use a parametric search with sympy for a reduced system.
        
        # Actually, let me just try random U,V with entries from a small set
        # and check if W can be solved exactly.
        
        import random
        random.seed(42)
        
        candidate_values = [0, 1, -1, 0.5, -0.5]
        
        for attempt in range(50000):
            if time.time() > deadline - 10:
                break
            
            # Generate random U and V with entries from candidate_values
            U = np.array([[random.choice(candidate_values) for _ in range(10)] for _ in range(4)])
            V = np.array([[random.choice(candidate_values) for _ in range(10)] for _ in range(6)])
            
            # Check if W can be solved for all c
            W_sol = []
            valid = True
            for c in range(6):
                sol = solve_for_W(U, V, c)
                if sol is None:
                    valid = False
                    break
                W_sol.append(sol)
            
            if valid:
                W = np.array(W_sol)
                # Verify the full decomposition
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    result = try_construct_rank10()
    if result is not None:
        return result
    
    # Fallback: return schoolbook (rank 12)
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
