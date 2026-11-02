"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    def schoolbook():
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

    def verify(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def to_half_int_list(lst):
        """Convert list of floats to half-integers, return None if not all are."""
        result = []
        for x in lst:
            h = round(x * 2) / 2.0
            if abs(x - h) > 1e-12:
                return None
            result.append(h)
        return result

    def strassen_3x3x3():
        """
        Return the Strassen rank-13 decomposition for 3x3x3.
        U[i*3+j], V[j*3+k], W[k*3+i] for i,j,k in 0..2.
        
        Strassen's algorithm for 3x3 matrix multiplication:
        We partition into 2x2 blocks and use Strassen, but that gives rank 21.
        
        Actually, the minimal known rank for 3x3x3 is 19 (Bini et al.), 
        but Strassen's original 2x2x2 rank-7 can be extended.
        
        Let me use a known rank-19 or better decomposition.
        
        For simplicity and correctness, let me use the approach where we
        decompose 3x3x3 by embedding 2x2x2 Strassen blocks.
        
        Actually, let me just use a verified rank-19 decomposition or 
        construct one from known results.
        
        The known minimal rank for 3x3x3 is 19. Let me use a specific one.
        
        I'll use the decomposition from Bini, Capovani, Lotti, Romani (1979)
        or a simpler known construction.
        
        Let me use a different approach: start with schoolbook for 3x3x3 (rank 27)
        and try to find a lower rank one using linear algebra.
        
        Actually, for the 3x3x5 problem, let me think differently.
        
        The tensor for 3x3x5 has 3*3*5 = 45 entries.
        Schoolbook rank = 45.
        
        A key insight: we can use the Strassen rank-7 for 2x2x2 subproblems.
        Partition the 3x3x5 into overlapping 2x2x2 blocks.
        
        Let me try a direct linear algebra approach:
        Assume rank r. We have U (9 x r), V (15 x r), W (15 x r).
        The constraint is: for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
        
        This is a system of 45 cubic equations in 9r + 15r + 15r = 39r unknowns.
        
        For r = 19: 39*19 = 741 unknowns, 45 equations. Very underdetermined.
        
        Let me try a structured approach:
        Use the fact that the 3x3x5 tensor contains a 3x3x3 sub-tensor.
        If we have a rank-r decomposition for 3x3x3, we can extend it to 3x3x5
        by adding rank-1 terms for the extra two output columns.
        
        For the extra columns k=3,4:
        We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for k=3,4.
        
        If we add 2 new rank-1 terms (one for k=3, one for k=4), we need:
        U[i,j,c_new] * V[j,3,c_new] * W[3,i,c_new] = 1 for all i,j
        U[i,j,c_new2] * V[j,4,c_new2] * W[4,i,c_new2] = 1 for all i,j
        
        This requires U[i,j,c_new] to be constant in i,j (say a), 
        V[j,3,c_new] to be constant in j (say b), 
        W[3,i,c_new] to be constant in i (say c), with a*b*c = 1.
        Similarly for the other column.
        
        So we can extend any 3x3x3 rank-r decomposition to 3x3x5 with rank r+2.
        
        If we can find a rank-17 decomposition for 3x3x3, we get rank 19 for 3x3x5.
        If rank 19 for 3x3x3, we get rank 21 for 3x3x5.
        
        The known minimal rank for 3x3x3 is 19 (Bini et al.).
        So the best we can do with this approach is rank 21 for 3x3x5.
        
        But wait - maybe we can do better by not just adding independent rank-1 terms.
        The extra columns might share structure with the existing terms.
        
        Let me first get a rank-19 decomposition for 3x3x3, then try to extend to rank 19 or 20 for 3x3x5.
        """
        # Known rank-19 decomposition for 3x3x3 (Bini-Capovani-Lotti-Romani)
        # This is complex to hardcode. Let me use a simpler approach.
        
        # For now, use schoolbook for 3x3x3 (rank 27) and extend to rank 29 for 3x3x5.
        # Then try to find lower rank.
        
        # Actually, let me try to find a good decomposition using linear algebra.
        # Start with the assumption that we can achieve rank 19 for 3x3x5 directly.
        
        return None  # Placeholder - will be implemented below

    def find_decomposition_via_linear_algebra(target_rank):
        """
        Try to find a decomposition of given rank using a structured approach.
        
        Strategy: Assume the decomposition has a specific structure.
        For 3x3x5, we can try:
        - Use rank-7 Strassen for a 2x2x2 sub-block
        - Add terms for the remaining parts
        
        Let me try a different approach: use the fact that we can parameterize
        the solution and solve a system of equations.
        """
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        r = target_rank
        
        # Try a structured decomposition where we fix some entries
        # and solve for others.
        
        # Approach: Use the schoolbook as a starting point and try to
        # eliminate some rank-1 terms by finding linear dependencies.
        
        # For now, let me try a randomized approach with better initialization.
        # Initialize with a valid high-rank decomposition and try to reduce.
        
        # Start with schoolbook
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Fill with schoolbook-like structure for the first min(r, nm*mp*pn) terms
        for c in range(r):
            if c < n*m*p:
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
            else:
                # For extra columns, use random initialization
                U[:, c] = np.random.randn(nm) * 0.1
                V[:, c] = np.random.randn(mp) * 0.1
                W[:, c] = np.random.randn(pn) * 0.1
        
        # Now try to optimize to satisfy all constraints
        # Use a simple iterative method
        
        for iteration in range(500):
            if time.time() > deadline - 30:
                break
            
            # Compute errors
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = s - 1
                        max_err = max(max_err, abs(err))
            
            if max_err < 1e-10:
                # Check if we can round to half-integers
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                
                # Verify after rounding
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U_h[i*m+j, c] * V_h[j*p+k, c] * W_h[k*n+i, c]
                            if abs(s - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    U_list = [[float(U_h[i*m+j, c]) for c in range(r)] for i in range(n) for j in range(m)]
                    V_list = [[float(V_h[j*p+k, c]) for c in range(r)] for j in range(m) for k in range(p)]
                    W_list = [[float(W_h[k*n+i, c]) for c in range(r)] for k in range(p) for i in range(n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Gradient update
            lr = 0.005
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        e = s - 1
                        for c in range(r):
                            U[i*m+j, c] -= lr * e * V[j*p+k, c] * W[k*n+i, c]
                            V[j*p+k, c] -= lr * e * U[i*m+j, c] * W[k*n+i, c]
                            W[k*n+i, c] -= lr * e * U[i*m+j, c] * V[j*p+k, c]
        
        return None

    def try_strassen_extension():
        """
        Try to construct a decomposition using Strassen's algorithm for 2x2x2
        and extend to 3x3x5.
        
        Partition the 3x3 matrix into overlapping 2x2 blocks.
        Use Strassen's 7 multiplications for each 2x2x2 block.
        
        For 3x3x3:
        - Top-left 2x2x2: Strassen rank 7
        - Remaining parts: need additional terms
        
        This is complex. Let me try a simpler structured approach.
        """
        n, m, p = DIMENSIONS
        
        # Try rank 19 directly with a structured initialization
        # Based on the idea that 3x3x5 can be decomposed with rank 19
        
        # Use a known result: the rank of 3x3x5 is at most 19
        # (this needs verification)
        
        # For now, let me try to find a decomposition with rank 19
        # using a specific structured form.
        
        # One approach: use the fact that the tensor can be written as
        # a sum of rank-1 tensors with specific patterns.
        
        # Let me try a direct construction.
        
        # Actually, let me just return the schoolbook for now and 
        # rely on the optimization to find better solutions.
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    
    # Try to find lower rank decompositions
    for target_rank in [21, 20, 19, 18, 17]:
        if time.time() > deadline - 40:
            break
        
        np.random.seed(target_rank * 7 + 13)
        result = find_decomposition_via_linear_algebra(target_rank)
        if result is not None:
            if verify(result['U'], result['V'], result['W']):
                best_result = result
                break
    
    # Try Strassen extension
    if time.time() < deadline - 20:
        result = try_strassen_extension()
        if result is not None:
            if verify(result['U'], result['V'], result['W']):
                best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
