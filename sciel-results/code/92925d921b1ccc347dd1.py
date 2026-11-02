"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Vectorized verification using numpy."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=1e-9)

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Returns (U, V, W) as lists.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        return U, V, W

    def try_rank13_structured():
        """
        Attempt to find a rank-13 decomposition by exploiting the shared A matrix
        between the two Strassen blocks. 
        
        Key insight: Since both blocks use the same A, we can potentially share
        some of the intermediate products. Let's try a structured approach where
        we fix most entries and solve for a few free parameters.
        """
        if time.time() > deadline - 40:
            return None
            
        # Strategy: Try to merge terms that share the same A-factor
        # For example, M1 = (a00+a11)(b00+b11) and N1 = (a00+a11)(b02+b13)
        # Both share (a00+a11). Can we combine them into one term?
        # Combined would be: (a00+a11)(b00+b11+b02+b13)
        # But this gives a single product, not two separate ones.
        
        # Alternative: Try a different factorization that shares more structure
        # Let's try to find a rank-13 solution by solving a system of equations
        
        # The tensor equation is: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # With rank 13, we have 4*13 + 8*13 + 8*13 = 260 unknowns
        # and 2*2*4 = 16 equations. This is very underdetermined.
        
        # Let's try a more systematic approach: fix the structure based on the
        # rank-14 solution and try to eliminate one term
        
        U14, V14, W14 = get_rank14()
        
        # Try each possible term to eliminate
        for r_elim in range(14):
            if time.time() > deadline - 40:
                break
                
            # Create a rank-13 version by removing term r_elim
            U = [[0]*13 for _ in range(4)]
            V = [[0]*13 for _ in range(8)]
            W = [[0]*13 for _ in range(8)]
            
            # Copy terms from rank-14, skipping r_elim
            new_r = 0
            for r_old in range(14):
                if r_old == r_elim:
                    continue
                for i in range(4):
                    U[i][new_r] = U14[i][r_old]
                for j in range(8):
                    V[j][new_r] = V14[j][r_old]
                for k in range(8):
                    W[k][new_r] = W14[k][r_old]
                new_r += 1
            
            # Now try to adjust the remaining terms to compensate
            # Use a local search: try small integer adjustments
            for attempt in range(500):
                if time.time() > deadline - 40:
                    break
                    
                U_copy = [row[:] for row in U]
                V_copy = [row[:] for row in V]
                W_copy = [row[:] for row in W]
                
                # Make a few random small adjustments
                for _ in range(3):
                    mat_choice = random.randint(0, 2)
                    if mat_choice == 0:
                        row = random.randint(0, 3)
                        col = random.randint(0, 12)
                        delta = random.choice([-2, -1, 1, 2])
                        U_copy[row][col] += delta
                    elif mat_choice == 1:
                        row = random.randint(0, 7)
                        col = random.randint(0, 12)
                        delta = random.choice([-2, -1, 1, 2])
                        V_copy[row][col] += delta
                    else:
                        row = random.randint(0, 7)
                        col = random.randint(0, 12)
                        delta = random.choice([-2, -1, 1, 2])
                        W_copy[row][col] += delta
                
                if verify(U_copy, V_copy, W_copy):
                    return U_copy, V_copy, W_copy
        
        return None

    def try_rank13_grid_search():
        """
        More systematic search for rank-13 solutions using a different strategy.
        Try to find a solution where the two blocks share some structure.
        """
        if time.time() > deadline - 35:
            return None
            
        # Strategy: Try to find a rank-13 solution by exploiting the fact that
        # both blocks share the same A matrix. 
        # 
        # One approach: Try to find a solution where some of the intermediate
        # products are shared between the two blocks.
        #
        # Let's try a different factorization pattern:
        # Instead of two independent Strassen decompositions, try to find
        # a decomposition where some terms are shared.
        
        # Try a specific structured pattern:
        # Assume the solution has a certain symmetry between the two blocks
        
        # Pattern 1: Try to share M1 and N1 by combining them
        # M1 = (a00+a11)(b00+b11), N1 = (a00+a11)(b02+b13)
        # Combined: (a00+a11)(b00+b11+b02+b13)
        # This gives one product, but we need two separate outputs.
        # So we'd need to split it somehow.
        
        # Let's try a different approach: use a more systematic search
        # over structured patterns
        
        # Try various structured patterns for the U, V, W matrices
        patterns = [
            # Pattern 0: Standard Strassen-like
            lambda r: (random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])),
            # Pattern 1: More sparse
            lambda r: (random.choice([0, -1, 1]), random.choice([0, -1, 1]), random.choice([0, -1, 1])),
            # Pattern 2: Allow 2s
            lambda r: (random.choice([-2, -1, 1, 2]), random.choice([-2, -1, 1, 2]), random.choice([-2, -1, 1, 2])),
        ]
        
        for pat_idx, pattern_gen in enumerate(patterns):
            if time.time() > deadline - 35:
                break
                
            for attempt in range(1000):
                if time.time() > deadline - 35:
                    break
                    
                U = [[0]*13 for _ in range(4)]
                V = [[0]*13 for _ in range(8)]
                W = [[0]*13 for _ in range(8)]
                
                # For each rank term, choose random non-zero entries using the pattern
                for r in range(13):
                    # U: choose 1-2 non-zero entries
                    num_u = random.randint(1, 2)
                    u_rows = random.sample(range(4), num_u)
                    for row in u_rows:
                        U[row][r] = pattern_gen(r)[0]
                    
                    # V: choose 1-3 non-zero entries
                    num_v = random.randint(1, 3)
                    v_rows = random.sample(range(8), num_v)
                    for row in v_rows:
                        V[row][r] = pattern_gen(r)[1]
                    
                    # W: choose 1-2 non-zero entries
                    num_w = random.randint(1, 2)
                    w_rows = random.sample(range(8), num_w)
                    for row in w_rows:
                        W[row][r] = pattern_gen(r)[2]
                
                # Check if this gives a valid decomposition
                if verify(U, V, W):
                    return U, V, W
        
        return None

    # First, try the known rank-14 solution
    U, V, W = get_rank14()
    
    if verify(U, V, W):
        rank14_solution = {"U": U, "V": V, "W": W}
    else:
        rank14_solution = naive_decomp()

    # Try to find a rank-13 solution
    # Use a more time-efficient search strategy
    if time.time() < deadline - 50:
        result13 = try_rank13_structured()
        if result13 is not None:
            U13, V13, W13 = result13
            if verify(U13, V13, W13):
                return {"U": U13, "V": V13, "W": W13}
    
    if time.time() < deadline - 45:
        result13 = try_rank13_grid_search()
        if result13 is not None:
            U13, V13, W13 = result13
            if verify(U13, V13, W13):
                return {"U": U13, "V": V13, "W": W13}

    # Return the best solution found
    return rank14_solution

# EVOLVE-BLOCK-END
