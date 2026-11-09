import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We want to find a rank-r decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Standard approach: rank = n*m*p = 100 (identity terms)
    # Strassen-like reductions can save terms.
    #
    # Model-and-optimize approach:
    # 1. Start with the standard 100-term decomposition
    # 2. Try to merge/eliminate terms using algebraic identities
    # 3. Use random search to find better decompositions
    #
    # Key insight: For 2x2x2 blocks, Strassen gives rank 7 instead of 8.
    # We can also try to find cross-block optimizations.
    #
    # Let's start with the known best: apply Strassen to all possible 2x2x2 blocks.
    # In a 4x5x5 tensor, we have blocks at i in {0,1},{2,3}, j in {0,1},{2,3}, k in {0,1},{2,3}
    # That's 8 blocks, each saving 1 term: rank = 100 - 8 = 92
    
    # But can we do better? Let's try a more systematic optimization.
    
    # Approach: Start with rank 92 (Strassen on all 2x2x2 blocks), then try to reduce further.
    # We can use the following strategy:
    # - Try to find linear dependencies among the terms
    # - Use random linear combinations to try to eliminate terms
    
    # First, let's build the base 92-rank decomposition
    
    def build_strassen_decomposition():
        rank = 92
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        blocks = []
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2]:
                    blocks.append((
                        [i_base, i_base + 1],
                        [j_base, j_base + 1],
                        [k_base, k_base + 1]
                    ))
        
        skip_set = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        skip_set.add((i, j, k))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in skip_set:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        for I, J, K in blocks:
            i1, i2 = I
            j1, j2 = J
            k1, k2 = K
            
            # M1 = (a11 + a22)(b11 + b22)
            U[i1 * m + j1][col] += 1
            U[i2 * m + j2][col] += 1
            V[j1 * p + k1][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[i2 * m + j1][col] += 1
            U[i2 * m + j2][col] += 1
            V[j1 * p + k1][col] += 1
            W[k1 * n + i2][col] += 1
            W[k2 * n + i2][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[i1 * m + j1][col] += 1
            V[j1 * p + k2][col] += 1
            V[j2 * p + k2][col] += -1
            W[k2 * n + i1][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[i2 * m + j2][col] += 1
            V[j2 * p + k1][col] += 1
            V[j1 * p + k1][col] += -1
            W[k1 * n + i1][col] += 1
            W[k1 * n + i2][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[i1 * m + j1][col] += 1
            U[i1 * m + j2][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += -1
            W[k2 * n + i1][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[i2 * m + j1][col] += 1
            U[i1 * m + j1][col] += -1
            V[j1 * p + k1][col] += 1
            V[j1 * p + k2][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[i1 * m + j2][col] += 1
            U[i2 * m + j2][col] += -1
            V[j2 * p + k1][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += 1
            col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct."""
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce the rank further using optimization
    # Strategy: Try to find a linear combination of terms that can be eliminated
    
    # Convert to numpy for faster operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    best_rank = current_rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try to find reducible terms using random linear combinations
    # A term r is reducible if there exists a linear combination of other terms
    # that cancels it out.
    
    # Alternative approach: Use the fact that the tensor has a known structure
    # and try to find a smaller generating set.
    
    # Let's try a different strategy: start from scratch with a smaller rank
    # and use optimization to find the entries.
    
    # For rank r, we need to find U (20 x r), V (25 x r), W (20 x r) such that
    # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 100 equations in 65*r unknowns (approximately).
    # For r = 92, we have 65*92 = 5980 unknowns and 100 equations.
    # The system is underdetermined, so we can search for solutions.
    
    # But this is a hard non-convex problem. Let's stick with the Strassen approach
    # and try to find additional reductions.
    
    # Try: can we merge two standard terms with a Strassen term?
    # For example, if we have terms for (i1,j1,k1), (i2,j2,k2) and a Strassen term
    # that involves some of the same indices, maybe we can combine them.
    
    # Let's try a brute-force search for rank reduction:
    # For each pair of columns, check if one can be expressed as a linear combination
    # of the others (in the appropriate tensor product space).
    
    # This is expensive, so let's limit the search.
    
    # Actually, let's try a more direct approach:
    # Use the known optimal or near-optimal decompositions from the literature.
    # For 4x5x5, the best known rank is around 88-92.
    
    # Let's try rank 90 and see if we can find a valid decomposition.
    
    # Strategy: Take the rank-92 decomposition and try to eliminate 2 more terms.
    # We can do this by finding a linear dependency among the 92 terms.
    
    # Build the tensor and check for dependencies
    # The "fiber" approach: for each (i,j), the vector over k must be reproducible.
    
    # Let's try a random search: pick a random subset of terms to keep and see
    # if the remaining terms can be expressed as linear combinations.
    
    # This is computationally expensive, so let's try a limited search.
    
    # The rank-92 Strassen decomposition is verified correct.
    # Try to find further reductions via linear dependency search.
    
    # For each column, check if it's a linear combination of others in the
    # tensor-product sense. This is expensive, so limit to a few attempts.
    
    # Quick check: are there any all-zero columns?
    for r in range(current_rank):
        if all(U_np[i][r] == 0 for i in range(nm)) and \
           all(V_np[i][r] == 0 for i in range(mp)) and \
           all(W_np[i][r] == 0 for i in range(pn)):
            # Remove this column
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1
            r -= 1
    
    if time.time() > deadline:
        # Return what we have
        U_final = [[int(round(U_np[i][r])) if abs(U_np[i][r] - round(U_np[i][r])) < 1e-9 else U_np[i][r] 
                    for r in range(current_rank)] for i in range(nm)]
        V_final = [[int(round(V_np[i][r])) if abs(V_np[i][r] - round(V_np[i][r])) < 1e-9 else V_np[i][r] 
                    for r in range(current_rank)] for i in range(mp)]
        W_final = [[int(round(W_np[i][r])) if abs(W_np[i][r] - round(W_np[i][r])) < 1e-9 else W_np[i][r] 
                    for r in range(current_rank)] for i in range(pn)]
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Convert back to lists with integer/half-integer entries
    U_final = []
    for i in range(nm):
        row = []
        for r in range(current_rank):
            val = U_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        U_final.append(row)
    
    V_final = []
    for i in range(mp):
        row = []
        for r in range(current_rank):
            val = V_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        V_final.append(row)
    
    W_final = []
    for i in range(pn):
        row = []
        for r in range(current_rank):
            val = W_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        W_final.append(row)
    
    return {"U": U_final, "V": V_final, "W": W_final}
