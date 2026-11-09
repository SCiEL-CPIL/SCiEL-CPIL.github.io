import numpy as np
import time
from itertools import product, combinations
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 5, 5
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (25, rank)  
    # W shape: (p*n, rank) = (10, rank)
    
    # Build target tensor as sparse list of (a,b,c) indices where T[a,b,c]=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target_indices.append((a, b, c))
    
    # Strategy: Try to find rank-r decomposition
    # For each rank r, we need to assign to each of the r terms a "pattern"
    # that covers certain (i,j,k) triples with coefficients
    
    # Key insight: Since n=2, we can think of this as two 1x5x5 problems
    # that share the middle dimension structure
    
    # Try structured decompositions based on partitioning of j and k dimensions
    # For the 5x5 middle part, try various block structures
    
    # First, try the known rank-46 construction (Strassen on 2x2 blocks + schoolbook)
    # Then try to find lower rank
    
    # Let's try a more systematic approach:
    # For rank r, each term t has:
    #   U[:, t] in R^10, V[:, t] in R^25, W[:, t] in R^10
    # The constraint is: sum_t U[a,t]*V[b,t]*W[c,t] = 1 if (a,b,c) in target, 0 otherwise
    
    # This is a bilinear system. For fixed U and V, W is determined linearly.
    # Let's try: fix U and V to have sparse structure, then solve for W.
    
    # Approach: Try rank values from 40 to 46, for each try random structured initializations
    # and check if the resulting decomposition is valid.
    
    # Actually, let's be smarter. The minimum rank for 2x5x5 is at most 46 (known).
    # Can we do better? Let's try rank 44, 45, 46.
    
    # For a given rank r, we can set up the problem as:
    # Find U (10xr), V (25xr), W (10xr) such that the tensor matches.
    # This is 45r variables with 2500 constraints (but only 250 are equality to 1, rest to 0).
    
    # Let's try a different approach: use the fact that the problem has structure.
    # The (i,j,k) -> (a,b,c) mapping is: a=i*5+j, b=j*5+k, c=k*2+i
    # So for fixed j, the sub-problem is 2x1x5 (for varying i,k) with shared j.
    
    # Let me try to construct a rank-45 or lower solution by modifying the rank-46 one.
    
    # Start with the rank-46 solution and see if we can merge any terms.
    
    def build_rank46():
        rank = 46
        U = np.zeros((10, rank))
        V = np.zeros((25, rank))
        W = np.zeros((10, rank))
        
        col = 0
        
        def add_term(i, j, k):
            nonlocal col
            if col >= rank:
                raise ValueError("Rank exceeded")
            U[i*5+j, col] = 1
            V[j*5+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
        
        def add_strassen_block(ja, jb, ka, kb):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[0*5+ja, col] = 1
            U[1*5+jb, col] = 1
            V[ja*5+ka, col] = 1
            V[jb*5+kb, col] = 1
            W[ka*2+0, col] = 1
            W[kb*2+1, col] = 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[1*5+ja, col] = 1
            U[1*5+jb, col] = 1
            V[ja*5+ka, col] = 1
            W[ka*2+1, col] = 1
            W[kb*2+1, col] = -1
            col += 1
            
            # M3 = A00(B01-B11)
            U[0*5+ja, col] = 1
            V[ja*5+kb, col] = 1
            V[jb*5+kb, col] = -1
            W[kb*2+0, col] = 1
            W[kb*2+1, col] = 1
            col += 1
            
            # M4 = A11(B10-B00)
            U[1*5+jb, col] = 1
            V[jb*5+ka, col] = 1
            V[ja*5+ka, col] = -1
            W[ka*2+0, col] = 1
            W[ka*2+1, col] = 1
            col += 1
            
            # M5 = (A00+A01)B11
            U[0*5+ja, col] = 1
            U[0*5+jb, col] = 1
            V[jb*5+kb, col] = 1
            W[ka*2+0, col] = -1
            W[kb*2+0, col] = 1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[1*5+ja, col] = 1
            U[0*5+ja, col] = -1
            V[ja*5+ka, col] = 1
            V[ja*5+kb, col] = 1
            W[kb*2+1, col] = 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[0*5+jb, col] = 1
            U[1*5+jb, col] = -1
            V[jb*5+ka, col] = 1
            V[jb*5+kb, col] = 1
            W[ka*2+0, col] = 1
            col += 1
        
        add_strassen_block(0, 1, 0, 1)
        add_strassen_block(2, 3, 0, 1)
        add_strassen_block(0, 1, 2, 3)
        add_strassen_block(2, 3, 2, 3)
        
        for i in [0, 1]:
            for k in [0, 1, 2, 3]:
                add_term(i, 4, k)
        
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                add_term(i, j, 4)
        
        for i in [0, 1]:
            add_term(i, 4, 4)
        
        return U, V, W, rank
    
    # Try rank 46 first as baseline
    U46, V46, W46, _ = build_rank46()
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U46, V46, W46)
    T_target = np.zeros((10, 25, 10))
    for (a, b, c) in target_indices:
        T_target[a, b, c] = 1.0
    
    error46 = np.max(np.abs(T_recon - T_target))
    
    if error46 > 1e-9:
        # Should not happen, but fallback to schoolbook
        rank_school = 50
        U_s = np.zeros((10, rank_school))
        V_s = np.zeros((25, rank_school))
        W_s = np.zeros((10, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*5+j, c] = 1
                    V_s[j*5+k, c] = 1
                    W_s[k*2+i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Now try to find lower rank by searching
    # For each candidate rank r < 46, try to find a solution
    
    # Idea: Use the structure that n=2. Think of the tensor as two "slices":
    # Slice i=0: T[j, j*5+k, k*2] for j,k in 0..4
    # Slice i=1: T[5+j, j*5+k, k*2+1] for j,k in 0..4
    
    # Each slice is a 5x5 "matrix multiplication" tensor (but with different index structure)
    # Actually, for fixed i, the non-zero entries are at (i*5+j, j*5+k, k*2+i)
    # This is like a 5x5 matrix product but with permuted indices.
    
    # Let me try a randomized search for lower rank.
    # For rank r, initialize U, V, W with small random integers in {-1, 0, 1}
    # Then check if the tensor matches.
    
    # But this is very unlikely to work with random init. Let's be more structured.
    
    # Alternative: Try to see if we can reduce the rank by combining terms in the rank-46 solution.
    # Check if any two terms in the rank-46 solution can be merged.
    
    # For now, return the rank-46 solution.
    # The search for lower rank is complex and may not succeed within time limits.
    
    if time.time() > deadline - 5:
        return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
    
    # Try a simple search: for rank 45, 44, 43, try to find a solution
    # by modifying the rank-46 solution.
    
    # One idea: In the schoolbook tail terms, see if any can be combined.
    # The tail terms are:
    # (i,4,k) for i in {0,1}, k in {0,1,2,3} -> 8 terms
    # (i,j,4) for i in {0,1}, j in {0,1,2,3} -> 8 terms  
    # (i,4,4) for i in {0,1} -> 2 terms
    # Total 18 tail terms.
    
    # Can any two tail terms be combined into one?
    # Term (i1,4,k1) and (i2,4,k2) can be combined if i1=i2 and we can find a single term
    # that covers both. But a single term has fixed i,j,k, so no.
    
    # What about combining a tail term with a Strassen term?
    # This is complex. Let's just return rank 46.
    
    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
