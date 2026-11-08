import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 75
    
    # We'll try to find a rank-minimal decomposition.
    # Known: rank >= 2*max(n,m,p) + 3 = 13 (Bjorklund et al. bound for 3x5x5 is actually higher)
    # Actually for 3x5x5, the best known is 71 (from the current solution).
    # Let me try a different approach: use a smarter block partition.
    
    # Strategy: Try different block sizes and Strassen-like decompositions.
    # For 3x5x5, we can try:
    # 1. 3x2x2 blocks (Strassen rank 7 each) + schoolbook for the rest
    # 2. 2x3x3 blocks
    # 3. 3x3x2 blocks
    # 4. Mixed sizes
    
    # Let's enumerate possible block decompositions and compute their ranks.
    # A block at (i0, j0, k0) with size (si, sj, sk) covers:
    #   U indices: (i0..i0+si-1)*m + (j0..j0+sj-1)
    #   V indices: (j0..j0+sj-1)*p + (k0..k0+sk-1)
    #   W indices: (k0..k0+sk-1)*n + (i0..i0+si-1)
    
    # For a 2x2x2 block, Strassen gives rank 7 vs schoolbook 8, saving 1.
    # For larger blocks, the saving may be more complex.
    
    # Let me try: use 2x2x2 Strassen blocks where possible, then fill the rest with schoolbook.
    # The question is: what's the best placement of 2x2x2 blocks?
    
    # Current solution uses 4 blocks: (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    # This covers i=0,1; j=0,1,2,3; k=0,1,2,3 => 2*4*4=32 terms covered by Strassen
    # Remaining: 75-32=43 schoolbook terms
    # Total rank: 4*7+43=71
    
    # Can we do better? Let's try other placements.
    # Key insight: we want to maximize the number of 2x2x2 blocks we can fit.
    # Max number of disjoint 2x2x2 blocks in 3x5x5:
    #   i: 3 rows -> can fit 1 full block of height 2 (rows 0-1 or 1-2)
    #   j: 5 cols -> can fit 2 full blocks of width 2 (cols 0-1 and 2-3, or 1-2 and 3-4)
    #   k: 5 rows -> can fit 2 full blocks of depth 2 (rows 0-1 and 2-3, or 1-2 and 3-4)
    # So max 1*2*2=4 blocks, same as current.
    
    # But wait - we can also use non-2x2x2 Strassen-like decompositions.
    # E.g., a 3x2x2 block: rank ? vs schoolbook 12
    # A 2x3x2 block: rank ?
    # A 2x2x3 block: rank ?
    
    # For a 3x2x2 block, we can think of it as 3 separate 1x2x2 blocks.
    # A 1x2x2 block: A[i,j0..j0+1] * B[j0..j0+1, k0..k0+1] -> C[i, k0..k0+1]
    # This is: C[i,k0] = A[i,j0]*B[j0,k0] + A[i,j0+1]*B[j0+1,k0]
    #          C[i,k0+1] = A[i,j0]*B[j0,k0+1] + A[i,j0+1]*B[j0+1,k0+1]
    # This is just 2 terms (no Strassen saving for 1x2x2).
    
    # What about a 2x3x2 block?
    # Think of it as 2 1x3x2 blocks. Each 1x3x2 is 3 1x1x2... no Strassen saving.
    
    # So 2x2x2 is the smallest block where Strassen gives a saving.
    
    # Alternative: use a different algebraic identity for a 2x2x2 block.
    # The standard Strassen 7-multiplication is optimal for 2x2x2.
    
    # Let me try a completely different approach: use a random search / optimization
    # to find a lower-rank decomposition.
    
    # Actually, let me check: what's the theoretical lower bound?
    # For n x m x p, rank >= 2*max(n,m,p) + 3 = 13 (for n=m=p, it's 2n+3)
    # For 3x5x5: 2*5+3=13. But this is a weak bound.
    # Better bound: rank >= (nm + mp + pn)/3 * ... no, that's not right.
    # Actually, rank >= max(nm, mp, pn) = 15 (since 3*5=15, 5*5=25, 5*3=15)
    # So rank >= 25? No, that's not a valid lower bound.
    
    # The correct lower bound: rank >= (nm + mp + pn)/... 
    # Actually, the best known lower bound for 3x5x5 is 40 (from literature).
    # Best known upper bound is 71 (current solution).
    
    # Given the 120s time limit, let me try a randomized search for a better decomposition.
    # Approach: start from the rank-71 solution and try to merge columns.
    
    # Build the rank-71 solution first.
    def build_base_solution():
        blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
        total_strassen_rank = len(blocks) * 7
        strassen_terms_count = len(blocks) * 8
        schoolbook_terms_count = total_terms - strassen_terms_count
        total_rank = total_strassen_rank + schoolbook_terms_count
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i0, j0, k0, col_offset):
            c = col_offset + 0
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 1
            U[(i0+1)*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += -1
            
            c = col_offset + 2
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 3
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            
            c = col_offset + 4
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1
            W[(k0+1)*n + i0][c] += 1
            
            c = col_offset + 5
            U[(i0+1)*m + j0][c] += 1
            U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 6
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
        
        for i0, j0, k0 in blocks:
            add_strassen_block(i0, j0, k0, col)
            col += 7
        
        covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered_3d[i0+di][j0+dj][k0+dk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return U, V, W, col
    
    U, V, W, rank = build_base_solution()
    
    if time.time() > deadline:
        return {"U": U, "V": W, "W": W}  # Fallback
    
    # Try to reduce rank by finding redundant columns or merging.
    # A column c is redundant if there exist other columns c1, c2 such that
    # U[:,c] = a*U[:,c1] + b*U[:,c2], etc.
    
    # Simpler approach: try replacing pairs of columns with a single column.
    # For each pair (c1, c2), check if there's a linear combination that matches.
    
    # This is expensive. Let's try a heuristic: look for columns that are
    # "similar" in the sense that their support overlaps significantly.
    
    # Convert to numpy for efficient operations.
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Verify the decomposition.
    # The tensor T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Actually, the convention is: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta (i.e., the identity in the appropriate sense)
    # The factorization is: T = sum_c U_c (outer) V_c (outer) W_c
    # where U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n
    # and T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let me verify with a small test.
    # Actually, let's just try to reduce the rank by removing linearly dependent columns.
    
    def is_valid(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        r = U_np.shape[1]
        
        # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For this to be valid, T[i,j,k] should be 1 if k==i (for the identity part)
        # Wait, no. For matrix multiplication C = A*B, we have:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
        # Actually no, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
        # So T[i,j,k] = 1 for all i,j,k where the indices are compatible.
        # In our case, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k].
        
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
        
        # This is expensive to check fully. Let's just check a few random indices.
        np.random.seed(42)
        for _ in range(100):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = 0.0
            for c in range(r):
                val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
            if abs(val - 1.0) > 1e-6:
                return False
        return True
    
    # Try to reduce rank by checking if any column is a linear combination of others.
    # This is O(r^3 * dim) which might be too slow.
    
    # Simpler: try removing each column and checking if the remaining still works.
    # This is O(r^2) checks, each O(r * dim) = O(71 * 75 * something) which might be OK.
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try removing one column at a time.
    for col_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Remove column col_to_remove
        mask = [i for i in range(rank) if i != col_to_remove]
        U_trial = [row[mask] for row in U]
        V_trial = [row[mask] for row in V]
        W_trial = [row[mask] for row in W]
        
        # Quick check: is the decomposition still valid?
        # Check a few random (i,j,k) indices.
        np.random.seed(col_to_remove)
        valid = True
        for _ in range(20):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = 0.0
            for c in range(len(mask)):
                val += U_trial[i*m+j][c] * V_trial[j*p+k][c] * W_trial[k*n+i][c]
            if abs(val - 1.0) > 1e-6:
                valid = False
                break
        
        if valid:
            best_rank = len(mask)
            best_U, best_V, best_W = U_trial, V_trial, W_trial
            # Update U, V, W to the reduced version for further attempts
            U, V, W = U_trial, V_trial, W_trial
            rank = best_rank
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
