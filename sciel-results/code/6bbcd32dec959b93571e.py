import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # We'll try to find a better decomposition by using Strassen on 2x2x2 blocks
    # and then handling the remaining terms. But first, let's think about whether
    # we can pack more 2x2x2 blocks or use a different strategy.
    
    # For 3x5x5: n=3 rows of A, m=5 columns of A / rows of B, p=5 columns of B
    # Total terms = 3*5*5 = 75
    
    # A 2x2x2 Strassen block covers 8 terms with rank 7 (saves 1)
    # We can pack at most: floor(3/2)*floor(5/2)*floor(5/2) = 1*2*2 = 4 blocks
    # That covers 4*8 = 32 terms, leaving 75-32 = 43 terms
    # Rank = 4*7 + 43 = 28 + 43 = 71
    
    # Can we do better? What if we use some 2x1x2 or 1x2x2 blocks?
    # 2x1x2: covers 4 terms. Schoolbook needs 4, Strassen-like? 
    # For 2x1*1x2: C is 2x2, A is 2x1, B is 1x2
    # C00 = A0*B0, C01 = A0*B1, C10 = A1*B0, C11 = A1*B1
    # This is just 4 independent products, no savings possible.
    
    # 1x2x2: A is 1x2, B is 2x2, C is 1x2
    # C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11
    # This is 2 products of scalars with 2 terms each. No savings.
    
    # What about a 3x2x2 block? 
    # A is 3x2, B is 2x2, C is 3x2
    # 12 terms, schoolbook rank 12
    # Can we save? Probably not easily.
    
    # What if we use a different partition? Instead of 2x2x2 blocks aligned at even indices,
    # maybe we can find a better arrangement.
    
    # Actually, let's reconsider. The 4 blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    # cover rows 0-1 of A (i=0,1), all 5 columns of A/rows of B (j=0-4), all 5 cols of B (k=0-4)
    # Wait no. Each 2x2x2 block at (i0,j0,k0) covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    
    # Block at (0,0,0): i=0,1; j=0,1; k=0,1
    # Block at (0,2,0): i=0,1; j=2,3; k=0,1
    # Block at (0,0,2): i=0,1; j=0,1; k=2,3
    # Block at (0,2,2): i=0,1; j=2,3; k=2,3
    
    # So covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # That's 2*4*4 = 32 terms. Correct.
    
    # Uncovered: 
    # i=2: all j,k (5*5=25 terms)
    # i=0,1: j=4, all k (2*1*5=10 terms)
    # i=0,1: j=0,1,2,3, k=4 (2*4*1=8 terms)
    # Total: 25+10+8 = 43. Correct.
    
    # What if we use blocks at different positions?
    # E.g., (0,0,0), (0,2,2), (1,0,2), (1,2,0)?
    # (0,0,0): i=0,1; j=0,1; k=0,1
    # (0,2,2): i=0,1; j=2,3; k=2,3
    # (1,0,2): i=1,2; j=0,1; k=2,3
    # (1,2,0): i=1,2; j=2,3; k=0,1
    
    # Covered: 
    # From (0,0,0) and (0,2,2): i=0,1; j=0,1,2,3; k=0,1,2,3 (but only specific (j,k) pairs)
    # Actually (0,0,0) covers (j,k) in {(0,0),(0,1),(1,0),(1,1)} for i=0,1
    # (0,2,2) covers (j,k) in {(2,2),(2,3),(3,2),(3,3)} for i=0,1
    # (1,0,2) covers (j,k) in {(0,2),(0,3),(1,2),(1,3)} for i=1,2
    # (1,2,0) covers (j,k) in {(2,0),(2,1),(3,0),(3,1)} for i=1,2
    
    # So for i=0: (j,k) in {(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)}
    # For i=1: all 16 (j,k) pairs! Since (0,0,0) gives first 4, (0,2,2) gives next 4, (1,0,2) gives next 4, (1,2,0) gives last 4.
    # For i=2: (j,k) in {(0,2),(0,3),(1,2),(1,3),(2,0),(2,1),(3,0),(3,1)}
    
    # Total covered: 8 + 16 + 8 = 32. Same count.
    
    # Uncovered:
    # i=0: (j,k) in {(0,2),(0,3),(1,2),(1,3),(2,0),(2,1),(3,0),(3,1)} - 8 terms
    # i=2: (j,k) in {(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)} - 8 terms
    # i=1: none
    # j=4: all i,k (3*1*5=15 terms)
    # k=4: i=0,1,2; j=0,1,2,3 (3*4*1=12 terms)
    # Total: 8+8+15+12 = 43. Same.
    
    # So the number of uncovered terms is the same regardless of block placement.
    # The question is whether we can save on the uncovered terms.
    
    # Idea: for the uncovered terms, check if any group of them can be handled
    # with fewer than schoolbook rank by exploiting structure.
    
    # Let's build the uncovered set for the first placement and analyze.
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    covered = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    uncovered.append((i,j,k))
    
    # Try to find subgroups that can be compressed.
    # For each (i,j) pair, collect the k values.
    from collections import defaultdict
    by_ij = defaultdict(list)
    for (i,j,k) in uncovered:
        by_ij[(i,j)].append(k)
    
    # If for some (i,j) we have all 5 k values, that's a 1x1x5 problem.
    # 1x1 * 1x5: C is 1x5, A is 1x1, B is 1x5. 5 products, no savings.
    
    # For (i,k) pairs:
    by_ik = defaultdict(list)
    for (i,j,k) in uncovered:
        by_ik[(i,k)].append(j)
    
    # For (j,k) pairs:
    by_jk = defaultdict(list)
    for (i,j,k) in uncovered:
        by_jk[(j,k)].append(i)
    
    # Check for 2x2x1 or 2x1x2 or 1x2x2 subproblems in uncovered.
    # 2x2x1: two i values, two j values, one k. 4 terms.
    # But 2x2 * 2x1 = 2x1. Schoolbook is 4. Can we do better?
    # C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # This is 4 scalar products. No algebraic trick saves a product here.
    
    # It seems 71 is the natural answer for this approach.
    # Let's just build it efficiently.
    
    total_rank = len(blocks) * 7 + len(uncovered)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0, c0):
        c = c0
        U[i0*m+j0][c] = 1
        U[(i0+1)*m+(j0+1)][c] = 1
        V[j0*p+k0][c] = 1
        V[(j0+1)*p+(k0+1)][c] = 1
        W[k0*n+i0][c] = 1
        W[(k0+1)*n+(i0+1)][c] = 1
        
        c = c0+1
        U[(i0+1)*m+j0][c] = 1
        U[(i0+1)*m+(j0+1)][c] = 1
        V[j0*p+k0][c] = 1
        W[k0*n+(i0+1)][c] = 1
        W[(k0+1)*n+(i0+1)][c] = -1
        
        c = c0+2
        U[i0*m+j0][c] = 1
        V[j0*p+(k0+1)][c] = 1
        V[(j0+1)*p+(k0+1)][c] = -1
        W[(k0+1)*n+i0][c] = 1
        W[(k0+1)*n+(i0+1)][c] = 1
        
        c = c0+3
        U[(i0+1)*m+(j0+1)][c] = 1
        V[(j0+1)*p+k0][c] = 1
        V[j0*p+k0][c] = -1
        W[k0*n+i0][c] = 1
        W[k0*n+(i0+1)][c] = 1
        
        c = c0+4
        U[i0*m+j0][c] = 1
        U[i0*m+(j0+1)][c] = 1
        V[(j0+1)*p+(k0+1)][c] = 1
        W[k0*n+i0][c] = -1
        W[(k0+1)*n+i0][c] = 1
        
        c = c0+5
        U[(i0+1)*m+j0][c] = 1
        U[i0*m+j0][c] = -1
        V[j0*p+k0][c] = 1
        V[j0*p+(k0+1)][c] = 1
        W[(k0+1)*n+(i0+1)][c] = 1
        
        c = c0+6
        U[i0*m+(j0+1)][c] = 1
        U[(i0+1)*m+(j0+1)][c] = -1
        V[(j0+1)*p+k0][c] = 1
        V[(j0+1)*p+(k0+1)][c] = 1
        W[k0*n+i0][c] = 1
    
    for i0, j0, k0 in blocks:
        add_strassen(i0, j0, k0, col)
        col += 7
    
    for (i,j,k) in uncovered:
        if time.time() > deadline:
            break
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
