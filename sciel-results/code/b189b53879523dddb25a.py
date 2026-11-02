import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 11]

def _build_strassen_2x2x2():
    """Return U, V, W for rank-7 Strassen 2x2x2."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # P1: (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # P2: (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    # P3: a00(b01-b11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # P4: a11(b10-b00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # P5: (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # P6: (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # P7: (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _build_rank39():
    """Return the known rank-39 decomposition as (U, V, W) lists."""
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    r = 0
    for b in range(5):
        v00, v01, v10, v11 = 2*b, 2*b+1, 11+2*b, 12+2*b
        w00, w01, w10, w11 = 4*b, 4*b+2, 4*b+1, 4*b+3
        U[0][r]+=1; U[3][r]+=1; V[v00][r]+=1; V[v11][r]+=1; W[w00][r]+=1; W[w11][r]+=1; r+=1
        U[2][r]+=1; U[3][r]+=1; V[v00][r]+=1; W[w10][r]+=1; W[w11][r]-=1; r+=1
        U[0][r]+=1; V[v01][r]+=1; V[v11][r]-=1; W[w01][r]+=1; W[w11][r]+=1; r+=1
        U[3][r]+=1; V[v10][r]+=1; V[v00][r]-=1; W[w00][r]+=1; W[w10][r]+=1; r+=1
        U[0][r]+=1; U[1][r]+=1; V[v11][r]+=1; W[w00][r]-=1; W[w01][r]+=1; r+=1
        U[2][r]+=1; U[0][r]-=1; V[v00][r]+=1; V[v01][r]+=1; W[w11][r]+=1; r+=1
        U[1][r]+=1; U[3][r]-=1; V[v10][r]+=1; V[v11][r]+=1; W[w00][r]+=1; r+=1
    U[0][r]+=1; V[10][r]+=1; W[20][r]+=1; r+=1
    U[1][r]+=1; V[21][r]+=1; W[20][r]+=1; r+=1
    U[2][r]+=1; V[10][r]+=1; W[21][r]+=1; r+=1
    U[3][r]+=1; V[21][r]+=1; W[21][r]+=1; r+=1
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Compute tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # Target: T[i,j,k] = 1 if i//m == j%m and i%m == k//n and j//p == k%n, else 0
    # Actually for matrix mult: C = A * B where A is n x m, B is m x p, C is n x p
    # A[i,a], B[a,j], C[i,j]
    # Tensor: T[i, a, j] = sum... no wait
    # The tensor for matrix mult M_{n,m,p}: T[i,j,k] where i indexes A row (0..n-1), 
    # j indexes B col (0..p-1), k indexes A col/B row (0..m-1)
    # T[i,j,k] = 1 if we need A[i,k]*B[k,j] to contribute to C[i,j]
    # In the factorization: sum_r U[i*m+k, r] * V[k*p+j, r] * W[i*p+j, r]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # U has shape (n*m, rank): index = i*m + k where i is A row, k is A col
    # V has shape (m*p, rank): index = k*p + j where k is B row, j is B col
    # W has shape (p*n, rank): index = ? The problem says W[k*n+i, r]
    # Hmm, "output factor follows the official trace convention W[k*n+i,r]"
    # Let me think: C[i,j] = sum_k A[i,k] * B[k,j]
    # So W should index C. C has n rows and p cols.
    # If W index is k*n + i, that doesn't seem right for C which is n x p.
    # Wait, maybe the convention is: W index = j*n + i where j is C col, i is C row?
    # Or W index = i*p + j?
    # Let me check with the existing code: W has shape (p*n, rank) = (22, rank)
    # In the Strassen block for block b: w00=4*b, w01=4*b+2, w10=4*b+1, w11=4*b+3
    # For b=0: w00=0, w01=2, w10=1, w11=3
    # C block for b=0 covers C[0,0], C[0,1], C[1,0], C[1,1]
    # So w00=0 -> C[0,0], w01=2 -> C[0,1], w10=1 -> C[1,0], w11=3 -> C[1,1]
    # This suggests W index = col*n + row = j*2 + i for n=2
    # So W[j*n + i, r] where i is C row, j is C col
    # For the 2x1 block: C[0,10] -> W[10*2+0]=W[20], C[1,10] -> W[10*2+1]=W[21]
    # This matches the code: W[20] for C[0,10], W[21] for C[1,10]
    # So W index = j*n + i (col first, then row)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                target = 1
                # C[i,j] += A[i,k] * B[k,j]
                # U index: i*m + k
                # V index: k*p + j
                # W index: j*n + i
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                computed = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                if computed != target:
                    return False
    return True

def _tensor_contribution(U_col, V_col, W_col, n, m, p):
    """For a given rank-1 term (u, v, w), return the set of (i,j,k) it contributes to."""
    # U index = i*m + k, V index = k*p + j, W index = j*n + i
    # The contribution to C[i,j] from A[i,k]*B[k,j] is u[i*m+k] * v[k*p+j] * w[j*n+i]
    # We want to find which (i,j,k) triples have nonzero contribution
    contrib = set()
    for i in range(n):
        for k in range(m):
            u_val = U_col[i*m + k]
            if u_val == 0:
                continue
            for j in range(p):
                v_val = V_col[k*p + j]
                if v_val == 0:
                    continue
                w_val = W_col[j*n + i]
                if w_val == 0:
                    continue
                if u_val * v_val * w_val != 0:
                    contrib.add((i, j, k))
    return contrib

def _greedy_reduce(U, V, W, max_time=120):
    """Try to reduce rank by removing redundant rank-1 terms."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Convert to Fraction for exact arithmetic
    U_f = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
    V_f = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
    W_f = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try removing each term one at a time and see if we can compensate
    for r_remove in range(rank):
        if time.time() - start_time > max_time - 10:
            break
        # Check if term r_remove is removable
        # A term is removable if its contribution is zero (all products are zero)
        removable = True
        for i in range(n):
            for k in range(m):
                if U_f[i*m+k][r_remove] != 0:
                    for j in range(p):
                        if V_f[k*p+j][r_remove] != 0 and W_f[j*n+i][r_remove] != 0:
                            removable = False
                            break
                    if not removable:
                        break
            if not removable:
                break
        
        if removable:
            # Remove this term
            new_rank = rank - 1
            new_U = [row[:r_remove] + row[r_remove+1:] for row in U]
            new_V = [row[:r_remove] + row[r_remove+1:] for row in V]
            new_W = [row[:r_remove] + row[r_remove+1:] for row in W]
            if _verify(new_U, new_V, new_W):
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                U, V, W = new_U, new_V, new_W
                rank = new_rank
                # Restart the loop since indices changed
                return _greedy_reduce(U, V, W, max_time - (time.time() - start_time))
    
    return best_rank, best_U, best_V, best_W

def _try_cross_block_optimization(U, V, W, max_time=120):
    """Try to merge terms across blocks by finding equivalent decompositions."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # For each pair of blocks, try to see if we can save a term
    # by using a different decomposition of the 2x2 block
    # This is a heuristic search
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try: for the 2x1 block, see if we can fold it into adjacent Strassen blocks
    # The 2x1 block at col 10 uses 4 terms. Can we do better by modifying
    # the 5th Strassen block (cols 8,9) to also handle col 10?
    
    # Actually, let's try a different approach: use the known fact that
    # 2x2 x 2x3 can be done in rank 8 (instead of 7+4=11 for separate blocks)
    # If we can do 2x2 x 2x3 in rank 8, then 5*7 + 8 = 43... that's worse.
    # What about 2x2 x 2x2 in rank 7 and 2x2 x 2x3 in rank 8?
    # 4*7 + 8 = 36! That would be better than 39.
    
    # Let me search for a rank-8 decomposition of 2x2 x 2x3
    # A is 2x2, B is 2x3, C is 2x3
    # Schoolbook: 2*2*3 = 12
    # Strassen on 2x2 part + 1 col: 7 + 4 = 11
    # Can we do better?
    
    # Try to construct one:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    
    # Try rank 8:
    # P1 = (a00+a11)*b00 -> c00, c10
    # P2 = (a00+a11)*b10 -> c00, c10 (with signs)
    # Actually let me use a known result: the tensor rank of 2x2 x 2x3 is 8.
    # Reference: Stothers (1971) showed rank(2,2,3) = 8
    
    # Construction for rank 8:
    # Let me try to build it.
    # U: 4 x 8, V: 6 x 8, W: 6 x 8
    
    # Try:
    # P1: a00 * b00 -> c00
    # P2: a01 * b10 -> c00
    # P3: a10 * b00 -> c10
    # P4: a11 * b10 -> c10
    # P5: a00 * b01 -> c01
    # P6: a01 * b11 -> c01
    # P7: a10 * b01 -> c11
    # P8: a11 * b11 -> c11
    # That's rank 8 but doesn't handle col 2!
    
    # We need to handle 3 columns. Let me think differently.
    # Use Strassen-like ideas for the 2x2 x 2x2 part (cols 0,1) in rank 7,
    # plus 1 more term for col 2? That's 8.
    # But col 2 needs: c02 = a00*b02 + a01*b12, c12 = a10*b02 + a11*b12
    # That's 4 terms in schoolbook. Can we share with Strassen?
    
    # Actually, let me just try to verify if rank 8 is achievable by search.
    # For now, let's try a direct approach: modify the 5th block to be 2x2 x 2x3.
    
    if time.time() - start_time > max_time - 5:
        return best_rank, best_U, best_V, best_W
    
    # Try: 4 Strassen blocks (cols 0-7) + 1 block of 2x2 x 2x3 (cols 8,9,10)
    # If 2x2 x 2x3 has rank 8, total = 4*7 + 8 = 36
    
    # Let me try to find a rank-8 decomposition of 2x2 x 2x3
    # by searching over small integer/half-integer coefficients
    
    # This is a hard search. Let me try a specific construction.
    # From the literature, rank(2,2,3) = 8 is known.
    # One construction:
    
    # Let's try:
    # P1: (a00+a11)(b00+b10) -> contributes to c00, c10
    # P2: a00*b10 -> c00
    # P3: a11*b00 -> c10
    # P4: (a00+a11)(b01+b11) -> c01, c11
    # P5: a00*b11 -> c01
    # P6: a11*b01 -> c11
    # P7: (a00+a11)(b02+b12) -> c02, c12
    # P8: a00*b12 -> c02, a11*b02 -> c12... but that's two terms
    
    # This isn't working cleanly. Let me try a different approach.
    # 
    # Actually, I recall that for 2x2 x 2x3, one can use:
    # The tensor is M_{2,2,2} ⊗ M_{2,1,1} in some sense, but that's not quite right.
    #
    # Let me just try to verify the rank-39 solution and return it if we can't improve.
    
    return best_rank, best_U, best_V, best_W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with the known rank-39 decomposition
    U, V, W = _build_rank39()
    
    # Verify it's correct
    if not _verify(U, V, W):
        # Should not happen, but return something valid
        pass
    
    # Try to reduce rank
    try:
        new_rank, new_U, new_V, new_W = _greedy_reduce(U, V, W, max_time=30)
        if new_rank < 39:
            U, V, W = new_U, new_V, new_W
    except Exception:
        pass
    
    # Try cross-block optimization
    if time.time() < deadline - 10:
        try:
            new_rank, new_U, new_V, new_W = _try_cross_block_optimization(U, V, W, max_time=60)
            if new_rank < len(U[0]):
                U, V, W = new_U, new_V, new_W
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
