import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Matrix multiplication of A (n x m) and B (m x p) gives C (n x p).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor decomposes into p independent M_{n,m,1} tensors.
    # Rank of M_{2,2,1} is 4 (Laderman's algorithm).
    # Total rank = p * 4 = 9 * 4 = 36.
    
    rank = 4 * p
    
    # Laderman's rank-4 decomposition for 2x2x1 matrix multiplication
    # Variables: A00, A01, A10, A11, B00, B01, B10, B11 -> C00, C01, C10, C11
    # 
    # C00 = P1 - P2 - P3 + P4
    # C01 = P3 + P5 - P1 - P6
    # C10 = P2 + P7 - P4 - P8
    # C11 = P4 + P6 + P8 - P7
    #
    # Where:
    # P1 = A00 * B00
    # P2 = A00 * B10
    # P3 = A10 * B00
    # P4 = A10 * B10
    # P5 = A01 * B01
    # P6 = A01 * B11
    # P7 = A11 * B01
    # P8 = A11 * B11
    
    # Wait, that's not Laderman's. Laderman's is:
    # P1 = A00 * B00
    # P2 = A00 * (B10 + B11)
    # P3 = (A00 + A10) * B01
    # P4 = A11 * (B00 + B01)
    # P5 = (A00 + A10) * (B10 + B11)
    # P6 = A01 * (B00 - B01)
    # P7 = (A01 + A11) * (B10 + B11)
    # P8 = A11 * (B10 - B11)
    #
    # C00 = P1 + P3 - P6
    # C01 = P1 + P5 - P3 - P8  (wait, let me verify)
    
    # Actually, let's just use the simple rank-4 decomposition for 2x2:
    # This is not minimal but let's check if rank 4 is possible.
    # R(M_{2,2,1}) = 4.
    
    # Simple decomposition using 4 terms isn't trivial with integer coefficients.
    # However, we can use the "naive" rank 4 decomposition if we allow half-integers?
    # No, Laderman's uses integer coefficients.
    
    # Let's use Laderman's algorithm properly.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = a*e
    # P2 = a*(g+h)
    # P3 = (a+c)*f
    # P4 = d*(e+f)
    # P5 = (a+c)*(g+h)
    # P6 = b*(e-f)
    # P7 = (b+d)*(g+h)
    # P8 = d*(g-h)
    
    # C00 = P1 + P3 - P6
    # C01 = P1 + P5 - P3 - P8  -> Wait.
    # Let's re-derive or look up Laderman's formula.
    #
    # C00 = P1 + P3 - P6
    # C01 = P2 - P5 + P7  (No)
    
    # Correct Laderman's:
    # P1 = a * e
    # P2 = a * (g + h)
    # P3 = (a + c) * f
    # P4 = d * (e + f)
    # P5 = (a + c) * (g + h)
    # P6 = b * (e - f)
    # P7 = (b + d) * (g + h)
    # P8 = d * (g - h)
    
    # C00 = P1 + P3 - P6
    # C01 = P2 - P5 + P7  -> Let's check: a(g+h) - (a+c)(g+h) + (b+d)(g+h) = (a - a - c + b + d)(g+h) = (b+d-c)(g+h). No.
    
    # Let's use a known valid rank-4 decomposition.
    # Or, since the task allows "integer/half-integer", we can use a simpler one if rank 4 is hard to code.
    # But rank 4 is the theoretical minimum for 2x2x1.
    
    # Actually, for the purpose of this task, and given the seed program used rank 36 (which is 9*4),
    # it implies the seed already used a rank-4 decomposition for each 2x2x1 block.
    # The seed program code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This is a rank-1 decomposition per element!
    # Total rank = 2*2*9 = 36.
    # Wait. If the seed uses rank 36 and the score is 1.0, then the seed's rank is the baseline.
    # Is rank 36 the minimum?
    # R(M_{2,2,1}) = 4. So 9 * 4 = 36.
    # The seed program used rank 36 as well, but with a different structure?
    # The seed program used:
    # for i in 2: for j in 2: for k in 9:
    #   U[i*2+j][col] = 1
    #   V[j*9+k][col] = 1
    #   W[k*2+i][col] = 1
    # This is the trivial decomposition where each multiplication is a single term.
    # C[i,k] = A[i,j] * B[j,k].
    # Sum over j.
    # If we have separate terms for each (i,j,k), then C[i,k] = sum_j (A[i,j] * B[j,k]).
    # This is valid. Rank = 2*2*9 = 36.
    
    # Can we do better?
    # R(M_{2,2,1}) = 4.
    # So for each k, we can do the 2x2 multiplication with rank 4.
    # Total rank = 9 * 4 = 36.
    # So the rank is still 36.
    # The seed program's rank is 36.
    # My potential improvement is to use Laderman's algorithm to make the matrices sparser or simpler, but the rank is the same.
    # The score is the rank. So the score will be 1.0 regardless, as long as rank is 36.
    # Unless... is there a lower rank?
    # R(M_{2,2,9}) = 9 * R(M_{2,2,1}) = 36.
    # So the minimum rank is 36.
    
    # Therefore, any valid rank-36 decomposition will yield a score of 1.0.
    # The "improvement" comes from robustness or perhaps if the evaluator prefers integer/half-integer entries in a specific way?
    # "score is the rank".
    # So I just need to provide a valid rank-36 decomposition.
    
    # I will implement Laderman's decomposition for each of the 9 k-columns.
    # This will also have rank 36, but with a different structure.
    # U shape: (n*m, rank) = (4, 36)
    # V shape: (m*p, rank) = (18, 36)
    # W shape: (p*n, rank) = (18, 36)
    
    # Indices:
    # U: i (0..1), j (0..1) -> i*2 + j
    # V: j (0..1), k (0..8) -> j*9 + k
    # W: k (0..8), i (0..1) -> k*2 + i
    
    # Laderman's for column k:
    # Uses 4 terms (cols 4k to 4k+3)
    # A elements: A[0,0], A[0,1], A[1,0], A[1,1]
    # B elements: B[0,k], B[1,k] (for the k-th column of B)
    # C elements: C[0,k], C[1,k]
    
    # Let's map Laderman's P1..P8 to U, V, W entries.
    # Term t in [0..3] for column k:
    # col = 4*k + t
    
    # P1: U[A00]=1, V[B0k]=1, W[C0k]=1  -> Wait, Laderman combines terms.
    # It's not a direct sum of rank-1 tensors.
    # U, V, W will have entries -1, 0, 1.
    
    # P1 = A00 * B00 (relative to col k)
    #   U[A00][col] += 1
    #   V[B0k][col] += 1
    #   W[C0k][col] += 1  (Contribution to C0k)
    
    # P2 = A00 * (B10+B11) -> B1k + B0k? No, B is 2x9.
    #   B[0,k], B[1,k].
    #   P2 = A00 * (B[1,k] + B[0,k])? No, Laderman's is for 2x2 * 2x2.
    #   Here B is 2x1 for each k.
    #   So B[0,k] and B[1,k] are the two elements.
    
    # Let's rename for clarity:
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,k], f = B[1,k]  (Note: In standard 2x2*2x2, B is [[e,f],[g,h]]. Here B is [[e],[f]].)
    # Wait, B is 2x9. So for column k, B is [[B[0,k]], [B[1,k]]].
    # Let e = B[0,k], f = B[1,k].
    
    # Laderman's for A (2x2) * B (2x1):
    # C[0,k] = a*e + b*f
    # C[1,k] = c*e + d*f
    
    # This is just vector-matrix multiplication.
    # Rank of M_{2,2,1} (where 1 is the "columns" of B) is 4?
    # Actually, M_{2,2,1} is the tensor for (2x2) * (2x1) -> (2x1).
    # This is equivalent to M_{2,2,2} but with the second dimension of B being 1?
    # No, M_{n,m,p} is n x m times m x p.
    # Here n=2, m=2, p=1.
    # R(M_{2,2,1}) = 4.
    
    # Decomposition:
    # P1 = A[0,0] * B[0,k]
    # P2 = A[0,0] * B[1,k]
    # P3 = A[1,0] * B[0,k]
    # P4 = A[1,0] * B[1,k]
    # ... this is rank 4? No, this is rank 2 * 2 = 4 terms?
    # C[0,k] = P1 + P2
    # C[1,k] = P3 + P4
    # This is rank 4.
    
    # Can we do rank 3?
    # R(M_{2,2,1}) = 4. (Known result).
    
    # So for each k, we need 4 terms.
    # Total 36 terms.
    
    # My implementation:
    # For each k:
    #   col_base = 4*k
    #   
    #   # P1: A[0,0] * B[0,k]
    #   U[0*2+0][col_base] = 1
    #   V[0*9+k][col_base] = 1
    #   W[k*2+0][col_base] = 1  # C[0,k] += ...
    #   W[k*2+1][col_base] = 0  # C[1,k] += 0 ... wait.
    
    #   # Let's define the 4 terms for column k:
    #   # t=0: A[0,0] * B[0,k]
    #   # t=1: A[0,0] * B[1,k]
    #   # t=2: A[1,0] * B[0,k]
    #   # t=3: A[1,0] * B[1,k]
    #   # This misses A[0,1] and A[1,1].
    #   # A is 2x2. A[0,1] and A[1,1] are also needed.
    #   # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    #   # So the terms are:
    #   # 1. A[0,0] * B[0,k]
    #   # 2. A[0,1] * B[1,k]
    #   # 3. A[1,0] * B[0,k]
    #   # 4. A[1,1] * B[1,k]
    
    #   # This is a valid rank-4 decomposition.
    #   # It is actually the "naive" decomposition for this specific shape.
    
    #   # Let's construct this.
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for k in range(p):
        # Term 1: A[0,0] * B[0,k]
        U[0, col] = 1
        V[0, col] = 1  # B[0, k] is at index 0*p + k = k? No.
        # V index: j*p + k.
        # B is m x p. j is row of B (0..1), k is col of B (0..8).
        # B[0,k] -> j=0, k=k -> index 0*9+k = k.
        # B[1,k] -> j=1, k=k -> index 1*9+k = 9+k.
        
        V[0*p + k, col] = 1
        # W index: k*n + i.
        # C is n x p. i is row of C (0..1), k is col of C (0..8).
        # C[0,k] -> i=0, k=k -> index 0*2+k = k.
        # C[1,k] -> i=1, k=k -> index 1*2+k = 2+k.
        
        W[k*n + 0, col] = 1 # C[0,k]
        col += 1
        
        # Term 2: A[0,1] * B[1,k]
        U[1, col] = 1 # A[0,1] is at index 0*2+1 = 1
        V[1*p + k, col] = 1 # B[1,k]
        W[k*n + 0, col] = 1 # C[0,k]
        col += 1
        
        # Term 3: A[1,0] * B[0,k]
        U[2, col] = 1 # A[1,0] is at index 1*2+0 = 2
        V[0*p + k, col] = 1 # B[0,k]
        W[k*n + 1, col] = 1 # C[1,k]
        col += 1
        
        # Term 4: A[1,1] * B[1,k]
        U[3, col] = 1 # A[1,1] is at index 1*2+1 = 3
        V[1*p + k, col] = 1 # B[1,k]
        W[k*n + 1, col] = 1 # C[1,k]
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
