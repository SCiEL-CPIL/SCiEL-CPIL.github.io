"""Matrix multiplication 3x3x4 rank-minimal tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 3, 4

    # Schoolbook rank = n*m*p = 36
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    def build_schoolbook():
        """Build the schoolbook decomposition as fallback."""
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
        return U, V, W

    U, V, W = build_schoolbook()
    best_U, best_V, best_W = U, V, W

    # Try to find a better decomposition using a structured search
    # The tensor T_{i,j,k,l} = delta_{i,l} for (i,j) in A=(n x m), (k,l) in B=(p x n)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,l}
    # where i in [0,n), j in [0,m), k in [0,p), l in [0,n)
    # Note: W index is k*n + i, and the output index l must equal i for the delta.
    
    # Actually, let me re-read the convention: W[k*n+i, r]
    # The matrix product C = A * B where A is n x m, B is m x p
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] where the "output" index pairs are (i, k) and 
    # the "input" index is j. But the task says output factor W[k*n+i, r].
    # So the output space is indexed by (k, i) with k in [0,p), i in [0,n).
    # Wait, that's p*n = 12 indices.
    
    # Let me re-read: "multiplying a 3 by 3 matrix by a 3 by 4 matrix"
    # A is 3x3 (n=3, m=3), B is 3x4 (m=3, p=4)
    # C = A*B is 3x4, so C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[?,c]
    # But W has shape (p*n, rank) = (4*3, rank) = (12, rank)
    # And W[k*n+i, r] means W is indexed by (k, i) where k in [0,4), i in [0,3)
    
    # Hmm, this is a bit unusual. Let me think about it differently.
    # The standard tensor for matrix multiplication A(n x m) * B(m x p) -> C(n x p)
    # has T[i, j, k] = delta_{i,?} ... no.
    # Actually T_{i,j,k} where the multiplication is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T[i, j, k] and the "free" indices are i (row of A/C) and k (col of B/C)
    # The "contracted" index is j.
    # In the rank decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[?,c]
    # But W should not depend on j. The standard form is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where W has shape (n*p, rank).
    
    # But the task says W has shape (p*n, rank) with W[k*n+i, r].
    # So W is indexed by (k, i) in that order.
    # So the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should equal delta_{i,?}... 
    
    # Wait, I think the tensor here is not the standard one. Let me re-read.
    # "multiplying a 3 by 3 matrix by a 3 by 4 matrix"
    # The output is 3x4. The output factor W[k*n+i, r] with k in [0,4), i in [0,3).
    # So the output index is (k, i) which corresponds to C[i, k] if we think of
    # i as row and k as column. But the index is k*n+i, so k is the "outer" index.
    
    # I think the tensor is defined such that:
    # The multiplication tensor T has T[i, j, k] = 1 if the (i,k)-th entry of the product
    # depends on A[i,j] and B[j,k]. Since C[i,k] = sum_j A[i,j]*B[j,k],
    # we have T[i,j,k] = 1 for all i,j,k (each triple contributes to exactly one output entry).
    # But that's not a delta. 
    
    # Actually, I think the standard formulation is:
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T in the space U*V*W where U = R^{n*m}, V = R^{m*p}, W = R^{n*p}
    # is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # where e_{i,j} is the basis vector for A[i,j], e_{j,k} for B[j,k], e_{i,k} for C[i,k].
    
    # But the task says W has shape (p*n, rank) with W[k*n+i, r].
    # So the output basis is indexed by (k, i) in that order, i.e., e_{k,n+i} = e_{i,k} in standard order.
    # So W[k*n+i, c] corresponds to the coefficient for C[i,k] in the c-th rank-1 term.
    
    # So the decomposition is:
    # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other combinations (which don't exist since i,j,k range over all valid values), it's 0.
    # Wait, but every (i,j,k) triple does contribute to the product C[i,k].
    # So actually the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # That means we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL i,j,k.
    
    # Hmm, but that's not right either. In the standard matrix multiplication tensor,
    # T[i,j,k,l] where l is the column index of B... no.
    
    # Let me look at this more carefully. The task says:
    # "multiplying a 3 by 3 matrix by a 3 by 4 matrix"
    # A is 3x3, B is 3x4, C = A*B is 3x4.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # 
    # The tensor for this bilinear map lives in R^{3*3} \otimes R^{3*4} \otimes R^{3*4}
    # = R^9 \otimes R^{12} \otimes R^{12}
    # 
    # The tensor is T = sum_{i=0}^{2} sum_{j=0}^{2} sum_{k=0}^{3} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # where e_{i,j} is the (i,j)-th basis vector in R^9 (flattened A),
    # e_{j,k} is the (j,k)-th basis vector in R^{12} (flattened B),
    # e_{i,k} is the (i,k)-th basis vector in R^{12} (flattened C).
    # 
    # But the task says W has shape (p*n, rank) = (4*3, rank) = (12, rank) with W[k*n+i, r].
    # So the output index is (k, i) -> index k*n+i.
    # This means e_{i,k} in standard order is e_{k*n+i} in the task's ordering.
    # 
    # So the tensor is: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    # And T[a, b, c] = 0 otherwise.
    # 
    # For the rank decomposition: T[a, b, c] = sum_{r=0}^{R-1} U[a, r] * V[b, r] * W[c, r]
    # 
    # So we need: for all i, j, k: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    # And for all (a, b, c) not of that form, the sum = 0.
    
    # Let me verify: are there any (a, b, c) triples that are NOT of the form (i*m+j, j*p+k, k*n+i)?
    # a in [0,9), b in [0,12), c in [0,12).
    # a = i*m+j = 3i+j, so i = a//3, j = a%3.
    # b = j*p+k = 3j+k, so given j from a, k = b - 3j.
    # For this to be valid, k must be in [0,4).
    # c = k*n+i = 3k+i.
    # 
    # So for a given a, we have i = a//3, j = a%3.
    # For b, k = b - 3j must be in [0,4), i.e., b in [3j, 3j+4).
    # Then c must equal 3k + i = 3(b-3j) + i = 3b - 9j + i.
    # 
    # So the constraint is: for all a, b, c:
    #   if b in [3*(a%3), 3*(a%3)+4) and c == 3*(b - 3*(a%3)) + a//3:
    #       sum_r U[a,r]*V[b,r]*W[c,r] = 1
    #   else:
    #       sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # This is a system of 9*12*12 = 1296 equations.
    
    # Let's try a simple approach: search for low-rank decompositions with small integer/half-integer entries.
    
    # First, let's try rank = 18 (half of schoolbook) or lower.
    # Actually, for 3x3x3, the known minimal rank is 23 (Strassen is 7 for 2x2x2, 
    # but 3x3x3 is 23 or 24). For 3x3x4, it might be around 20-25.
    
    # Let me try a different approach: use the known structure.
    # For the standard matrix multiplication tensor, a rank-1 term is:
    # U[a,r] * V[b,r] * W[c,r]
    # where a = 3i+j, b = 3j+k, c = 3k+i.
    
    # A simple rank-1 term that contributes to exactly one (i,j,k) triple:
    # U[3i+j, r] = 1, V[3j+k, r] = 1, W[3k+i, r] = 1, all others 0.
    # This gives the schoolbook decomposition.
    
    # To get a lower rank, we need rank-1 terms that cover multiple (i,j,k) triples.
    # For example, if U[a,r] = 1 for multiple a values, etc.
    
    # Let me try to find a decomposition of rank R by solving a system of equations.
    # For a fixed R, we have R^2 variables (roughly) and 1296 equations.
    # This is underdetermined for small R, but we need exact solutions.
    
    # Alternative: use the fact that the tensor has a specific structure and try
    # to find a decomposition using a greedy or structured approach.
    
    # Let me try rank = 24, 25, 26, ... and see if we can find a solution.
    # Actually, let's start with a higher rank and try to reduce.
    
    # For now, let's just return the schoolbook decomposition if we can't find better.
    # But let me try a simple search for rank < 36.
    
    # Try to find a decomposition of rank R where each rank-1 term has entries in {-1, 0, 1}.
    # We can use a randomized approach: pick random sparse U, V, W and check if they work.
    
    import random
    random.seed(42)
    
    for target_rank in range(20, 36):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try a few random attempts for this rank
        attempts = 0
        max_attempts = 50 if target_rank >= 30 else 200
        while attempts < max_attempts and time.time() - start_time < DEADLINE:
            attempts += 1
            
            # Create random sparse matrices
            U_try = [[0] * target_rank for _ in range(n * m)]
            V_try = [[0] * target_rank for _ in range(m * p)]
            W_try = [[0] * target_rank for _ in range(p * n)]
            
            for r in range(target_rank):
                # Pick random non-zero entries
                for a in range(n * m):
                    if random.random() < 0.3:
                        U_try[a][r] = random.choice([-1, 1])
                for b in range(m * p):
                    if random.random() < 0.3:
                        V_try[b][r] = random.choice([-1, 1])
                for c in range(p * n):
                    if random.random() < 0.3:
                        W_try[c][r] = random.choice([-1, 1])
            
            # Check if this decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0
                        for r in range(target_rank):
                            s += U_try[a][r] * V_try[b][r] * W_try[c][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also check that non-matching triples give 0
            for a in range(n * m):
                if not valid:
                    break
                i = a // m
                j = a % m
                for b in range(m * p):
                    if not valid:
                        break
                    # Determine if (a, b) is a valid pair
                    k_val = b - j * p
                    if k_val < 0 or k_val >= p:
                        # This (a,b) pair doesn't correspond to any (i,j,k)
                        for c in range(p * n):
                            s = 0
                            for r in range(target_rank):
                                s += U_try[a][r] * V_try[b][r] * W_try[c][r]
                            if s != 0:
                                valid = False
                                break
                        if not valid:
                            break
                    else:
                        # Check the correct c
                        c_correct = k_val * n + i
                        # c_correct already checked above
                        # Check other c values
                        for c in range(p * n):
                            if c == c_correct:
                                continue
                            s = 0
                            for r in range(target_rank):
                                s += U_try[a][r] * V_try[b][r] * W_try[c][r]
                            if s != 0:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                best_rank = target_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                break
    
    if time.time() - start_time > DEADLINE:
        # Return best found so far
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
