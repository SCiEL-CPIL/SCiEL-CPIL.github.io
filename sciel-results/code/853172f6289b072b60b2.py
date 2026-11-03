"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def _verify_exact(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    # Check that all other entries are 0
    # The tensor T has T[i,j,k,r] = sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For matrix multiplication, T should be 1 if (i,j,k) corresponds to a valid multiplication term
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m and B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k,r] structure... let me think again.
    
    # The matrix multiplication tensor T_{i,k,j} = delta_{i,k,j} ... no.
    # Actually, for C = A * B, C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The bilinear form is: C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} T_{i,k,j,r}
    # where T_{i,k,j,r} is 1 if the r-th term uses A_{i,j} and B_{j,k}.
    
    # In the schoolbook decomposition:
    # For each (i,j,k), we have a rank-1 term with U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # The output is W indexed by k*n+i, which is the (k,i) position in the output.
    
    # So the tensor is: for each (i,j,k), the (i*m+j) row of U, (j*p+k) row of V, and (k*n+i) row of W
    # The bilinear product gives sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 if (i,j,k) is a valid term, else 0.
    # But in matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So for fixed i,k, we sum over j. Each term A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor should have T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # And all other entries should be 0.
    
    # Let's check that for any (a, b, c) not corresponding to a valid (i,j,k) triple, the sum is 0.
    # a = i*m + j for some i in [0,n), j in [0,m)
    # b = j*p + k for some j in [0,m), k in [0,p)
    # c = k*n + i for some k in [0,p), i in [0,n)
    
    # For the sum to be nonzero, we need U[a,r]*V[b,r]*W[c,r] != 0 for some r.
    # In the schoolbook, each r corresponds to a unique (i,j,k).
    # So if a = i1*m + j1, b = j2*p + k2, c = k3*n + i3, then the only r that contributes
    # is the one where (i,j,k) = (i1,j1,k2) = (i3,j2,k3), which requires i1=i3, j1=j2, k2=k3.
    # Otherwise the product is 0.
    
    # So we need to verify that for all a, b, c, the sum equals 1 if they form a valid triple, else 0.
    # Let's check a few cases to be thorough, but the main check above covers the valid cases.
    
    # For efficiency, let's just check the valid cases and trust the structure.
    # Actually, let's do a more thorough check for small cases.
    
    # Check that invalid combinations give 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                valid = (i1 == i3 and j1 == j2 and k2 == k3)
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if valid and s != 1:
                    return False
                if not valid and s != 0:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find a lower rank decomposition using randomized search
    # Strategy: try ranks from a lower bound up to schoolbook_rank - 1
    # For each rank, try random sparse integer vectors
    
    # Lower bound for matrix multiplication rank
    # For n x m * m x p, the rank is at least max(n*m, m*p, p*n) ... actually it's more subtle.
    # The rank of the matrix multiplication tensor [n x m x p] is at least max(n*p, n*m, m*p)
    # Actually, the rank is at least the maximum of the dimensions of the "slices".
    # For [n, m, p], the rank is at least max(n*p, n*m, m*p).
    # Here n=3, m=3, p=5, so max(15, 9, 15) = 15.
    # Schoolbook is 45.
    
    # Known results: rank of 3x3 x 3x3 is at most 23 (Strassen-like), at least 19 or so.
    # For 3x3 x 3x5, let's try to find something below 45.
    
    # Use randomized algebraic identity verification
    # Sample sparse integer vectors from a small alphabet
    
    alphabet = [-2, -1, 0, 1, 2]
    
    # Try ranks from 15 up to schoolbook_rank - 1
    # But this could be very slow, so let's be strategic.
    
    # Actually, let's try a different approach: use the known Strassen-style identities.
    # For 3x3 matrix multiplication, there are known decompositions of rank 19-23.
    # For 3x3 x 3x5, we can extend these.
    
    # Let me try a pragmatic randomized search with limited attempts.
    
    random.seed(42)
    
    # Try a few specific ranks
    for target_rank in range(max(n*p, n*m, m*p), schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try a limited number of random attempts for this rank
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse U, V, W
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with sparse random values
            for a in range(n * m):
                for r in range(target_rank):
                    if random.random() < 0.3:  # 30% density
                        U[a][r] = random.choice(alphabet)
            
            for b in range(m * p):
                for r in range(target_rank):
                    if random.random() < 0.3:
                        V[b][r] = random.choice(alphabet)
            
            for c in range(p * n):
                for r in range(target_rank):
                    if random.random() < 0.3:
                        W[c][r] = random.choice(alphabet)
            
            # Verify
            if _verify_exact(U, V, W, n, m, p, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # If we couldn't find a better rank, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
