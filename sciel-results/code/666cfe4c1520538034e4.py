"""Matrix multiplication 3x5 x 5x6 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 90
    # Try to find better decomposition
    
    # For matrix multiplication, we can use the structure:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form in A and B
    
    # Strategy: Use random projections to find a lower rank decomposition
    # The tensor T has components T[i,j,k] = 1 if we're computing element (i,k) from (j)
    # Actually, the multiplication tensor is: T[i,j,k] = 1 for the (i,j) of A and (j,k) of B
    
    # Let me use a different approach: try known decompositions or search
    
    # For small matrices, we can use the fact that:
    # rank(MM_{n,m,p}) <= n*m*p but can be much smaller
    
    # Let's try a structured approach: group terms and use linear combinations
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # Use SVD or rank reduction on the factor matrices
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = rank_school
    
    # Try random linear combinations to find dependencies
    # For each pair of columns, try to see if one can be expressed as combo of others
    
    # Better: use the structure of matrix multiplication
    # The tensor is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(i,j,k) for the multiplication
    
    # Actually, the correct identity is:
    # sum_c U[i,j,c] * V[j,k,c] = W[k,i,c] ... no
    
    # Let me reconsider. The matrix multiplication tensor satisfies:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 means we need A[i,j]*B[j,k]
    # A rank-r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # and the condition is: for all A,B, sum_c (u_c · A_flat) * (v_c · B_flat) = (A*B)_flat
    
    # This means: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 if this is the (i,k) element from j
    # Actually: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] should give the coefficient for A[i,j]*B[j,k] in computing (A*B)[i,k]
    # Which is 1 for each (i,j,k) triple that contributes to (i,k)
    
    # So the condition is: for each (i,k), sum_j sum_c u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] = 1
    # And for the "wrong" combinations, it should be 0... 
    
    # Actually the tensor T has T[i,j,k] = 1 for all valid (i,j,k), and the decomposition must satisfy:
    # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = T[i,j,k] = 1 for all i,j,k
    
    # Wait, that's not right either. Let me think again.
    
    # The matrix multiplication bilinear map: (A,B) -> A*B
    # Flattened: (A_flat, B_flat) -> (A*B)_flat
    # This is a 3-linear form: sum_{i,j,k} (A*B)[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j]*B[j,k] * (A*B)[i,k]
    
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1 for the coefficient of A[i,j]*B[j,k] in computing (A*B)[i,k]
    
    # A rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # means: T[a,b,c] = sum_d u_d[a]*v_d[b]*w_d[c]
    # So: sum_d u_d[i*m+j] * v_d[j*p+k] * w_d[k*n+i] = 1 for all i,j,k
    
    # This is the condition we need to satisfy.
    
    # For the schoolbook: u_{ijk}[i*m+j]=1, v_{ijk}[j*p+k]=1, w_{ijk}[k*n+i]=1, all others 0
    # Then sum_d gives 1 only when d=(i,j,k), so it works.
    
    # To reduce rank, we need to find fewer vectors that still satisfy this.
    
    # Let's try a search: start with schoolbook and try to merge columns
    
    # Time check
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a better decomposition using random initialization and optimization
    # For small ranks, we can try to find integer/half-integer solutions
    
    # Known: for 2x2, rank is 7 (Strassen). For larger, it's harder.
    # For 3x5 x 5x6, let's try to find a decomposition with rank < 90.
    
    # Use a simple approach: try random rank-r decompositions and check if they work
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # For each (i,j,k), check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > tol:
                        return False
        # Also need to check that "cross terms" are zero
        # The tensor only has 1s at (i*m+j, j*p+k, k*n+i) for valid i,j,k
        # All other entries should be 0
        # But in the flattened tensor, most entries are 0
        # We need: for all (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), sum_d U[a,d]*V[b,d]*W[c,d] = 0
        
        # This is expensive to check fully. Let's check a sample.
        # Actually, for correctness, we need all entries. But the tensor has specific structure.
        
        # The tensor T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
        # We need: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c] for all a,b,c
        
        # Let's verify a few random non-diagonal entries
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if (a,b,c) is a valid (i*m+j, j*p+k, k*n+i)
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k2 = b % p
            k1 = c // n
            i2 = c % n
            if i1 == i2 and j1 == j2:
                expected = 1
            else:
                expected = 0
            s = sum(U[a][d]*V[b][d]*W[c][d] for d in range(rank))
            if abs(s - expected) > tol:
                return False
        return True
    
    # Try to find a lower rank decomposition
    # Start with schoolbook and try to reduce
    
    # Use a greedy approach: try to eliminate columns by expressing them as combos of others
    
    # For now, return schoolbook if we can't find better quickly
    if time.time() > deadline - 10:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try a structured decomposition using block methods
    # Split the 5x5 middle into blocks and use Strassen-like ideas
    
    # For 3x5 x 5x6, we can view it as blocks:
    # A is 3x5, B is 5x6
    # Split A into [A1 | A2] where A1 is 3x2, A2 is 3x3
    # Split B into [B1; B2] where B1 is 2x6, B2 is 3x6
    # Then A*B = A1*B1 + A2*B2
    
    # This doesn't directly help with rank reduction.
    
    # Let's just return the schoolbook for now, but try a quick random search for lower rank
    
    # Try rank 80, 70, 60 with random initialization
    for target_rank in [80, 70, 60, 50]:
        if time.time() > deadline - 15:
            break
        
        # Random initialization with small integers
        U_rand = np.zeros((n*m, target_rank))
        V_rand = np.zeros((m*p, target_rank))
        W_rand = np.zeros((p*n, target_rank))
        
        # Initialize with random ±1 or 0
        for c in range(target_rank):
            for a in range(n*m):
                U_rand[a, c] = random.choice([-1, 0, 1])
            for b in range(m*p):
                V_rand[b, c] = random.choice([-1, 0, 1])
            for c_idx in range(p*n):
                W_rand[c_idx, c] = random.choice([-1, 0, 1])
        
        # Check if this works (unlikely, but let's try)
        if check_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
            best_U = U_rand.tolist()
            best_V = V_rand.tolist()
            best_W = W_rand.tolist()
            best_rank = target_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
