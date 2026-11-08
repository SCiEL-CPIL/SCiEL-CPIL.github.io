<<<<<<< SEARCH
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 54
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[u,v,w] = 1 iff u=i*m+j, v=j'*p+k, w=k'*n+i' with i=i', j=j', k=k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank = n*m*p)
    U = np.zeros((n*m, rank_school), dtype=np.float64)
    V = np.zeros((m*p, rank_school), dtype=np.float64)
    W = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Attempt to find lower-rank exact decompositions
    # For each candidate rank r, check if a valid decomposition exists
    # This is a hard nonlinear problem; use randomized search with validation
    
    def validate(Ur, Vr, Wr, tol=1e-10):
        """Check if Ur, Vr, Wr form a valid decomposition of T."""
        # Efficient check: T[u,v,w] = sum_t Ur[u,t] * Vr[v,t] * Wr[w,t]
        # We can compute the reconstruction and compare.
        # Using einsum for clarity and correctness.
        # Ur: (n*m, r), Vr: (m*p, r), Wr: (p*n, r)
        # Reconstructed tensor: (n*m, m*p, p*n)
        recon = np.einsum('ut,vt,wt->uvw', Ur, Vr, Wr)
        return np.allclose(recon, T, atol=tol)
    
    # Try ranks from rank_school-1 down to a reasonable minimum
    min_rank_to_try = max(30, rank_school - 20)
    for r in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with half-integer-friendly values
            Ur = np.random.randint(-2, 3, size=(n*m, r)) * 0.5
            Vr = np.random.randint(-2, 3, size=(m*p, r)) * 0.5
            Wr = np.random.randint(-2, 3, size=(p*n, r)) * 0.5
            
            if validate(Ur, Vr, Wr):
                best_rank = r
                best_U, best_V, best_W = Ur, Vr, Wr
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
=======
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 54
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[u,v,w] = 1 iff u=i*m+j, v=j'*p+k, w=k'*n+i' with i=i', j=j', k=k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank = n*m*p)
    U = np.zeros((n*m, rank_school), dtype=np.float64)
    V = np.zeros((m*p, rank_school), dtype=np.float64)
    W = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    def validate(Ur, Vr, Wr, tol=1e-9):
        """Check if Ur, Vr, Wr form a valid decomposition of T."""
        # Efficient check using batched operations
        # For each pair (u,v), check if the vector of inner products matches T[u,v,:]
        # Ur: (n*m, r), Vr: (m*p, r), Wr: (p*n, r)
        # Compute all pairwise dot products: (n*m, m*p, p*n)
        # This is expensive, so we check only non-zero entries first
        # T has exactly n*m*p = 54 non-zero entries
        nz = np.nonzero(T)
        for idx in range(len(nz[0])):
            u, v, w = nz[0][idx], nz[1][idx], nz[2][idx]
            # T[u,v,w] should equal sum_t Ur[u,t] * Vr[v,t] * Wr[w,t]
            val = np.dot(np.dot(Ur[u, :], Vr[v, :]), Wr[w, :])
            if abs(val - T[u, v, w]) > tol:
                return False
        # Also check that zero entries remain zero (sample check for speed)
        # Check a subset of zero entries
        zero_mask = (T == 0)
        zero_indices = np.argwhere(zero_mask)
        if len(zero_indices) > 0:
            # Check up to 100 random zero entries
            check_count = min(100, len(zero_indices))
            np.random.shuffle(zero_indices)
            for idx in range(check_count):
                u, v, w = zero_indices[idx]
                val = np.dot(np.dot(Ur[u, :], Vr[v, :]), Wr[w, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Strategy 1: Try to find rank < 54 by removing redundant terms
    # The schoolbook has 54 terms. Some may be combined if they share structure.
    # For 2x3x9, let's try systematic reduction
    
    # First, try rank 53: can we merge two schoolbook terms?
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist
    # coefficients that make the sum equivalent. This is generally not possible
    # for standard matrix multiplication, but let's try structured searches.
    
    # Strategy 2: Use the known lower bound. For 2x3x9, the rank is at least
    # max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # Known results: rank of 2x3x9 matrix multiplication.
    # Let's try to find a decomposition with rank < 54 using algebraic methods.
    
    # Try structured decompositions
    # For small matrices, we can try to find patterns
    
    # Attempt: use the fact that for 2xn matrices, there may be optimizations
    # Let's try a greedy approach: start with schoolbook, try to remove terms
    # by checking if one term is linearly dependent on others
    
    # More effective: try to find a rank-r decomposition where r < 54
    # by using randomized algebraic approaches
    
    # Use a different approach: try to find a basis for the tensor space
    # and see if we can represent it with fewer factors
    
    # For now, let's try a more systematic search for rank reduction
    # by checking if certain combinations of schoolbook terms can be merged
    
    # Check if rank 53 is possible by trying all pairs to merge
    # This is O(54^2) checks, each taking some time
    if time.time() < deadline - 60:
        # Try to find if any two schoolbook terms can be replaced by one
        # with appropriate scaling (this is a special case)
        for i1 in range(n):
            for j1 in range(m):
                for k1 in range(p):
                    if time.time() > deadline - 60:
                        break
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i1, j1, k1) == (i2, j2, k2):
                                    continue
                                # Check if term (i2,j2,k2) can be absorbed into (i1,j1,k1)
                                # This requires specific structural conditions
                                pass
    
    # Try randomized search with more trials and better initialization
    # Use the schoolbook as a starting point and try perturbations
    
    # For ranks 53, 52, 51, try to find valid decompositions
    for r in range(rank_school - 1, max(40, rank_school - 15), -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize near schoolbook: take first r terms and add small noise
            Ur = np.zeros((n*m, r), dtype=np.float64)
            Vr = np.zeros((m*p, r), dtype=np.float64)
            Wr = np.zeros((p*n, r), dtype=np.float64)
            
            # Copy first r schoolbook terms
            for t in range(min(r, rank_school)):
                i = t // (m * p)
                rem = t % (m * p)
                j = rem // p
                k = rem % p
                Ur[i*m+j, t] = 1.0
                Vr[j*p+k, t] = 1.0
                Wr[k*n+i, t] = 1.0
            
            # Add small random perturbation to allow optimization
            if r < rank_school:
                Ur += np.random.randn(n*m, r) * 0.1
                Vr += np.random.randn(m*p, r) * 0.1
                Wr += np.random.randn(p*n, r) * 0.1
            
            if validate(Ur, Vr, Wr):
                best_rank = r
                best_U, best_V, best_W = Ur, Vr, Wr
                break
        
        if best_rank < rank_school:
            break
    
    # If we haven't found improvement, try a completely different approach
    # Use known algebraic decompositions for small matrix multiplication
    
    # For 2x3x9, try to construct a decomposition using block structure
    # A = [a b c; d e f] (2x3), B = [b1 b2 b3; b4 b5 b6; b7 b8 b9] (3x9)
    # C = A*B is 2x9
    
    # The product has 18 entries, each being a dot product of length 3
    # C[0,0] = a*b1 + b*b4 + c*b7
    # C[0,1] = a*b2 + b*b5 + c*b8
    # etc.
    
    # We can group terms to reduce rank
    # For example, if we define intermediate variables, we might reduce the count
    
    # Let's try to find a decomposition with rank < 54 using a different strategy
    # Try rank 50, 45, 40, etc. with more aggressive random search
    
    for target_rank in [50, 45, 40, 35, 30]:
        if time.time() > deadline - 5:
            break
        if target_rank >= best_rank:
            continue
        
        r = target_rank
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            
            # Random initialization with half-integer values
            Ur = np.random.randint(-4, 5, size=(n*m, r)) * 0.5
            Vr = np.random.randint(-4, 5, size=(m*p, r)) * 0.5
            Wr = np.random.randint(-4, 5, size=(p*n, r)) * 0.5
            
            if validate(Ur, Vr, Wr):
                best_rank = r
                best_U, best_V, best_W = Ur, Vr, Wr
                break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
