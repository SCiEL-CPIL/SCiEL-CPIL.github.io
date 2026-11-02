"""Matrix multiplication 4x5 x 5x5 with rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
DEADLINE = 140  # seconds; evaluator kills at 150s

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    """
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1
    # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
    # This is equivalent to: the tensor U ⊗ V (outer product structure) matches W in the right way
    # Actually the condition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * (something)
    # No wait. The standard tensor decomposition for matrix mult:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 (the multiplication tensor)
    # Rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i*m+j] = U[i*m+j, c], v_c[j*p+k] = V[j*p+k, c], w_c[k*n+i] = W[k*n+i, c]
    # So: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k
    # Wait, that's not right either. Let me think again.
    
    # The multiplication tensor T ∈ R^{n×m × m×p × p×n} has T[i,j,k] = 1.
    # A rank-r decomposition is T = sum_{c=1}^r u^{(c)} ⊗ v^{(c)} ⊗ w^{(c)}
    # where u^{(c)} ∈ R^{n*m}, v^{(c)} ∈ R^{m*p}, w^{(c)} ∈ R^{p*n}
    # So: 1 = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] for all i,j,k
    
    # That means: for every (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # But wait, the task says the output factor follows W[k*n+i, r]. Let me re-read.
    # "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify this way
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using the structure."""
    rank = len(U[0])
    # For each c, we have vectors u_c, v_c, w_c
    # T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] must equal 1
    
    # Group by j: for fixed j, we have a matrix M_j[i,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    # This must equal 1 for all i,k.
    
    # For each j, build M_j as a matrix product:
    # M_j[i,k] = sum_c (u_c[i*m+j] * w_c[k*n+i]) * v_c[j*p+k]
    # Hmm, this is still O(n*p*rank) per j, total O(m*n*p*rank)
    
    # Alternative: for each j, form A_j[i, c] = u_c[i*m+j] * ... no.
    # M_j[i,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    # = sum_c (u_c[i*m+j]) * (v_c[j*p+k] * w_c[k*n+i])
    # Let's precompute for each j:
    #   a_j[i] = [u_c[i*m+j] for c in range(rank)]  -- length rank
    #   b_j[k] = [v_c[j*p+k] for c in range(rank)]  -- length rank  
    #   Then M_j[i,k] = sum_c a_j[i][c] * b_j[k][c] * w_c[k*n+i]
    # This isn't a simple matrix product because w depends on both k and i.
    
    # Let's just do the direct check but with early termination
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Verification using numpy for speed."""
    import numpy as np
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each c, the contribution to T is outer product of u_c reshaped to (n,m), 
    # v_c reshaped to (m,p), w_c reshaped to (p,n)
    # T = sum_c np.einsum('ij,jk,ki->ijk', Ua[:,c].reshape(n,m), Va[:,c].reshape(m,p), Wa[:,c].reshape(p,n))
    
    # More efficiently: T[i,j,k] = sum_c Ua[i*m+j,c] * Va[j*p+k,c] * Wa[k*n+i,c]
    # Let's compute for a few random (i,j,k) first, then all if needed
    # Actually just compute all
    # T = np.zeros((n,m,p))
    # for c in range(rank):
    #     T += np.einsum('ij,jk,ki->ijk', Ua[:,c].reshape(n,m), Va[:,c].reshape(m,p), Wa[:,c].reshape(p,n))
    
    # But this is slow for large rank. Let's use a different approach.
    # For fixed j: M_j[i,k] = sum_c Ua[i*m+j,c] * Va[j*p+k,c] * Wa[k*n+i,c]
    # This is like a "matrix multiplication" where the middle index is contracted with rank.
    
    # Let's just check all (i,j,k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook rank = 100
    rank_school = n * m * p  # 100
    
    def make_schoolbook():
        U = [[0]*rank_school for _ in range(n*m)]
        V = [[0]*rank_school for _ in range(m*p)]
        W = [[0]*rank_school for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Known results: 
    # - 3x3x3: rank 19 (Lader)
    # - 4x4x4: rank 49 (Strassen-like)
    # - 5x5x5: rank 97 (best known exact)
    # - 4x5x5: unknown, but at most 100 (schoolbook)
    
    # Strategy: Try various rank values and use randomized search / structured decomposition
    
    # First, try a structured approach:
    # Split the 5x5 into blocks and use known low-rank decompositions
    
    # For 4x5 x 5x5:
    # A is 4x5, B is 5x5, C = AB is 4x5
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Try rank from 90 down to 60
    # Use a heuristic: random initialization + gradient-free optimization
    
    # Actually, let's try a more structured approach first.
    # The key insight: if we can find a rank-r decomposition where the entries are in {0, 1/2, 1},
    # we can use integer programming or exhaustive search over small alphabets.
    
    # For now, let's try random search with small alphabets for various ranks.
    
    def try_random_rank(r, num_trials=1000, alphabet=None):
        """Try random decompositions of rank r."""
        if alphabet is None:
            alphabet = [0, 1, -1, 0.5, -0.5, 2, -2]
        
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            U = [[random.choice(alphabet) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(alphabet) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(alphabet) for _ in range(r)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                return U, V, W
        return None
    
    # Try structured decompositions based on block structure
    # 4x5 x 5x5: Think of A as [A1 A2] where A1 is 4x3, A2 is 4x2
    # B as [[B11][B12]; [B21][B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    # C = [A1 A2] [[B11][B12]; [B21][B22]] = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # This gives:
    # C[:, :3] = A1*B11 + A2*B21  (4x3 = 4x3 * 3x3 + 4x2 * 2x3)
    # C[:, 3:] = A1*B12 + A2*B22  (4x2 = 4x3 * 3x2 + 4x2 * 2x2)
    
    # Rank contributions:
    # A1*B11: 4*3*3 = 36 (schoolbook) or lower
    # A2*B21: 4*2*3 = 24
    # A1*B12: 4*3*2 = 24
    # A2*B22: 4*2*2 = 16
    # Total schoolbook: 100
    
    # With Strassen-like for 3x3: rank 19 instead of 27 for B11 part
    # But the structure is more complex because it's rectangular.
    
    # Let's try a different approach: use the fact that we need
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 100 equations in r variables per "term"
    # Actually, it's r^3 variables total (r entries in each of U, V, W)
    # with 100 constraints.
    
    # For r < 100, this is an underdetermined system in a sense, but nonlinear.
    
    # Let's try a more targeted search:
    # Fix most entries to 0 or 1, and search for the remaining degrees of freedom.
    
    # Approach: Start from schoolbook and try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if their contributions can be combined.
    
    # Actually, let's try a completely different approach:
    # Use the known rank-19 decomposition for 3x3x3 and build up.
    
    # Lader's rank-19 decomposition for 3x3x3:
    # This is complex to encode from scratch. Let's try a simpler structured approach.
    
    # For 4x5 x 5x5, one approach:
    # Pad to 5x5 x 5x5 (rank 97 known) and then project.
    # But projection may not preserve rank.
    
    # Let's try random search with a deadline guard.
    
    # Try ranks from 95 down
    for r in range(95, 60, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try with small alphabet
        result = try_random_rank(r, num_trials=500, alphabet=[0, 1, -1])
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        
        # Also try with half-integers
        if time.time() - start_time > DEADLINE:
            break
        result = try_random_rank(r, num_trials=200, alphabet=[0, 1, -1, 0.5, -0.5])
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    # If we found something better, use it; otherwise use schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = make_schoolbook()
        return {"U": U, "V": V, "W": W}
