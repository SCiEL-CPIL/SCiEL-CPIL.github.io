import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T as a sparse representation
    # T[u_idx, v_idx, w_idx] = 1 where u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i
    # For verification, we need: sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
    # T has exactly n*m*p = 48 non-zero entries (all 1s)
    
    # Store target as dict for sparse checking
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[(u_idx, v_idx, w_idx)] = 1
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Verify exact decomposition using Fraction arithmetic."""
        # Convert to Fraction for exactness
        Uf = [[Fraction(int(round(2*x))) / 2 for x in row] for row in U]
        Vf = [[Fraction(int(round(2*x))) / 2 for x in row] for row in V]
        Wf = [[Fraction(int(round(2*x))) / 2 for x in row] for row in W]
        
        # Check all target entries equal 1
        for (u_idx, v_idx, w_idx), expected in target.items():
            actual = Fraction(0)
            for c in range(rank):
                actual += Uf[u_idx][c] * Vf[v_idx][c] * Wf[w_idx][c]
            if actual != Fraction(expected):
                return False
        
        # Check all non-target entries equal 0
        # There are nm*mp*pn - n*m*p = 12*16*12 - 48 = 2304 - 48 = 2256 zero entries
        # But checking all is expensive. Check a subset or all if rank is small.
        # For correctness, we must check all.
        all_u = range(nm)
        all_v = range(mp)
        all_w = range(pn)
        for u_idx in all_u:
            for v_idx in all_v:
                for w_idx in all_w:
                    key = (u_idx, v_idx, w_idx)
                    if key in target:
                        continue
                    actual = Fraction(0)
                    for c in range(rank):
                        actual += Uf[u_idx][c] * Vf[v_idx][c] * Wf[w_idx][c]
                    if actual != 0:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy, then exact check if close."""
        Un = np.array(U, dtype=float)
        Vn = np.array(V, dtype=float)
        Wn = np.array(W, dtype=float)
        
        # Compute all entries
        # T_recon[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
        # Use einsum
        T_recon = np.einsum('uc,vc,wc->uvw', Un, Vn, Wn)
        
        # Check target entries
        for (u_idx, v_idx, w_idx), expected in target.items():
            if abs(T_recon[u_idx, v_idx, w_idx] - expected) > 1e-6:
                return False
        
        # Check a sample of non-target entries
        # If all target entries are correct and the residual on non-targets is small,
        # then do exact verification
        # Build full target tensor
        T_target = np.zeros((nm, mp, pn))
        for (u_idx, v_idx, w_idx), val in target.items():
            T_target[u_idx, v_idx, w_idx] = val
        
        residual = np.linalg.norm(T_recon - T_target)
        if residual > 1e-4:
            return False
        
        # Do exact verification
        return verify(U, V, W, rank)
    
    # Slice rank lower bound
    # The matrix flattening along mode 1: (nm) x (mp*pn)
    # Its rank is at most nm = 12
    # Along mode 2: (mp) x (nm*pn) = 16 x 144, rank at most 16
    # Along mode 3: (pn) x (nm*mp) = 12 x 192, rank at most 12
    # So slice rank >= 16
    
    # For matrix multiplication, the tensor rank is known to be between
    # max(nm, mp, pn) and nm*mp*pn
    # For 3x4x4: between 16 and 48
    
    # Strategy: Try ranks from a lower bound upward
    # Use a structured random search with half-integer vectors
    
    # Precompute: for each rank, we need to find U, V, W with half-integer entries
    # such that the tensor decomposition is exact.
    
    # Key insight: The matrix multiplication tensor has a lot of symmetry.
    # We can use the fact that for each fixed i, the "slice" T[i,:,:] has structure.
    
    # Let's try a different approach: use the known structure of small matrix multiplication
    # tensors. For 3x4x4, we can try to find a decomposition by searching over
    # rank-1 terms that are "compatible" with the tensor structure.
    
    # A rank-1 term (u, v, w) contributes u[a]*v[b]*w[c] to T[a,b,c].
    # For the term to be useful, it should contribute to target entries and
    # cancel out on non-target entries when summed with other terms.
    
    # Approach: Start with rank = lower_bound and incrementally add rank-1 terms.
    # At each step, try to find a half-integer rank-1 term that reduces the residual.
    
    # But this is complex. Let me try a simpler approach:
    # Random search with half-integer vectors, checking verification.
    
    def random_half_integer_vector(length, max_val=3):
        """Generate a random half-integer vector with entries in [-max_val, max_val]."""
        vec = []
        for _ in range(length):
            # Choose from -max_val to max_val in steps of 0.5
            val = (np.random.randint(-2*max_val, 2*max_val + 1)) / 2.0
            vec.append(val)
        return vec
    
    def try_rank(rank, num_trials=20000):
        """Try to find a rank decomposition with given rank using random search."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random half-integer vectors
            U = [random_half_integer_vector(nm, max_val=2) for _ in range(rank)]
            V = [random_half_integer_vector(mp, max_val=2) for _ in range(rank)]
            W = [random_half_integer_vector(pn, max_val=2) for _ in range(rank)]
            
            # Transpose to row-major: U[i][c] -> U_rows[i][c]
            U_rows = [[U[c][i] for c in range(rank)] for i in range(nm)]
            V_rows = [[V[c][j] for c in range(rank)] for j in range(mp)]
            W_rows = [[W[c][k] for c in range(rank)] for k in range(pn)]
            
            # Fast verification
            if verify_fast(U_rows, V_rows, W_rows, rank):
                return {"U": U_rows, "V": V_rows, "W": W_rows}
        
        return None
    
    # The random search is unlikely to find a solution for rank < 48.
    # Let me think differently.
    
    # For matrix multiplication, there's a known construction using Strassen-like
    # methods. For 3x4x4, let's try to construct it systematically.
    
    # Actually, let me try a smarter approach: use the algebraic structure.
    # The tensor T[i,j,k] = 1 for all i,j,k in the multiplication pattern.
    # This means for each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    
    # This is a system of 48 equations in rank*48 variables (but with structure).
    
    # Let me try: for small ranks, use a greedy approach to build terms.
    
    # Alternative: Use the fact that the tensor is "combinatorial" and try to
    # find a decomposition using 0, 1, -1, 0.5, -0.5 entries.
    
    # Let me try a more systematic search for rank 44-48.
    
    # Actually, the most promising approach for this specific problem:
    # Use the known result that 3x4x4 matrix multiplication has rank 44 or similar.
    # Let me try to find it by searching over structured decompositions.
    
    # Key idea: For each rank-1 term, the vectors u, v, w should have a specific
    # pattern. In particular, for the term to contribute to T[i,j,k]=1, we need
    # u[i*m+j] * v[j*p+k] * w[k*n+i] != 0 for some (i,j,k).
    
    # Let me try a different structural approach: use the "partition" method.
    # Partition the 48 target entries into groups, and for each group, find a
    # rank-1 term that covers multiple entries.
    
    # For a rank-1 term (u, v, w), the set of (i,j,k) where u[i*m+j]*v[j*p+k]*w[k*n+i] != 0
    # is determined by the supports of u, v, w.
    
    # Let me try: for each possible support pattern, check if it can cover target entries.
    
    # This is getting complex. Let me just try the random search with a better
    # initialization and more trials, focusing on ranks 44-48.
    
    # First, let me try a deterministic construction for a known rank.
    # For 3x4x4, I'll try rank 44.
    
    # Actually, let me just return the schoolbook if we can't find better.
    # But let me try a few ranks with random search.
    
    best_result = None
    
    # Try ranks from 44 to 48 (just below schoolbook)
    for rank in range(44, 49):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_trials=5000)
        if result:
            best_result = result
            break
    
    if best_result:
        return best_result
    
    # Fallback: schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
