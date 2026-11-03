"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import random
import itertools
import sys
import time

# Known rank for 3x3 x 3x5 matrix multiplication is 31 (from Baur-Strassen and others)
# Schoolbook rank is 3*3*5 = 45
# We need to find a decomposition with rank <= 31

# The tensor for matrix multiplication of (3x3) * (3x5) = (3x5)
# T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
# The tensor is T = sum_{i,j,k} E_i^A ⊗ E_j^B ⊗ E_k^C where E_i, E_j, E_k are basis vectors

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # We'll use a randomized algorithm to find a low-rank decomposition
    # of the matrix multiplication tensor
    
    # The tensor T has shape (n*m, m*p, p*n) = (9, 15, 15)
    # T[a, b, c] where a = i*m+j (row of A), b = j*p+k (col of A = row of B), c = k*n+i (col of B)
    # T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent, else 0
    
    # More precisely:
    # A is n x m = 3x3
    # B is m x p = 3x5
    # C = A*B is n x p = 3x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Tensor T of shape (n*m, m*p, p*n):
    # T[i*m+j, j*p+k, k*n+i] = 1 (all others 0)
    
    # We want to find U, V, W of shapes (n*m, r), (m*p, r), (p*n, r)
    # such that T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # The rank of 3x3 * 3x5 matrix multiplication:
    # Lower bounds: max(n*m, m*p, n*p) = max(9, 15, 15) = 15
    # Known upper bounds from various results...
    
    # For 3x3 * 3x5, let's think about this more carefully.
    # The bilinear complexity of 3x3 * 3x5 is known to be at most 31 based on certain algorithms.
    
    # Let's try to construct an explicit decomposition with rank 31.
    
    # Actually, let me think about this differently. Let me use a known result:
    # The bilinear rank of M_{3,3,5} is at most... 
    
    # Let me try a different approach: use the fact that we can pad or exploit structure.
    
    # For M_{n,m,p}, one known upper bound comes from Kronecker substitution or similar.
    
    # Let me try to find a decomposition with rank < 45 (schoolbook).
    
    # One approach: use the "banded" decomposition where we group operations.
    
    # Actually, let me try a practical approach: use random initialization and 
    # optimize to find a low-rank decomposition.
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), else 0.
    
    # Let's build the tensor explicitly:
    n, m, p = 3, 3, 5
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    # Build tensor T as a list of non-zero entries
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    tensor_nnz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_nnz.append((a, b, c, 1.0))
    
    # Schoolbook rank:
    rank_schoolbook = len(tensor_nnz)  # 45
    
    def verify(U, V, W, tensor_nnz, nm, mp, pn, tol=1e-6):
        """Verify that U ⊗ V ⊗ W equals the tensor defined by tensor_nnz."""
        r = len(U[0]) if U else 0
        # Compute the decomposed tensor at non-zero positions
        for (a, b, c, val) in tensor_nnz:
            s = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
            if abs(s - val) > tol:
                return False
        # Also check that off-diagonal entries are zero (sample check)
        # Build full tensor from decomposition and check against expected
        # For efficiency, just check a random sample of off-diagonal entries
        return True
    
    def verify_full(U, V, W, nm, mp, pn, tol=1e-6):
        """Full verification: check all (a,b,c) positions."""
        r = len(U[0]) if U else 0
        # Expected tensor: T[a,b,c] = 1 if exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        expected = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    expected[(a, b, c)] = 1.0
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in expected:
                        s = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
                        if abs(s - 1.0) > tol:
                            return False
                    else:
                        s = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
                        if abs(s) > tol:
                            return False
        return True
    
    # Try to find a decomposition with rank < 45
    
    # Strategy 1: Start with schoolbook and try to merge terms
    # Strategy 2: Random initialization + optimization
    # Strategy 3: Use known constructions
    
    # Let me try a practical optimization approach:
    # Initialize with random vectors and use coordinate descent or similar
    
    # First, let me try small rank values and see if we can find a valid decomposition.
    
    def try_rank(r, num_trials=10, seed_base=42):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            seed = seed_base + trial * 137
            random.seed(seed)
            
            # Initialize U, V, W randomly
            U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(nm)]
            V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(mp)]
            W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(pn)]
            
            # Simple iterative method: for each term, adjust to fit
            # This is very basic and may not converge well, but let's try
            
            # Better: use the structure. The tensor is sparse with exactly n*m*p=45 non-zero entries.
            # We need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # Let's try a different approach: fix some structure based on the problem.
            # 
            # For M_{3,3,5}, note that:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # 
            # One known trick: if we can factor one dimension, we reduce rank.
            # E.g., if m is prime and small, we might use FFT-like tricks.
            # For m=3, we can write A[i,j]*B[j,k] using 3 terms naturally.
            #
            # Actually, the schoolbook IS the natural m=3 sum.
            #
            # Let me try: use the fact that 5 = 3 + 2 and see if there's structure.
            
            pass  # placeholder
            
        return None
    
    # Let me try a more systematic approach:
    # The key insight is that the matrix multiplication tensor M_{n,m,p} has rank related to 
    # the complexity of multiplying polynomials of degree < n and < p with coefficients in 
    # a ring of size m (or similar).
    
    # For our specific case, let me try to explicitly construct a low-rank decomposition.
    
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{2,2,3}) <= 9
    # rank(M_{3,3,3}) <= 23 (Laderman)
    # rank(M_{3,3,5}) <= ?
    
    # Let me try an approach based on Kronecker products:
    # M_{3,3,5} can be embedded in M_{4,4,8} or similar, but that doesn't directly help.
    
    # Alternative: use the decomposition via convolution.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Think of A as having 3 rows, each with 3 entries; B as having 3 rows, each with 5 entries.
    
    # Let me try to find a rank-31 construction (the known upper bound for this case from 
    # certain papers on bilinear complexity).
    
    # Actually, I should just try to find ANY rank less than 45.
    
    # Let me try a greedy/structured approach:
    # Since the problem is small, I'll try to build the decomposition term by term.
    
    # Approach: Use the "sliding window" or "overlap" method.
    # For each pair (i,k), C[i,k] = sum_j A[i,j]*B[j,k].
    # We have 15 such sums, each with 3 terms.
    
    # Idea: If two (i,k) pairs share structure, we can combine them.
    
    # Let me try yet another approach: integer/half-integer based constructions.
    # Many known fast matrix multiplication algorithms use small integers.
    
    # Let me try to construct using a specific pattern:
    # We want 31 (or fewer) trilinear terms.
    
    # Given the complexity of finding an optimal decomposition analytically,
    # let me try a numerical search with random restarts and local optimization.
    
    def random_search(r_max, total_iters=5000, deadline=None):
        """Random search for low-rank decomposition."""
        best_rank = r_max + 1
        best_result = None
        
        for r in range(15, r_max + 1):  # Start from lower bound
            if deadline and time.time() > deadline - 10:
                break
            if best_rank <= r:
                continue
                
            # Try to find a rank-r decomposition
            for attempt in range(total_iters // (r_max + 1)):
                if deadline and time.time() > deadline - 5:
                    break
                    
                random.seed(attempt * 7919 + r * 104729)
                
                # Random initialization with some structure
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
                
                # Quick check
                # Verify at the 45 non-zero positions
                valid = True
                for (a, b, c, val) in tensor_nnz:
                    s = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
                    if s != val:
                        valid = False
                        break
                
                if valid:
                    # Full verification
                    if verify_full(U, V, W, nm, mp, pn, tol=1e-9):
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
                        return best_rank, best_result
        
        return best_rank, best_result
    
    # First, return schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try random search
    # This is unlikely to find a solution by pure random, but let's try
    # A better approach: use the structure of the problem
    
    # Let me think about this more carefully.
    # 
    # The tensor M_{3,3,5} has 45 non-zero entries in a 9x15x15 space.
    # The lower bound on rank is max(n*m, m*p, n*p) = max(9,15,15) = 15.
    # 
    # Known results:
    # - rank(M_{2,2,2}) = 7
    # - rank(M_{2,2,3}) <= 9 (and >= 8)
    # - rank(M_{3,3,3}) <= 23 (Laderman), lower bound 19
    # - rank(M_{3,3,4}) <= ?
    # - rank(M_{3,3,5}) <= ?
    #
    # For M_{3,3,5}, a reasonable upper bound might be around 31-35.
    
    # Let me try a structured construction:
    # Write the 3x3 * 3x5 multiplication using overlapping blocks.
    
    # A = [a11 a12 a13]
    #     [a21 a22 a23]
    #     [a31 a32 a33]
    #
    # B = [b11 b12 b13 b14 b15]
    #     [b21 b22 b23 b24 b25]
    #     [b31 b32 b33 b34 b35]
    #
    # C = A*B:
    # c11 = a11*b11 + a12*b21 + a13*b31
    # c12 = a11*b12 + a12*b22 + a13*b32
    # ... etc.
    
    # Schoolbook: 45 multiplications.
    # 
    # One way to save: notice that c11..c15 all use a11, a12, a13.
    # c1k = a11*b1k + a12*b2k + a13*b3k for k=1..5
    # That's 15 multiplications for row 1 of C.
    # Similarly for rows 2 and 3.
    # Total: 45.
    
    # To save multiplications, we need to share some.
    # E.g., if we compute s1 = a11*b11 + a12*b22 (shared between c11 and c22 if they share structure).
    # But c11 and c22 don't share the same products generally.
    
    # Actually, for non-square products, the structure is different.
    
    # Let me just try to find a rank-39, 38, ... decomposition by modifying the schoolbook.
    
    # Idea: In schoolbook, each term is U=1 at one position, V=1 at one, W=1 at one.
    # If we can merge two schoolbook terms into one trilinear term, we save rank.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged into:
    # U has 1 at (i1,j1) and (i2,j2)
    # V has 1 at (j1,k1) and (j2,k2)
    # W has 1 at (k1,i1) and (k2,i2)
    # BUT: this creates extra cross terms! U[i1*m+j1]*V[j2*p+k2]*W[k2*n+i2] etc.
    # So naive merging doesn't work unless the cross terms cancel.
    
    # With signs: U has +1 at (i1,j1) and -1 at (i2,j2), etc., we can cancel cross terms
    # but this requires specific patterns.
    
    # This is getting complex. Let me just try to find the best we can within the time budget.
    
    # Given the difficulty, let me try a different strategy:
    # Use known fast multiplication for the 3x3 part and handle the 5 columns.
    
    # For 3x3 * 3x3, Laderman uses rank 23.
    # For 3x3 * 3x5 = 3x3 * (3x3 + 3x2), we can do:
    # 3x3*3x3 (rank 23) + 3x3*3x2 (rank ?) 
    # But this isn't quite right because the addition doesn't simply add ranks.
    
    # Let me just go with a reasonable attempt.
    
    # Actually, let me try a much simpler but effective approach:
    # Use the schoolbook decomposition and try to find rank reductions 
    # by checking if any linear dependencies exist among the terms.
    
    # In the schoolbook decomposition, each column c corresponds to (i,j,k).
    # The three vectors are:
    #   U_c = e_{i*m+j}  (standard basis vector in R^9)
    #   V_c = e_{j*p+k}  (standard basis vector in R^15)
    #   W_c = e_{k*n+i}  (standard basis vector in R^15)
    
    # Two columns c1, c2 can potentially be "merged" if there's a way to write
    # e1⊗f1⊗g1 + e2⊗f2⊗g2 = something of rank 1.
    # This is only possible if (e1,f1,g1) and (e2,f2,g2) are proportional, which they're not
    # for distinct (i,j,k) pairs.
    
    # So the schoolbook terms are linearly independent in the tensor sense.
    
    # To reduce rank, we need a completely different construction.
    
    # Given time constraints, let me try one more thing: a semi-structured random search
    # with the constraint that entries are in {-1, 0, 1} or half-integers.
    
    # For small rank, let's try to search more intelligently.
    
    def structured_search(max_rank, num_rounds=1000, deadline=None):
        """Search with structure-aware initialization."""
        best = None
        best_r = max_rank + 1
        
        for target_r in range(15, max_rank + 1):
            if deadline and time.time() > deadline - 5:
                break
            if best_r <= target_r:
                continue
                
            found = False
            for rnd in range(num_rounds):
                if deadline and time.time() > deadline - 3:
                    break
                if found:
                    break
                    
                random.seed(rnd * 15485863 + target_r * 7919)
                
                # Initialize with a partial schoolbook structure
                # Start with all schoolbook terms, then randomly perturb and see if rank reduces
                r = target_r
                
                # Random integer matrix
                U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(nm)]
                V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(mp)]
                W = [[random.randint(-2, 2) for _ in range(r)] for _ in range(pn)]
                
                # Quick verification
                ok = True
                for (a, b, c, val) in tensor_nnz:
                    s = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
                    if abs(s - val) > 1e-9:
                        ok = False
                        break
                
                if ok:
                    # Full check
                    if verify_full(U, V, W, nm, mp, pn, tol=1e-9):
                        best = {"U": U, "V": V, "W": W}
                        best_r = target_r
                        found = True
                        
            if best_r <= 35:  # Good enough
                break
        
        return best_r, best
    
    # Try structured search
    # This is unlikely to work with pure random, so let's also try a greedy construction
    
    # Greedy: build decomposition term by term
    def greedy_construct(deadline=None):
        """Build decomposition by adding terms one at a time."""
        # Target: T[a,b,c] for all (a,b,c) in tensor_nnz
        # We'll build U, V, W incrementally.
        
        # Start with empty
        U = []
        V = []
        W = []
        
        # Remaining tensor to cover
        remaining = {}
        for (a, b, c, val) in tensor_nnz:
            remaining[(a, b, c)] = val
        
        # Also need off-diagonal entries to be 0
        # This makes greedy hard because each term affects many entries.
        
        # Alternative: fix the target and solve.
        # For a given rank r, we need to find U,V,W such that
        # sum_r U[a,r]V[b,r]W[c,r] = T[a,b,c] for all (a,b,c).
        
        # This is a system of polynomial equations. Hard in general.
        
        # Let me try a different tactic: exploit the specific structure.
        # 
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # For i=0: c_0k = a_00*b_0k + a_01*b_1k + a_02*b_2k
        # For i=1: c_1k = a_10*b_0k + a_11*b_1k + a_12*b_2k  
        # For i=2: c_2k = a_20*b_0k + a_21*b_1k + a_22*b_2k
        #
        # Note that for fixed j, the term a_ij * b_jk appears in C[i,k].
        # 
        # Idea: compute all products a_i0*b_0k for i=0,1,2 and k=0,1,2,3,4 (15 products)
        #       all products a_i1*b_1k for i=0,1,2 and k=0,1,2,3,4 (15 products)
        #       all products a_i2*b_2k for i=0,1,2 and k=0,1,2,3,4 (15 products)
        # Total: 45. This is just schoolbook.
        
        # To save, we need to compute combinations.
        # E.g., s0 = (a_00+a_01)*b_00, s1 = (a_00+a_01)*b_10, etc.
        # But this introduces extra cross terms that need cancellation.
        
        # Given the time, let me just try to find a rank-39 decomposition
        # by slightly modifying the schoolbook.
        
        # Actually, I'll try one more thing: use numpy to do a proper numerical search.
        return None
    
    # Let me try using numpy for a more serious search
    import numpy as np
    
    def numpy_search(deadline):
        """Use numpy-based random search."""
        T_nnz = tensor_nnz
        
        # Try decreasing rank
        for r in range(40, 14, -1):  # Try from 40 down to 15
            if time.time() > deadline - 10:
                break
                
            # Try multiple random initializations
            max_tries = 50 if r > 35 else 200
            
            for t in range(max_tries):
                if time.time() > deadline - 5:
                    break
                    
                np.random.seed(t * 31 + r)
                
                # Random initialization
                U = np.random.randint(-1, 2, size=(nm, r)).astype(float)
                V = np.random.randint(-1, 2, size=(mp, r)).astype(float)
                W = np.random.randint(-1, 2, size=(pn, r)).astype(float)
                
                # Quick check at non-zero positions
                max_err = 0
                for (a, b, c, val) in T_nnz:
                    s = U[a] @ (V[b][:, None] * W[c][:, None]).T
                    # Actually: s = sum_r U[a,r]*V[b,r]*W[c,r]
                    s = float(U[a] * V[b] * W[c]).sum()
                    max_err = max(max_err, abs(s - val))
                
                if max_err < 1e-9:
                    # Full verification
                    full_ok = True
                    for a in range(nm):
                        for b in range(mp):
                            for c in range(pn):
                                s = float(U[a] * V[b] * W[c]).sum()
                                if (a, b, c) in {(x[0], x[1], x[2]) for x in T_nnz}:
                                    if abs(s - 1.0) > 1e-9:
                                        full_ok = False
                                        break
                                else:
                                    if abs(s) > 1e-9:
                                        full_ok = False
                                        break
                            if not full_ok:
                                break
                        if not full_ok:
                            break
                    
                    if full_ok:
                        return r, {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        return None, None
    
    # Try numpy search
    result_r, result = numpy_search(deadline)
    
    if result:
        return result
    
    # If nothing found, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
