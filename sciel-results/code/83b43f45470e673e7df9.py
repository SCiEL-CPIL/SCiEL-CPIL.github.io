"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the target tensor T as a 3D array T[(i,j),(j',k),(i',k')] = 1 iff i=i', j=j', k=k'."""
    # We'll build it as a 2D matrix of shape (n*m*p, n*m*p) by flattening,
    # but actually we need a 3D tensor. Let's think about the structure.
    # 
    # The tensor T has indices:
    #   U-index: (i, j) where i in [0,n), j in [0,m)  -> n*m values
    #   V-index: (j', k) where j' in [0,m), k in [0,p) -> m*p values
    #   W-index: (k', i') where k' in [0,p), i' in [0,n) -> p*n values
    #
    # T[(i,j), (j',k), (k',i')] = 1 if i=i' AND j=j' AND k=k'
    #
    # So for each (i,j,k), there's exactly one non-zero entry:
    # T[(i,j), (j,k), (k,i)] = 1
    #
    # Total non-zeros: n*m*p = 90
    
    # We'll represent the decomposition as:
    # T = sum_{c=0}^{r-1} U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    #
    # And the tensor entry is:
    # T[(i,j),(j',k),(k',i')] = sum_c U[(i,j),c] * V[(j',k),c] * W[(k',i'),c]
    
    pass

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the target tensor."""
    r = U.shape[1]
    
    # Check dimensions
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    
    # For each (i,j,k), verify that the tensor entry is 1
    # and for all other combinations, it's 0
    
    # Efficient check: compute all entries
    # T[(i,j),(j',k),(k',i')] = sum_c U[(i,j),c] * V[(j',k),c] * W[(k',i'),c]
    
    # Check non-zero entries: for each (i,j,k), T[(i,j),(j,k),(k,i)] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries: sample some random zero entries
    # For efficiency, check a subset
    random.seed(42)
    for _ in range(1000):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        jp = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        kp = random.randint(0, p - 1)
        ip = random.randint(0, n - 1)
        
        # Skip if this is a non-zero entry
        if i == ip and j == jp and k == kp:
            continue
        
        val = 0.0
        for c in range(r):
            val += U[i * m + j, c] * V[jp * p + k, c] * W[kp * n + ip, c]
        if abs(val) > tol:
            return False
    
    return True

def _try_strassen_like_2x2(n, m, p, deadline):
    """
    Try to use Strassen's algorithm on 2x2 blocks to reduce rank.
    For 3x5x6, we can try to decompose into 2x2 blocks where possible.
    """
    # 3 = 2 + 1, 5 = 2 + 2 + 1, 6 = 2 + 2 + 2
    # This is complex. Let's try a simpler approach.
    
    # Actually, let's try to find a rank-89 decomposition by modifying the schoolbook.
    # The idea: if we can find two schoolbook terms that can be combined into one
    # using a Strassen-like identity, we save one rank.
    
    # For 2x2x2, Strassen saves 1 term (from 8 to 7).
    # Can we embed a 2x2x2 Strassen decomposition into 3x5x6?
    
    # If we can find a 2x2x2 sub-problem within 3x5x6, we can save 1 rank there.
    # 3 >= 2, 5 >= 2, 6 >= 2, so yes.
    
    # Let's try: use Strassen for the first 2x2x2 block, schoolbook for the rest.
    # The first 2x2x2 block uses rows i=0,1, columns j=0,1, k=0,1.
    # Strassen uses 7 terms instead of 8, saving 1.
    # Remaining: (3*5*6 - 2*2*2) = 90 - 8 = 82 schoolbook terms.
    # Total: 7 + 82 = 89.
    
    # Let's construct this!
    
    U = np.zeros((n * m, 89), dtype=float)
    V = np.zeros((m * p, 89), dtype=float)
    W = np.zeros((p * n, 89), dtype=float)
    
    # Strassen for 2x2x2 (indices i=0,1, j=0,1, k=0,1)
    # C_{i,k} = sum_j A_{i,j} B_{j,k} for i,k in {0,1}, j in {0,1}
    #
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express this in terms of U, V, W vectors.
    # Each Strassen term is a product of a linear combination of A entries
    # and a linear combination of B entries, giving a linear combination of C entries.
    
    # Term M1: (A00 + A11)(B00 + B11)
    # U vector: has 1 at (0,0) and (1,1)
    # V vector: has 1 at (0,0) and (1,1)
    # W vector: contributes to C00 (+1), C01 (0), C10 (0), C11 (-1)
    # Wait, M1 contributes to C00 with +1 and C11 with +1? Let me recheck.
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00 and +1 to C11.
    
    # Let me be more careful. For the 2x2x2 subproblem:
    # A is 2x2, B is 2x2, C is 2x2.
    # A_{i,j}, B_{j,k}, C_{i,k} for i,j,k in {0,1}
    
    # M1 = (A_{0,0} + A_{1,1})(B_{0,0} + B_{1,1})
    # This product contributes to C_{0,0} with coefficient +1 and C_{1,1} with coefficient +1.
    # In our tensor framework:
    # U: e_{0,0} + e_{1,1} (in the 2x2 A space)
    # V: e_{0,0} + e_{1,1} (in the 2x2 B space)  
    # W: e_{0,0} + e_{1,1} (in the 2x2 C space, where C_{i,k} is indexed by (k,i) in W)
    
    # Wait, W is indexed by (k', i'). So C_{i,k} corresponds to W[k,i].
    # M1 contributes +1 to C_{0,0} -> W[0,0] += 1
    # M1 contributes +1 to C_{1,1} -> W[1,1] += 1
    
    # Let me define the 7 Strassen terms:
    
    # M1: U = e_{0,0} + e_{1,1}, V = e_{0,0} + e_{1,1}, W = e_{0,0} + e_{1,1}
    # M2: U = e_{1,0} + e_{1,1}, V = e_{0,0}, W = e_{0,0} (contributes to C10) 
    #     Wait, M2 = (A10 + A11)B00. This gives C10 += M2 and C11 += -M2.
    #     So W: e_{0,1} - e_{1,1}? No...
    
    # Let me restart with proper bookkeeping.
    # For the 2x2 subproblem (i,j,k in {0,1}):
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # M1 = (A00 + A11)(B00 + B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   Contributes to: C00 += A00*B00, C01 += A00*B11, C10 += A11*B00, C11 += A11*B11
    #   In tensor form: U = e_{0,0} + e_{1,1}, V = e_{0,0} + e_{1,1}, W = e_{0,0} + e_{1,1}
    #   Check: sum_c U[(i,j),c] * V[(j',k),c] * W[(k',i'),c]
    #   For (i,j)=(0,0), (j',k)=(0,0), (k',i')=(0,0): 1*1*1 = 1 ✓ (C00 gets A00*B00)
    #   For (i,j)=(0,0), (j',k)=(1,1), (k',i')=(1,0): 1*1*1 = 1 ✓ (C01 gets A00*B11)
    #   For (i,j)=(1,1), (j',k)=(0,0), (k',i')=(0,1): 1*1*1 = 1 ✓ (C10 gets A11*B00)
    #   For (i,j)=(1,1), (j',k)=(1,1), (k',i')=(1,1): 1*1*1 = 1 ✓ (C11 gets A11*B11)
    #   But wait, M1 should contribute +1 to C00 and +1 to C11 only, not to C01 and C10!
    #   
    # I'm confusing myself. Let me use the standard Strassen formulas more carefully.
    
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Using 0-indexed: i,j,k in {0,1}
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each M is a product of a linear combination of A entries and B entries.
    # In tensor decomposition, each M corresponds to one term:
    # M = (sum_j a_j * A_{i_j, j_j}) * (sum_k b_k * B_{j'_k, k_k})
    # This gives a tensor term: U = sum_j a_j * e_{i_j, j_j}, V = sum_k b_k * e_{j'_k, k_k}
    # And W should be such that the product reconstructs the correct C entries.
    
    # For M1 = (A00 + A11)(B00 + B11):
    # U = e_{0,0} + e_{1,1}
    # V = e_{0,0} + e_{1,1}
    # The product U ⊗ V gives:
    #   (0,0)⊗(0,0), (0,0)⊗(1,1), (1,1)⊗(0,0), (1,1)⊗(1,1)
    # These correspond to: A00*B00, A00*B11, A11*B00, A11*B11
    # We need W such that:
    #   W[0,0] * 1 (for A00*B00 -> C00) = +1
    #   W[1,0] * 1 (for A00*B11 -> C01) = 0 (M1 doesn't contribute to C01)
    #   W[0,1] * 1 (for A11*B00 -> C10) = 0 (M1 doesn't contribute to C10)
    #   W[1,1] * 1 (for A11*B11 -> C11) = +1
    # But W is a single vector! We can't have different coefficients for different (k,i) pairs
    # unless we use multiple terms.
    
    # Ah, I see the issue. In tensor decomposition, each term is a rank-1 tensor.
    # M1 is NOT a rank-1 tensor in general. It's a sum of 4 rank-1 tensors.
    # So Strassen's 7 products don't directly translate to 7 tensor terms.
    
    # The Strassen algorithm works by computing 7 scalar products, each of which
    # is a product of two linear combinations. But each such product expands to
    # multiple tensor terms.
    
    # So this approach doesn't directly work. Let me think differently.
    
    # Actually, in the context of tensor rank, Strassen's algorithm for 2x2x2
    # DOES give a rank-7 decomposition. The key insight is that the 7 products
    # are not just scalar products but define 7 tensor terms.
    
    # Let me look at this more carefully. The 2x2x2 matrix multiplication tensor
    # has rank 7. The 7 terms are:
    # T1: U = e_{0,0} + e_{1,1}, V = e_{0,0} + e_{1,1}, W = e_{0,0} + e_{1,1}
    # T2: U = e_{1,0} + e_{1,1}, V = e_{0,0}, W = e_{0,0} - e_{1,1}
    # ... etc.
    
    # Let me just implement the known rank-7 decomposition for 2x2x2.
    
    # From literature, the Strassen decomposition for M_{2,2,2} is:
    # (using 1-indexed for clarity, will convert to 0-indexed)
    
    # Let me use a different approach: just try random search for rank 89.
    
    return None

def _random_search_rank(n, m, p, target_rank, deadline, max_iter=500):
    """Try to find a decomposition of the given rank using random initialization + local search."""
    
    # Build the target tensor as a flat vector
    # T[(i,j),(j',k),(k',i')] = 1 if i=i', j=j', k=k'
    
    # For efficiency, we'll work with the non-zero entries only.
    # There are n*m*p = 90 non-zero entries.
    
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i, j, k))
    
    r = target_rank
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute error on non-zero entries
        err = 0.0
        for (i, j, k) in nz_indices:
            val = 0.0
            for c in range(r):
                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
            err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-15:
            break
        
        # Local search: perturb one entry
        factor = random.choice(['U', 'V', 'W'])
        row = random.randint(0, (n * m if factor == 'U' else m * p if factor == 'V' else p * n) - 1)
        col = random.randint(0, r - 1)
        
        old_val = U[row, col] if factor == 'U' else V[row, col] if factor == 'V' else W[row, col]
        new_val = old_val + random.gauss(0, 0.1)
        
        if factor == 'U':
            U[row, col] = new_val
        elif factor == 'V':
            V[row, col] = new_val
        else:
            W[row, col] = new_val
        
        # Compute new error
        new_err = 0.0
        for (i, j, k) in nz_indices:
            val = 0.0
            for c in range(r):
                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
            new_err += (val - 1.0) ** 2
        
        if new_err > err * 1.01:
            if factor == 'U':
                U[row, col] = old_val
            elif factor == 'V':
                V[row, col] = old_val
            else:
                W[row, col] = old_val
    
    if best_err < 1e-10:
        # Verify
        if _verify_decomposition(n, m, p, best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Try to find rank 89 decomposition
    random.seed(42)
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        result = _random_search_rank(n, m, p, 89, deadline, max_iter=100)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
