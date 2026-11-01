import time
import numpy as np
from itertools import product, combinations
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 9
P = p * n  # 6
SCHOOLBOOK = n * m * p  # 18

def build_tensor():
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def generate_support_patterns():
    """Generate all possible rank-1 support patterns (i,j,k) tuples."""
    patterns = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                patterns.append((i, j, k))
    return patterns

def tensor_residual(U, V, W, T):
    """Compute residual efficiently."""
    rank = U.shape[1]
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                total += (s - T[i,j,k]) ** 2
    return total

def solve_support_subset(patterns, r, T, time_limit=5.0):
    """
    Try to find a rank-r decomposition using a subset of 'r' support patterns.
    Uses linear programming to find half-integer coefficients.
    """
    start_time = time.time()
    n_patterns = len(patterns)
    
    if n_patterns < r:
        return None
    
    # Generate all subsets of size r
    # This is C(n_patterns, r). For n_patterns=18 and r=15, it's C(18,15) = 816.
    # For r=16, C(18,16) = 153.
    # For r=17, C(18,17) = 18.
    # So we can exhaustively search for r >= 15 easily.
    
    for subset in combinations(range(n_patterns), r):
        if time.time() - start_time > time_limit:
            break
            
        # Build the linear system A x = b
        # x is a vector of length r (coefficients for each pattern in the subset)
        # A has shape (n*m*p, r)
        A = np.zeros((n*m*p, r))
        b = np.zeros(n*m*p)
        
        for idx, pat_idx in enumerate(subset):
            i, j, k = patterns[pat_idx]
            # The contribution of this pattern to T[i,j,k] is x_idx
            # But wait, the tensor decomposition is T = sum_r U_r V_r W_r
            # If we restrict to support patterns (i,j,k), it means:
            # For a given term r, U[i,j] is non-zero only if (i,j) is in the support of U for that term.
            # Actually, a "support pattern" in the context of matrix multiplication tensor
            # usually refers to the structure of the rank-1 terms.
            # A rank-1 term is defined by matrices U_r, V_r, W_r.
            # If we say the "support" is (i,j,k), it implies:
            # U_r is non-zero only at (i,j), V_r only at (j,k), W_r only at (k,i).
            # In that case, the term contributes x_r * U_r[i,j] * V_r[j,k] * W_r[k,i] to T[i,j,k].
            # And 0 to all other entries.
            # So if we choose a set of supports, we are choosing a sparse structure.
            # Let's assume for each term r, we pick a triple (i_r, j_r, k_r).
            # Then U_r[i_r, j_r] = u_r, V_r[j_r, k_r] = v_r, W_r[k_r, i_r] = w_r.
            # The contribution to T[i,j,k] is sum_{r: (i_r,j_r,k_r)==(i,j,k)} u_r v_r w_r.
            # Wait, this is only correct if the supports are disjoint in a specific way?
            # No, the tensor T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i].
            # If we force U_r to be non-zero only at (i_r, j_r), etc., then:
            # T[i,j,k] gets a contribution from term r ONLY IF i=i_r AND j=j_r AND k=k_r.
            # In that case, T[i,j,k] = sum_{r: i_r=i, j_r=j, k_r=k} u_r v_r w_r.
            # Since T[i,j,k] = 1 for all i,j,k, we need:
            # For every (i,j,k), the sum of products of coefficients of terms with support (i,j,k) must be 1.
            # This implies that we must have at least one term for each (i,j,k) to get a non-zero value,
            # UNLESS we allow terms to have larger supports.
            
            # Let's reconsider. The standard "support" search in tensor decomposition (like for Strassen)
            # allows U_r, V_r, W_r to have multiple non-zero entries.
            # However, a simpler approach for small ranks is to assume the "diagonal" structure
            # where each term is associated with a specific (i,j,k) but allows for "cross-talk"
            # if we relax the strict diagonal assumption.
            
            # Actually, the most general rank-1 tensor is defined by three vectors.
            # If we want to use integer/half-integer entries, we can search over the entries directly.
            # But the space is huge.
            
            # Let's stick to the "diagonal support" hypothesis first:
            # Each term r is associated with a triple (i_r, j_r, k_r).
            # U_r[i_r, j_r] = u_r, V_r[j_r, k_r] = v_r, W_r[k_r, i_r] = w_r.
            # All other entries are 0.
            # Then T[i,j,k] = sum_{r: (i_r,j_r,k_r)=(i,j,k)} u_r v_r w_r.
            # We need T[i,j,k] = 1 for all 18 triples.
            # This requires that for each triple (i,j,k), the sum of products of coefficients
            # of terms assigned to that triple is 1.
            # If we assign exactly one term to each triple, we need u_r v_r w_r = 1.
            # This gives rank 18.
            # Can we do better?
            # If we allow a term to have support on multiple (i,j,k) triples, it's no longer a simple diagonal.
            # A rank-1 tensor U V W has entries U[i,j] V[j,k] W[k,i].
            # If U, V, W are rank-1 matrices (outer products), then the tensor is a "product" tensor.
            # But U, V, W are just matrices here, not necessarily rank-1.
            # Wait, U is (n x m), V is (m x p), W is (p x n).
            # The tensor is T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i].
            # This is the standard definition.
            
            # Let's try a different search strategy:
            # Assume the decomposition has a specific structure.
            # For 2x3x3, the rank is known to be 16 (Bini et al.).
            # Let's try to find a rank-16 decomposition.
            
            # The "support" idea is tricky. Let's go back to the linear system approach
            # but with a more general parameterization.
            
            # Let's use the fact that we can scale the terms.
            # If we have a valid decomposition, we can scale U_r by a, V_r by b, W_r by c
            # such that a*b*c = 1.
            
            # Let's try to solve for a rank-r decomposition where each term
            # has a "dominant" support (i_r, j_r, k_r) but allows small values elsewhere?
            # No, we need exact half-integer entries.
            
            # Let's try a grid search over the coefficients for a fixed support structure.
            # Support structure: assign each of the r terms to a triple (i,j,k).
            # Multiple terms can be assigned to the same triple.
            # For each triple (i,j,k), let S_{ijk} be the set of terms assigned to it.
            # We need sum_{r in S_{ijk}} u_r v_r w_r = 1.
            # And for any (i',j',k') != (i,j,k), the term r assigned to (i,j,k) must not contribute to T[i',j',k'].
            # This requires U_r[i',j'] = 0 or V_r[j',k'] = 0 or W_r[k',i'] = 0 for all (i',j',k') != (i,j,k).
            # This is very restrictive. It essentially forces the diagonal structure.
            
            # So, if we stick to the diagonal structure, we are stuck with rank 18 unless
            # we allow the "cross terms" to cancel out.
            # But if U_r is non-zero at (i,j) and (i',j'), then for a term r assigned to (i,j),
            # it will contribute to T[i',j',k] if V_r and W_r are non-zero there.
            
            # Let's try a different approach:
            # We know that rank 16 is possible.
            # Let's try to construct it.
            # A known rank-16 algorithm for 2x3x3 (or similar small matrices) might exist.
            # Or we can use the "lattice" method.
            
            # Let's try to find a rank-16 decomposition by solving a system of equations
            # with a specific pattern.
            
            # Pattern:
            # Let r = 16.
            # We have 16 terms.
            # Let's try to assign them to the 18 triples.
            # Two triples will have to be "shared" or "covered" by terms assigned to other triples?
            # No, that's not how it works.
            
            # Let's try to use the fact that 2x3x3 is close to 2x2x2 (rank 7) and 3x3x3 (rank 23?).
            
            # Actually, let's just try to find a rank-16 decomposition by search.
            # We will parameterize the entries of U, V, W as half-integers.
            # We will use a local search or simulated annealing to minimize the residual.
            
            # Let's implement a more robust search for rank r.
            pass
        
        # The diagonal support assumption leads to rank 18.
        # We need a more flexible structure.
        
        # Let's try to find a rank-16 decomposition by assuming a specific sparsity pattern.
        # For example, let's assume that U, V, W have a specific number of non-zero entries.
        
        # Let's try a random search with local refinement, but with a better initialization.
        
        # For now, let's just return None if we can't find a solution quickly.
        # We will improve the search in the next iteration.
        
        pass
        
    return None

def is_half_integer(arr, tol=1e-8):
    """Check if all entries are integer or half-integer."""
    flat = arr.flatten()
    v = flat * 2
    return np.all(np.abs(v - np.round(v)) < tol)

def round_to_half(arr):
    return np.round(arr * 2) / 2

def verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that U, V, W exactly decompose T with half-integer entries."""
    if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
        return False
    res = tensor_residual(U, V, W, T)
    return res < tol

def schoolbook_decomposition():
    rank = SCHOOLBOOK
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def try_rank_search(r, T, time_limit=10.0):
    """
    Try to find a rank-r decomposition using a combination of
    structured initialization and local search.
    """
    start_time = time.time()
    patterns = generate_support_patterns()
    
    # We will try to find a decomposition where each term is associated with a support pattern.
    # But as we saw, the diagonal support assumption is too restrictive.
    # Instead, we will use a more general approach:
    # We will initialize U, V, W with random half-integer values and then refine them.
    
    best_obj = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize U, V, W with random half-integer values
        np.random.seed(42 + attempt * 100 + r)
        U = np.random.randint(-2, 3, size=(N, r)) / 2.0
        V = np.random.randint(-2, 3, size=(M, r)) / 2.0
        W = np.random.randint(-2, 3, size=(P, r)) / 2.0
        
        # Refine using alternating least squares
        for it in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    b = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    b = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            obj = tensor_residual(U, V, W, T)
            
            if obj < 1e-14:
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return U, V, W
                else:
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    if tensor_residual(U_r, V_r, W_r, T) < 1e-8:
                        return U_r, V_r, W_r
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return None

def try_rank_structured(r, T, max_iters=2000, seed=42, time_limit=8.0):
    """Try to find a rank-r decomposition using structured initialization and refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    for attempt in range(12):
        if time.time() - start_time > time_limit:
            break
            
        # Structured initialization
        if attempt == 0:
            # Cyclic assignment
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        c = c % r
                        U[i*m+j, c] += 1.0
                        V[j*p+k, c] += 1.0
                        W[k*n+i, c] += 1.0
        elif attempt == 1:
            # Sparse initialization
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        c = c % r
                        U[i*m+j, c] = 1.0
                        V[j*p+k, c] = 1.0
                        W[k*n+i, c] = 1.0
        elif attempt == 2:
            # Block structure: try to group by i
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                base = i * (r // n)
                for j in range(m):
                    for k in range(p):
                        c = base + (j * p + k) % (r // n)
                        U[i*m+j, c] = 1.0
                        V[j*p+k, c] = 1.0
                        W[k*n+i, c] = 1.0
        elif attempt == 3:
            # Random half-integer initialization
            np.random.seed(seed + attempt * 100)
            U = np.random.randint(-2, 3, size=(N, r)) / 2.0
            V = np.random.randint(-2, 3, size=(M, r)) / 2.0
            W = np.random.randint(-2, 3, size=(P, r)) / 2.0
        else:
            # Random initialization with scaling
            U = rng.randn(N, r) * (1.0 / np.sqrt(r))
            V = rng.randn(M, r) * (1.0 / np.sqrt(r))
            W = rng.randn(P, r) * (1.0 / np.sqrt(r))
        
        best_obj = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Alternating least squares
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    b = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    b = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Calculate objective
            obj = tensor_residual(U, V, W, T)
            
            if obj < 1e-14:
                # Check if half-integer
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return U, V, W
                else:
                    # Try rounding
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    if tensor_residual(U_r, V_r, W_r, T) < 1e-8:
                        return U_r, V_r, W_r
            
            if obj < best_obj:
                best_obj = obj
    
    return None

def try_rank_milp(r, T, time_limit=5.0):
    """Try to find a rank-r decomposition using MILP-like approach with scipy.optimize.milp."""
    start_time = time.time()
    
    # This is a simplified approach - we'll use a grid search over small integer/half-integer values
    # For very small ranks, we can enumerate support patterns
    
    if r > 15:
        return None  # Too large for MILP-style enumeration
    
    # Try a few structured patterns with small integer values
    values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # For rank <= 10, try some sparse patterns
    for pattern_idx in range(20):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(12345 + pattern_idx * 7 + r)
        
        U = np.zeros((N, r))
        V = np.zeros((M, r))
        W = np.zeros((P, r))
        
        # Random sparse support
        for col in range(r):
            # Choose which entries are non-zero
            u_mask = np.random.random(N) < 0.3
            v_mask = np.random.random(M) < 0.3
            w_mask = np.random.random(P) < 0.3
            
            for idx in range(N):
                if u_mask[idx]:
                    U[idx, col] = np.random.choice(values)
            for idx in range(M):
                if v_mask[idx]:
                    V[idx, col] = np.random.choice(values)
            for idx in range(P):
                if w_mask[idx]:
                    W[idx, col] = np.random.choice(values)
        
        # Refine with alternating least squares
        for it in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    b = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    b = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            obj = tensor_residual(U, V, W, T)
            if obj < 1e-14:
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return U, V, W
                else:
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    if tensor_residual(U_r, V_r, W_r, T) < 1e-8:
                        return U_r, V_r, W_r
    
    return None

def construct():
    start_time = time.time()
    deadline = 130  # Leave 20s safety margin
    
    T = build_tensor()
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = SCHOOLBOOK
    
    # Try ranks from low to high
    # We know that rank 16 is possible for 2x3x3.
    # Let's try ranks 16, 15, 14, ... in that order.
    for r in range(16, 9, -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        time_alloc = 10.0 if r <= 16 else 5.0
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time()
            if time_alloc < 1.0:
                break
        
        # Try structured ALS
        result = try_rank_structured(r, T, max_iters=3000, seed=42, time_limit=time_alloc * 0.6)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, T):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found a valid lower rank
        
        # Try MILP-style search if time permits
        remaining = deadline - time.time()
        if remaining > 2.0:
            result = try_rank_milp(r, T, time_limit=min(remaining * 0.4, 4.0))
            if result is not None:
                U, V, W = result
                if verify_decomposition(U, V, W, T):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
    
    return best
