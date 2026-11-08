"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s limit, stop at 145s
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Initialize with schoolbook decomposition
    # Tensors are indexed by (i, j, k) -> U[i*m+j], V[j*p+k], W[k*n+i]
    # We represent the tensor as a list of rank-1 components.
    # Each component c is a tuple (u_vec, v_vec, w_vec) where these are basis vectors in their respective spaces.
    # Actually, it's easier to work with the index sets.
    # A rank-1 term is defined by (i, j, k).
    # The tensor T = sum_{i,j,k} T_{ijk} E_i \otimes E_j \otimes E_k
    # For matrix multiplication, T_{ijk} is 1 if i,j,k are valid indices, else 0? 
    # No, the multiplication tensor M_{(i,j),(k), (l,m)} is 1 if i=l and j=k.
    # Let's stick to the standard definition:
    # U is (n*m) x R, V is (m*p) x R, W is (p*n) x R.
    # The product C = A B is given by C_{l,m} = sum_{j,k} A_{l,j} B_{j,k}.
    # The tensor is T = sum_{l,j,k} E_l \otimes E_j \otimes E_k.
    # In the task's convention:
    # U rows indexed by (l, j) -> l*m + j
    # V rows indexed by (j, k) -> j*p + k
    # W rows indexed by (k, l) -> k*n + l
    # So the rank-1 terms correspond to triples (l, j, k).
    
    # Total possible terms: n * m * p
    # Each term t has:
    # U index: l * m + j
    # V index: j * p + k
    # W index: k * n + l
    
    # We want to find the minimal number of such terms that span the tensor.
    # This is equivalent to finding the rank of the tensor.
    
    # Strategy: 
    # 1. Start with all n*m*p terms.
    # 2. Try to remove one term at a time by checking if it is in the span of the others.
    #    "In the span" means there exist coefficients for the other terms such that 
    #    their linear combination equals the target term in the U, V, W spaces simultaneously.
    #    This is a system of linear equations.
    #    Let terms be t_1, ..., t_R. 
    #    We want to check if t_target = sum_{c != target} alpha_c t_c.
    #    This implies:
    #    U[t_target] = sum alpha_c U[t_c]
    #    V[t_target] = sum alpha_c V[t_c]
    #    W[t_target] = sum alpha_c W[t_c]
    #    Here U[t] is the unit vector e_{u_idx(t)}.
    #    So this is a system of (n*m + m*p + p*n) equations for (R-1) unknowns.
    #    We can solve this using least squares. If the residual is ~0, it's dependent.
    
    # To make this efficient and robust:
    # We will maintain a list of active terms.
    # We iterate through potential terms to remove.
    # To speed up, we can randomize the order or use a greedy approach: 
    # Try to remove the term that is "most dependent".
    
    # Let's represent each term by its (u_idx, v_idx, w_idx).
    terms = []
    for l in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = l * m + j
                v_idx = j * p + k
                w_idx = k * n + l
                terms.append((u_idx, v_idx, w_idx))
    
    # Current set of active terms
    active_terms = list(terms)
    
    # To check dependency of term t against set S:
    # We construct a matrix A where columns are the concatenated (u, v, w) vectors of terms in S \ {t}.
    # And vector b is the concatenated (u, v, w) vector of t.
    # We solve A x = b.
    # If ||A x - b|| < eps, then t is in span(S).
    
    def is_dependent(target, others):
        # target: (u, v, w)
        # others: list of (u, v, w)
        if not others:
            return False, None
            
        # Dimensions
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        total_dim = dim_u + dim_v + dim_w
        
        # Build matrix A
        # Number of columns = len(others)
        A = np.zeros((total_dim, len(others)))
        
        for idx, (ou, ov, ow) in enumerate(others):
            A[ou, idx] = 1.0
            A[dim_u + ov, idx] = 1.0
            A[dim_u + dim_v + ow, idx] = 1.0
            
        # Build vector b
        b = np.zeros(total_dim)
        tu, tv, tw = target
        b[tu] = 1.0
        b[dim_u + tv] = 1.0
        b[dim_u + dim_v + tw] = 1.0
        
        # Solve least squares
        # Use np.linalg.lstsq
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return False, None
            
        # Calculate residual
        Ax = A @ x
        residual = np.linalg.norm(Ax - b)
        
        # Tolerance
        # Since entries are 0/1, and we are looking for exact integer/half-integer solutions,
        # a very small residual indicates exact dependency in floating point.
        eps = 1e-9
        if residual < eps:
            # Check if x has simple values? The task allows integer/half-integer.
            # Lstsq might give floating point numbers. 
            # If it's exactly in the span, the coefficients should be representable.
            # We just need to return the coefficients if we want to keep them, 
            # but for removal, we just need to know it's dependent.
            # However, if we remove it, we don't need the coefficients for the final output, 
            # we just output the remaining basis vectors as the U, V, W matrices.
            # Wait, the output requires U, V, W matrices.
            # If we remove a term, the remaining terms form the decomposition.
            # The U, V, W matrices will have columns corresponding to the remaining terms.
            # The entries in U, V, W will be 0 or 1 (from the basis vectors).
            # This satisfies the "integer/half-integer" requirement.
            
            return True, x
        else:
            return False, None

    # Greedy removal loop
    # To avoid O(R^2 * Cost_Lstsq) being too slow, we limit iterations or use a smarter order.
    # R starts at 60. 
    # Cost of one check: Lstsq on (60+30+20=110) x (59) matrix. Very fast.
    # Total checks if we try all pairs: 60 * 59 * ... is too much if we restart.
    # Strategy: 
    # 1. Iterate through current active terms.
    # 2. For each term, check if it's dependent on the others.
    # 3. If yes, remove it.
    # 4. Repeat until no more terms can be removed or time limit.
    
    # To ensure we find a good set, we can shuffle the order of checking to avoid local minima?
    # Or just try removing the first one found.
    
    max_iterations = 1000
    iteration = 0
    
    while time.time() < deadline - 5.0 and iteration < max_iterations:
        removed_any = False
        # Shuffle active terms to check different orders? 
        # Actually, dependency is a property of the set. 
        # If term A is dependent on B,C,D, it doesn't matter if we check A first or B first.
        # But if we remove A, the set changes.
        # Let's just iterate through the current list.
        
        # To be safe against getting stuck in a specific order, we can randomize the check order each pass.
        indices = list(range(len(active_terms)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
                
            target = active_terms[idx]
            # Others are all active_terms except target
            others = [t for i, t in enumerate(active_terms) if i != idx]
            
            is_dep, _ = is_dependent(target, others)
            if is_dep:
                # Remove target
                active_terms.pop(idx)
                removed_any = True
                # Restart the pass with the new list
                break
                
        if not removed_any:
            # No more terms can be removed in this pass
            # To try to find more, we could try a different randomization or stop.
            # Given the small size, one pass might be enough, or a few passes.
            # Let's do a few passes with different randomizations.
            pass_count = 0
            while pass_count < 10 and time.time() < deadline - 5.0:
                removed_any_pass = False
                indices = list(range(len(active_terms)))
                np.random.shuffle(indices)
                for idx in indices:
                    target = active_terms[idx]
                    others = [t for i, t in enumerate(active_terms) if i != idx]
                    is_dep, _ = is_dependent(target, others)
                    if is_dep:
                        active_terms.pop(idx)
                        removed_any_pass = True
                        break
                if not removed_any_pass:
                    break
                pass_count += 1
            
            if not removed_any_pass:
                # If we didn't remove anything in the extra passes, we are likely at a local minimum 
                # or a global minimum for this greedy strategy.
                break
                
        iteration += 1

    # Construct the final U, V, W matrices from active_terms
    rank = len(active_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col, (u_idx, v_idx, w_idx) in enumerate(active_terms):
        U[u_idx, col] = 1.0
        V[v_idx, col] = 1.0
        W[w_idx, col] = 1.0

    # Convert to lists of lists of floats (or ints if preferred, but floats are fine)
    # The task says "integer/half-integer entries". 1.0 is an integer.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
