"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Deadline guard
    deadline = time.time() + 115.0
    
    # 1. Initialization Strategies
    
    # Strategy 1: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Strategy 2: Spectral Initialization
    # Use the Fiedler vector (2nd smallest eigenvector) of A and B
    # For QAP, we want to align the structure.
    try:
        # Symmetrize A and B for spectral analysis if not already
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get eigenvectors
        # scipy is available, but np.linalg.eigh is fine for n=256
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        # Fiedler vector is index 1 (0 is the trivial all-ones vector, usually)
        # Sort to ensure we get the right one if eigenvalues are degenerate
        fiedler_a = evecs_a[:, 1]
        fiedler_b = evecs_b[:, 1]
        
        # Partition facilities based on sign of fiedler_a
        # Partition locations based on sign of fiedler_b
        # We want to map "positive" facilities to "positive" locations or vice versa?
        # Usually, we want to minimize cut, so align signs.
        
        # Create a permutation that respects the spectral ordering
        # Sort facilities by fiedler value
        order_f = np.argsort(fiedler_a)
        order_l = np.argsort(fiedler_b)
        
        p_spec = np.empty(n, dtype=np.int64)
        p_spec[order_f] = order_l
        
        # Also try the reverse mapping (anti-correlation)
        p_spec_rev = np.empty(n, dtype=np.int64)
        p_spec_rev[order_f] = order_l[::-1]
        
    except Exception:
        p_spec = p_rank
        p_spec_rev = p_rank

    # Evaluate initial costs
    def calc_cost(p):
        # p[i] is location of facility i
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # Use einsum for speed
        return np.einsum('ij,ik,jk->', a, b[np.ix_(p, p)], optimize=True)

    cost_rank = calc_cost(p_rank)
    cost_spec = calc_cost(p_spec)
    cost_spec_rev = calc_cost(p_spec_rev)
    
    best_p = p_rank.copy()
    best_cost = cost_rank
    if cost_spec < best_cost:
        best_cost = cost_spec
        best_p = p_spec.copy()
    if cost_spec_rev < best_cost:
        best_cost = cost_spec_rev
        best_p = p_spec_rev.copy()
        
    # 2. Local Search: Simulated Annealing with 2-Swaps
    
    # Pre-compute nothing specific for SA, just use the delta formula
    # Delta for swapping u, v:
    # Let x = p[u], y = p[v]
    # Delta = sum_{k != u,v} (a[u,k] - a[v,k]) * (b[x, p[k]] - b[y, p[k]])
    #       + (a[u,v] * b[x,y] + a[v,u] * b[y,x] - a[u,v] * b[y,x] - a[v,u] * b[x,y])
    # Assuming symmetry a[u,v]=a[v,u] and b[x,y]=b[y,x], the mutual term cancels out?
    # Old mutual: a[u,v]b[x,y] + a[v,u]b[y,x] = 2 a[u,v] b[x,y]
    # New mutual: a[u,v]b[y,x] + a[v,u]b[x,y] = 2 a[u,v] b[x,y]
    # So mutual term change is 0.
    
    # So Delta = sum_{k != u,v} (a[u,k] - a[v,k]) * (b[x, p[k]] - b[y, p[k]])
    
    # We can vectorize the inner sum for a given u, v
    # But doing it in a loop is O(n). n=256 is small.
    
    T = 1000.0
    T_min = 1.0
    cooling = 0.999
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Randomize start slightly if time permits? No, stick to best init.
    
    # Helper to get delta
    # We can speed this up by precomputing row differences?
    # a_diff = a[u, :] - a[v, :]
    # b_diff = b[x, :] - b[y, :]  # Note: b is indexed by locations, but we need b[x, p[k]]
    # So we need b[x, current_p] - b[y, current_p]
    
    steps = 0
    while time.time() < deadline:
        # Batch of moves? No, simple loop.
        # To speed up, we can do multiple moves per time check
        
        # Pick random u, v
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        x = current_p[u]
        y = current_p[v]
        
        # Calculate Delta
        # a_diff = a[u, :] - a[v, :]
        # b_vals_u = b[x, current_p]
        # b_vals_v = b[y, current_p]
        # b_diff = b_vals_u - b_vals_v
        # delta = np.dot(a_diff, b_diff)
        
        # Correction for k=u and k=v terms included in the dot product:
        # Term k=u: (a[u,u] - a[v,u]) * (b[x, x] - b[y, x])
        # Term k=v: (a[u,v] - a[v,v]) * (b[x, y] - b[y, y])
        # We want to EXCLUDE these from the sum.
        # So delta = dot - term_u - term_v
        
        a_diff = a[u, :] - a[v, :]
        b_diff = b[x, current_p] - b[y, current_p]
        
        delta = np.dot(a_diff, b_diff)
        
        # Subtract the u and v self-interaction terms that were included in the dot product
        # k=u term: a_diff[u] * b_diff[u]
        # a_diff[u] = a[u,u] - a[v,u]
        # b_diff[u] = b[x, p[u]] - b[y, p[u]] = b[x, x] - b[y, x]
        term_u = a_diff[u] * b_diff[u]
        
        # k=v term: a_diff[v] * b_diff[v]
        # a_diff[v] = a[u,v] - a[v,v]
        # b_diff[v] = b[x, p[v]] - b[y, p[v]] = b[x, y] - b[y, y]
        term_v = a_diff[v] * b_diff[v]
        
        delta -= term_u
        delta -= term_v
        
        # Acceptance
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            # Swap
            current_p[u] = y
            current_p[v] = x
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
        T *= cooling
        
        steps += 1
        if steps % 1000 == 0:
            if time.time() > deadline:
                break
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
