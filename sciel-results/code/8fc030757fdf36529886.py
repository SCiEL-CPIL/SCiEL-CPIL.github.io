import time
import random
import numpy as np
import itertools

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    # Step 1: Find a valid graph on n=20
    adj_20 = None
    for seed in range(1000):
        if time.time() > deadline - 60:
            break
        rng = random.Random(seed)
        # Generate random graph
        adj = [0] * 20
        for i in range(20):
            for j in range(i + 1, 20):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if _is_valid_fast(adj, 20):
            adj_20 = adj
            break
            
    if adj_20 is None:
        # Fallback: try smaller n or return known valid small graph
        # But we need to maximize n. If we can't find 20, we fail the main goal.
        # However, the prompt says "Graphs with 20 vertices are known".
        # So we should find one.
        # If not found in 1000 seeds, try harder or different strategy.
        # For now, return a trivially valid small graph to avoid crash, 
        # but score will be low.
        # Let's try a few more seeds if time permits.
        for seed in range(1000, 5000):
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            adj = [0] * 20
            for i in range(20):
                for j in range(i + 1, 20):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if _is_valid_fast(adj, 20):
                adj_20 = adj
                break
                
    if adj_20 is None:
        # Last resort fallback
        return {'n': 5, 'edges': []}

    # Step 2: Try to extend to n=21
    adj_21 = _try_extend(adj_20, deadline - 10)
    
    if adj_21 is not None:
        edges = _adj_to_edges(adj_21, 21)
        return {'n': 21, 'edges': edges}
    else:
        edges = _adj_to_edges(adj_20, 20)
        return {'n': 20, 'edges': edges}

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _is_valid_fast(adj, n):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Uses vectorized NumPy checks for each subset size.
    """
    # Convert adj list to numpy matrix for easier indexing?
    # Or stick to bitmasks?
    # Bitmasks are fast for individual checks, but vectorized subset checks
    # are easier with matrices or pre-computed subset indices.
    
    # Pre-compute all subsets of size k for k=6..n
    # For n=20, this is fine.
    
    # We check sizes 6 to n
    for k in range(6, n + 1):
        # Generate all combinations of size k
        # This is the expensive part in Python if done naively.
        # But for n=20, k=6, it's 38k combos.
        # We can generate them as a list of tuples.
        
        # Optimization: If we find a violation, return False immediately.
        
        # Vectorized check:
        # Create an array of all subsets
        # This might use too much memory for large k?
        # C(20, 10) = 184756. 184756 * 10 * 10 floats = ~7MB. OK.
        
        subsets = list(itertools.combinations(range(n), k))
        if not subsets:
            continue
            
        S = np.array(subsets, dtype=np.int32) # Shape (M, k)
        
        # Extract sub-adjacency matrices
        # We need A[S[:, i], S[:, j]] for all i, j
        # A is a list of ints. Convert to numpy 2D array?
        # Let's convert adj to numpy uint8 matrix once.
        
    # Re-do with matrix
    A = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        mask = adj[i]
        # Set bits
        for j in range(n):
            if mask & (1 << j):
                A[i, j] = 1

    for k in range(6, n + 1):
        subsets = list(itertools.combinations(range(n), k))
        if not subsets:
            continue
            
        S = np.array(subsets, dtype=np.int32)
        
        # Get sub-matrices
        # Advanced indexing: A[S[:, :, None], S[:, None, :]]
        # This creates an array of shape (M, k, k)
        # M = C(n, k)
        
        # For n=20, k=6, M=38760. Size = 38760 * 36 = 1.4M. OK.
        # For n=20, k=10, M=184756. Size = 184756 * 100 = 18M. OK.
        
        # Memory might be an issue for k around n/2 if n is larger, but n=20 is fine.
        
        try:
            # Efficient extraction
            # A_sub[m, i, j] = A[S[m, i], S[m, j]]
            A_sub = A[S[:, :, None], S[:, None, :]]
            
            # Compute degrees: sum over j for each i
            # deg[m, i] = sum_j A_sub[m, i, j]
            degs = A_sub.sum(axis=2)
            
            # Check if all degrees in each subset are equal
            # degs shape (M, k)
            # Compare each column to the first column
            # is_reg[m] = all(degs[m, i] == degs[m, 0] for i)
            is_reg = (degs == degs[:, :1]).all(axis=1)
            
            if is_reg.any():
                return False
        except MemoryError:
            # Fallback to loop if memory error
            for subset in subsets:
                if _is_regular_subset_loop(adj, subset, k):
                    return False
                    # Actually _is_regular_subset_loop takes adj list
            # Continue to next k

    return True

def _is_regular_subset_loop(adj, subset, size):
    """Fallback loop check."""
    degs = []
    for v in subset:
        deg = 0
        for u in subset:
            if u != v and (adj[v] & (1 << u)):
                deg += 1
        degs.append(deg)
    return all(d == degs[0] for d in degs)

def _try_extend(adj_20, deadline):
    """
    Try to extend a valid n=20 graph to n=21.
    """
    n_old = 20
    n_new = 21
    
    # Pre-compute properties of subsets of old vertices
    # We need to check subsets S of V_old such that S U {new} is regular.
    # Condition:
    # Let x be the bitmask of neighbors of new vertex in V_old.
    # For a subset S (bitmask or list), let d_S(u) be degree of u in G[S].
    # Let X_S = sum_{u in S} x_u.
    # Regular iff for all u in S: d_S(u) + x_u == X_S.
    
    # We will iterate over all subsets S of V_old with size >= 5.
    # For each S, we can pre-compute d_S(u) for u in S.
    # Then the condition becomes a constraint on x.
    
    # Instead of pre-computing for all S, we can just check on the fly.
    # But checking all S for each x is expensive.
    # Strategy:
    # 1. Start with random x.
    # 2. Find violations.
    # 3. Fix violations by flipping bits in x.
    
    # To find violations efficiently:
    # Iterate over all subsets S of V_old with size >= 5.
    # For current x, check if S U {new} is regular.
    # This requires computing d_S(u) and X_S.
    
    # Pre-compute d_S(u) for all S?
    # Number of subsets S of size >= 5 is ~500k.
    # Storing d_S(u) for all S is too much memory.
    
    # Alternative:
    # Iterate over S.
    # For a given S, the condition is linear in x?
    # d_S(u) + x_u = X_S = sum_{w in S} x_w
    # => d_S(u) + x_u - sum_{w in S} x_w = 0
    # => d_S(u) + x_u - (x_u + sum_{w in S, w!=u} x_w) = 0
    # => d_S(u) - sum_{w in S, w!=u} x_w = 0
    # => sum_{w in S, w!=u} x_w = d_S(u)
    
    # This must hold for ALL u in S.
    # So, for a fixed S, x must satisfy:
    # For all u in S: sum_{w in S, w!=u} x_w = d_S(u)
    
    # Let y_w = x_w.
    # This is a system of equations.
    # Note that sum_{w in S, w!=u} x_w = X_S - x_u.
    # So X_S - x_u = d_S(u) => x_u = X_S - d_S(u).
    # This implies x_u is determined by X_S and d_S(u).
    # Also, summing over u:
    # sum_{u in S} (X_S - d_S(u)) = sum_{u in S} x_u = X_S
    # |S| X_S - sum_{u in S} d_S(u) = X_S
    # (|S|-1) X_S = sum_{u in S} d_S(u)
    # Note that sum_{u in S} d_S(u) = 2 * E(S), where E(S) is number of edges in G[S].
    # So (|S|-1) X_S = 2 E(S).
    # This gives a necessary condition for X_S.
    # If (|S|-1) does not divide 2 E(S), then no x exists for this S?
    # No, X_S is an integer. So 2 E(S) must be divisible by |S|-1.
    # If not, then S U {new} CANNOT be regular for ANY x.
    # This is a great filter!
    
    # So, for each S, check if 2 E(S) is divisible by |S|-1.
    # If yes, then X_S is fixed to 2 E(S) / (|S|-1).
    # Then x_u must be X_S - d_S(u).
    # We also need x_u to be 0 or 1.
    # And x_u must be consistent for all u.
    
    # Algorithm for Local Search:
    # 1. Start with random x.
    # 2. Find a violating S.
    #    - Iterate S.
    #    - Check if 2 E(S) % (|S|-1) == 0.
    #    - If yes, compute X_S.
    #    - Compute required x_u = X_S - d_S(u) for all u in S.
    #    - Check if current x matches these values.
    #    - If not, no violation?
    #      Wait. The condition for regularity is:
    #      Current x must satisfy x_u = X_S - d_S(u) for all u in S.
    #      AND X_S must equal sum_{w in S} x_w.
    #      If 2 E(S) is divisible by |S|-1, let X_req = 2 E(S) / (|S|-1).
    #      If current sum_{w in S} x_w != X_req, then S U {new} is NOT regular.
    #      If current sum_{w in S} x_w == X_req, we still need to check if x_u == X_req - d_S(u) for all u.
    #      If yes, it IS regular.
    
    # 3. If violation found, flip a bit in x to break it.
    
    # This is efficient because we can quickly filter out most S.
    
    # Implementation:
    
    # Pre-compute E(S) and d_S(u) for all S?
    # Too much memory.
    # Compute on the fly?
    # Iterating 500k subsets and computing E(S) and d_S(u) is slow.
    
    # Optimization:
    # We only need to check S that are "potential" violations.
    # Most S will fail the divisibility test.
    
    # Let's just iterate and check.
    # To speed up, we can pre-compute the adjacency matrix A.
    
    A = np.zeros((n_old, n_old), dtype=np.uint8)
    for i in range(n_old):
        mask = adj_20[i]
        for j in range(n_old):
            if mask & (1 << j):
                A[i, j] = 1
                
    # List of all subsets S of V_old with size >= 5
    # We generate them lazily or pre-compute?
    # Pre-computing list of 500k subsets is fine.
    
    # However, generating combinations is slow.
    # We can use a bitmask iteration?
    
    # For now, let's use itertools.combinations.
    
    # Local Search Loop
    rng = random.Random(42)
    
    # Initial x
    x = [rng.random() < 0.5 for _ in range(n_old)]
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        violation_found = False
        violating_S = None
        violating_x_req = None
        
        # Check all subsets S
        # To speed up, we can check random subsets first?
        # Or just iterate.
        
        # Optimization: Only check subsets S where 2 E(S) is divisible by |S|-1.
        # We can pre-compute E(S) for all S?
        # No, compute on the fly.
        
        # To make this faster, we can pre-compute a list of "critical" subsets S
        # where the divisibility condition holds.
        # This depends only on G_20, not on x.
        
        # Let's pre-compute critical_S.
        if not hasattr(_try_extend, 'critical_S'):
            _try_extend.critical_S = []
            for size in range(5, n_old + 1):
                for S in itertools.combinations(range(n_old), size):
                    # Compute E(S)
                    e_count = 0
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            if A[S[i], S[j]]:
                                e_count += 1
                    if (2 * e_count) % (size - 1) == 0:
                        _try_extend.critical_S.append(S)
                        
        # Now iterate over critical_S
        for S in _try_extend.critical_S:
            size = len(S)
            # Compute X_req
            e_count = 0
            for i in range(size):
                for j in range(i+1, size):
                    if A[S[i], S[j]]:
                        e_count += 1
            X_req = (2 * e_count) // (size - 1)
            
            # Check if current x matches
            # Sum of x in S
            X_cur = sum(x[u] for u in S)
            
            if X_cur != X_req:
                continue
                
            # Check if x_u == X_req - d_S(u) for all u in S
            valid = True
            for u in S:
                # Compute d_S(u)
                d = 0
                for v in S:
                    if v != u and A[u, v]:
                        d += 1
                req_x = X_req - d
                if x[u] != req_x:
                    valid = False
                    break
                    
            if valid:
                violation_found = True
                violating_S = S
                # We need to change x to break this.
                # Flip a random bit in S.
                break
                
        if not violation_found:
            # Success!
            adj_21 = adj_20[:]
            # Add vertex 20
            adj_21.append(0)
            for i in range(n_old):
                if x[i]:
                    adj_21[i] |= (1 << 20)
                    adj_21[20] |= (1 << i)
            return adj_21
            
        # Flip a bit in violating_S
        # Pick random u in violating_S
        u = rng.choice(violating_S)
        x[u] = not x[u]
        
    return None
