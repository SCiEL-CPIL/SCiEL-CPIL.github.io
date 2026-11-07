import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute valid masks (clique or independent set)
    valid_masks = []
    # We skip 0 as it's not a part
    for mask in range(1, 1 << n):
        # Check clique
        is_c = True
        m = mask
        while m and is_c:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != (mask ^ lsb):
                is_c = False
            m ^= lsb
        
        if is_c:
            valid_masks.append(mask)
            continue
        
        # Check independent set
        is_i = True
        m = mask
        while m and is_i:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != 0:
                is_i = False
            m ^= lsb
        
        if is_i:
            valid_masks.append(mask)

    # If no valid masks found (should not happen for n>=1), return singletons
    if not valid_masks:
        return [[i] for i in range(n)]

    # Formulate MILP
    # Variables x_j for each valid mask j
    # Minimize sum x_j
    # Subject to sum_{j: v in mask_j} x_j = 1 for all v
    # x_j in {0, 1}
    
    num_vars = len(valid_masks)
    
    # Objective
    c = np.ones(num_vars)
    
    # Constraints
    # A is n x num_vars
    A = np.zeros((n, num_vars))
    for j, mask in enumerate(valid_masks):
        for v in range(n):
            if mask & (1 << v):
                A[v, j] = 1.0
                
    b = np.ones(n)
    
    constraints = LinearConstraint(A, b, b)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_vars)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # Extract partition
            partition = []
            for j, mask in enumerate(valid_masks):
                if res.x[j] > 0.5:
                    part = [i for i in range(n) if mask & (1 << i)]
                    partition.append(part)
            
            # Verify partition covers all vertices exactly once
            # If verification fails, fall back to DP
            seen = [0] * n
            for part in partition:
                for v in part:
                    seen[v] += 1
            if all(count == 1 for count in seen):
                return [[int(v) for v in part] for part in partition]
    except Exception:
        pass

    # Fallback: Subset DP
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    valid_bool = [False] * (1 << n)
    for mask in valid_masks:
        valid_bool[mask] = True

    for mask in range(1, 1 << n):
        sub = mask
        while sub > 0:
            if valid_bool[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub == -1:
            # Should not happen if valid_masks is correct
            break
        part = [i for i in range(n) if sub & (1 << i)]
        partition.append(part)
        mask ^= sub
    
    return [[int(v) for v in part] for part in partition]

# EVOLVE-BLOCK-END
