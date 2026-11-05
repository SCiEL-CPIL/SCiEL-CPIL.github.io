import random
import itertools
import numpy as np
from scipy.optimize import linprog

def construct():
    n = 20
    # 1. Create complete graph edges
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v, 0]) # Placeholder color
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx
            idx += 1
            
    # 2. Assign a Proper Edge Coloring (1-factorization for K_20)
    # Pattern: center = n-1. 
    # Edges (center, i) -> color i % (n-1)
    # Edges (i, j) where i,j != center -> color (i+j) % (n-1)
    # Note: This is a proper edge coloring.
    
    k = n // 2
    center = n - 1
    for i in range(len(edges)):
        u, v, _ = edges[i]
        if u == center:
            other = v
        elif v == center:
            other = u
        else:
            other = -1
            
        if other != -1:
            edges[i][2] = other % (n - 1)
        else:
            edges[i][2] = (u + v) % (n - 1)

    # 3. Precompute 4-cycles and identify "bad" ones
    # A 4-cycle is bad if opposite edges share a color.
    # In a proper coloring, adjacent edges have different colors.
    # So we check c(e1)==c(e3) or c(e2)==c(e4).
    
    bad_cycles = [] # List of tuples of edge indices (e1, e2, e3, e4)
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,c)]
        e3 = edge_to_idx[(c,d)]
        e4 = edge_to_idx[(d,a)]
        if edges[e1][2] == edges[e3][2] or edges[e2][2] == edges[e4][2]:
            bad_cycles.append((e1, e2, e3, e4))
            
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,d)]
        e3 = edge_to_idx[(d,c)]
        e4 = edge_to_idx[(c,a)]
        if edges[e1][2] == edges[e3][2] or edges[e2][2] == edges[e4][2]:
            bad_cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(a,c)]
        e2 = edge_to_idx[(c,b)]
        e3 = edge_to_idx[(b,d)]
        e4 = edge_to_idx[(d,a)]
        if edges[e1][2] == edges[e3][2] or edges[e2][2] == edges[e4][2]:
            bad_cycles.append((e1, e2, e3, e4))

    # 4. Branch and Bound with LP Relaxation
    
    best_edges_found = 0
    best_mask = [True] * len(edges)
    
    # Precompute LP matrices for active edges? 
    # Since active set changes, we rebuild constraints each time or use a mask.
    # For speed, we will filter bad_cycles based on active mask.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def get_lp_bound(active_mask):
        active_indices = [i for i, m in enumerate(active_mask) if m]
        if not active_indices:
            return 0.0
        
        num_active = len(active_indices)
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c_obj = np.full(num_active, -1.0)
        
        # Constraints: For each bad cycle fully contained in active_indices, sum(x_i) <= 3
        # We need to map global edge idx to local active idx
        global_to_local = {g: l for l, g in enumerate(active_indices)}
        
        A_ub = []
        b_ub = []
        
        for cycle in bad_cycles:
            # Check if all edges in cycle are active
            if all(active_mask[e] for e in cycle):
                row = np.zeros(num_active)
                for e in cycle:
                    row[global_to_local[e]] = 1.0
                A_ub.append(row)
                b_ub.append(3.0)
                
        if not A_ub:
            return float(num_active)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * num_active, method='highs')
        
        if res.success:
            return -res.fun
        else:
            # Fallback to greedy bound if LP fails
            return float(num_active)

    def branch_and_bound(active_mask, depth=0):
        nonlocal best_edges_found, best_mask
        
        current_count = sum(active_mask)
        
        # Prune 1: If current count + remaining possible <= best
        # Simple bound: current_count is the max we can get from this branch if we remove 0 more.
        # Actually, we can only remove edges. So max is current_count.
        if current_count <= best_edges_found:
            return
        
        # Find all bad cycles in current active set
        active_bad_cycles = []
        for cycle in bad_cycles:
            if all(active_mask[e] for e in cycle):
                active_bad_cycles.append(cycle)
                
        if not active_bad_cycles:
            # Valid configuration
            if current_count > best_edges_found:
                best_edges_found = current_count
                best_mask = active_mask.copy()
            return
        
        # LP Bound
        lb = get_lp_bound(active_mask)
        if lb <= best_edges_found:
            return
            
        # Select a bad cycle to branch on
        # Heuristic: Pick cycle with edges having highest degree in the "conflict graph" of active bad cycles
        # For simplicity, pick the first one, or one with max frequency
        # Let's pick the cycle that appears most frequently? No, just pick one.
        # To improve, pick the edge with highest conflict count.
        
        # Let's just pick the first active bad cycle for now, or randomize to avoid bias?
        # Deterministic is better for reproducibility.
        cycle = active_bad_cycles[0]
        
        # Branch on removing each edge in the cycle
        # Order edges by how many other bad cycles they are in (remove the most critical one first? 
        # No, we want to find a solution. Removing the "least critical" might leave more edges?
        # Actually, we branch on ALL removals. The order affects which branch is explored first.
        # Exploring the branch that removes the "most conflicting" edge first might find a valid solution faster?
        
        # Calculate conflict counts for edges in this cycle
        counts = {e: 0 for e in cycle}
        for c in active_bad_cycles:
            for e in cycle:
                if e in c:
                    counts[e] += 1
                    
        # Sort edges in cycle by count descending (try removing the most constrained one first)
        sorted_edges = sorted(cycle, key=lambda e: counts[e], reverse=True)
        
        for e in sorted_edges:
            new_mask = active_mask.copy()
            new_mask[e] = False
            branch_and_bound(new_mask, depth + 1)

    initial_mask = [True] * len(edges)
    branch_and_bound(initial_mask)
    
    # Construct result
    result = []
    for i, (u, v, c) in enumerate(edges):
        if best_mask[i]:
            result.append([u, v, c])
            
    return result
