import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 140  # Safety margin
    
    best_p = None
    best_cost = float('inf')
    
    # Initial solutions
    starts = []
    
    # 1. Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    starts.append(p_rank.copy())
    
    # 2. Greedy construction
    p_greedy = np.zeros(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        best_loc = -1
        best_cost_loc = float('inf')
        for u in range(n):
            if assigned[u]:
                continue
            # Cost of assigning i to u
            # Sum over assigned j
            # a[i, assigned] @ b[u, p[assigned]] + a[assigned, i] @ b[p[assigned], u]
            # Since a, b symmetric: 2 * sum a[i,j] b[u, p[j]]
            # But careful with unassigned.
            # We can just compute full cost incrementally?
            # Or just compute partial cost.
            # Partial cost = sum_{j assigned} a[i,j] b[u, p[j]] + sum_{j assigned} a[j,i] b[p[j], u]
            # = 2 * sum_{j assigned} a[i,j] b[u, p[j]] (if symmetric)
            # Let's just compute it directly to be safe.
            cost = 0
            for j in range(n):
                if assigned[j]:
                    cost += a[i, j] * b[u, p_greedy[j]]
                    cost += a[j, i] * b[p_greedy[j], u]
            if cost < best_cost_loc:
                best_cost_loc = cost
                best_loc = u
        p_greedy[i] = best_loc
        assigned[best_loc] = True
    starts.append(p_greedy.copy())
    
    # 3. Random starts
    for _ in range(10):
        starts.append(np.random.permutation(n).astype(np.int64))
    
    for p_start in starts:
        if time.time() > deadline:
            break
            
        # Local search
        p = p_start.copy()
        current_cost = np.sum(a * b[np.ix_(p, p)])
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                for j in range(i+1, n):
                    u = p[i]
                    v = p[j]
                    diff_a = a[i] - a[j]
                    diff_b = b[v, p] - b[u, p]
                    dot = np.dot(diff_a, diff_b)
                    
                    # Correction terms
                    term_i = (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
                    term_j = (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
                    
                    sum_excl = dot - term_i - term_j
                    delta = 2 * sum_excl
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                        improved = True
            
            if improved:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
    return best_p.tolist()
