import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute quads
    # Vertices 0..15
    # Edges (i, j) with i < j
    # Edge index: i * 15 - i * (i+1) // 2 + (j - i - 1) ?
    # Simpler: map (i,j) to index
    edge_to_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edge_to_idx[(i, j)] = idx
            idx += 1
    # idx should be 120
    
    # List of quads: each is a list of 6 edge indices
    quads = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                    q_edges = [
                        edge_to_idx[(a, b)],
                        edge_to_idx[(a, c)],
                        edge_to_idx[(a, d)],
                        edge_to_idx[(b, c)],
                        edge_to_idx[(b, d)],
                        edge_to_idx[(c, d)]
                    ]
                    quads.append(q_edges)
    
    # Map edge index to list of quads containing it
    edge_to_quads = [[] for _ in range(120)]
    for q_idx, q_edges in enumerate(quads):
        for e_idx in q_edges:
            edge_to_quads[e_idx].append(q_idx)
            
    def solve_for_k(k):
        # Initialize random
        edges = [random.randint(0, k - 1) for _ in range(120)]
        quad_bad = [False] * len(quads)
        bad_count = 0
        
        # Calculate initial badness
        for q_idx, q_edges in enumerate(quads):
            if len(set(edges[e] for e in q_edges)) < 5:
                quad_bad[q_idx] = True
                bad_count += 1
                
        best_edges = edges.copy()
        best_bad_count = bad_count
        
        if best_bad_count == 0:
            return best_edges, best_bad_count
            
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < deadline:
            # Pick random edge
            e_idx = random.randint(0, 119)
            old_c = edges[e_idx]
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            delta = 0
            affected_quads = edge_to_quads[e_idx]
            
            # Store old states to rollback if rejected
            # We can just calculate delta and if rejected, revert edges[e_idx]
            # But we modified quad_bad? No, we only calculate delta first.
            
            # Optimization: only calculate delta for affected quads
            # We need to know if they were bad and if they will be bad.
            
            # Temporarily change edge
            edges[e_idx] = new_c
            
            for q_idx in affected_quads:
                q_edges = quads[q_idx]
                new_distinct = len(set(edges[e] for e in q_edges))
                new_bad = new_distinct < 5
                old_bad = quad_bad[q_idx]
                
                if new_bad and not old_bad:
                    delta += 1
                elif not new_bad and old_bad:
                    delta -= 1
                    
                # We don't update quad_bad yet, only if accepted
                # But we need to know new_bad for the logic above?
                # Yes, we computed it.
                
            # SA Acceptance
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                # Update quad_bad for affected quads
                for q_idx in affected_quads:
                    q_edges = quads[q_idx]
                    new_distinct = len(set(edges[e] for e in q_edges))
                    quad_bad[q_idx] = (new_distinct < 5)
                bad_count += delta
                
                if bad_count < best_bad_count:
                    best_bad_count = bad_count
                    best_edges = edges.copy()
                    if best_bad_count == 0:
                        return best_edges, best_bad_count
            else:
                # Reject
                edges[e_idx] = old_c
                
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_edges, best_bad_count

    # Try k=15, 14, 13
    # We want the minimum k that yields a valid solution.
    # But we don't know which will work.
    # Let's try 15 first as it's most likely.
    # If we find a solution for 15, we can try 14 with remaining time.
    
    best_solution = None
    best_k = 100
    
    # Try 15
    if time.time() < deadline - 10:
        sol, cost = solve_for_k(15)
        if cost == 0:
            best_solution = sol
            best_k = 15
            
    # Try 14
    if best_k > 14 and time.time() < deadline - 10:
        sol, cost = solve_for_k(14)
        if cost == 0:
            best_solution = sol
            best_k = 14
            
    # Try 13
    if best_k > 13 and time.time() < deadline - 5:
        sol, cost = solve_for_k(13)
        if cost == 0:
            best_solution = sol
            best_k = 13
            
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
