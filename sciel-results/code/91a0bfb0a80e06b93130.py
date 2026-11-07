import time
import random
import numpy as np

N = 128
DEADLINE = 55.0  # 55s to be safe

def build_hypergraph():
    """
    Returns list of hyperedges. Each hyperedge is a tuple of unique ints in 1..N.
    Forbidden if ALL elements in the tuple are present in S.
    """
    edges = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s3 > N:
                    continue
                # The set of vertices involved
                vs = [a, b, c, s1, s2, s3]
                # Create a canonical frozenset for uniqueness
                edge = frozenset(vs)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(tuple(edge))
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    edges = build_hypergraph()
    n_edges = len(edges)
    
    # Map vertex -> list of edge indices
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge sizes (all are 6, but frozenset might merge? 
    # If a+b = c, then {a,b,c,s1,s2,s3} might have size < 6.
    # The constraint is: if a,b,c in S AND a+b in S AND a+c in S AND b+c in S.
    # If a+b=c, then the condition is "a,b,c in S" (since c=a+b is in S).
    # So the "edge" is effectively the set of distinct vertices involved.
    # If all distinct vertices in that set are in S, it's a violation.
    # So using frozenset is correct.
    edge_sizes = [len(e) for e in edges]
    
    best_set = set()
    best_size = 0
    
    # Greedy function
    def greedy(weights):
        sel = set()
        # counts[i] = number of vertices in edge i that are in sel
        counts = np.zeros(n_edges, dtype=np.int8)
        # Order vertices by weight
        order = sorted(range(1, N + 1), key=lambda v: -weights[v])
        
        for v in order:
            if time.time() > deadline:
                break
            conflict = False
            for ei in vert_to_edges[v]:
                if counts[ei] == edge_sizes[ei] - 1:
                    conflict = True
                    break
            if not conflict:
                sel.add(v)
                for ei in vert_to_edges[v]:
                    counts[ei] += 1
        return sel, counts, sel  # return sel and counts for potential reuse
    
    # 1. Randomized Greedy Restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Generate random weights
        weights = np.random.rand(N + 1)
        # Bias towards higher numbers? Or random? 
        # Let's try uniform first.
        
        sel, counts, _ = greedy(weights)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel.copy()
            
        # Check time, if low, do final polish
        if time.time() > deadline - 5.0:
            break

    # 2. Polish the best set found
    if time.time() < deadline:
        sel = best_set.copy()
        # Rebuild counts for current best_set
        counts = np.zeros(n_edges, dtype=np.int8)
        for v in sel:
            for ei in vert_to_edges[v]:
                counts[ei] += 1
                
        # Try adding any missing vertex
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if v in sel:
                    continue
                conflict = False
                for ei in vert_to_edges[v]:
                    if counts[ei] == edge_sizes[ei] - 1:
                        conflict = True
                        break
                if not conflict:
                    sel.add(v)
                    for ei in vert_to_edges[v]:
                        counts[ei] += 1
                    improved = True
                    if len(sel) > best_size:
                        best_size = len(sel)
                        best_set = sel.copy()
            
        # If stuck, try 1-remove-then-add-all
        # This is more expensive.
        if time.time() < deadline:
            # Try removing each v in sel, then greedy-add
            current_sel = sel.copy()
            current_counts = counts.copy()
            
            for v_out in list(current_sel):
                if time.time() > deadline:
                    break
                
                # Temporarily remove v_out
                temp_sel = set(current_sel)
                temp_sel.remove(v_out)
                temp_counts = current_counts.copy()
                for ei in vert_to_edges[v_out]:
                    temp_counts[ei] -= 1
                
                # Greedy add from temp_sel
                # Use a fixed order (e.g., numerical) or random? 
                # Numerical is fast and deterministic.
                order = sorted(range(1, N + 1), key=lambda x: 0) # Just 1..N
                
                for v_in in order:
                    if v_in in temp_sel:
                        continue
                    conflict = False
                    for ei in vert_to_edges[v_in]:
                        if temp_counts[ei] == edge_sizes[ei] - 1:
                            conflict = True
                            break
                    if not conflict:
                        temp_sel.add(v_in)
                        for ei in vert_to_edges[v_in]:
                            temp_counts[ei] += 1
                            
                if len(temp_sel) > len(current_sel):
                    current_sel = temp_sel
                    current_counts = temp_counts
                    if len(current_sel) > best_size:
                        best_size = len(current_sel)
                        best_set = current_sel.copy()
                    # Restart the loop if improved? 
                    # For efficiency, just do one pass of removals.
        
        # Final check
        if len(current_sel) > len(best_set):
             best_set = current_sel.copy()

    return [int(v) for v in sorted(best_set)]
