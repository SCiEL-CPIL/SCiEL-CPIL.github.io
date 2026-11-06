import time
import random
import itertools

N = 40

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    
    # Precompute all non-rim edges
    # Rim edges are (i, i+1) % N
    rim_edges = set()
    for i in range(N):
        rim_edges.add((i, (i + 1) % N))
        rim_edges.add(((i + 1) % N, i))
        
    all_non_rim_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in rim_edges:
                all_non_rim_edges.append((u, v))
                
    # Universe of lengths to cover: 3 to 39
    # Length 40 is covered by the rim
    target_lengths = set(range(3, N))
    
    # Precompute coverage for each non-rim edge
    # Edge (u, v) creates cycles of length d+1 and N-d+1 where d = (v-u) % N
    # Note: if d=1, it's a rim edge. We excluded those.
    # d ranges from 2 to N-2 (since d=1 and d=N-1 are rim edges)
    edge_cover = {}
    for u, v in all_non_rim_edges:
        d = (v - u) % N
        if d == 0: continue
        l1 = d + 1
        l2 = N - d + 1
        covers = set()
        if l1 in target_lengths:
            covers.add(l1)
        if l2 in target_lengths:
            covers.add(l2)
        edge_cover[(u, v)] = covers
        
    # Filter out edges that don't cover anything useful (though all d>=2 should cover something)
    # d=2 -> 3, 39
    # d=38 -> 39, 3
    # All non-rim edges cover at least one length in 3..39?
    # Max d is 19 (for N=40, d and N-d are symmetric).
    # d=19 -> 20, 21.
    # Yes, all non-rim edges cover at least one target length.

    best_solution = None
    best_edge_count = float('inf')
    
    random.seed(42)
    
    # Run multiple greedy attempts with different randomizations
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        covered = set()
        selected_edges = []
        remaining = set(target_lengths)
        
        # Shuffle edges to break ties randomly
        shuffled_edges = all_non_rim_edges[:]
        random.shuffle(shuffled_edges)
        
        while remaining:
            best_edge = None
            best_new_cover = 0
            
            for e in shuffled_edges:
                if e in [se for se in selected_edges]:
                    continue
                cov = edge_cover[e]
                new_cover = len(cov & remaining)
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_edge = e
                    if best_new_cover == len(remaining):
                        # Perfect coverage found for this step
                        break
            
            if best_edge is None or best_new_cover == 0:
                # Cannot cover remaining lengths with single chords?
                # This shouldn't happen for N=40 with this model, 
                # but if it does, we fail this attempt.
                break
                
            selected_edges.append(best_edge)
            covered.update(edge_cover[best_edge])
            remaining -= covered
            
        if not remaining:
            # We have a set of chords that covers all lengths 3..39
            # Total edges = N (rim) + len(selected_edges)
            total_edges = N + len(selected_edges)
            
            if total_edges < best_edge_count:
                best_edge_count = total_edges
                best_solution = selected_edges[:]
                
                # Try to remove redundant edges
                # If we remove an edge, do we still cover all lengths?
                # Since coverage is union of sets, if a length is covered by >1 edge, we might remove one.
                # Let's do a quick reduction pass
                reduced_edges = best_solution[:]
                i = 0
                while i < len(reduced_edges):
                    # Check if reduced_edges[i] is redundant
                    test_edges = reduced_edges[:i] + reduced_edges[i+1:]
                    test_covered = set()
                    for e in test_edges:
                        test_covered.update(edge_cover[e])
                    if target_lengths.issubset(test_covered):
                        # Redundant, remove it
                        reduced_edges.pop(i)
                    else:
                        i += 1
                best_solution = reduced_edges
                best_edge_count = N + len(best_solution)
                
                # Early exit if we hit a theoretical lower bound?
                # Each edge covers at most 2 lengths. 37 lengths / 2 = 18.5 -> 19 edges.
                # If we get close to 19, we can stop.
                if best_edge_count <= N + 20:
                    break

    if best_solution is None:
        # Fallback: Fan graph (vertex 0 connected to all others)
        # Edges: (0, i) for i=1..39
        # This covers lengths:
        # (0, k) covers k+1 and 40-k+1
        # k=1 -> 2, 40 (rim, 40)
        # k=2 -> 3, 39
        # k=3 -> 4, 38
        # ...
        # k=19 -> 20, 21
        # k=20 -> 21, 20
        # So fan graph covers all lengths.
        # Edges: 39 chords + 40 rim? No, fan graph is usually star + path.
        # If we use rim + chords from 0:
        # Chords (0, k) for k=2..38 (19 chords) covers all 3..39.
        # Total edges = 40 (rim) + 19 (chords) = 59.
        # Score = 19.
        # This is a safe fallback.
        chords = [(0, k) for k in range(2, N-1)]
        best_solution = chords
        best_edge_count = N + len(chords)

    # Construct final output
    edges = []
    # Rim
    for i in range(N):
        edges.append((i, (i + 1) % N))
    # Chords
    for u, v in best_solution:
        edges.append((u, v))
        
    # Ensure edges are unique and sorted for consistency? 
    # The task says "edges are simple and distinct".
    # We should ensure no duplicates.
    unique_edges = set()
    for e in edges:
        a, b = e
        if a > b:
            a, b = b, a
        unique_edges.add((a, b))
    edges = list(unique_edges)
    
    # Generate cycles
    # We need to provide *one* cycle of every length 3..40.
    # We know the chords we picked cover the lengths directly.
    # We need to map each length L to a specific cycle in the graph.
    
    # Build a map from length to a witness cycle
    length_to_cycle = {}
    length_to_cycle[N] = list(range(N)) # Rim
    
    # For each chord, add the cycles it directly creates
    for u, v in best_solution:
        d = (v - u) % N
        if d == 0: continue
        l1 = d + 1
        l2 = N - d + 1
        
        # Cycle for l1: path from u to v along rim + chord (u,v)
        # Path: u, u+1, ..., v
        if 3 <= l1 <= N and l1 not in length_to_cycle:
            cycle = list(range(u, v+1))
            if v < u: # Wrap around
                cycle = list(range(u, N)) + list(range(0, v+1))
            length_to_cycle[l1] = cycle
            
        # Cycle for l2: path from v to u along rim + chord (u,v)
        # Path: v, v+1, ..., u
        if 3 <= l2 <= N and l2 not in length_to_cycle:
            cycle = list(range(v, u+1))
            if u < v: # Wrap around
                cycle = list(range(v, N)) + list(range(0, u+1))
            length_to_cycle[l2] = cycle
            
    # Check if all lengths are covered
    for L in range(3, N+1):
        if L not in length_to_cycle:
            # This means our greedy set cover didn't actually cover L?
            # Or the cycle generation failed?
            # If so, we need to find a cycle or fallback.
            # For robustness, if a length is missing, we might need to add more chords or use a different strategy.
            # However, the greedy set cover *guaranteed* coverage of the length by the chord's direct cycle.
            # So this block should not execute if the logic is correct.
            pass
            
    # Final verification and cycle list construction
    cycles = []
    for L in range(3, N+1):
        if L in length_to_cycle:
            cycles.append(length_to_cycle[L])
        else:
            # Fallback: This shouldn't happen. If it does, return a valid but potentially suboptimal graph?
            # Or raise an error? The task requires a valid result.
            # We will try to find a cycle in the graph using BFS/DFS for this specific length.
            # This is expensive.
            # Given the Set Cover logic, this is highly unlikely.
            # We'll just append an empty list or raise.
            # Let's assume it's covered.
            cycles.append([]) 

    return {'edges': edges, 'cycles': cycles}

# To ensure the fallback is actually valid if the greedy fails to cover everything (which it shouldn't),
# we can add a check. But the greedy covers 3..39.
# The rim covers 40.
# So all 3..40 are covered.

# One optimization: The greedy loop in Python can be slow.
# We can pre-sort or use a heap.
# But N=40 is small. Number of non-rim edges is ~780.
# 500 attempts * 780 edges * small constant = fast.

if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges']) - 40}")
    # Verify
    N = 40
    edges_set = set()
    for e in res['edges']:
        edges_set.add(tuple(sorted(e)))
    
    def has_cycle_of_length(L):
        # Simple check: is the cycle in res['cycles'] valid?
        c = res['cycles'][L-3]
        if len(c) != L: return False
        if len(set(c)) != L: return False
        for i in range(L):
            u = c[i]
            v = c[(i+1)%L]
            if (u, v) not in edges_set and (v, u) not in edges_set:
                return False
        return True
        
    for L in range(3, N+1):
        if not has_cycle_of_length(L):
            print(f"Failed for length {L}")
            break
    else:
        print("All cycles valid")
