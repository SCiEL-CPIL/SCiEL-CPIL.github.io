import time
import itertools

N = 40

def is_pancyclic(edges_set, n):
    # Check if graph contains cycles of all lengths 3..n
    # Simple DFS/BFS for each length is too slow if repeated.
    # But n=40 is small.
    # We can use a heuristic or simple check.
    # For a fixed set of edges, we can try to find a cycle of length k.
    # This is NP-hard in general, but for sparse graphs with n=40, it's manageable.
    # However, doing this 680 times might be slow.
    
    # Faster check:
    # A graph is pancyclic if it has a cycle of length k for all k.
    # We can just check if the number of distinct cycle lengths found is n-2.
    
    # To speed up, we can precompute adjacency.
    pass

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    # Seed solution
    seed_edges = [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)]
    seed_cycles = [list(range(k)) for k in range(3,N+1)]
    seed_result = {'edges': seed_edges, 'cycles': seed_cycles}
    
    # Try to find a 2-chord solution (N+2 edges)
    # Base: Cycle 0-1-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add(frozenset((i, (i+1)%N)))
    
    # Chords must be non-edges in base
    # Chords (0,i) and (0,j) where i,j != 1, 39 (since (0,1) and (0,39) are in base)
    # Actually (0,1) and (0,39) are in base cycle.
    # So chords must be to i in 2..38.
    
    candidates = list(range(2, N-1)) # 2 to 38
    
    best_result = seed_result
    
    for i, j in itertools.combinations(candidates, 2):
        if time.time() > deadline:
            break
            
        chord1 = frozenset((0, i))
        chord2 = frozenset((0, j))
        
        if chord1 in base_edges or chord2 in base_edges:
            continue
            
        current_edges = base_edges | {chord1, chord2}
        
        # Check if pancyclic
        # We need to verify cycles of length 3..40.
        # This check is expensive.
        # Optimization: Only check if the set of edges allows the specific cycles.
        # For (0,i) and (0,j), the cycles are constrained.
        
        # If we find a valid one, return immediately.
        if check_pancyclic_fast(current_edges, N):
            # Construct the cycle lists
            # We need to output the actual vertex lists.
            # For (0,i) and (0,j), we can generate them.
            cycles = generate_cycles_for_chords(N, i, j)
            if cycles:
                edge_list = [tuple(e) for e in current_edges]
                # Ensure unique and simple
                return {'edges': edge_list, 'cycles': cycles}
                
    return best_result

def check_pancyclic_fast(edges_set, n):
    # Heuristic or exact check for specific structure
    # Given the complexity, we might skip the full check and assume
    # that if we find a known good pair, we use it.
    # But we don't know the good pair.
    
    # For now, return False to be safe, unless we implement a real check.
    # A real check for n=40 is feasible if optimized.
    pass

def generate_cycles_for_chords(n, i, j):
    pass
