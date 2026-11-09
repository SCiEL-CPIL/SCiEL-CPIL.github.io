import time
import random

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    # Rim edges
    rim_edges = [(i, i + 1) for i in range(N - 1)]
    rim_edges.append((N - 1, 0))
    
    # Function to calculate covered lengths for a set of chords K
    def get_covered_lengths(K):
        lengths = {N}
        for k in K:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        for i in range(len(K)):
            for j in range(i + 1, len(K)):
                lengths.add(abs(K[i] - K[j]) + 2)
        return lengths
        
    # Check if K covers all required lengths 3..N
    def is_valid(K):
        covered = get_covered_lengths(K)
        for L in range(3, N + 1):
            if L not in covered:
                return False
        return True
        
    # Greedy search for minimal chords
    # We want to find a set K such that is_valid(K) is True, minimizing len(K)
    
    best_K = None
    best_size = N # Upper bound
    
    # Try greedy construction
    # Start with empty set, add chords that cover most new lengths
    K = set()
    uncovered = set(range(3, N + 1))
    
    # Pre-calculate potential chords 1..N-1
    candidates = list(range(1, N))
    
    while uncovered:
        best_gain = -1
        best_k = -1
        for k in candidates:
            if k in K:
                continue
            # Calculate new lengths covered by adding k
            new_lengths = set()
            new_lengths.add(k + 1)
            new_lengths.add(N - k + 1)
            for k2 in K:
                new_lengths.add(abs(k - k2) + 2)
            
            gain = len(new_lengths & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_k = k
                
        if best_k == -1:
            break
            
        K.add(best_k)
        # Update uncovered
        new_lengths = set()
        new_lengths.add(best_k + 1)
        new_lengths.add(N - best_k + 1)
        for k2 in K:
            new_lengths.add(abs(best_k - k2) + 2)
        uncovered -= new_lengths
        
    if is_valid(K):
        best_K = list(K)
        best_size = len(K)
        
    # Try to improve by removing redundant chords
    # If we have a valid K, try removing each chord and see if still valid
    if best_K:
        improved = True
        while improved:
            improved = False
            for k in list(best_K):
                temp_K = [x for x in best_K if x != k]
                if is_valid(temp_K):
                    best_K = temp_K
                    best_size = len(best_K)
                    improved = True
                    break # Restart loop with smaller K
                    
    # If we didn't find a smaller set, use the known 19-chord set
    if best_K is None or len(best_K) >= 19:
        # Use the standard 19 even chords
        K = [2 * k for k in range(1, N // 2)]
    else:
        K = best_K
        
    # Construct edges
    chord_edges = [(0, k) for k in K]
    
    edges = rim_edges + chord_edges
    
    # De-duplicate
    seen = set()
    unique_edges = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append((a, b))
    edges = unique_edges
    
    edge_set = set()
    for a, b in edges:
        edge_set.add((min(a, b), max(a, b)))
        
    # Construct cycles
    # We need to provide explicit cycles.
    # The logic for generating cycles depends on K.
    # For each L in 3..N, we need to find a cycle.
    
    cycles = []
    valid_construction = True
    
    for L in range(3, N + 1):
        cyc = None
        
        if L == N:
            cyc = list(range(N))
        else:
            # Try single chord cycles
            # Length L can be formed by chord k if L = k+1 or L = N-k+1
            # So k = L-1 or k = N-L+1
            found = False
            for k in K:
                if L == k + 1:
                    # Cycle 0-1-...-k-0
                    cyc = list(range(0, k + 1))
                    found = True
                    break
                elif L == N - k + 1:
                    # Cycle 0-N-1-...-k-0
                    # Vertices: 0, N-1, ..., k
                    # Order: 0, N-1, N-2, ..., k
                    cyc = [0] + list(range(N - 1, k - 1, -1))
                    found = True
                    break
            
            if not found:
                # Try difference cycles
                # L = |k1 - k2| + 2 => |k1 - k2| = L - 2
                diff = L - 2
                for i in range(len(K)):
                    for j in range(i + 1, len(K)):
                        if abs(K[i] - K[j]) == diff:
                            k1, k2 = K[i], K[j]
                            if k1 < k2:
                                # Cycle 0-k1-...-k2-0
                                cyc = [0] + list(range(k1, k2 + 1))
                            else:
                                # Cycle 0-k2-...-k1-0
                                cyc = [0] + list(range(k2, k1 + 1))
                            found = True
                            break
                    if found:
                        break
            
        if not found:
            valid_construction = False
            break
            
        # Validate cycle edges
        if len(set(cyc)) != len(cyc):
            valid_construction = False
            break
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                valid_construction = False
                break
                
        cycles.append(cyc)
        
    if not valid_construction:
        # Fallback to seed if something goes wrong
        path_edges = [(i, i + 1) for i in range(N - 1)]
        spoke_edges = [(i, 0) for i in range(2, N)]
        edges = path_edges + spoke_edges
        seen = set()
        unique_edges = []
        for a, b in edges:
            if a == b: continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_edges.append((a, b))
        edges = unique_edges
        cycles = [list(range(k)) for k in range(3, N + 1)]

    return {'edges': edges, 'cycles': cycles}
