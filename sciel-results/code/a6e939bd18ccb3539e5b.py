import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    n_max = 180
    all_elements = list(range(1, n_max + 1))

    def _build_edges(n_max=180):
        """Build all edges (triples) a,b,c distinct in [1,n_max] with 1/a=1/b+1/c."""
        edges = []
        # a < b, a < c. b and c can be in any order relative to each other?
        # 1/a = 1/b + 1/c.
        # c = ab / (b-a).
        # For c to be integer, b-a must divide ab.
        # Let d = b-a. b = a+d.
        # c = a(a+d)/d = a + a^2/d.
        # d must divide a^2.
        # d > 0.
        # b = a+d.
        # c = a + a^2/d.
        # Distinctness:
        # a != b since d>0.
        # a != c since a^2/d > 0.
        # b != c => a+d != a + a^2/d => d != a^2/d => d^2 != a^2 => d != a.
        
        for a in range(1, n_max + 1):
            # Find divisors of a^2
            a_sq = a * a
            for d in range(1, int(a_sq**0.5) + 1):
                if a_sq % d == 0:
                    candidates = [d]
                    other_d = a_sq // d
                    if other_d != d:
                        candidates.append(other_d)
                    
                    for d_val in candidates:
                        if d_val == a:
                            continue
                        b = a + d_val
                        c = a + a_sq // d_val
                        if b <= n_max and c <= n_max:
                            # The set is {a, b, c}
                            # Sort to make it a canonical edge? No, just store the set.
                            edge = (min(a,b,c), min(max(a,b), max(a,c)), max(a,b,c))
                            edges.append(edge)
        # Remove duplicates
        unique_edges = list(set(edges))
        return unique_edges

    edges = _build_edges(n_max)
    m = len(edges)
    
    # Precompute vertex to edges mapping
    vertex_to_edges = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(edges):
        vertex_to_edges[a].append(idx)
        vertex_to_edges[b].append(idx)
        vertex_to_edges[c].append(idx)

    def check_conflict_fast(x, S):
        """Check if adding x to set S creates any forbidden triple using precomputed edges."""
        for e_idx in vertex_to_edges[x]:
            a, b, c = edges[e_idx]
            others = [v for v in (a, b, c) if v != x]
            if others[0] in S and others[1] in S:
                return True
        return False

    best_set = set(all_elements) # Fallback
    best_size = 0
    
    num_runs = 2000 # Adjust based on time
    
    for run in range(num_runs):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        S = set(all_elements)
        removed = []
        
        # Counts
        edge_incount = [3] * m
        vertex_conflict = {i: 0 for i in range(1, n_max + 1)}
        
        # Initialize vertex_conflict
        for idx in range(m):
            a, b, c = edges[idx]
            vertex_conflict[a] += 1
            vertex_conflict[b] += 1
            vertex_conflict[c] += 1
        
        total_conflicts = m
        
        while total_conflicts > 0:
            # Find max conflict vertex
            max_conf = 0
            candidates = []
            for v in S:
                if vertex_conflict[v] > max_conf:
                    max_conf = vertex_conflict[v]
                    candidates = [v]
                elif vertex_conflict[v] == max_conf:
                    candidates.append(v)
            
            if not candidates or max_conf == 0:
                break
                
            v = random.choice(candidates)
            S.remove(v)
            removed.append(v)
            
            # Update
            for e_idx in vertex_to_edges[v]:
                edge_incount[e_idx] -= 1
                if edge_incount[e_idx] == 2:
                    # This edge was a conflict, now it is not
                    total_conflicts -= 1
                    a, b, c = edges[e_idx]
                    # Decrease conflict count for other vertices
                    for u in (a, b, c):
                        if u != v:
                            vertex_conflict[u] -= 1
        
        # Repair Phase 1: Add back removed vertices in reverse order
        for v in reversed(removed):
            if v in S:
                continue
            if not check_conflict_fast(v, S):
                S.add(v)
        
        # Repair Phase 2: Local search swaps (remove one, add two)
        # Try a few random swaps to escape local optima
        for _ in range(5):
            if time.time() > deadline:
                break
            if not S:
                break
            # Pick a random vertex to remove
            v_remove = random.choice(list(S))
            S.remove(v_remove)
            
            added = []
            # Try to add vertices that were removed or not in S
            candidates_add = [v for v in removed if v not in S]
            random.shuffle(candidates_add)
            
            for v_add in candidates_add:
                if v_add in S:
                    continue
                if not check_conflict_fast(v_add, S):
                    S.add(v_add)
                    added.append(v_add)
                if len(added) >= 2:
                    break
            
            if len(added) > 1:
                # Keep the change
                pass
            else:
                # Revert
                for v in added:
                    S.discard(v)
                S.add(v_remove)

        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    if not best_set:
        best_set = set(range(91, 181))
        
    return sorted(best_set)
