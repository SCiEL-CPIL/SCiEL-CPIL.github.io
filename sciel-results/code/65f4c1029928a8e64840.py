import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin for evaluator overhead
    
    def check_valid(n, edges):
        """Check if graph has no K5 and no independent set of size 5.
        Returns (is_valid, violation_type, violation_vertices)."""
        # Build adjacency set for fast lookup
        edge_set = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for K5
        for combo in combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if (min(u,v), max(u,v)) not in edge_set:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return (False, "clique", combo)
        
        # Check for independent set of size 5
        for combo in combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if (min(u,v), max(u,v)) in edge_set:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return (False, "independent", combo)
        
        return (True, None, None)
    
    def find_violation(n, adj, edge_set):
        """Find a K5 or independent 5-set. Returns (type, vertices) or None."""
        # Check for K5
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] not in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return ("clique", combo)
        
        # Check for independent set
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return ("independent", combo)
        
        return None
    
    def build_adj(n, edges):
        """Build adjacency sets and edge set from edge list."""
        edge_set = set()
        adj = [set() for _ in range(n)]
        for u, v in edges:
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return edge_set, adj
    
    def clebsch_complement():
        """Generate the complement of the Clebsch graph (16 vertices, 40 edges).
        The Clebsch graph is the folded 5-cube / Hoffman-Singleton related graph.
        Its complement has no K5 and no independent 5-set (both would be K5 in Clebsch).
        The Clebsch graph is (16, 5, 2)-strongly regular, so its complement is (16, 10, 6)-strongly regular.
        """
        # Clebsch graph: vertices are even-weight binary strings of length 5 (16 vertices)
        # Two vertices adjacent if Hamming distance is 2
        # Label vertices 0..15 by their 5-bit representation with even parity
        
        # Generate all 5-bit even-parity strings
        even_strings = []
        for x in range(32):
            if bin(x).count('1') % 2 == 0:
                even_strings.append(x)
        
        # Map each even string to an index 0..15
        idx_map = {s: i for i, s in enumerate(even_strings)}
        
        # Build Clebsch edges: adjacent if Hamming distance == 2
        clebsch_edges = set()
        for i in range(16):
            for j in range(i+1, 16):
                a = even_strings[i]
                b = even_strings[j]
                ham = bin(a ^ b).count('1')
                if ham == 2:
                    clebsch_edges.add((i, j))
        
        # Complement: all pairs minus clebsch edges
        all_pairs = set()
        for i in range(16):
            for j in range(i+1, 16):
                all_pairs.add((i, j))
        
        comp_edges = list(all_pairs - clebsch_edges)
        return comp_edges
    
    # Start with Clebsch complement (16 vertices)
    base_edges = clebsch_complement()
    base_n = 16
    valid, vtype, vset = check_valid(base_n, base_edges)
    
    if not valid:
        # Fallback: try Paley 37
        def paley_graph(n):
            if n < 2:
                return []
            def is_prime(x):
                if x < 2:
                    return False
                if x == 2:
                    return True
                if x % 2 == 0:
                    return False
                for i in range(3, int(x**0.5) + 1, 2):
                    if x % i == 0:
                        return False
                return True
            if not is_prime(n) or n % 4 != 1:
                return []
            residues = set()
            for x in range(1, n):
                residues.add((x * x) % n)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i) % n in residues:
                        edges.append((i, j))
            return edges
        
        best_n = 37
        best_edges = paley_graph(37)
        return best_n, best_edges
    
    # Now try to expand from 16 vertices toward 42 using vertex duplication + edge flipping
    best_n = base_n
    best_edges = list(base_edges)
    
    current_n = base_n
    current_edges = list(base_edges)
    edge_set, adj = build_adj(current_n, current_edges)
    
    # Target: get as close to 42 as possible
    target_max = 42
    
    while current_n < target_max and time.time() < deadline - 15:
        # Try to add one vertex via duplication heuristic
        found = False
        max_trials = 30
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            # Pick a random existing vertex to "duplicate"
            base_vertex = random.randint(0, current_n - 1)
            new_idx = current_n
            
            # New vertex connects to:
            # - neighbors of base_vertex (with some probability)
            # - non-neighbors of base_vertex (with some probability)
            new_edges = list(current_edges)
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            
            for i in range(current_n):
                if i == base_vertex:
                    # Connect to the base vertex itself (clone)
                    new_edges.append((i, new_idx))
                    new_adj[i].add(new_idx)
                    new_adj[new_idx].add(i)
                    continue
                
                if i in adj[base_vertex]:
                    # i is neighbor of base_vertex
                    # Clone should connect to neighbors of base with prob ~0.6
                    if random.random() < 0.6:
                        new_edges.append((i, new_idx))
                        new_adj[i].add(new_idx)
                        new_adj[new_idx].add(i)
                else:
                    # i is non-neighbor of base_vertex
                    # Clone should connect to non-neighbors with prob ~0.3
                    if random.random() < 0.3:
                        new_edges.append((i, new_idx))
                        new_adj[i].add(new_idx)
                        new_adj[new_idx].add(i)
            
            # Check validity
            valid, vtype, vset = check_valid(current_n + 1, new_edges)
            if valid:
                current_n += 1
                current_edges = new_edges
                edge_set, adj = build_adj(current_n, current_edges)
                found = True
                if current_n > best_n:
                    best_n = current_n
                    best_edges = list(current_edges)
                break
        
        if not found:
            # Try edge-flipping local search on current graph to create room
            # Then retry adding a vertex
            flipped = False
            for fix_attempt in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Find a violation in current graph (shouldn't have one, but check)
                violation = find_violation(current_n, adj, edge_set)
                if violation is not None:
                    vtype, vset = violation
                    if vtype == "clique":
                        # Remove a random edge from the clique
                        pairs = list(combinations(vset, 2))
                        random.shuffle(pairs)
                        u, v = pairs[0]
                        u, v = min(u,v), max(u,v)
                        if (u,v) in edge_set:
                            edge_set.remove((u,v))
                            adj[u].discard(v)
                            adj[v].discard(u)
                            current_edges = list(edge_set)
                            flipped = True
                            break
                    else:
                        # Add a random non-edge from the independent set
                        pairs = list(combinations(vset, 2))
                        random.shuffle(pairs)
                        u, v = pairs[0]
                        u, v = min(u,v), max(u,v)
                        if (u,v) not in edge_set:
                            edge_set.add((u,v))
                            adj[u].add(v)
                            adj[v].add(u)
                            current_edges = list(edge_set)
                            flipped = True
                            break
                else:
                    # No violation; try a random edge flip to diversify
                    all_pairs = [(i, j) for i in range(current_n) for j in range(i+1, current_n)]
                    random.shuffle(all_pairs)
                    for u, v in all_pairs[:5]:
                        if (u, v) in edge_set:
                            # Flip: remove edge
                            edge_set.remove((u, v))
                            adj[u].discard(v)
                            adj[v].discard(u)
                            current_edges = list(edge_set)
                            # Check if still valid
                            valid, _, _ = check_valid(current_n, current_edges)
                            if valid:
                                flipped = True
                            else:
                                # Revert
                                edge_set.add((u, v))
                                adj[u].add(v)
                                adj[v].add(u)
                                current_edges = list(edge_set)
                            break
                        else:
                            # Flip: add edge
                            edge_set.add((u, v))
                            adj[u].add(v)
                            adj[v].add(u)
                            current_edges = list(edge_set)
                            valid, _, _ = check_valid(current_n, current_edges)
                            if valid:
                                flipped = True
                            else:
                                # Revert
                                edge_set.remove((u, v))
                                adj[u].discard(v)
                                adj[v].discard(u)
                                current_edges = list(edge_set)
                            break
            
            if not flipped:
                break
    
    # Final verification
    valid, _, _ = check_valid(best_n, best_edges)
    if not valid:
        # Return the base graph
        return base_n, base_edges
    
    return best_n, best_edges
