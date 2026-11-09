import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency and edge set
    init_adj = [set() for _ in range(n)]
    init_edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        init_edge_set.add((u, v))
        init_adj[u].add(v)
        init_adj[v].add(u)

    def expand_biclique(L, R, adj):
        """Expand L and R to a maximal biclique in the given adjacency."""
        changed = True
        while changed:
            changed = False
            new_L = set()
            for x in L:
                if R.issubset(adj[x]):
                    new_L.add(x)
            if len(new_L) < len(L):
                L = new_L
                changed = True
            new_R = set()
            for y in R:
                if L.issubset(adj[y]):
                    new_R.add(y)
            if len(new_R) < len(R):
                R = new_R
                changed = True
        return L, R

    def get_biclique_edges(L, R, adj):
        """Get the set of edges covered by biclique (L, R) that exist in adj."""
        edge_set = set()
        for x in L:
            for y in R:
                if y in adj[x]:
                    edge_set.add((min(x, y), max(x, y)))
        return edge_set

    def generate_candidates(remaining, adj, max_candidates=200):
        """Generate candidate bicliques from remaining edges."""
        candidates = []
        seen = set()
        edge_list = list(remaining)
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if len(candidates) >= max_candidates:
                break
            
            # Try both orientations
            for L_init, R_init in [({u}, {v}), ({v}, {u})]:
                L, R = expand_biclique(set(L_init), set(R_init), adj)
                if len(L) == 0 or len(R) == 0:
                    continue
                # Ensure all cross-edges exist
                valid = True
                for x in L:
                    for y in R:
                        if y not in adj[x]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                e_set = get_biclique_edges(L, R, adj)
                # Filter to only remaining edges
                e_set_remaining = e_set & remaining
                if not e_set_remaining:
                    continue
                
                # Use a canonical key to avoid duplicates
                key = (frozenset(L), frozenset(R))
                key2 = (frozenset(R), frozenset(L))
                if key in seen or key2 in seen:
                    continue
                seen.add(key)
                seen.add(key2)
                
                candidates.append((list(L), list(R), e_set_remaining))
        
        return candidates

    def set_cover_greedy(remaining, adj, overlap_penalty=0.3):
        """Set-cover greedy with overlap penalty."""
        bicliques = []
        remaining_edges = set(remaining)
        selected_edge_sets = []
        
        while remaining_edges:
            candidates = generate_candidates(remaining_edges, adj, max_candidates=150)
            
            if not candidates:
                # Fallback: pick individual edges
                for u, v in list(remaining_edges):
                    bicliques.append([[u], [v]])
                    remaining_edges.discard((u, v))
                break
            
            # Score each candidate: new edges covered - penalty * overlap with selected
            best_score = -1
            best_cand = None
            
            for L, R, e_set in candidates:
                new_edges = len(e_set - set().union(*selected_edge_sets) if selected_edge_sets else e_set)
                total_selected = len(set().union(*selected_edge_sets)) if selected_edge_sets else 0
                overlap = 0
                if selected_edge_sets:
                    union_sel = set().union(*selected_edge_sets)
                    overlap = len(e_set & union_sel)
                
                score = new_edges - overlap_penalty * overlap
                if score > best_score:
                    best_score = score
                    best_cand = (L, R, e_set)
            
            if best_cand is None:
                # Fallback: pick individual edges
                for u, v in list(remaining_edges):
                    bicliques.append([[u], [v]])
                    remaining_edges.discard((u, v))
                break
            
            L, R, e_set = best_cand
            bicliques.append([L, R])
            selected_edge_sets.append(e_set)
            
            # Remove covered edges
            for e in e_set:
                remaining_edges.discard(e)
                u, v = e
                if v in adj[u]:
                    adj[u].discard(v)
                if u in adj[v]:
                    adj[v].discard(u)
        
        return bicliques

    def solve_once():
        adj = [set(s) for s in init_adj]
        return set_cover_greedy(init_edge_set, adj, overlap_penalty=0.2 + 0.1 * random.random())

    # Run multiple restarts with different random seeds
    best_sol = None
    best_count = float('inf')
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 90s limit
    
    attempts = 0
    while time.time() < deadline:
        sol = solve_once()
        attempts += 1
        if len(sol) < best_count:
            best_count = len(sol)
            best_sol = sol
        
        # Early exit if optimal (1 biclique is unlikely but check)
        if best_count <= 2:
            break
    
    if best_sol is None:
        # Fallback: one biclique per edge
        bicliques = []
        for u, v in P['edges']:
            bicliques.append([[u], [v]])
        return bicliques
        
    return best_sol
