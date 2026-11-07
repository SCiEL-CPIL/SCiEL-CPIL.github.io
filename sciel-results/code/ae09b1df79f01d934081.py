import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Precompute: for each K-vertex, which I-vertices are adjacent
        adj_k_to_i = {u: [] for u in range(k)}
        for v in range(k, n):
            for u in neighbors[v]:
                adj_k_to_i[u].append(v)
        
        best_cliques = None
        best_count = float('inf')
        
        # Strategy: assign each I-vertex a subset of K-vertices to form a clique
        # The savings for assigning a set S to v is |S| (if |S| >= 2) since
        # instead of |S| separate edges (each as a size-2 clique), we have 1 clique
        # But we must ensure that all pairs in S are edges in K (i.e., S is a clique in K)
        
        # Since K is a clique (all pairs connected), any subset of K is a clique
        # So we just need to partition K-edges among I-vertices
        # Each I-vertex v can "absorb" a set S of K-vertices where all u in S are adjacent to v
        # This forms clique S ∪ {v}
        
        # Objective: minimize total cliques = (number of big cliques) + (number of residual size-2 cliques)
        # = sum over v of [1 if S_v nonempty else 0] + (total K-edges - sum over v of C(|S_v|, 2)) + (I-K edges not in any S_v)
        # Actually: total cliques = (# big cliques) + (# remaining K-edges as size-2) + (# remaining I-K edges as size-2)
        
        # Let's think differently:
        # Total edges = C(k,2) + sum_v |N(v)|
        # Each big clique S_v ∪ {v} covers C(|S_v|, 2) K-edges + |S_v| I-K edges in 1 clique
        # Without it, those would be C(|S_v|,2) + |S_v| separate size-2 cliques
        # Savings per big clique = C(|S_v|, 2) + |S_v| - 1
        
        # We want to maximize total savings subject to:
        # - Each K-vertex can be in at most one big clique (since K-edges must be covered exactly once)
        # - Each I-vertex can be in at most one big clique
        # - S_v ⊆ N(v)
        
        # This is a complex combinatorial optimization. Use greedy with multiple restarts.
        
        rng_local = random.Random(seed + 1000)
        
        for attempt in range(200):
            if time.time() > deadline - 1.0:
                break
            
            # Track which K-vertices are used
            k_used = [False] * k
            # Track which I-vertices are used  
            i_used = [False] * (n - k)
            # Store the assignment: for each I-vertex, the set of K-vertices
            assignment = {v: set() for v in range(k, n)}
            
            # Order of processing I-vertices: vary this across attempts
            i_list = list(range(k, n))
            if attempt % 4 == 0:
                # Sort by decreasing degree
                i_list.sort(key=lambda v: -len(neighbors[v]))
            elif attempt % 4 == 1:
                # Sort by increasing degree
                i_list.sort(key=lambda v: len(neighbors[v]))
            elif attempt % 4 == 2:
                # Random shuffle
                rng_local.shuffle(i_list)
            else:
                # Sort by a random key
                i_list.sort(key=lambda v: rng_local.random())
            
            # For each I-vertex, decide how many K-vertices to assign
            for v in i_list:
                if i_used[v - k]:
                    continue
                
                # Available K-vertices adjacent to v and not yet used
                avail = [u for u in neighbors[v] if not k_used[u]]
                
                if len(avail) < 2:
                    continue
                
                # Heuristic: take a subset that maximizes marginal savings
                # Savings of taking m vertices = C(m, 2) + m - 1
                # This is increasing in m, so we want to take as many as possible
                # But we might want to leave some for other I-vertices
                # Use a threshold: take vertices that are "scarcely used" by others
                
                # For each available K-vertex u, count how many other I-vertices want it
                # (i.e., are adjacent to u and haven't been assigned yet)
                scarcity_scores = {}
                for u in avail:
                    # Count other I-vertices that are adjacent to u and could use it
                    count = 0
                    for w in adj_k_to_i[u]:
                        if w != v and not i_used[w - k]:
                            count += 1
                    scarcity_scores[u] = count
                
                # Sort by scarcity (fewest other candidates first)
                avail.sort(key=lambda u: scarcity_scores[u])
                
                # Decide how many to take: greedy, take top few
                # Try different sizes
                best_size = min(3, len(avail))
                for size in range(2, min(6, len(avail) + 1)):
                    # Check if taking this many is beneficial
                    # We'll just take the top few by scarcity
                    pass
                
                # Simple: take the top few by scarcity
                take_count = min(3, len(avail))
                chosen = avail[:take_count]
                
                # Verify: all pairs in chosen must be edges in K (they are, since K is a clique)
                # Mark as used
                assignment[v] = set(chosen)
                i_used[v - k] = True
                for u in chosen:
                    k_used[u] = True
            
            # Build the solution
            cliques = []
            covered_k_edges = set()
            covered_ik_edges = set()
            
            # Add big cliques
            for v in range(k, n):
                if i_used[v - k] and len(assignment[v]) >= 2:
                    clique = sorted(list(assignment[v]) + [v])
                    cliques.append(clique)
                    # Mark covered edges
                    kverts = list(assignment[v])
                    for a, b in combinations(kverts, 2):
                        covered_k_edges.add((min(a, b), max(a, b)))
                        covered_ik_edges.add((min(a, v), max(a, v)))
                        covered_ik_edges.add((min(b, v), max(b, v)))
            
            # Add residual K-edges
            for u in range(k):
                for w in range(u + 1, k):
                    if (u, w) not in covered_k_edges:
                        cliques.append([u, w])
            
            # Add residual I-K edges
            for v in range(k, n):
                for u in neighbors[v]:
                    e = (min(u, v), max(u, v))
                    if e not in covered_ik_edges:
                        cliques.append([u, v])
            
            # Convert to int lists
            clean_cliques = [[int(x) for x in c] for c in cliques]
            
            # Validate
            edge_count = {}
            for c in clean_cliques:
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        u, w = min(c[i], c[j]), max(c[i], c[j])
                        edge_count[(u, w)] = edge_count.get((u, w), 0) + 1
            
            # Check all edges covered exactly once
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
            
            valid = True
            for e in all_edges:
                if edge_count.get(e, 0) != 1:
                    valid = False
                    break
            for e, cnt in edge_count.items():
                if e not in all_edges:
                    valid = False
                    break
            
            if not valid:
                continue
            
            count = len(clean_cliques)
            if count < best_count:
                best_count = count
                best_cliques = clean_cliques
        
        if best_cliques is None:
            # Fallback: all edges as size-2 cliques
            cliques = []
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([min(u, v), max(u, v)])
            best_cliques = [[int(x) for x in c] for c in cliques]
        
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 1.0:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    
    while len(results) < 3:
        results.append([])
    
    return results
