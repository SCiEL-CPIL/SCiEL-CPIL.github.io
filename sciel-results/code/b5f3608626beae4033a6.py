import itertools
import random

def construct():
    n = 18
    
    # Helper to check if a graph H (adjacency list or matrix) has alpha <= 3
    # H is represented as a set of edges or a matrix. Let's use a set of tuples for edges.
    # Or better, a list of lists for adjacency.
    
    def has_independent_set_size_k(H_edges, k):
        # H_edges is a set of (u, v)
        # Check if there is an independent set of size k
        # For k=4, just check all subsets of size 4
        if k <= 1: return False
        if k == 2:
            # Independent set of size 2 means a non-edge
            # We can just return True if not complete
            pass
        # We only need to check k=4 for alpha(H) <= 3
        # If no independent set of size 4, alpha <= 3
        # We also need to check for the score later
        
        # For alpha(H) <= 3, we check if there is an independent set of size 4
        # An independent set in H is a set of vertices with no edges between them in H
        # i.e., a clique in G = complement(H)
        
        # Let's use a simpler check:
        # Generate all subsets of size 4
        # For each subset, check if it's independent in H
        # Independent in H means for all u,v in subset, (u,v) not in H_edges
        
        # This is O(C(18,4)) = 3060 checks.
        # Each check is O(1) if we use a set of edges.
        
        verts = range(n)
        for subset in itertools.combinations(verts, 4):
            is_ind = True
            for u, v in itertools.combinations(subset, 2):
                if (u, v) in H_edges or (v, u) in H_edges:
                    is_ind = False
                    break
            if is_ind:
                return True
        return False

    def calc_score(H_edges):
        # Score is max k such that exists S size k with alpha(H[S]) <= 2
        # alpha(H[S]) <= 2 means no independent set of size 3 in H[S]
        # Independent set of size 3 in H[S] means 3 vertices in S with no edges in H
        # i.e., a triangle in G[S]
        
        # We iterate k from n down to 1
        # For each k, check if there exists S of size k such that no 3 vertices in S form an independent set in H
        # i.e., for all triples u,v,w in S, at least one edge exists in H
        
        verts = range(n)
        for k in range(n, 0, -1):
            found = False
            for S in itertools.combinations(verts, k):
                # Check if H[S] has an independent set of size 3
                # i.e., are there u,v,w in S such that no edges between them in H?
                has_ind_3 = False
                for u, v, w in itertools.combinations(S, 3):
                    # Check if u,v,w are independent in H
                    if ((u, v) not in H_edges and (v, u) not in H_edges and
                        (v, w) not in H_edges and (w, v) not in H_edges and
                        (u, w) not in H_edges and (w, u) not in H_edges):
                        has_ind_3 = True
                        break
                if not has_ind_3:
                    found = True
                    break
            if found:
                return k
        return 0

    candidates = []
    
    # Candidate 1: Seed (Complement of K6,6,6)
    # H is 3 disjoint K6s
    H_seed_edges = set()
    for i in range(3):
        start = i * 6
        for u in range(start, start + 6):
            for v in range(u + 1, start + 6):
                H_seed_edges.add((u, v))
    candidates.append(H_seed_edges)
    
    # Generate random candidates
    rng = random.Random(42)
    for _ in range(20):
        p = rng.uniform(0.6, 0.8)
        H_rand_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    H_rand_edges.add((u, v))
        # Check alpha(H) <= 3
        if not has_independent_set_size_k(H_rand_edges, 4):
            candidates.append(H_rand_edges)
            
    best_score = float('inf')
    best_H = None
    
    for H_edges in candidates:
        score = calc_score(H_edges)
        if score < best_score:
            best_score = score
            best_H = H_edges
            
    # Convert best_H to G edges
    # G is complement of H
    G_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in best_H and (v, u) not in best_H:
                G_edges.append([u, v])
                
    return G_edges
