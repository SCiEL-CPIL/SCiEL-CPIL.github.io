import itertools
import random

def construct():
    n = 18
    
    # H is the complement of the output graph G.
    # G is K4-free <=> alpha(H) <= 3.
    # Score is max s such that exists S, |S|=s, alpha(H[S]) <= 2.
    # We want to minimize this score.
    
    def get_adj_mask(H_edges):
        # H_edges is a set of (u, v)
        masks = [0] * n
        for u, v in H_edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    def is_independent_set(H_masks, subset):
        # Check if subset is an independent set in H
        # i.e., no edges between vertices in subset in H
        for i, u in enumerate(subset):
            for v in subset[i+1:]:
                if H_masks[u] & (1 << v):
                    return False
        return True

    def alpha_le_3(H_masks):
        # Check if alpha(H) <= 3
        # i.e., no independent set of size 4
        verts = range(n)
        for subset in itertools.combinations(verts, 4):
            if is_independent_set(H_masks, subset):
                return False
        return True

    def has_independent_set_size_3(H_masks, S):
        # Check if H[S] has an independent set of size 3
        # S is a tuple/list of vertices
        for u, v, w in itertools.combinations(S, 3):
            if is_independent_set(H_masks, (u, v, w)):
                return True
        return False

    def calc_score_exact(H_masks):
        # Find max s such that exists S of size s with alpha(H[S]) <= 2
        # alpha(H[S]) <= 2 <=> NO independent set of size 3 in H[S]
        verts = list(range(n))
        for s in range(n, 0, -1):
            for S in itertools.combinations(verts, s):
                if not has_independent_set_size_3(H_masks, S):
                    return s
        return 0

    def find_large_bad_subset(H_masks, target_size):
        # Find a subset S of size target_size such that alpha(H[S]) <= 2
        # i.e., NO independent set of size 3 in H[S]
        verts = list(range(n))
        for S in itertools.combinations(verts, target_size):
            if not has_independent_set_size_3(H_masks, S):
                return S
        return None

    # Initialize H
    # Start with a dense graph to ensure alpha(H) <= 3
    # p = 0.8
    rng = random.Random(42)
    H_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.8:
                H_edges.add((u, v))
    
    # Ensure alpha(H) <= 3
    # If not, add edges
    while not alpha_le_3(get_adj_mask(H_edges)):
        # Find an independent set of size 4
        verts = range(n)
        found = False
        for subset in itertools.combinations(verts, 4):
            if is_independent_set(get_adj_mask(H_edges), subset):
                # Add an edge within this subset to break it
                # Pick random pair
                u, v = rng.sample(list(subset), 2)
                if u > v: u, v = v, u
                H_edges.add((u, v))
                found = True
                break
        if not found:
            break # Should not happen

    H_masks = get_adj_mask(H_edges)
    current_score = calc_score_exact(H_masks)
    
    best_score = current_score
    best_H_edges = set(H_edges)
    
    # Local Search
    # We want to minimize score.
    # If score is s, there exists S of size s with alpha(H[S]) <= 2.
    # We want to destroy this property by removing edges in H[S] to increase alpha(H[S]) to 3.
    # But we must maintain alpha(H) <= 3.
    
    for step in range(1000):
        # Find a bad subset of size current_score
        S = find_large_bad_subset(H_masks, current_score)
        if S is None:
            # Score might have dropped? Recalculate
            new_score = calc_score_exact(H_masks)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_H_edges = set(H_edges)
            continue
            
        # Try to remove an edge in H[S] to increase alpha(H[S])
        # Removing edge (u,v) in H means adding edge (u,v) in G.
        # This might create an independent set of size 4 in H.
        
        # Get edges in H[S]
        edges_in_S = []
        for u, v in itertools.combinations(S, 2):
            if (u, v) in H_edges or (v, u) in H_edges:
                edges_in_S.append((min(u,v), max(u,v)))
                
        if not edges_in_S:
            # H[S] is already empty? Then alpha(H[S]) = |S| >= 3.
            # This contradicts alpha(H[S]) <= 2.
            # So this case shouldn't happen if S was found correctly.
            continue

        # Randomly pick an edge to remove
        u, v = rng.choice(edges_in_S)
        
        # Check if removing (u,v) from H creates an independent set of size 4
        # New H' = H \ {(u,v)}
        # Check if alpha(H') <= 3
        # We can do a quick check: 
        # An independent set of size 4 in H' must have been an independent set of size 4 in H 
        # OR it uses the new non-edge (u,v).
        # Since H had no independent set of size 4, any new independent set of size 4 in H' 
        # must contain both u and v (since if it doesn't, it was in H).
        # So we just need to check if there exist x, y such that {u, v, x, y} is independent in H'.
        # Conditions:
        # 1. u, v are non-adjacent in H' (true by construction)
        # 2. x, y are non-adjacent to u and v in H'
        # 3. x, y are non-adjacent to each other in H'
        
        # Let's just run alpha_le_3 on the modified graph for simplicity, 
        # but optimize by only checking sets containing u and v.
        
        new_H_edges = set(H_edges)
        new_H_edges.discard((u, v))
        new_H_edges.discard((v, u))
        
        # Check alpha <= 3
        # Fast check:
        # Find all vertices x that are non-adjacent to u in H and non-adjacent to v in H
        # (Since we only removed (u,v), adjacency to others is same as H)
        # Wait, if we remove (u,v) from H, u and v become non-adjacent in H.
        # So we look for x, y such that:
        # - x, y not in {u, v}
        # - x, y non-adjacent to u in H
        # - x, y non-adjacent to v in H
        # - x, y non-adjacent to each other in H
        # - x, y non-adjacent to u/v in H' (same as H for x,y != u,v)
        
        candidates = []
        for x in range(n):
            if x == u or x == v: continue
            # Check non-adjacent to u in H
            if H_masks[u] & (1 << x): continue
            # Check non-adjacent to v in H
            if H_masks[v] & (1 << x): continue
            candidates.append(x)
            
        # Check if there is an independent pair in candidates in H
        valid = False
        for i, x in enumerate(candidates):
            for y in candidates[i+1:]:
                if not (H_masks[x] & (1 << y)):
                    # {u, v, x, y} is independent in H'
                    valid = True
                    break
            if valid: break
            
        if not valid:
            # Safe to remove edge
            H_edges = new_H_edges
            H_masks = get_adj_mask(H_edges)
            
            # Update score
            new_score = calc_score_exact(H_masks)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_H_edges = set(H_edges)

    # Convert best_H_edges to G edges
    G_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in best_H_edges and (v, u) not in best_H_edges:
                G_edges.append([u, v])
                
    return G_edges
