import time
import random
from itertools import combinations

def _precompute():
    n = 16
    all_triples = list(combinations(range(n), 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    k222_list = []
    triple_to_k = [[] for _ in range(n_triples)]
    pairs_in_k = {} # key: (min_idx, max_idx), value: list of k_indices

    # Generate all K(2,2,2)
    for points in combinations(range(n), 6):
        pts = list(points)
        # Generate matchings
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m

        for matching in gen_matchings(pts):
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(triple_to_idx[t])
            
            k_idx = len(k222_list)
            k222_list.append(k_triples)
            
            for t_idx in k_triples:
                triple_to_k[t_idx].append(k_idx)
            
            # Add pairs
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = k_triples[i], k_triples[j]
                    if u > v: u, v = v, u
                    if (u, v) not in pairs_in_k:
                        pairs_in_k[(u, v)] = []
                    pairs_in_k[(u, v)].append(k_idx)
    
    return all_triples, triple_to_k, k222_list, pairs_in_k

def construct():
    start_time = time.time()
    deadline = 120.0
    
    all_triples, triple_to_k, k222_list, pairs_in_k = _precompute()
    n_triples = len(all_triples)
    n_k = len(k222_list)
    
    best_size = 0
    best_set = set()
    
    # Initial greedy seed to get a good starting point
    random.seed(42)
    order = list(range(n_triples))
    random.shuffle(order)
    initial_S = set()
    k_counts = [0] * n_k
    for t in order:
        valid = True
        for k in triple_to_k[t]:
            if k_counts[k] >= 7:
                valid = False
                break
        if valid:
            initial_S.add(t)
            for k in triple_to_k[t]:
                k_counts[k] += 1
    
    best_size = len(initial_S)
    best_set = set(initial_S)
    
    # Branch and Bound
    # State: S (set), C (set of candidates), k_counts (list)
    # We use recursion
    
    def compatible(u, v, k_counts):
        key = (u, v) if u < v else (v, u)
        if key not in pairs_in_k:
            return True
        for k in pairs_in_k[key]:
            if k_counts[k] == 6:
                return False
        return True

    def get_coloring(candidates, k_counts):
        # Greedy coloring of the compatibility graph H on 'candidates'
        # H edge (u,v) if compatible(u,v)
        # We want to color H.
        # Order candidates by degree in H (descending) to get tighter bound?
        # Or just arbitrary? Tomita uses specific ordering.
        # Let's try sorting by degree in H.
        
        # Calculate degrees
        degs = {t: 0 for t in candidates}
        for i, u in enumerate(candidates):
            for v in candidates[i+1:]:
                if compatible(u, v, k_counts):
                    degs[u] += 1
                    degs[v] += 1
        
        # Sort by degree desc
        sorted_cands = sorted(candidates, key=lambda x: degs[x], reverse=True)
        
        colors = {}
        next_color = 0
        for u in sorted_cands:
            # Find smallest color not used by neighbors in H
            # Neighbors in H are compatible vertices
            used_colors = set()
            for v in sorted_cands:
                if v == u: continue
                if u in colors and v in colors:
                    if compatible(u, v, k_counts):
                        used_colors.add(colors[v])
                # Note: We are coloring sequentially. Only previously colored vertices matter.
            
            c = 0
            while c in used_colors:
                c += 1
            colors[u] = c
            next_color = max(next_color, c + 1)
            
        return sorted_cands, colors, next_color

    def search(S, C, k_counts, depth=0):
        nonlocal best_size, best_set
        
        if time.time() - start_time > deadline - 5.0:
            return
            
        if not C:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return
            
        # Check validity of C
        C_valid = set()
        for t in C:
            ok = True
            for k in triple_to_k[t]:
                if k_counts[k] >= 7:
                    ok = False
                    break
            if ok:
                C_valid.add(t)
                
        if not C_valid:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return
            
        # Bound
        sorted_cands, colors, max_color = get_coloring(C_valid, k_counts)
        
        if len(S) + max_color <= best_size:
            return
            
        # Pick a vertex
        # Heuristic: Pick the one with highest color? Or highest degree?
        # In MCQ, we pick the last one in the ordering (highest degree/color).
        u = sorted_cands[-1]
        
        # Branch 1: Include u
        new_S = S | {u}
        new_k_counts = k_counts[:]
        for k in triple_to_k[u]:
            new_k_counts[k] += 1
            
        # New candidates: neighbors of u in H, excluding u
        # Must be compatible with u given new_k_counts
        new_C = set()
        for v in C_valid:
            if v == u: continue
            if compatible(u, v, new_k_counts):
                new_C.add(v)
                
        search(new_S, new_C, new_k_counts, depth + 1)
        
        # Branch 2: Exclude u
        # C_valid without u
        C_valid.discard(u)
        search(S, C_valid, k_counts, depth + 1)

    # Start search
    # To avoid deep recursion issues, we might need to increase recursion limit
    import sys
    sys.setrecursionlimit(10000)
    
    # Initial call
    # We can start with empty S or the greedy S?
    # Starting with greedy S might be better for pruning?
    # But the bound uses |S| + bound(C).
    # If we start with S=greedy, C=all, we might miss solutions that don't contain the greedy set.
    # So we should start with S=empty, C=all.
    
    search(set(), set(range(n_triples)), [0]*n_k)
    
    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
