import random
import itertools

def construct():
    n = 18
    
    # H is the complement of G.
    # G is K4-free <=> alpha(H) <= 3.
    # Score is max s such that exists S, |S|=s, alpha(H[S]) <= 2.
    # We want to minimize this score.
    
    def get_adj_mask(H_edges):
        masks = [0] * n
        for u, v in H_edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    def is_independent_set(H_masks, subset):
        for i, u in enumerate(subset):
            for v in subset[i+1:]:
                if H_masks[u] & (1 << v):
                    return False
        return True

    def alpha_le_3(H_masks):
        verts = range(n)
        for subset in itertools.combinations(verts, 4):
            if is_independent_set(H_masks, subset):
                return False
        return True

    def has_independent_set_size_3(H_masks, S):
        for u, v, w in itertools.combinations(S, 3):
            if is_independent_set(H_masks, (u, v, w)):
                return True
        return False

    def calc_score_exact(H_masks):
        verts = list(range(n))
        for s in range(n, 0, -1):
            for S in itertools.combinations(verts, s):
                if not has_independent_set_size_3(H_masks, S):
                    return s
        return 0

    def heuristic_score(H_masks):
        # Greedy construction of a large set S with alpha(H[S]) <= 2
        # This is an upper bound on the true score? 
        # No, it finds *a* set. The true score is the *max* such set.
        # So heuristic_score <= true_score.
        # We want to minimize true_score.
        # If heuristic is low, true score is likely low.
        
        # Try to build S greedily
        S = []
        remaining = list(range(n))
        # Sort remaining by degree in H? Or random?
        # Let's try random order
        rng = random.Random(42)
        rng.shuffle(remaining)
        
        for v in remaining:
            # Check if adding v to S keeps alpha(H[S+{v}]) <= 2
            # This means no independent set of size 3 in H[S+{v}]
            # An independent set of size 3 in H[S+{v}] must contain v?
            # If it doesn't contain v, it's in H[S], which we know has no independent set of size 3.
            # So we only need to check independent sets of size 3 containing v.
            # {v, x, y} is independent in H[S+{v}] iff:
            # 1. v is non-adjacent to x in H
            # 2. v is non-adjacent to y in H
            # 3. x is non-adjacent to y in H
            # 4. x, y in S
            
            # Let N_H(v) be neighbors of v in H.
            # We need x, y in S such that x, y not in N_H(v) and x not in N_H(y).
            # i.e., x, y in S \ N_H(v) and x, y non-adjacent in H.
            
            candidates = [u for u in S if not (H_masks[v] & (1 << u))]
            
            # Check if there is an independent pair in candidates in H
            valid = True
            for i, x in enumerate(candidates):
                for y in candidates[i+1:]:
                    if not (H_masks[x] & (1 << y)):
                        # {v, x, y} is independent
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(v)
                
        return len(S)

    # Initialize H
    rng = random.Random(123)
    H_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.7:
                H_edges.add((u, v))
                
    H_masks = get_adj_mask(H_edges)
    while not alpha_le_3(H_masks):
        found = False
        for subset in itertools.combinations(range(n), 4):
            if is_independent_set(H_masks, subset):
                u, v = rng.sample(list(subset), 2)
                if u > v: u, v = v, u
                H_edges.add((u, v))
                H_masks = get_adj_mask(H_edges)
                found = True
                break
        if not found:
            break

    current_score = heuristic_score(H_masks)
    best_score = current_score
    best_H_edges = set(H_edges)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    steps = 10000
    
    for step in range(steps):
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        new_H_edges = set(H_edges)
        if (u, v) in new_H_edges:
            new_H_edges.remove((u, v))
        else:
            new_H_edges.add((u, v))
            
        new_H_masks = get_adj_mask(new_H_edges)
        
        if not alpha_le_3(new_H_masks):
            T *= cooling_rate
            continue
            
        new_score = heuristic_score(new_H_masks)
        
        if new_score < current_score:
            H_edges = new_H_edges
            H_masks = new_H_masks
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_H_edges = set(H_edges)
        else:
            delta = new_score - current_score
            if rng.random() < (2.718281828 ** (-delta / T)):
                H_edges = new_H_edges
                H_masks = new_H_masks
                current_score = new_score
                
        T *= cooling_rate
        
    # Verify best with exact score
    best_H_masks = get_adj_mask(best_H_edges)
    exact_score = calc_score_exact(best_H_masks)
    
    # Convert to G
    G_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in best_H_edges:
                G_edges.append([u, v])
                
    return G_edges
