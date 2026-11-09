import random
import time
import math

_START = time.time()
_LIMIT = 110.0  # Safety margin

def _compute_alpha_fast(edges):
    """
    Compute independence number of hypergraph with 16 vertices.
    edges: list of ints (bitmasks)
    """
    n = 16
    all_vertices_mask = (1 << n) - 1
    
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in range(n):
            if e & (1 << v):
                vertex_edges[v].append(i)
                
    best = [0]
    
    def is_valid_add(v, S):
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            others = e ^ (1 << v)
            if (others & S) == others:
                return False
        return True

    def get_new_avail(S, v_bit, old_avail):
        new_S = S | v_bit
        new_avail = old_avail ^ v_bit
        final_avail = 0
        temp = new_avail
        while temp:
            u_bit = temp & -temp
            u = u_bit.bit_length() - 1
            temp ^= u_bit
            
            blocked = False
            for e_idx in vertex_edges[u]:
                e = edges[e_idx]
                rest = e ^ u_bit
                if (rest & new_S) == rest:
                    blocked = True
                    break
            
            if not blocked:
                final_avail |= u_bit
                
        return final_avail

    def dfs(S, avail):
        if S.bit_count() + avail.bit_count() <= best[0]:
            return
        
        if avail == 0:
            if S.bit_count() > best[0]:
                best[0] = S.bit_count()
            return
            
        v_bit = avail & -avail
        v = v_bit.bit_length() - 1
        
        dfs(S, avail ^ v_bit)
        
        if is_valid_add(v, S):
            new_avail = get_new_avail(S, v_bit, avail)
            dfs(S | v_bit, new_avail)
            
    dfs(0, all_vertices_mask)
    return best[0]

def _is_valid_triple_set(triples):
    """Check if the set of triples is linear (any two share at most 1 point)."""
    n = len(triples)
    for i in range(n):
        a, b, c = triples[i]
        for j in range(i + 1, n):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z: inter += 1
            if b == x or b == y or b == z: inter += 1
            if c == x or c == y or c == z: inter += 1
            if inter > 1:
                return False
    return True

def _build_covering_design(rng):
    """
    Build a covering design: a set of triples such that every 3-subset of {0..15}
    is contained in at least one triple. Use greedy + local search.
    """
    # Precompute all 3-subsets
    all_3subsets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                all_3subsets.append((i, j, k))
    
    # Map each 3-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_3subsets)}
    n_subsets = len(all_3subsets)
    
    # For each triple, which 3-subsets does it cover? (just itself)
    # Actually, a triple {a,b,c} covers exactly the 3-subset {a,b,c}
    # So we need to select triples such that every 3-subset is selected.
    # But that's trivial: just take all C(16,3) = 560 triples.
    # The constraint is linearity: any two triples share at most 1 point.
    
    # So the problem is: find a linear triple system that covers all 3-subsets.
    # That's a Steiner system S(3,3,16) or similar.
    
    # Actually, let me reconsider. The independence number is the largest set
    # with no triple entirely contained in it. To minimize this, we want to
    # "hit" all large independent sets with triples.
    
    # New approach: Use a different strategy. Start with a known good structure
    # and optimize.
    
    # Let's try: partition 16 points into groups and create triples across groups.
    # Or use a greedy approach that prioritizes covering large independent sets.
    
    # Greedy: repeatedly add the triple that reduces the independence number the most.
    triples = []
    used_pairs = set()
    
    def get_used_pairs(triples):
        pairs = set()
        for a, b, c in triples:
            pairs.add((a, b))
            pairs.add((a, c))
            pairs.add((b, c))
        return pairs
    
    def is_conflict(triple, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
            return True
        return False
    
    def greedy_init():
        triples = []
        used_pairs = set()
        remaining = list(all_3subsets)
        rng.shuffle(remaining)
        
        for t in remaining:
            if not is_conflict(t, used_pairs):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
        return triples
    
    return greedy_init()

def construct():
    rng = random.Random(42)
    
    # Precompute all possible triples as bitmasks
    all_triples_tuples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples_tuples.append((i, j, k))
                
    def get_used_pairs(triples):
        pairs = set()
        for a, b, c in triples:
            pairs.add((a, b))
            pairs.add((a, c))
            pairs.add((b, c))
        return pairs

    def is_conflict(triple, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
            return True
        return False

    def greedy_init():
        triples = []
        used_pairs = set()
        remaining = all_triples_tuples[:]
        rng.shuffle(remaining)
        
        for t in remaining:
            if not is_conflict(t, used_pairs):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
        return triples

    def local_search(triples, time_budget):
        """
        Local search with multiple move types: add, remove, swap, and 2-opt.
        """
        current_triples = list(triples)
        current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
        current_alpha = _compute_alpha_fast(current_masks)
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        T = 1.0
        cooling = 0.999
        start_time = time.time()
        
        while (time.time() - start_time) < time_budget:
            if time.time() - _START > _LIMIT:
                break
                
            move_type = rng.random()
            new_triples = None
            new_alpha = None
            delta = 0
            
            if move_type < 0.25:
                # Add a triple
                t = rng.choice(all_triples_tuples)
                if t in current_triples:
                    continue
                used_pairs = get_used_pairs(current_triples)
                if not is_conflict(t, used_pairs):
                    new_triples = current_triples + [t]
                    new_masks = current_masks + [(1<<t[0])|(1<<t[1])|(1<<t[2])]
                    new_alpha = _compute_alpha_fast(new_masks)
                    delta = new_alpha - current_alpha
            elif move_type < 0.5:
                # Remove a triple
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                new_triples = current_triples[:idx] + current_triples[idx+1:]
                new_masks = current_masks[:idx] + current_masks[idx+1:]
                new_alpha = _compute_alpha_fast(new_masks)
                delta = new_alpha - current_alpha
            elif move_type < 0.75:
                # Swap
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                t_remove = current_triples[idx]
                t_add = rng.choice(all_triples_tuples)
                if t_add == t_remove or t_add in current_triples:
                    continue
                
                temp_triples = current_triples[:idx] + current_triples[idx+1:]
                used_pairs = get_used_pairs(temp_triples)
                
                if not is_conflict(t_add, used_pairs):
                    new_triples = temp_triples + [t_add]
                    new_masks = current_masks[:idx] + current_masks[idx+1:] + [(1<<t_add[0])|(1<<t_add[1])|(1<<t_add[2])]
                    new_alpha = _compute_alpha_fast(new_masks)
                    delta = new_alpha - current_alpha
            else:
                # 2-opt: remove two, add two
                if len(current_triples) < 2:
                    continue
                idx1 = rng.randrange(len(current_triples))
                idx2 = rng.randrange(len(current_triples))
                if idx1 == idx2:
                    continue
                
                # Remove both
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                temp_triples = current_triples[:idx2] + current_triples[idx2+1:idx1] + current_triples[idx1+1:]
                used_pairs = get_used_pairs(temp_triples)
                
                # Add two random non-conflicting triples
                t1 = rng.choice(all_triples_tuples)
                if t1 in current_triples or is_conflict(t1, used_pairs):
                    continue
                used_pairs.add((t1[0], t1[1]))
                used_pairs.add((t1[0], t1[2]))
                used_pairs.add((t1[1], t1[2]))
                
                t2 = rng.choice(all_triples_tuples)
                if t2 in current_triples or t2 == t1 or is_conflict(t2, used_pairs):
                    continue
                
                new_triples = temp_triples + [t1, t2]
                new_masks = current_masks[:idx2] + current_masks[idx2+1:idx1] + current_masks[idx1+1:] + \
                           [(1<<t1[0])|(1<<t1[1])|(1<<t1[2]), (1<<t2[0])|(1<<t2[1])|(1<<t2[2])]
                new_alpha = _compute_alpha_fast(new_masks)
                delta = new_alpha - current_alpha

            if new_triples is None:
                continue

            if delta < 0:
                current_triples = new_triples
                current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                if T > 0:
                    if delta == 0:
                        if rng.random() < 0.5:
                            current_triples = new_triples
                            current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
                            current_alpha = new_alpha
                    else:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            current_triples = new_triples
                            current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
                            current_alpha = new_alpha
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_triples, best_alpha

    def tabu_search(triples, time_budget):
        """
        Tabu search with a different neighborhood structure.
        """
        current_triples = list(triples)
        current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
        current_alpha = _compute_alpha_fast(current_masks)
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        tabu_list = []  # List of (triple, expiry)
        tabu_tenure = 20
        start_time = time.time()
        
        while (time.time() - start_time) < time_budget:
            if time.time() - _START > _LIMIT:
                break
            
            # Clean expired tabu entries
            now = time.time()
            tabu_list = [(t, exp) for t, exp in tabu_list if exp > now]
            
            # Generate neighbors
            neighbors = []
            
            # Try adding each possible triple
            used_pairs = get_used_pairs(current_triples)
            for t in all_triples_tuples:
                if t in current_triples:
                    continue
                if not is_conflict(t, used_pairs):
                    # Check if t is tabu
                    if t in [tt for tt, _ in tabu_list]:
                        continue
                    new_triples = current_triples + [t]
                    new_masks = current_masks + [(1<<t[0])|(1<<t[1])|(1<<t[2])]
                    new_alpha = _compute_alpha_fast(new_masks)
                    neighbors.append((new_alpha, new_triples, new_masks, t, 'add'))
            
            # Try removing each triple
            for idx, t in enumerate(current_triples):
                if t in [tt for tt, _ in tabu_list]:
                    continue
                new_triples = current_triples[:idx] + current_triples[idx+1:]
                new_masks = current_masks[:idx] + current_masks[idx+1:]
                new_alpha = _compute_alpha_fast(new_masks)
                neighbors.append((new_alpha, new_triples, new_masks, t, 'remove'))
            
            if not neighbors:
                break
            
            # Sort by alpha
            neighbors.sort(key=lambda x: x[0])
            
            # Pick best non-tabu move
            best_move = None
            for alpha, triples, masks, triple, move_type in neighbors:
                if triple not in [tt for tt, _ in tabu_list]:
                    best_move = (alpha, triples, masks, triple, move_type)
                    break
            
            if best_move is None:
                best_move = neighbors[0]
            
            alpha, new_triples, new_masks, triple, move_type = best_move
            
            # Accept if it improves or is the best available
            if alpha <= current_alpha or len(neighbors) == 1:
                current_triples = new_triples
                current_masks = new_masks
                current_alpha = alpha
                
                # Add to tabu list
                tabu_list.append((triple, time.time() + tabu_tenure * 0.01))
                
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
        
        return best_triples, best_alpha

    best_global_alpha = 16
    best_global_triples = None
    
    start_time = time.time()
    restarts = 0
    while (time.time() - start_time) < (_LIMIT - 10):
        init_triples = greedy_init()
        if not init_triples:
            continue
        
        # Alternate between local search and tabu search
        if restarts % 2 == 0:
            budget = min(5.0, max(1.0, _LIMIT - 15 - (time.time() - start_time)))
            opt_triples, alpha = local_search(init_triples, budget)
        else:
            budget = min(5.0, max(1.0, _LIMIT - 15 - (time.time() - start_time)))
            opt_triples, alpha = tabu_search(init_triples, budget)
        
        if alpha < best_global_alpha:
            best_global_alpha = alpha
            best_global_triples = opt_triples
            
        restarts += 1
        if best_global_alpha <= 5:
            break

    if best_global_triples is None:
        best_global_triples = greedy_init()
        if not _is_valid_triple_set(best_global_triples):
            best_global_triples = []
            
    return best_global_triples

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
    print(f"Alpha: {_compute_alpha_fast([(1<<a)|(1<<b)|(1<<c) for a,b,c in res])}")
