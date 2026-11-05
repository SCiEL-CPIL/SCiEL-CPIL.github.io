import random
import itertools

def construct(k, g):
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    # Target: k=5, g=9
    # Try S_6 Cayley graph
    
    # Group: S_6
    # Represent permutations as tuples of 0..5
    # Identity
    id_perm = tuple(range(6))
    
    # Generate all involutions in S_6
    involutions = []
    for perm in itertools.permutations(range(6)):
        # Check if perm is involution: perm(perm(x)) == x for all x
        is_inv = True
        for i in range(6):
            if perm[perm[i]] != i:
                is_inv = False
                break
        if is_inv and perm != id_perm:
            involutions.append(perm)
            
    # We need k=5 generators
    # Randomly sample sets of 5 involutions
    # Check girth
    
    def multiply(p, q):
        # p(q(x))
        return tuple(p[q[i]] for i in range(6))
        
    def has_short_cycle(gens, max_len):
        # Check non-backtracking words of length 1 to max_len
        # Start with each gen
        # State: (current_perm, last_gen_index)
        # Actually, just track current perm.
        # If we return to id_perm, we have a cycle.
        # But we must avoid backtracking s_i s_i.
        
        # BFS/DFS for words
        # visited: set of (perm, last_gen) to avoid immediate backtracking?
        # Actually, we just need to see if ANY word of length L is identity.
        # We can just generate all words.
        
        # Length 1: gens are not identity.
        # Length 2: g_i g_j. If g_i == g_j, product is id. But we pick distinct gens.
        # So length 2 is safe if gens are distinct and involutions.
        
        # Length 3: g_i g_j g_k.
        
        # We'll do a BFS of states (perm, last_gen_idx)
        # Start: (g_i, i) for each i.
        # If any state reaches id_perm with last_gen_idx != -1 (meaning we used at least one step), 
        # and the path length is < 9, we fail.
        
        # Actually, if we reach id_perm at depth d, the cycle length is d.
        # We need d >= 9.
        
        visited = set()
        # Queue: (perm, last_idx, depth)
        queue = []
        for i, g in enumerate(gens):
            queue.append((g, i, 1))
            visited.add((g, i))
            
        while queue:
            perm, last_idx, depth = queue.pop(0)
            
            # Check if we are at identity
            if perm == id_perm:
                # Cycle found of length depth
                if depth < 9:
                    return True
                else:
                    # We found a cycle of length >= 9, but we are looking for SHORT cycles.
                    # If we reach identity at depth 9, it's ok.
                    # But we should stop expanding this path?
                    # No, we just need to know if ANY path reaches identity with depth < 9.
                    # So if depth >= 9, we don't care about this specific path for failure,
                    # but we should not expand it further?
                    # Actually, if we reach identity at depth 9, any extension is longer.
                    # So we can skip expansion.
                    continue

            if depth >= 9:
                continue
                
            for i, g in enumerate(gens):
                if i == last_idx:
                    continue
                next_perm = multiply(perm, g)
                state = (next_perm, i)
                if state not in visited:
                    visited.add(state)
                    queue.append((next_perm, i, depth + 1))
                    
        return False
        
    # Try random sets
    random.seed(42)
    for _ in range(10000):
        gens = random.sample(involutions, 5)
        if not has_short_cycle(gens, 8):
            # Construct edges
            # Group elements are all permutations
            # Map perm to index
            all_perms = list(itertools.permutations(range(6)))
            perm_to_idx = {p: i for i, p in enumerate(all_perms)}
            
            edges = []
            for i, p in enumerate(all_perms):
                for g in gens:
                    q = multiply(p, g)
                    j = perm_to_idx[q]
                    if i < j:
                        edges.append((i, j))
                        
            return len(all_perms), edges
            
    # Fallback to L-U-W if S_6 fails
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
        
    return luw_construct(k, g)
