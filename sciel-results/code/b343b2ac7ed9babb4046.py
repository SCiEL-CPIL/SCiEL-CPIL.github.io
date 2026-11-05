import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute configurations
    # A configuration is 3 disjoint pairs.
    # We iterate over all sets of 6 vertices, and all partitions into 3 pairs.
    configs = []
    triple_configs = [[] for _ in range(num_triples)]
    
    # Helper to generate partitions of 6 elements into 3 pairs
    def get_partitions(elements):
        # elements is a tuple of 6 distinct ints
        # Returns list of (pair1, pair2, pair3)
        e = list(elements)
        p1 = (e[0], e[1])
        remaining = e[2:]
        # Partitions of remaining 4 into 2 pairs
        # (r0,r1), (r2,r3)
        # (r0,r2), (r1,r3)
        # (r0,r3), (r1,r2)
        r = remaining
        yield (p1, (r[0], r[1]), (r[2], r[3]))
        yield (p1, (r[0], r[2]), (r[1], r[3]))
        yield (p1, (r[0], r[3]), (r[1], r[2]))
        
    # Wait, the pairs are unordered, and the set of pairs is unordered.
    # The above generates specific pairings.
    # For a fixed set of 6 vertices, there are 15 ways to partition into 3 pairs.
    # My generator above only gives 3.
    # I need to iterate over all ways to pick the first pair, then partition the rest.
    
    # Better approach:
    # Iterate over all 6-subsets of 16.
    for vertices in itertools.combinations(range(N), 6):
        # Generate all 15 pairings
        # Standard way:
        v = list(vertices)
        # Pair v[0] with v[i] for i in 1..5
        for i in range(1, 6):
            p1 = (v[0], v[i])
            rem = [v[j] for j in range(6) if j != 0 and j != i]
            # Pair rem[0] with rem[k] for k in 1,2,3
            for k in range(1, 4):
                p2 = (rem[0], rem[k])
                p3 = (rem[1 if k==2 else 2], rem[2 if k==2 else 1]) # This logic is messy
                
    # Let's just use a known list of 15 pairings for indices 0..5
    # Pairs are (0,1), (2,3), (4,5) etc.
    # Permutations of (0,1,2,3,4,5) where we take (0,1), (2,3), (4,5)
    # And divide by symmetries?
    # Easier: hardcode the 15 pairings of indices 0-5.
    
    pairings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    for vertices in itertools.combinations(range(N), 6):
        v_map = {vertices[i]: i for i in range(6)} # Not needed, just use indices
        # We need to map the abstract indices 0..5 to the actual vertex values
        # For each pairing of indices, map to actual vertices
        for p in pairings:
            # p is list of 3 pairs of indices
            # Construct the 8 transversal triples
            # Vertices are vertices[0]...vertices[5]
            # Pair 1: vertices[p[0][0]], vertices[p[0][1]]
            # Pair 2: vertices[p[1][0]], vertices[p[1][1]]
            # Pair 3: vertices[p[2][0]], vertices[p[2][1]]
            
            a1, a2 = vertices[p[0][0]], vertices[p[0][1]]
            b1, b2 = vertices[p[1][0]], vertices[p[1][1]]
            c1, c2 = vertices[p[2][0]], vertices[p[2][1]]
            
            # 8 triples
            t_indices = []
            for a in (a1, a2):
                for b in (b1, b2):
                    for c in (c1, c2):
                        t = tuple(sorted((a, b, c)))
                        t_indices.append(triple_to_idx[t])
            
            configs.append(t_indices)
            for idx in t_indices:
                triple_configs[idx].append(len(configs) - 1)
                
    num_configs = len(configs)
    
    in_set = [False] * num_triples
    config_count = [0] * num_configs
    
    def can_add(i):
        for c in triple_configs[i]:
            if config_count[c] == 7:
                return False
        return True

    def do_add(i):
        in_set[i] = True
        for c in triple_configs[i]:
            config_count[c] += 1

    def do_remove(i):
        in_set[i] = False
        for c in triple_configs[i]:
            config_count[c] -= 1
            
    # Greedy Initialization
    indices = list(range(num_triples))
    random.shuffle(indices)
    for i in indices:
        if can_add(i):
            do_add(i)
            
    current_size = sum(in_set)
    
    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any single triple
        # Randomize order to avoid bias
        indices = list(range(num_triples))
        random.shuffle(indices)
        for i in indices:
            if not in_set[i]:
                if can_add(i):
                    do_add(i)
                    current_size += 1
                    improved = True
                    break
        
        if improved:
            continue
            
        # 2. Try to swap
        # Pick a random triple to remove
        # Only consider removing triples that are in some "full" or "nearly full" config?
        # No, just random.
        in_indices = [i for i in range(num_triples) if in_set[i]]
        if not in_indices:
            break
            
        i_out = random.choice(in_indices)
        do_remove(i_out)
        
        added_count = 0
        added_indices = []
        
        # Try to add any triple
        indices = list(range(num_triples))
        random.shuffle(indices)
        for i in indices:
            if not in_set[i]:
                if can_add(i):
                    do_add(i)
                    added_count += 1
                    added_indices.append(i)
                    
        if added_count > 1:
            current_size += added_count - 1
            improved = True
        else:
            # Revert
            for i in added_indices:
                do_remove(i)
            do_add(i_out)
            
        if not improved and random.random() < 0.1:
             # Stuck for a while, maybe restart?
             # But we have a good current size.
             # Let's just continue.
             pass

    return [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
