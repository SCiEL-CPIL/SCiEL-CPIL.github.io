import random
from itertools import combinations

# Pre-compute Q3 patterns on 0..7
def get_q3_edge_sets():
    std_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                std_edges.add((i, j))
    
    unique_sets = set()
    import itertools
    for perm in itertools.permutations(range(8)):
        mapped_edges = set()
        for u, v in std_edges:
            mu = perm[u]
            mv = perm[v]
            if mu > mv:
                mu, mv = mv, mu
            mapped_edges.add((mu, mv))
        unique_sets.add(frozenset(mapped_edges))
    return list(unique_sets)

Q3_PATTERNS = get_q3_edge_sets()

# Pre-compute all subsets and their mapped patterns
n = 16
subsets = list(combinations(range(n), 8))
# Map subset to index
subset_to_idx = {s: i for i, s in enumerate(subsets)}

# Pre-compute mapped patterns for each subset
# subset_patterns[i] is a list of 48 frozensets of global edges
subset_patterns = []
for s in subsets:
    patterns_for_s = []
    for p in Q3_PATTERNS:
        mapped = set()
        for u, v in p:
            gu = s[u]
            gv = s[v]
            if gu > gv:
                gu, gv = gv, gu
            mapped.add((gu, gv))
        patterns_for_s.append(frozenset(mapped))
    subset_patterns.append(patterns_for_s)

def construct():
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    best_edge_count = 0
    best_edges = set()
    
    # Run a few times
    for _ in range(3):
        current_edges = set(all_edges)
        
        while True:
            found = False
            # Shuffle subset order?
            # To be fast, we just iterate.
            # We can randomize the order of subsets in each restart
            indices = list(range(len(subsets)))
            random.shuffle(indices)
            
            for idx in indices:
                s = subsets[idx]
                # Check if any pattern is a subset of current_edges
                for p in subset_patterns[idx]:
                    if p.issubset(current_edges):
                        found = True
                        # Remove a random edge from p
                        # p is a frozenset of tuples
                        edge_to_remove = random.choice(tuple(p))
                        current_edges.discard(edge_to_remove)
                        break
                if found:
                    break
                    
            if not found:
                break
        
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = set(current_edges)
            
    return [[u, v] for u, v in best_edges]
