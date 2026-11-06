"""Rewritten program using probabilistic method with local search for hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

def _generate_triples(n):
    """Generate all triples in lexicographic order and map to indices."""
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    return triples, triple_index

def _count_violations(n, colors, triples, triple_index):
    """Count red K4^(3) and blue K5^(3) violations."""
    # Precompute for each pair, the set of vertices forming a triple with them
    # Actually, let's just iterate over all 4-sets and 5-sets for counting
    # For n=34, C(34,4)=46376 and C(34,5)=278256, which is manageable
    
    red_violations = 0
    blue_violations = 0
    
    # Check all 4-sets for red K4^(3)
    for quad in combinations(range(n), 4):
        all_red = True
        for triple in combinations(quad, 3):
            idx = triple_index[triple]
            if colors[idx] != 0:
                all_red = False
                break
        if all_red:
            red_violations += 1
    
    # Check all 5-sets for blue K5^(3)
    for quint in combinations(range(n), 5):
        all_blue = True
        for triple in combinations(quint, 3):
            idx = triple_index[triple]
            if colors[idx] != 1:
                all_blue = False
                break
        if all_blue:
            blue_violations += 1
    
    return red_violations, blue_violations

def _local_search(n, colors, triples, triple_index, max_iterations=50000):
    """Local search to fix violations."""
    colors = list(colors)
    
    for iteration in range(max_iterations):
        red_v, blue_v = _count_violations(n, colors, triples, triple_index)
        if red_v == 0 and blue_v == 0:
            return colors
        
        if red_v > 0:
            # Pick a random red K4^(3) and flip one of its triples to blue
            found = False
            for _ in range(100):
                quad = tuple(sorted(random.sample(range(n), 4)))
                all_red = True
                triple_indices = []
                for triple in combinations(quad, 3):
                    idx = triple_index[triple]
                    if colors[idx] != 0:
                        all_red = False
                        break
                    triple_indices.append(idx)
                if all_red:
                    idx_to_flip = random.choice(triple_indices)
                    colors[idx_to_flip] = 1
                    found = True
                    break
            if not found:
                # Fallback: random flip
                idx = random.randint(0, len(colors) - 1)
                colors[idx] = 1 - colors[idx]
        else:
            # Pick a random blue K5^(3) and flip one of its triples to red
            found = False
            for _ in range(100):
                quint = tuple(sorted(random.sample(range(n), 5)))
                all_blue = True
                triple_indices = []
                for triple in combinations(quint, 3):
                    idx = triple_index[triple]
                    if colors[idx] != 1:
                        all_blue = False
                        break
                    triple_indices.append(idx)
                if all_blue:
                    idx_to_flip = random.choice(triple_indices)
                    colors[idx_to_flip] = 0
                    found = True
                    break
            if not found:
                # Fallback: random flip
                idx = random.randint(0, len(colors) - 1)
                colors[idx] = 1 - colors[idx]
    
    return colors

def construct():
    # Try to find a valid coloring for n=34 using probabilistic method with local search
    target_n = 34
    
    triples, triple_index = _generate_triples(target_n)
    num_triples = len(triples)
    
    # Use biased probability: p_red slightly less than 0.5 to balance expected violations
    # E[red K4] = C(n,4) * p^4
    # E[blue K5] = C(n,5) * (1-p)^10
    # For n=34: C(34,4)=46376, C(34,5)=278256
    # Minimize 46376*p^4 + 278256*(1-p)^10
    # Try a few values of p
    
    best_p = 0.45  # Slightly favor blue to reduce blue K5 violations
    
    random.seed(42)
    
    # Try multiple random seeds and probabilities
    for p in [0.4, 0.42, 0.45, 0.48, 0.5]:
        for seed in range(20):
            random.seed(seed * 1000 + int(p * 100))
            np.random.seed(seed * 1000 + int(p * 100))
            
            # Generate random coloring
            colors = [1 if random.random() < p else 0 for _ in range(num_triples)]
            
            # Local search
            colors = _local_search(target_n, colors, triples, triple_index, max_iterations=30000)
            
            red_v, blue_v = _count_violations(target_n, colors, triples, triple_index)
            if red_v == 0 and blue_v == 0:
                return {'n': target_n, 'colors': colors}
    
    # If we couldn't find a valid coloring for n=34, try n=33
    target_n = 33
    triples, triple_index = _generate_triples(target_n)
    num_triples = len(triples)
    
    for p in [0.45, 0.5]:
        for seed in range(20):
            random.seed(seed * 1000 + int(p * 100) + 10000)
            np.random.seed(seed * 1000 + int(p * 100) + 10000)
            
            colors = [1 if random.random() < p else 0 for _ in range(num_triples)]
            colors = _local_search(target_n, colors, triples, triple_index, max_iterations=30000)
            
            red_v, blue_v = _count_violations(target_n, colors, triples, triple_index)
            if red_v == 0 and blue_v == 0:
                return {'n': target_n, 'colors': colors}
    
    # Fallback: return a small valid coloring
    n = 9
    triples, triple_index = _generate_triples(n)
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
