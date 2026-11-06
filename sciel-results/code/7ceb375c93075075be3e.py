"""Rewritten program: Optimized greedy strategies for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import math
import random


def construct():
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    best_classes = []
    best_score = 0.0

    def evaluate_and_update(chosen):
        nonlocal best_classes, best_score
        score = sum(1.0 / moduli[idx] for idx, _ in chosen)
        if score > best_score:
            best_score = score
            best_classes = [[moduli[idx], r] for idx, r in chosen]

    def run_greedy(strategy):
        chosen = []
        for i in range(n_moduli):
            m = moduli[i]
            found_r = None
            # Check residues
            # Optimization: We can iterate r in a specific order
            if strategy == 'smallest':
                rng = range(m)
            elif strategy == 'largest':
                rng = range(m - 1, -1, -1)
            else: # random
                rng = list(range(m))
                random.shuffle(rng)
            
            for r in rng:
                is_ok = True
                for (j, r2) in chosen:
                    g = gcd_matrix[i][j]
                    if r % g == r2 % g:
                        is_ok = False
                        break
                if is_ok:
                    found_r = r
                    break
            
            if found_r is not None:
                chosen.append((i, found_r))
        
        evaluate_and_update(chosen)

    # Run deterministic strategies
    run_greedy('smallest')
    run_greedy('largest')
    
    # Run a few random strategies
    random.seed(42)
    for _ in range(20):
        run_greedy('random')
        
    return best_classes
# EVOLVE-BLOCK-END
