"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    n = 16
    all_points = list(range(n))
    
    # Precompute all pairs for each triple
    all_triples = list(itertools.combinations(range(n), 3))
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset({(a, b), (a, c), (b, c)})
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        pair_count = defaultdict(int)
        for t in triples:
            for pair in triple_pairs[t]:
                pair_count[pair] += 1
                if pair_count[pair] > 1:
                    return False
        return True
    
    def independence_number(triples):
        """Find the largest subset of points containing no triple."""
        if not triples:
            return n
        
        triple_set = set(triples)
        
        # Check from largest size down
        for size in range(n, 0, -1):
            for subset in itertools.combinations(all_points, size):
                subset_set = set(subset)
                has_triple = False
                for t in triple_set:
                    if all(p in subset_set for p in t):
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 0
    
    def greedy_construct(seed):
        """Greedy construction with random ordering."""
        rng = random.Random(seed)
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        used_pairs = set()
        selected = []
        
        for t in shuffled:
            pairs = triple_pairs[t]
            if not any(p in used_pairs for p in pairs):
                used_pairs.update(pairs)
                selected.append(list(t))
        
        return selected
    
    def local_search(triples, rng, max_iters=1000):
        """Try to improve the independence number by swapping triples."""
        if not is_linear(triples):
            return triples
        
        current_alpha = independence_number(triples)
        best = triples[:]
        best_alpha = current_alpha
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try adding a random triple
            if len(triples) < 20:  # Limit size to keep computation manageable
                candidates = []
                for t in all_triples:
                    if t not in triples:
                        pairs = triple_pairs[t]
                        if not any(p in set().union(*[triple_pairs[tt] for tt in triples]) for p in pairs):
                            candidates.append(t)
                
                if candidates:
                    t = rng.choice(candidates)
                    new_triples = triples + [t]
                    if is_linear(new_triples):
                        new_alpha = independence_number(new_triples)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best = new_triples[:]
                            triples = new_triples
                        elif new_alpha == best_alpha and rng.random() < 0.1:
                            triples = new_triples
            
            # Try removing a random triple
            elif triples:
                idx = rng.randrange(len(triples))
                removed = triples.pop(idx)
                new_alpha = independence_number(triples)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best = triples[:]
                else:
                    triples.append(removed)
        
        return best
    
    # Phase 1: Multiple greedy constructions
    best_system = None
    best_alpha = n + 1
    num_greedy = 50
    
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        
        triples = greedy_construct(i)
        if not is_linear(triples):
            continue
        
        alpha = independence_number(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = triples[:]
            print(f"Greedy {i}: alpha = {alpha}, triples = {len(triples)}")
    
    if best_system is None:
        # Fallback: return the seed's structure
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Phase 2: Local search from the best greedy solution
    rng = random.Random(42)
    improved = local_search(best_system[:], rng, max_iters=500)
    improved_alpha = independence_number(improved)
    
    if improved_alpha < best_alpha:
        best_alpha = improved_alpha
        best_system = improved
        print(f"After local search: alpha = {best_alpha}")
    
    # Phase 3: More local search attempts with different seeds
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(seed + 1000)
        # Start from a slightly perturbed version of the best
        perturbed = best_system[:]
        if perturbed:
            idx = rng.randrange(len(perturbed))
            perturbed.pop(idx)
        
        improved = local_search(perturbed, rng, max_iters=300)
        improved_alpha = independence_number(improved)
        
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_system = improved
            print(f"Local search seed {seed}: alpha = {best_alpha}")
    
    return best_system
