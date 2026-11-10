"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    """
    Construct a linear triple system on 16 points with small independence number.
    
    Models the problem as a maximum clique search in the compatibility graph of triples,
    where edges connect triples that intersect in at most one point (linear constraint).
    Uses greedy clique expansion with randomized restarts and local search.
    """
    points = list(range(16))
    n = 16
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(n), 3))
    n_triples = len(all_triples)
    
    # Precompute compatibility: two triples are compatible if they share at most 1 point
    # Use bitmask for fast intersection checking
    triple_masks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Compatibility matrix (symmetric)
    # compatible[i][j] = True if triples i and j share at most 1 point
    # For efficiency, precompute as list of lists of compatible indices
    compatible = [[] for _ in range(n_triples)]
    for i in range(n_triples):
        for j in range(i + 1, n_triples):
            # Count common points via bitmask
            common = triple_masks[i] & triple_masks[j]
            # Count bits
            bit_count = bin(common).count('1')
            if bit_count <= 1:
                compatible[i].append(j)
                compatible[j].append(i)
    
    def compute_independence_number(triples):
        """
        Compute the independence number: largest set of points containing no triple.
        Uses bitmask representation for efficiency.
        """
        if not triples:
            return 16
        
        # Convert triples to bitmasks
        triple_bitmasks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_bitmasks.append(mask)
        
        # Try to find the largest independent set
        # Check from size 16 down to 1
        for size in range(16, 0, -1):
            # Check if there exists a subset of this size with no triple
            for subset in itertools.combinations(range(n), size):
                subset_mask = 0
                for p in subset:
                    subset_mask |= (1 << p)
                # Check if any triple is entirely within this subset
                has_triple = False
                for tm in triple_bitmasks:
                    if (tm & subset_mask) == tm:
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 1
    
    def greedy_clique(seed=42):
        """
        Greedy clique construction with randomization.
        Builds a maximal set of mutually compatible triples.
        """
        rng = random.Random(seed)
        
        # Start with a random triple
        start = rng.randint(0, n_triples - 1)
        clique = [start]
        in_clique = set([start])
        
        # Available candidates: those compatible with all in clique
        # Use a list and filter
        candidates = compatible[start][:]
        rng.shuffle(candidates)
        
        while candidates:
            # Pick a random candidate from the front
            idx = candidates.pop(0)
            if idx in in_clique:
                continue
            
            # Check if compatible with all in clique
            is_compatible = True
            for c in clique:
                if idx not in compatible[c]:
                    is_compatible = False
                    break
            
            if is_compatible:
                clique.append(idx)
                in_clique.add(idx)
                # Remove incompatible candidates
                candidates = [c for c in candidates if c not in in_clique and c in compatible[idx]]
        
        # Convert clique indices to actual triples
        result = [all_triples[i] for i in clique]
        return result
    
    def local_search(triples, seed=0, max_iters=100):
        """
        Local search: try to swap triples to improve independence number.
        """
        rng = random.Random(seed)
        current = list(triples)
        current_score = compute_independence_number(current)
        
        for _ in range(max_iters):
            # Try to add a random compatible triple
            added = False
            for _ in range(50):
                idx = rng.randint(0, n_triples - 1)
                # Check if compatible with all in current
                compatible_with_all = True
                for t in current:
                    t_idx = all_triples.index(t)
                    if idx not in compatible[t_idx]:
                        compatible_with_all = False
                        break
                if compatible_with_all and idx not in [all_triples.index(t) for t in current]:
                    new_current = current + [all_triples[idx]]
                    new_score = compute_independence_number(new_current)
                    if new_score < current_score:
                        current = new_current
                        current_score = new_score
                        added = True
                        break
            
            if not added:
                break
        
        return current, current_score
    
    # Try multiple seeds with greedy clique
    best_triples = None
    best_score = 17
    
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    for seed in range(200):
        if time.time() - start_time > timeout:
            break
        
        triples = greedy_clique(seed)
        if not triples:
            continue
        
        score = compute_independence_number(triples)
        
        if score < best_score:
            best_score = score
            best_triples = triples
            # Try local search from this point
            if time.time() - start_time < timeout - 5:
                improved, improved_score = local_search(triples, seed=seed, max_iters=50)
                if improved_score < best_score:
                    best_score = improved_score
                    best_triples = improved
    
    # If we didn't find anything, return a default
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
