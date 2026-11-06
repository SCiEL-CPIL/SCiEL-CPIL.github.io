# EVOLVE-BLOCK-START
import random
import time
import itertools
import numpy as np

def _is_sidon_fast(subset):
    """Fast Sidon check with early termination using a set of sums."""
    n = len(subset)
    if n <= 1:
        return True
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _count_additive_relations(ambient):
    """Count the number of (not necessarily distinct) 4-tuples (a,b,c,d) in ambient
    with a+b=c+d where {a,b} != {c,d}. More relations = harder to find large Sidon subsets."""
    n = len(ambient)
    relations = 0
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            s = ambient[i] + ambient[j]
            if s in sum_count:
                relations += sum_count[s]
            sum_count[s] = sum_count.get(s, 0) + 1
    return relations

def _max_sidon_size_efficient(ambient):
    """Find largest Sidon subset using binary search on size with early termination."""
    n = len(ambient)
    # First check if the whole set is Sidon
    if _is_sidon_fast(list(ambient)):
        return n
    
    # Try from n-1 down, but limit the number of combinations checked
    # For efficiency, we'll use a smarter approach
    for k in range(n - 1, 1, -1):
        # Limit combinations checked for large k
        max_checks = 500 if k > 8 else 2000
        checked = 0
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_fast(list(combo)):
                return k
            checked += 1
            if checked >= max_checks:
                break
        # If we didn't find one in limited checks, assume smaller
        if checked >= max_checks:
            # Do a more thorough check for this k if k is small enough
            if k <= 6:
                for combo in itertools.combinations(ambient, k):
                    if _is_sidon_fast(list(combo)):
                        return k
                # If still not found, continue to smaller k
            else:
                # For large k with limited checks, be conservative
                # Try a few more random samples
                for _ in range(200):
                    sample = random.sample(list(ambient), k)
                    if _is_sidon_fast(sample):
                        return k
                # If not found, continue
                continue
    return 2  # At least 2 elements form a Sidon set (unless they're equal, but they're distinct)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    random.seed(12345)
    
    best_set = None
    best_sidon_size = 999
    best_relation_count = -1
    
    # Strategy 1: Greedy construction maximizing additive relations
    # Start with a small seed set, then greedily add elements that create most relations
    
    def greedy_construct(seed_elements, max_relations_target=200):
        """Build a set by greedily adding elements that maximize additive relations."""
        current = list(seed_elements)
        used = set(current)
        
        while len(current) < 12:
            best_candidate = None
            best_score = -1
            
            # Try all available values
            candidates = [x for x in range(129) if x not in used]
            
            for cand in candidates:
                # Compute how many new relations this would create
                new_rels = 0
                # Relations involving the new element
                for i in range(len(current)):
                    for j in range(len(current)):
                        # Check if current[i] + current[j] == cand + something
                        # or current[i] + cand == current[j] + something
                        pass
                
                # Simpler: just check if adding cand creates collisions
                # A collision means a+b=c+d for some a,b,c,d in new set
                # Focus on: for each pair (a,b) in current, check if a+b = cand + x for some x in current
                # or a+cand = b+x
                
                # Count potential new relations
                rels_with_new = 0
                for a in current:
                    for b in current:
                        # a + b = cand + x => x = a + b - cand
                        x = a + b - cand
                        if x in used and x != cand:
                            rels_with_new += 1
                        # a + cand = b + x => x = a + cand - b
                        x = a + cand - b
                        if x in used and x != cand:
                            rels_with_new += 1
                
                # Also check self-relations: cand + cand = a + b
                for a in current:
                    for b in current:
                        if 2 * cand == a + b and a != b:
                            rels_with_new += 1
                        if 2 * cand == a + a:
                            rels_with_new += 1
                
                if rels_with_new > best_score:
                    best_score = rels_with_new
                    best_candidate = cand
            
            if best_candidate is None:
                break
            
            current.append(best_candidate)
            used.add(best_candidate)
        
        return current
    
    # Try multiple greedy constructions with different seeds
    seed_patterns = [
        [0, 1, 2, 4],       # Powers of 2 style
        [0, 1, 2, 3],       # Dense small numbers
        [0, 1, 3, 7],       # Exponential
        [0, 2, 4, 8],       # Doubling
        [0, 1, 4, 9],       # Squares
        [0, 1, 2, 5],       # Fibonacci-ish
        [0, 3, 6, 9],       # AP with step 3
        [0, 2, 5, 8],       # AP with step 3, offset
    ]
    
    evaluated_sets = set()
    
    for seed in seed_patterns:
        if time.time() > deadline:
            break
        
        # Generate several variations
        for _ in range(3):
            # Perturb the seed slightly
            perturbed_seed = list(seed)
            for i in range(len(perturbed_seed)):
                if random.random() < 0.3:
                    perturbed_seed[i] = max(0, min(128, perturbed_seed[i] + random.randint(-2, 2)))
            
            if len(set(perturbed_seed)) != len(perturbed_seed):
                continue
            
            candidate = greedy_construct(perturbed_seed)
            if len(candidate) != 12:
                continue
            
            key = tuple(sorted(candidate))
            if key in evaluated_sets:
                continue
            evaluated_sets.add(key)
            
            rel_count = _count_additive_relations(candidate)
            sidon_size = _max_sidon_size_efficient(candidate)
            
            if sidon_size < best_sidon_size:
                best_sidon_size = sidon_size
                best_set = candidate
                best_relation_count = rel_count
            elif sidon_size == best_sidon_size and rel_count > best_relation_count:
                best_relation_count = rel_count
    
    # Strategy 2: Local search from best found
    if best_set is not None and best_sidon_size > 2:
        current = list(best_set)
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            # Random perturbation
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in current:
                continue
            
            current[idx] = new_val
            sidon_size = _max_sidon_size_efficient(current)
            rel_count = _count_additive_relations(current)
            
            if sidon_size < best_sidon_size:
                best_sidon_size = sidon_size
                best_set = list(current)
                best_relation_count = rel_count
            elif sidon_size == best_sidon_size and rel_count > best_relation_count:
                best_relation_count = rel_count
                # Keep this as a good candidate but don't necessarily commit
            else:
                current[idx] = old_val
    
    # Strategy 3: Try structured dense sets
    if time.time() < deadline - 10:
        # Dense clusters: group elements in tight clusters
        for cluster_gap in range(2, 8):
            for cluster_size in range(2, 5):
                base_positions = []
                for c in range(12 // cluster_size + 1):
                    base = c * cluster_gap * cluster_size
                    for i in range(cluster_size):
                        if len(base_positions) < 12:
                            base_positions.append(base + i)
                
                if len(base_positions) >= 12:
                    candidate = base_positions[:12]
                    if len(set(candidate)) != 12 or any(x < 0 or x > 128 for x in candidate):
                        continue
                    
                    key = tuple(sorted(candidate))
                    if key in evaluated_sets:
                        continue
                    evaluated_sets.add(key)
                    
                    rel_count = _count_additive_relations(candidate)
                    sidon_size = _max_sidon_size_efficient(candidate)
                    
                    if sidon_size < best_sidon_size:
                        best_sidon_size = sidon_size
                        best_set = candidate
                        best_relation_count = rel_count
    
    # Final fallback
    if best_set is None or len(set(best_set)) != 12 or any(x < 0 or x > 128 for x in best_set):
        # Return a known valid set
        return [0, 1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40]
    
    return best_set
# EVOLVE-BLOCK-END
