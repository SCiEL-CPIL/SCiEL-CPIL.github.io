import random
import time
from itertools import combinations

def construct():
    n = 16
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Generate all possible K(2,2,2) configurations
    # For each 6-set of vertices, there are 3 ways to partition into 3 pairs
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # For 6 vertices {a,b,c,d,e,f}, the 3 partitions are:
    # ({a,b},{c,d},{e,f}), ({a,c},{b,e},{d,f}), ({a,d},{b,f},{c,e})
    # Each partition gives 8 transversals
    k222_configs = []
    for six in combinations(range(n), 6):
        a, b, c, d, e, f = six
        partitions = [
            ((a,b), (c,d), (e,f)),
            ((a,c), (b,e), (d,f)),
            ((a,d), (b,f), (c,e)),
        ]
        for p1, p2, p3 in partitions:
            transversals = set()
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.add(t)
            if len(transversals) == 8:
                k222_configs.append(frozenset(transversals))
    
    # For each triple, find which K(2,2,2) configs it belongs to
    triple_to_configs = {t: [] for t in all_triples}
    for i, config in enumerate(k222_configs):
        for t in config:
            triple_to_configs[t].append(i)
    
    # Convert to lists for faster access
    config_list = [list(c) for c in k222_configs]
    
    # Simulated annealing
    best_score = 0
    best_triples = set()
    
    # Initial solution: random subset
    random.seed(42)
    current = set()
    # Start with a moderate random set
    for t in all_triples:
        if random.random() < 0.3:
            current.add(t)
    
    # Count conflicts for current solution
    config_counts = [0] * len(k222_configs)
    for t in current:
        for ci in triple_to_configs[t]:
            config_counts[ci] += 1
    
    def count_conflicts(counts):
        return sum(1 for c in counts if c >= 8)
    
    current_conflicts = count_conflicts(config_counts)
    current_size = len(current)
    current_score = current_size - 1000 * current_conflicts
    
    if current_score > best_score:
        best_score = current_score
        best_triples = set(current)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        if time.time() > deadline - 5:
            break
        
        # Try to add a random triple
        t = random.choice(all_triples)
        if t not in current:
            # Check if adding creates new conflicts
            new_conflicts = 0
            for ci in triple_to_configs[t]:
                if config_counts[ci] == 7:
                    new_conflicts += 1
            delta_score = 1 - 1000 * new_conflicts
            if delta_score > 0 or random.random() < pow(2.718281828, delta_score / T):
                current.add(t)
                for ci in triple_to_configs[t]:
                    config_counts[ci] += 1
                current_conflicts = count_conflicts(config_counts)
                current_size += 1
                current_score = current_size - 1000 * current_conflicts
                if current_score > best_score:
                    best_score = current_score
                    best_triples = set(current)
        else:
            # Try to remove a random triple from current
            if current:
                t = random.choice(list(current))
                # Check if removing fixes conflicts
                old_conflicts = current_conflicts
                current.remove(t)
                for ci in triple_to_configs[t]:
                    config_counts[ci] -= 1
                new_conflicts = count_conflicts(config_counts)
                delta_score = -1 + 1000 * (old_conflicts - new_conflicts)
                if delta_score > 0 or random.random() < pow(2.718281828, delta_score / T):
                    current_size -= 1
                    current_conflicts = new_conflicts
                    current_score = current_size - 1000 * new_conflicts
                    if current_score > best_score:
                        best_score = current_score
                        best_triples = set(current)
                else:
                    # Revert
                    current.add(t)
                    for ci in triple_to_configs[t]:
                        config_counts[ci] += 1
        
        # Periodic greedy improvement
        if len(current) % 100 == 0 and time.time() < deadline - 10:
            # Try to greedily add non-conflicting triples
            added_any = True
            while added_any and time.time() < deadline - 10:
                added_any = False
                for t in all_triples:
                    if t not in current:
                        can_add = True
                        for ci in triple_to_configs[t]:
                            if config_counts[ci] >= 8:
                                can_add = False
                                break
                        if can_add:
                            current.add(t)
                            for ci in triple_to_configs[t]:
                                config_counts[ci] += 1
                            current_size += 1
                            added_any = True
            current_conflicts = count_conflicts(config_counts)
            current_score = current_size - 1000 * current_conflicts
            if current_score > best_score:
                best_score = current_score
                best_triples = set(current)
        
        T *= cooling
    
    # Final greedy cleanup
    if time.time() < deadline - 2:
        # Remove conflicting triples greedily
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i, config in enumerate(config_list):
                if config_counts[i] >= 8:
                    # Remove a random triple from this config
                    candidates = [t for t in config if t in current]
                    if candidates:
                        t = random.choice(candidates)
                        current.remove(t)
                        for ci in triple_to_configs[t]:
                            config_counts[ci] -= 1
                        current_size -= 1
                        changed = True
                        break
        current_conflicts = count_conflicts(config_counts)
        current_score = current_size - 1000 * current_conflicts
        if current_score > best_score:
            best_score = current_score
            best_triples = set(current)
    
    # Return best valid solution (no conflicts)
    # If best has conflicts, try to find a conflict-free subset
    result = list(best_triples)
    
    # Verify and clean if needed
    if time.time() < deadline - 1:
        # Check for conflicts in best_triples
        final_counts = [0] * len(k222_configs)
        for t in best_triples:
            for ci in triple_to_configs[t]:
                final_counts[ci] += 1
        
        has_conflict = any(c >= 8 for c in final_counts)
        if has_conflict:
            # Greedily remove to eliminate conflicts
            current_clean = set(best_triples)
            clean_counts = [0] * len(k222_configs)
            for t in current_clean:
                for ci in triple_to_configs[t]:
                    clean_counts[ci] += 1
            
            while any(c >= 8 for c in clean_counts) and time.time() < deadline - 0.5:
                for i, config in enumerate(config_list):
                    if clean_counts[i] >= 8:
                        candidates = [t for t in config if t in current_clean]
                        if candidates:
                            t = random.choice(candidates)
                            current_clean.remove(t)
                            for ci in triple_to_configs[t]:
                                clean_counts[ci] -= 1
                            break
            
            if len(current_clean) > len(result):
                result = list(current_clean)
    
    return [list(t) for t in sorted(result)]
