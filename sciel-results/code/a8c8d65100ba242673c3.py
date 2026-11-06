import random
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    # Generate all triples
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # For each triple, find which K(2,2,2) configs it participates in
    # A K(2,2,2) is defined by 3 disjoint pairs (a,b), (c,d), (e,f)
    # The 8 transversal triples are: (a,c,e), (a,c,f), (a,d,e), (a,d,f),
    #                                 (b,c,e), (b,c,f), (b,d,e), (b,d,f)
    
    # Precompute all K(2,2,2) configurations
    all_configs = []
    for ab in combinations(range(n), 2):
        for cd in combinations(range(n), 2):
            if set(ab) & set(cd):
                continue
            for ef in combinations(range(n), 2):
                if set(ab) & set(ef) or set(cd) & set(ef):
                    continue
                # Generate the 8 transversal triples
                a, b = ab
                c, d = cd
                e, f = ef
                config_triples = [
                    (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                    (b, c, e), (b, c, f), (b, d, e), (b, d, f)
                ]
                # Sort each triple for consistent lookup
                config_triples = [tuple(sorted(t)) for t in config_triples]
                all_configs.append(config_triples)
    
    num_configs = len(all_configs)
    
    # For each triple, which configs does it belong to?
    triple_in_configs = [[] for _ in range(num_triples)]
    for ci, config in enumerate(all_configs):
        for t in config:
            triple_in_configs[triple_to_idx[t]].append(ci)
    
    def is_valid(triple_set):
        """Check if no K(2,2,2) is fully contained in triple_set."""
        for config in all_configs:
            if all(t in triple_set for t in config):
                return False
        return True
    
    def count_violations(triple_set):
        """Count how many K(2,2,2) configs are fully present."""
        count = 0
        for config in all_configs:
            if all(t in triple_set for t in config):
                count += 1
        return count
    
    def greedy_min_removal():
        """Start with all triples, greedily remove to break all configs."""
        present = set(range(num_triples))
        # Count how many configs each triple is in
        config_count = [0] * num_configs
        for t in range(num_triples):
            for ci in triple_in_configs[t]:
                config_count[ci] += 1
        
        # Greedily remove triples that break the most configs
        removed = set()
        while config_count.count(8) > 0:  # while there are fully-present configs
            if time.time() > deadline:
                break
            # Find triple that breaks most fully-present configs
            best_t = -1
            best_breaks = 0
            for t in range(num_triples):
                if t in removed:
                    continue
                breaks = 0
                for ci in triple_in_configs[t]:
                    if config_count[ci] == 8:
                        breaks += 1
                if breaks > best_breaks:
                    best_breaks = breaks
                    best_t = t
            if best_t == -1 or best_breaks == 0:
                break
            removed.add(best_t)
            for ci in triple_in_configs[best_t]:
                config_count[ci] -= 1
        
        return list(present - removed)
    
    def local_improve(triples_list):
        """Try to add back removed triples or swap to improve."""
        present = set(triples_list)
        removed = set(range(num_triples)) - present
        
        # Recompute config counts
        config_count = [0] * num_configs
        for t in present:
            for ci in triple_in_configs[t]:
                config_count[ci] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding back a removed triple if it doesn't create a violation
            for t in list(removed):
                would_violate = False
                for ci in triple_in_configs[t]:
                    if config_count[ci] == 7:  # adding t would make it 8
                        would_violate = True
                        break
                if not would_violate:
                    present.add(t)
                    removed.discard(t)
                    for ci in triple_in_configs[t]:
                        config_count[ci] += 1
                    improved = True
                    break
            
            # Try swaps: remove one present, add one removed
            if not improved:
                for t_remove in list(present):
                    if time.time() > deadline:
                        return list(present)
                    # Temporarily remove t_remove
                    for ci in triple_in_configs[t_remove]:
                        config_count[ci] -= 1
                    
                    for t_add in list(removed):
                        would_violate = False
                        for ci in triple_in_configs[t_add]:
                            if config_count[ci] >= 7:  # after adding, would be >=8
                                would_violate = True
                                break
                        if not would_violate:
                            # Check if this swap is beneficial (it maintains size, 
                            # but might enable future additions)
                            present.discard(t_remove)
                            present.add(t_add)
                            removed.add(t_remove)
                            removed.discard(t_add)
                            for ci in triple_in_configs[t_add]:
                                config_count[ci] += 1
                            improved = True
                            break
                    
                    # Restore if no swap found
                    if not improved:
                        for ci in triple_in_configs[t_remove]:
                            config_count[ci] += 1
                    else:
                        break
        
        return list(present)
    
    def greedy_with_randomness(seed):
        """Greedy construction with random tie-breaking."""
        rng = random.Random(seed)
        present = set(range(num_triples))
        
        config_count = [0] * num_configs
        for t in range(num_triples):
            for ci in triple_in_configs[t]:
                config_count[ci] += 1
        
        # Greedily remove triples
        while any(c == 8 for c in config_count):
            if time.time() > deadline:
                break
            candidates = []
            for t in range(num_triples):
                if t not in present:
                    continue
                breaks = 0
                for ci in triple_in_configs[t]:
                    if config_count[ci] == 8:
                        breaks += 1
                if breaks > 0:
                    candidates.append((breaks, t))
            
            if not candidates:
                break
            
            # Pick randomly among top candidates
            max_breaks = max(c[0] for c in candidates)
            top = [c[1] for c in candidates if c[0] == max_breaks]
            t_remove = rng.choice(top)
            
            present.discard(t_remove)
            for ci in triple_in_configs[t_remove]:
                config_count[ci] -= 1
        
        return list(present)
    
    # Try multiple strategies
    best_result = None
    best_size = 0
    
    # Strategy 1: Greedy min removal + local improve
    for attempt in range(5):
        if time.time() > deadline:
            break
        triples = greedy_min_removal()
        triples = local_improve(triples)
        if len(triples) > best_size:
            best_size = len(triples)
            best_result = triples
    
    # Strategy 2: Randomized greedy + local improve
    for seed in range(20):
        if time.time() > deadline:
            break
        triples = greedy_with_randomness(seed)
        triples = local_improve(triples)
        if len(triples) > best_size:
            best_size = len(triples)
            best_result = triples
    
    if best_result is None:
        # Fallback: star construction
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    # Convert to list of lists
    return [list(t) for t in best_result]
