import random
import time
import itertools
from collections import defaultdict

def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    points = list(range(16))
    
    def is_linear(triples, new_triple):
        """Check if adding new_triple maintains linearity."""
        new_set = set(new_triple)
        for t in triples:
            overlap = len(new_set & set(t))
            if overlap > 1:
                return False
        return True
    
    def calc_alpha(triples):
        """Calculate independence number: largest set with no triple contained."""
        triple_sets = [frozenset(t) for t in triples]
        
        # Try subsets from largest to smallest
        n = 16
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(points, size):
                subset_set = frozenset(subset)
                has_triple = False
                for ts in triple_sets:
                    if ts <= subset_set:
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 0
    
    def greedy_random(max_triples=40, time_limit=None):
        """Greedy randomized search for linear triple system."""
        triples = []
        used_pairs = set()
        
        # Pre-generate all possible triples
        all_triples = list(itertools.combinations(range(16), 3))
        
        while len(triples) < max_triples:
            if time_limit and time.time() > time_limit:
                break
            
            random.shuffle(all_triples)
            added = False
            for t in all_triples:
                t_set = set(t)
                t_pairs = set(itertools.combinations(sorted(t), 2))
                
                # Check if any pair is already used
                if any(p in used_pairs for p in t_pairs):
                    continue
                
                # Check linearity with existing triples
                linear = True
                for existing in triples:
                    if len(t_set & set(existing)) > 1:
                        linear = False
                        break
                
                if linear:
                    triples.append(list(t))
                    used_pairs.update(t_pairs)
                    added = True
                    break
            
            if not added:
                break
        
        return triples
    
    # Initial greedy solution
    best_triples = greedy_random()
    best_alpha = calc_alpha(best_triples)
    
    # Try multiple random restarts with different strategies
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Vary the approach
        strategy = restart % 3
        
        if strategy == 0:
            # Standard greedy
            triples = greedy_random()
        elif strategy == 1:
            # Greedy with bias toward covering uncovered points
            triples = []
            used_pairs = set()
            uncovered = set(range(16))
            
            while uncovered:
                if time.time() > deadline:
                    break
                
                # Find points that appear in fewest triples
                point_count = defaultdict(int)
                for t in triples:
                    for p in t:
                        point_count[p] += 1
                
                # Pick a point with minimum coverage
                candidates = sorted(uncovered, key=lambda p: point_count.get(p, 0))
                random.shuffle(candidates)
                
                added = False
                for center in candidates[:3]:  # Try top 3 undercovered points
                    # Find good triples containing center
                    possible = []
                    for a in range(16):
                        for b in range(a+1, 16):
                            if a == center or b == center:
                                continue
                            t = sorted([center, a, b])
                            t_set = set(t)
                            t_pairs = set(itertools.combinations(t, 2))
                            
                            if any(p in used_pairs for p in t_pairs):
                                continue
                            
                            linear = True
                            for existing in triples:
                                if len(t_set & set(existing)) > 1:
                                    linear = False
                                    break
                            
                            if linear:
                                possible.append(t)
                    
                    if possible:
                        random.shuffle(possible)
                        t = possible[0]
                        triples.append(t)
                        used_pairs.update(set(itertools.combinations(t, 2)))
                        uncovered.difference_update(t)
                        added = True
                        break
                
                if not added:
                    break
            
        else:
            # Randomized greedy with different shuffle
            triples = greedy_random()
        
        alpha = calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # Final polish: try local improvements
    if time.time() < deadline - 5:
        # Try to add more triples to reduce alpha
        used_pairs = set()
        for t in best_triples:
            used_pairs.update(set(itertools.combinations(sorted(t), 2)))
        
        all_triples = list(itertools.combinations(range(16), 3))
        random.shuffle(all_triples)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for t in all_triples:
                if time.time() > deadline - 2:
                    break
                
                t_set = set(t)
                t_pairs = set(itertools.combinations(sorted(t), 2))
                
                if any(p in used_pairs for p in t_pairs):
                    continue
                
                linear = True
                for existing in best_triples:
                    if len(t_set & set(existing)) > 1:
                        linear = False
                        break
                
                if linear:
                    new_triples = best_triples + [list(t)]
                    new_alpha = calc_alpha(new_triples)
                    if new_alpha <= best_alpha:
                        best_triples = new_triples
                        best_alpha = new_alpha
                        used_pairs.update(t_pairs)
                        improved = True
                        break
    
    return best_triples
