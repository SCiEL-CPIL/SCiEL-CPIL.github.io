import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    def compute_independence_number(triples):
        """Find the largest subset of points containing no full triple."""
        # Convert triples to a set of frozensets for fast lookup
        triple_set = set(frozenset(t) for t in triples)
        
        # Check subsets from largest to smallest
        # For n=16, we can iterate by size
        for size in range(n, 0, -1):
            for subset in itertools.combinations(points, size):
                fs = frozenset(subset)
                # Check if any triple is fully contained
                found = False
                for t in triple_set:
                    if t <= fs:
                        found = True
                        break
                if not found:
                    return size
        return 0
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                shared = set(triples[i]) & set(triples[j])
                if len(shared) > 1:
                    return False
        return True
    
    def greedy_build(seed):
        """Build a linear hypergraph greedily with a given seed."""
        rng = random.Random(seed)
        used_pairs = set()
        triples = []
        
        # Try to add triples, prioritizing those that cover under-covered points
        point_coverage = [0] * n
        all_triples = list(itertools.combinations(range(n), 3))
        rng.shuffle(all_triples)
        
        for t in all_triples:
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            # Check linearity: no pair can be in two triples
            if any(p in used_pairs for p in pairs):
                continue
            # Add the triple
            triples.append(list(t))
            for p in pairs:
                used_pairs.add(p)
            point_coverage[a] += 1
            point_coverage[b] += 1
            point_coverage[c] += 1
        
        return triples
    
    def local_search(triples, rng):
        """Improve the hypergraph via local search."""
        best_triples = [list(t) for t in triples]
        best_indep = compute_independence_number(best_triples)
        
        all_triples = list(itertools.combinations(range(n), 3))
        used_pairs = set()
        for t in best_triples:
            a, b, c = t
            used_pairs.add((min(a,b), max(a,b)))
            used_pairs.add((min(a,c), max(a,c)))
            used_pairs.add((min(b,c), max(b,c)))
        
        improvements = 0
        while time.time() < deadline - 5:
            improved = False
            # Try to swap a triple with another
            for i in range(len(best_triples)):
                if time.time() > deadline - 5:
                    break
                old_triple = best_triples[i]
                # Remove old triple's pairs
                a, b, c = old_triple
                old_pairs = [(min(a,b), max(a,b)), (min(a,c), max(a,c)), (min(b,c), max(b,c))]
                for p in old_pairs:
                    used_pairs.discard(p)
                
                # Try to find a better triple
                candidates = []
                for t in all_triples:
                    ta, tb, tc = t
                    tpairs = [(min(ta,tb), max(ta,tb)), (min(ta,tc), max(ta,tc)), (min(tb,tc), max(tb,tc))]
                    if any(p in used_pairs for p in tpairs):
                        continue
                    candidates.append(t)
                
                if not candidates:
                    # Restore old triple
                    for p in old_pairs:
                        used_pairs.add(p)
                    continue
                
                # Try a few random candidates
                rng.shuffle(candidates)
                tried = 0
                for t in candidates:
                    if tried >= 20:
                        break
                    tried += 1
                    ta, tb, tc = t
                    tpairs = [(min(ta,tb), max(ta,tb)), (min(ta,tc), max(ta,tc)), (min(tb,tc), max(tb,tc))]
                    
                    # Tentatively add
                    for p in tpairs:
                        used_pairs.add(p)
                    new_triples = best_triples[:i] + [list(t)] + best_triples[i+1:]
                    new_indep = compute_independence_number(new_triples)
                    
                    if new_indep < best_indep:
                        best_triples = new_triples
                        best_indep = new_indep
                        improved = True
                        improvements += 1
                        break
                    else:
                        # Revert
                        for p in tpairs:
                            used_pairs.discard(p)
                
                if not improved:
                    # Restore old triple if we didn't replace it
                    current = best_triples[i]
                    if list(current) != list(old_triple):
                        pass  # already replaced
                    else:
                        for p in old_pairs:
                            used_pairs.add(p)
            
            if not improved:
                break
        
        return best_triples, best_indep
    
    best_result = None
    best_score = n + 1
    
    # Try multiple seeds
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        
        triples = greedy_build(seed)
        if len(triples) == 0:
            continue
        
        indep = compute_independence_number(triples)
        
        if indep < best_score:
            best_score = indep
            best_result = triples
        
        # Local search on promising candidates
        if indep <= best_score and time.time() < deadline - 20:
            rng = random.Random(seed + 10000)
            improved_triples, improved_indep = local_search(triples, rng)
            
            if improved_indep < best_score:
                best_score = improved_indep
                best_result = improved_triples
    
    if best_result is None:
        # Fallback: the original
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_result
