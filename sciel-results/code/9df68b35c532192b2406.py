import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # 1. Generate STS(15)
    # We will use a known STS(15) or generate one.
    # Generating one via backtracking is fast.
    
    def generate_sts_15():
        points = list(range(15))
        pairs_to_cover = set(frozenset(p) for p in itertools.combinations(points, 2))
        triples = []
        
        def backtrack(remaining_pairs):
            if not remaining_pairs:
                return True
            if time.time() > deadline - 10.0:
                return False
                
            # Pick a pair with fewest options? Or just first?
            # For STS(15), backtracking is very fast.
            pair = next(iter(remaining_pairs))
            a, b = tuple(pair)
            
            candidates = []
            for c in range(15):
                if c == a or c == b:
                    continue
                t = frozenset([a, b, c])
                compatible = True
                for existing in triples:
                    if len(t & existing) > 1:
                        compatible = False
                        break
                if compatible:
                    p1 = frozenset([a, b])
                    p2 = frozenset([a, c])
                    p3 = frozenset([b, c])
                    if p1 in remaining_pairs and p2 in remaining_pairs and p3 in remaining_pairs:
                        candidates.append(t)
                        
            random.shuffle(candidates)
            
            for t in candidates:
                triples.append(t)
                p1, p2, p3 = [frozenset(p) for p in itertools.combinations(list(t), 2)]
                new_remaining = remaining_pairs - {p1, p2, p3}
                
                if backtrack(new_remaining):
                    return True
                triples.pop()
            return False
        
        if backtrack(pairs_to_cover):
            return [list(t) for t in triples]
        return None

    sts_15 = generate_sts_15()
    if not sts_15:
        # Fallback: disjoint triples
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    best_system = None
    best_alpha = 16
    
    # Map pairs to triple indices in sts_15
    pair_to_triple_idx = {}
    for i, t in enumerate(sts_15):
        for p in itertools.combinations(t, 2):
            pair_to_triple_idx[frozenset(p)] = i

    # Try different matchings of size k
    # Max matching size is 7.
    # We want to minimize alpha.
    
    for k in range(7, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Generate random matchings of size k
        # A matching is a set of disjoint pairs.
        # We can generate this by randomly selecting pairs and checking disjointness.
        
        attempts = 2000 if k == 7 else 5000
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Generate a random matching of size k from 15 points
            points = list(range(15))
            random.shuffle(points)
            matching = []
            used = set()
            for p in points:
                if len(matching) == k:
                    break
                if p in used:
                    continue
                # Find an available partner
                partners = [q for q in points if q > p and q not in used]
                if not partners:
                    continue
                partner = random.choice(partners)
                matching.append(frozenset([p, partner]))
                used.add(p)
                used.add(partner)
                
            if len(matching) < k:
                continue
                
            # Check if the triples corresponding to these pairs are distinct
            # They must be, because pairs are disjoint in a matching?
            # If pairs are disjoint, the triples in STS(15) containing them are distinct.
            # Proof: If T1 contains {a,b} and T2 contains {c,d} with {a,b} disjoint {c,d}, can T1=T2?
            # No, T1 has 3 points, T2 has 3 points. If they were the same triple, they would share points.
            # So the removed triples are distinct.
            
            removed_indices = set()
            for pair in matching:
                idx = pair_to_triple_idx[pair]
                removed_indices.add(idx)
                
            # Construct new system
            new_triples = [t for i, t in enumerate(sts_15) if i not in removed_indices]
            for pair in matching:
                a, b = tuple(pair)
                new_triples.append([15, a, b])
                
            # Calculate alpha
            # We need the largest independent set.
            # For 16 points, we can check if there is an independent set of size 8.
            # If not, alpha <= 7.
            # If yes, alpha >= 8.
            
            # To be precise, let's find the max independent set.
            # Since 16 is small, we can use a simple branch and bound or just check sizes.
            # Check if alpha <= 7 is equivalent to: no independent set of size 8.
            
            if has_independent_set_size_8(new_triples):
                # Alpha >= 8
                # We can try to calculate exact alpha if we want, but for score, 
                # we just need to know if it's better than best.
                # If best_alpha is 8, this is not better.
                # If best_alpha is 9, this is better.
                # Let's just store it if it's better than current best.
                # But calculating exact alpha is expensive.
                # We can estimate: if no set of size 8, alpha=7.
                # If set of size 8 exists, alpha >= 8.
                # Let's assume if we find a set of size 8, alpha is at least 8.
                # We want to minimize alpha.
                # So if we find a system with NO set of size 8, we have alpha=7.
                # This is the best possible (likely).
                
                pass
            else:
                # No independent set of size 8.
                # So alpha <= 7.
                # Since we have 16 points, and it's a linear system, alpha is likely 7.
                return new_triples
                
            # If we didn't return, it means alpha >= 8.
            # We can keep track of the best one (smallest alpha).
            # But without exact alpha, we can't compare easily.
            # However, the goal is to find alpha=7.
            # If we don't find it, we return the best we have.
            # Let's just return the first valid one if we don't find alpha=7 quickly?
            # No, we should search.
            
    # If no system with alpha=7 found, return the first valid one we found (alpha=8?)
    # Or just return a default.
    # Let's return the system with k=7 from the first attempt, which should have alpha=8 or 7.
    # Actually, the loop above returns immediately if alpha=7 is found.
    # If not, we need a fallback.
    
    # Let's just return the result of the first k=7 attempt, even if alpha=8.
    # It will score 1.375 which is better than 1.0.
    
    # Re-run one specific instance to ensure we have a result
    if time.time() < deadline - 5.0:
         # Use the last generated matching or just generate one
         points = list(range(15))
         random.shuffle(points)
         matching = []
         used = set()
         for p in points:
             if len(matching) == 7:
                 break
             if p in used:
                 continue
             partners = [q for q in points if q > p and q not in used]
             if not partners:
                 continue
             partner = random.choice(partners)
             matching.append(frozenset([p, partner]))
             used.add(p)
             used.add(partner)
             
         removed_indices = set()
         for pair in matching:
             idx = pair_to_triple_idx[pair]
             removed_indices.add(idx)
             
         new_triples = [t for i, t in enumerate(sts_15) if i not in removed_indices]
         for pair in matching:
             a, b = tuple(pair)
             new_triples.append([15, a, b])
             
         return new_triples

    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

def has_independent_set_size_8(triples):
    # Check if there is a subset of 8 points containing no triple
    triples_set = set(frozenset(t) for t in triples)
    points = range(16)
    
    # Optimization: If a point is in many triples, it's less likely to be in an independent set?
    # But 16 choose 8 is 12870. 12870 * (num_triples) is small.
    # Num triples is ~35.
    # 12870 * 35 = 450,000 operations. Very fast.
    
    for subset in itertools.combinations(points, 8):
        subset_f = frozenset(subset)
        independent = True
        for t in triples_set:
            if t.issubset(subset_f):
                independent = False
                break
        if independent:
            return True
    return False
