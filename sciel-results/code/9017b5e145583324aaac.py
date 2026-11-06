import random
import time
import itertools

def calc_alpha(triples):
    """
    Calculate the independence number of the 3-uniform hypergraph defined by triples.
    n=16.
    """
    n = 16
    # Map triple to set for fast lookup
    triple_sets = [set(t) for t in triples]
    
    # Check subsets of size k from n down to 1
    # Optimization: if we find one, return k.
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            s = set(subset)
            is_independent = True
            for t_set in triple_sets:
                if t_set.issubset(s):
                    is_independent = False
                    break
            if is_independent:
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_triples = None
    best_alpha = 16
    
    # Initial trivial solution
    trivial = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    best_triples = trivial
    best_alpha = calc_alpha(trivial)
    
    # Local search / Randomized construction
    for _ in range(10000): # Limit iterations, break on time
        if time.time() > deadline:
            break
            
        # Generate a random linear 3-graph
        used_pairs = set()
        triples = []
        
        # Shuffle points to break symmetry
        points = list(range(16))
        random.shuffle(points)
        
        # Try to add as many triples as possible
        # We can use a greedy approach with random restarts inside the generation
        for _ in range(100): # Try 100 times to add a triple
            a, b, c = random.sample(points, 3)
            p1 = (min(a,b), max(a,b))
            p2 = (min(a,c), max(a,c))
            p3 = (min(b,c), max(b,c))
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                triples.append([a,b,c])
                
        if not triples:
            continue
            
        alpha = calc_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 5: # 5 is a strong lower bound candidate
                break

        # Local improvement: try to swap or add
        # This part is complex, let's rely on many random restarts first.
        # If we find a good one, we can try to improve it.
        if alpha < 8:
            # Try to improve this specific instance
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Try adding a new random triple
                a, b, c = random.sample(range(16), 3)
                p1 = (min(a,b), max(a,b))
                p2 = (min(a,c), max(a,c))
                p3 = (min(b,c), max(b,c))
                
                # Check linearity with existing triples
                valid = True
                for p in [p1, p2, p3]:
                    if p in used_pairs:
                        valid = False
                        break
                
                if valid:
                    new_triples = triples + [[a,b,c]]
                    new_used_pairs = used_pairs | {p1, p2, p3}
                    new_alpha = calc_alpha(new_triples)
                    if new_alpha < alpha:
                        alpha = new_alpha
                        triples = new_triples
                        used_pairs = new_used_pairs
                        if alpha < best_alpha:
                            best_alpha = alpha
                            best_triples = triples
                            if best_alpha <= 5:
                                break
    return best_triples
