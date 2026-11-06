import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin
    
    N = 16
    
    def check_alpha(triples):
        """
        Returns the independence number of the hypergraph.
        Optimized for N=16.
        """
        # Map triples to bitmasks
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
        
        # If no triples, alpha is 16
        if not triple_masks:
            return N
            
        # We want the largest k such that there exists a subset of size k 
        # with no triple contained in it.
        # Equivalently, find the smallest k such that EVERY subset of size k 
        # contains a triple? No.
        # Alpha is the max size of an independent set.
        # Independent set S: no triple is fully contained in S.
        
        # Iterate k from N down to 1
        # Check if there exists a subset of size k that is independent.
        # A subset S is independent if for all t in triples, S does not contain t.
        # i.e., (mask_S & mask_t) != mask_t for all t.
        
        # To speed up, we can just check all subsets of size k.
        # Number of subsets of size k is C(16, k).
        # Max is C(16, 8) = 12870.
        # Total subsets to check if alpha is small: sum C(16, k) for k > alpha.
        
        for k in range(N, 0, -1):
            for subset in combinations(range(N), k):
                # Create mask for subset
                mask_s = 0
                for p in subset:
                    mask_s |= (1 << p)
                
                # Check if any triple is contained in mask_s
                independent = True
                for mask_t in triple_masks:
                    if (mask_s & mask_t) == mask_t:
                        independent = False
                        break
                
                if independent:
                    return k
        return 0

    def is_linear(triples):
        used_pairs = set()
        for t in triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            for p in pairs:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    return False
                used_pairs.add(p)
        return True

    best_triples = None
    best_alpha = N + 1
    
    num_restarts = 1000
    
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        random.seed(i * 101 + 7)
        
        # Greedy construction
        triples = []
        used_pairs = set()
        
        # Shuffle all possible triples
        all_triples = list(combinations(range(N), 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            conflict = False
            for p in pairs:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    conflict = True
                    break
            if not conflict:
                triples.append(list(t))
                for p in pairs:
                    used_pairs.add(p)
        
        if not triples:
            continue
            
        # Calculate alpha
        alpha = check_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
        # If we found the theoretical minimum (5), we can stop early?
        # We proved alpha >= 5.
        if best_alpha <= 5:
            break

    if best_triples is None:
        # Fallback: return a valid linear system (even if alpha is high)
        # Simple construction: disjoint triples
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    return best_triples
