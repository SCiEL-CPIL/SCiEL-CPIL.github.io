"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Weight-4 vectors in R^11: norm^2 = 4
    # Condition: min pairwise dist^2 >= max norm^2 = 4
    # For weight-4 vectors: dist^2 = 8 - 2*u.v >= 4 => u.v <= 2
    # 
    # Conflict condition for distinct weight-4 vectors:
    # They share exactly 3 coordinates AND have the same sign on all 3 shared coordinates.
    #
    # Strategy: Greedy with triple-sign tracking
    # Each vector has 4 triples (support minus one element).
    # Track (triple, sign_pattern) counts. A new vector conflicts if any of its
    # (triple, sign_pattern) pairs already has count > 0.
    
    # Generate all weight-4 vectors
    all_vecs = []
    vec_triples = []
    
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            all_vecs.append(v)
            
            # Precompute the 4 triples and their sign patterns
            triples = []
            for i in range(4):
                T = support[:i] + support[i+1:]
                signs_on_T = tuple(signs[j] for j in range(4) if j != i)
                triples.append((T, signs_on_T))
            vec_triples.append(triples)
    
    n_vecs = len(all_vecs)
    indices = list(range(n_vecs))
    
    best_set = []
    best_count = 0
    
    np.random.seed(42)
    
    # Try multiple random orderings
    n_attempts = 500
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        np.random.shuffle(indices)
        
        triple_sign_count = defaultdict(int)
        current_set = []
        
        for idx in indices:
            triples = vec_triples[idx]
            
            # Check if any triple has a conflict
            conflict = False
            for T, signs_on_T in triples:
                if triple_sign_count[(T, signs_on_T)] > 0:
                    conflict = True
                    break
            
            if not conflict:
                current_set.append(all_vecs[idx])
                for T, signs_on_T in triples:
                    triple_sign_count[(T, signs_on_T)] += 1
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
    
    # Also try a few deterministic orders if time permits
    if time.time() < deadline:
        for order_seed in range(5):
            if time.time() > deadline:
                break
            # Different deterministic orderings
            if order_seed == 0:
                ordered = list(range(n_vecs))
            elif order_seed == 1:
                ordered = list(reversed(range(n_vecs)))
            else:
                # Sort by some hash
                rng = np.random.RandomState(order_seed * 123 + 456)
                ordered = list(rng.permutation(n_vecs))
            
            triple_sign_count = defaultdict(int)
            current_set = []
            
            for idx in ordered:
                triples = vec_triples[idx]
                
                conflict = False
                for T, signs_on_T in triples:
                    if triple_sign_count[(T, signs_on_T)] > 0:
                        conflict = True
                        break
                
                if not conflict:
                    current_set.append(all_vecs[idx])
                    for T, signs_on_T in triples:
                        triple_sign_count[(T, signs_on_T)] += 1
            
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set
    
    if not best_set:
        # Fallback: return at least one vector
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
