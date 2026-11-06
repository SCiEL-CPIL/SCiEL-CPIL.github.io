import time
import random
from itertools import combinations, product

def construct():
    """
    Find a large triple system on 16 points with no K(2,2,2).
    A K(2,2,2) is formed by three disjoint pairs {a,a'}, {b,b'}, {c,c'}
    and all 8 transversal triples {x,y,z} where x in {a,a'}, y in {b,b'}, z in {c,c'}.
    """
    random.seed(12345)
    start_time = time.time()
    deadline = 140  # 20s margin
    
    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    total_triples = len(all_triples)  # 560
    
    # Precompute: for each triple, which K(2,2,2)s it participates in
    # A K(2,2,2) is defined by 3 disjoint pairs. 
    # Number of ways to choose 3 disjoint pairs from 16 points:
    # Choose 6 points, partition into 3 pairs: C(16,6) * 15 = 8008 * 15 = 120120
    # But each K(2,2,2) has exactly 8 triples.
    
    # Better approach: precompute for each triple which other triples would complete a K(2,2,2)
    # For triple (a,b,c), a K(2,2,2) containing it requires:
    # - a' != a, a' not in {b,c}
    # - b' != b, b' not in {a,c}  
    # - c' != c, c' not in {a,b}
    # - {a',b',c'} all distinct
    # Then the 8 triples are: (a,b,c), (a',b,c), (a,b',c), (a,b,c'), 
    #                         (a',b',c), (a',b,c'), (a,b',c'), (a',b',c')
    
    # Map each triple to list of K(2,2,2) indices it belongs to
    # We'll assign each K(2,2,2) a unique index and track which triples are present
    
    # Generate all K(2,2,2) structures
    # A K(2,2,2) is determined by 3 disjoint pairs: ({a,a'}, {b,b'}, {c,c'})
    # The 8 triples are all combinations picking one from each pair
    
    k222_list = []  # Each entry is a frozenset of 8 triples
    triple_to_k222 = {t: [] for t in all_triples}
    
    # Efficient generation: pick 6 distinct points, partition into 3 ordered pairs
    # But we need to avoid duplicates. Each K(2,2,2) can be represented uniquely
    # by the unordered set of 3 unordered pairs.
    
    seen_k222 = set()
    
    for combo in combinations(points, 6):
        # Partition 6 points into 3 pairs: 15 ways
        for p1 in combo:
            for p2 in combo:
                if p2 <= p1:
                    continue
                for p3 in combo:
                    if p3 <= p2:
                        continue
                    # We have pairs (p1,p2), and 4 remaining, need to split into 2 pairs
                    remaining = [x for x in combo if x != p1 and x != p2]
                    remaining.sort()
                    r1, r2, r3, r4 = remaining
                    # Ways to partition 4 elements into 2 pairs: 3 ways
                    # (r1,r2),(r3,r4) | (r1,r3),(r2,r4) | (r1,r4),(r2,r3)
                    for (r_a, r_b, r_c, r_d) in [(r1,r2,r3,r4), (r1,r3,r2,r4), (r1,r4,r2,r3)]:
                        pairs = frozenset([frozenset({p1,p2}), frozenset({r_a,r_b}), frozenset({r_c,r_d})])
                        if pairs in seen_k222:
                            continue
                        seen_k222.add(pairs)
                        
                        # Generate the 8 transversal triples
                        triples_in_k222 = set()
                        for x in p1, p2:
                            for y in r_a, r_b:
                                for z in r_c, r_d:
                                    triple = tuple(sorted((x, y, z)))
                                    triples_in_k222.add(triple)
                        
                        if len(triples_in_k222) == 8:
                            k222_frozen = frozenset(triples_in_k222)
                            idx = len(k222_list)
                            k222_list.append(k222_frozen)
                            for t in triples_in_k222:
                                if t in triple_to_k222:
                                    triple_to_k222[t].append(idx)
    
    num_k222 = len(k222_list)
    
    # k222_counts[i] = number of triples from k222_list[i] currently in the system
    k222_counts = [0] * num_k222
    
    def is_valid_add(triple, S, k222_counts):
        """Check if adding triple to S would create a K(2,2,2)."""
        if triple in S:
            return False
        for k_idx in triple_to_k222[triple]:
            if k222_counts[k_idx] == 7:
                # This would be the 8th triple in a K(2,2,2)
                return False
        return True
    
    def add_triple(triple, S, k222_counts):
        """Add triple to system, updating counts."""
        S.add(triple)
        for k_idx in triple_to_k222[triple]:
            k222_counts[k_idx] += 1
    
    def remove_triple(triple, S, k222_counts):
        """Remove triple from system, updating counts."""
        S.remove(triple)
        for k_idx in triple_to_k222[triple]:
            k222_counts[k_idx] -= 1
    
    def greedy_build(order):
        """Greedy construction following given order."""
        S = set()
        k222_counts = [0] * num_k222
        for triple in order:
            if is_valid_add(triple, S, k222_counts):
                add_triple(triple, S, k222_counts)
        return S, k222_counts
    
    def local_search(S, k222_counts, max_iters=50):
        """Local search with remove-readd moves."""
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            
            if time.time() > deadline - 5:
                break
            
            # Try removing one triple and re-adding blocked ones
            triples_list = list(S)
            random.shuffle(triples_list)
            
            for t_remove in triples_list:
                if time.time() > deadline - 5:
                    break
                
                # Remove t_remove
                remove_triple(t_remove, S, k222_counts)
                
                # Try to add back all triples that were blocked by t_remove
                added_back = 0
                candidates = [t for t in all_triples if t not in S]
                random.shuffle(candidates)
                
                for t_add in candidates:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_add(t_add, S, k222_counts):
                        add_triple(t_add, S, k222_counts)
                        added_back += 1
                
                # If we added back more than we removed, keep the change
                if added_back > 1:
                    improved = True
                else:
                    # Revert
                    add_triple(t_remove, S, k222_counts)
        
        return S, k222_counts
    
    # Best solution found
    best_set = set()
    best_k222_counts = None
    best_size = 0
    
    # Multiple restarts with different random orders
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random order
        order = list(all_triples)
        random.shuffle(order)
        
        S, k222_counts = greedy_build(order)
        
        # Local search improvement
        S, k222_counts = local_search(S, k222_counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_k222_counts = k222_counts[:]
    
    # Final verification and return
    result = [list(t) for t in best_set]
    return result
