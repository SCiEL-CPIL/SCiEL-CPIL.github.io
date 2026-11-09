import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_system = None
    best_score = 16  # Max possible independence number
    
    # We will try to construct a system that is a Steiner Triple System (STS) on 16 points.
    # Since 16 is not of the form 6k+1 or 6k+3, a perfect STS doesn't exist.
    # However, we can get close. The goal is to maximize the number of pairs covered by triples.
    # A triple covers 3 pairs.
    # Total pairs = C(16, 2) = 120.
    # If we have T triples, they cover 3T pairs (if no overlaps in pairs).
    # Linearity constraint: any two triples share at most 1 point.
    # This implies no pair is repeated in two different triples.
    # So we are looking for a packing of triples.
    
    # Greedy approach:
    # 1. Start with empty set of triples.
    # 2. Maintain a set of "uncovered" pairs.
    # 3. At each step, find a triple (a,b,c) such that all 3 pairs (a,b), (b,c), (a,c) are uncovered.
    #    Among all valid candidates, pick the one that maximizes some heuristic (e.g., random, or covers points with high degree).
    #    Actually, since all valid triples cover exactly 3 new pairs, the "gain" is constant.
    #    The structure is determined by which triples we pick.
    #    To minimize independence number, we want the hypergraph to be as "dense" as possible.
    #    A greedy packing usually yields a good packing.
    
    # Let's try multiple random restarts of a greedy packing algorithm.
    # We can also try to repair or optimize the resulting system using local search (simulated annealing) 
    # but the directive asks for a structural change. Greedy construction is structurally different from 
    # "start with random system and anneal". It builds from scratch.
    
    def greedy_packing(random_seed):
        random.seed(random_seed)
        points = list(range(16))
        triples = []
        # Set of used pairs
        used_pairs = set()
        
        # Pre-compute all possible triples
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)
        
        # To make it more effective, we can prioritize triples that involve points with fewer connections?
        # Or just random order is a standard greedy.
        
        # A better greedy: 
        # At each step, look at all remaining valid triples.
        # If we just pick the first valid one in a shuffled list, it's a "random greedy".
        # Let's try to be slightly smarter: 
        # We want to avoid leaving isolated points.
        # But with 16 points and max ~16-20 triples, it's hard to leave many isolated.
        
        # Let's implement a simple greedy:
        # Shuffle all triples.
        # Iterate. If a triple is valid (no pair shared with existing), add it.
        # This produces a maximal packing.
        
        # To get variety, we can shuffle the points or the triples differently.
        
        valid_triples = []
        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (b, c) if b < c else (c, b)
            p3 = (a, c) if a < c else (c, a)
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                valid_triples.append(t)
        
        # This simple "filter once" doesn't work because adding a triple invalidates others.
        # We must iterate.
        
        triples = []
        used_pairs = set()
        
        # To improve on pure random greedy, we can use a priority queue or just re-shuffle remaining valid candidates?
        # Re-shuffling is expensive.
        # Let's just do random greedy:
        # Shuffle the list of all possible triples.
        # Iterate through the list.
        # If triple is valid, add it.
        
        # This is O(N^2) roughly, which is fine for N=560.
        
        shuffled_triples = list(combinations(points, 3))
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (b, c) if b < c else (c, b)
            p3 = (a, c) if a < c else (c, a)
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                triples.append(t)
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                
                # Check time
                if time.time() > deadline:
                    break
        
        return triples

    def calculate_independence_number(system):
        # Find the largest subset of points containing no triple from the system.
        # This is the maximum independent set in the 3-uniform hypergraph.
        # NP-hard, but n=16 is small.
        # We can use a branch and bound or simply iterate over subsets?
        # 2^16 = 65536 subsets. This is very fast.
        
        # Represent points as a bitmask.
        # A subset S is independent if no triple in `system` is fully contained in S.
        
        # Pre-compute bitmask for each triple
        triple_masks = []
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
        
        # Iterate through all subsets of 16 points
        # To find the MAXIMUM independent set, we can iterate by size descending?
        # Or just iterate all 2^16 and keep track of max size.
        # 65536 iterations is trivial.
        
        max_size = 0
        # Optimization: only check subsets of size > current max?
        # No, we need to find the max.
        # We can iterate k from 16 down to 0, and check if there is an independent set of size k.
        # If we find one, return k.
        
        # Generating combinations of size k is C(16, k).
        # Sum of C(16, k) for k=16 down to 0 is 2^16.
        
        # Let's just iterate all masks from 0 to 2^16 - 1.
        # If mask is independent, update max_size.
        
        for mask in range(1 << 16):
            # Check if mask contains any triple
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                # Count bits
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
        
        return max_size

    # Run multiple restarts of greedy packing
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:  # Leave 10s for evaluation
            break
            
        system = greedy_packing(i)
        
        # Evaluate
        score = calculate_independence_number(system)
        
        if score < best_score:
            best_score = score
            best_system = system
            
            # If we reach a very good score (e.g., 5 or 6), we might stop early?
            # The theoretical lower bound for independence number of a 3-uniform hypergraph
            # with high density is related to the codegree.
            # For an STS(15), alpha is 5. For 16 points, likely 5 or 6.
            if best_score <= 5:
                break

    # Fallback: if best_system is None or we didn't find a good one, return the seed?
    # The seed returns 5 disjoint triples, alpha=11 (16-5*3 + 5? No. 
    # Seed: [0,1,2], [3,4,5]...
    # Independent set can pick 1 from each triple + remaining 1 point = 5+1=6?
    # Wait, seed: 5 triples cover 15 points. 1 point left.
    # Pick 1 point from each triple: 5 points. Plus the 16th point. Total 6.
    # Can we pick 2 from a triple? No, that's not a triple itself, but the condition is "contains no triple".
    # So a set of 2 points from a triple is independent.
    # So for seed: Pick 2 from each of 5 triples = 10 points. Plus the 16th. Total 11.
    # So seed score is 11.
    
    if best_system is None:
        # Return seed behavior
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_system
# EVOLVE-BLOCK-END
