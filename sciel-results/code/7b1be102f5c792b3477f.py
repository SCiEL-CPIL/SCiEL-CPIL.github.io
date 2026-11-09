"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-element sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def construct():
    deadline = time.time() + 85  # 5s safety margin under 90s wall-clock
    best_family = []
    
    # All 4-subsets of [16], represented as frozensets for hashing
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_total = len(all_sets)
    
    # Precompute pairwise sunflower check helper:
    # For a new candidate s, it conflicts with existing pair (a,b) if
    # (a & b) == (a & s) == (b & s)
    
    # Greedy randomized restarts to find a large sunflower-free family
    best_size = 0
    best_result = []
    
    for restart in range(50):
        if time.time() > deadline - 1:
            break
        
        # Randomize order of all sets
        order = list(range(n_total))
        random.shuffle(order)
        
        chosen = []
        # For fast conflict checking, maintain a set of (intersection, frozenset) pairs?
        # Actually, for each pair in chosen, we need to check if adding s creates a sunflower.
        # A sunflower with s and two existing sets a,b requires:
        #   a & b == a & s == b & s
        # Let I = a & b. Then we need a & s == I and b & s == I.
        # Equivalently: s contains I, and s contains no other elements of a\I or b\I that would make intersections larger.
        # More precisely: a & s == I means s ∩ (a \ I) = ∅ and I ⊆ s.
        # Similarly b & s == I means s ∩ (b \ I) = ∅ and I ⊆ s.
        # So: I ⊆ s, and s ∩ (a\I) = ∅, and s ∩ (b\I) = ∅.
        # Combined: I ⊆ s ⊆ [16] \ (a\I) \ (b\I) = [16] \ (a ∪ b \ I)
        # Note a ∪ b \ I = (a \ I) ∪ (b \ I) ∪ I = a ∪ b.
        # Wait: a\I and b\I are disjoint from I, so (a\I) ∪ (b\I) ∪ I = a ∪ b.
        # So the condition is: I ⊆ s ⊆ [16] \ (a ∪ b \ I) ... let me re-derive.
        # a & s = I means: I ⊆ s and s ∩ (a \ I) = ∅.
        # b & s = I means: I ⊆ s and s ∩ (b \ I) = ∅.
        # So: I ⊆ s and s ∩ ((a\I) ∪ (b\I)) = ∅.
        # (a\I) ∪ (b\I) = (a ∪ b) \ I.
        # So s must satisfy: I ⊆ s ⊆ [16] \ ((a ∪ b) \ I).
        # The available elements for s beyond I are: [16] \ (a ∪ b).
        # So s must be a 4-element set containing I, with remaining (4 - |I|) elements chosen from [16] \ (a ∪ b).
        
        # For efficiency, for each pair (a,b) in chosen, compute I = a & b.
        # If |I| > 4, skip (impossible since sets are size 4, so |I| ≤ 4).
        # The number of forbidden s for this pair is: C(16 - |a ∪ b|, 4 - |I|) if 4 - |I| ≤ 16 - |a ∪ b|, else 0.
        # But we just need to check if a specific s is forbidden.
        
        # For each candidate s, check against all pairs in chosen. This is O(|chosen|^2) per candidate.
        # To speed up, we can maintain for each possible intersection I (a subset of [16] of size ≤ 4),
        # the set of pairs (a,b) that have a & b = I.
        # Then for candidate s, for each I ⊆ s with |I| ≤ 4, check if there's a pair with intersection I
        # such that s ⊆ [16] \ ((a ∪ b) \ I) for that pair.
        
        # Simpler approach: for each candidate s, iterate over all pairs in chosen.
        # With |chosen| up to ~250, pairs up to ~31k, and 1820 candidates, that's ~56M checks per restart.
        # Each check is a few set ops. Might be too slow.
        
        # Better: maintain a dictionary mapping intersection frozenset I to list of pairs (a,b) with a&b=I.
        # For candidate s, for each I that is a subset of s (with |I| ≤ 4), look up pairs with that I.
        # For each such pair (a,b), check if s ∩ ((a\I) ∪ (b\I)) = ∅.
        
        # Number of subsets I of s: at most C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 1+4+6+4+1 = 16.
        # For each I, the number of pairs with that intersection could be large.
        
        # Let's use a different approach: maintain for each pair (i,j) in chosen (by index),
        # precompute a&b. When adding s, check all pairs.
        
        # Given the small size, let's just do the O(|chosen|^2) check per candidate but optimize the inner loop.
        
        chosen = []
        # Map from frozenset intersection to list of indices of pairs with that intersection
        # Actually, let's just store chosen and do direct checks.
        
        for idx in order:
            if time.time() > deadline - 1:
                break
            s = all_sets[idx]
            conflict = False
            for i in range(len(chosen)):
                a = chosen[i]
                for j in range(i+1, len(chosen)):
                    b = chosen[j]
                    if _is_sunflower_triple(a, b, s):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                chosen.append(s)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_result = [sorted(s) for s in chosen]
        
        # Early stop if we hit the theoretical max or a good threshold
        if best_size >= 250:
            break
    
    return best_result
# EVOLVE-BLOCK-END
