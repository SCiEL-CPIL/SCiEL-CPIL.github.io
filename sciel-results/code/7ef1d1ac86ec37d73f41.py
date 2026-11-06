"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    # 255 = 3 * 5 * 17
    # We need a subset of [1, 255] containing 255 where all pairs have gcd > 1.
    # Since 255 = 3*5*17, every element in the set must share a factor with 255,
    # i.e., must be divisible by 3, 5, or 17.
    
    # The problem is equivalent to finding a maximum clique in the graph where
    # vertices are numbers in [1,255] that share a factor with 255, and edges
    # connect pairs with gcd > 1.
    
    # However, we can use a structural approach:
    # Partition candidates by their "type" based on which of {3,5,17} divide them.
    # Two numbers are compatible if they share at least one prime factor from {3,5,17}
    # OR if they share some other common factor.
    
    # Actually, let's think more carefully. Two numbers a, b have gcd(a,b) > 1 if
    # they share ANY prime factor, not just from {3,5,17}.
    
    # Let's use a branch-and-bound / greedy approach.
    # First, build the candidate set: all x in [1,255] such that gcd(x,255) > 1.
    
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    # This includes 255 itself.
    
    # For each candidate, compute its prime factors (for checking compatibility).
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute prime factors for all candidates
    factors_map = {x: prime_factors(x) for x in candidates}
    
    # Build compatibility: two numbers are compatible if they share at least one prime factor.
    # We want the largest subset where all pairs are compatible (i.e., a clique in the
    # compatibility graph).
    
    # Greedy approach: sort candidates by some heuristic, then greedily add.
    # But let's try a smarter approach using the structure.
    
    # Key insight: If we pick all multiples of 3, they all share factor 3, so they're pairwise compatible.
    # Similarly for multiples of 5, or 17.
    # But we can mix: a multiple of 3 and a multiple of 5 are compatible only if they share
    # another factor (like 15 divides both, or they share some other prime).
    
    # Actually, a multiple of 3 and a multiple of 5: gcd could be 1 if they're like 3 and 5.
    # So mixing is tricky.
    
    # Let's try: take all multiples of 3 in [1,255]. That's 85 numbers (3,6,9,...,255).
    # All share factor 3, so pairwise gcd >= 3 > 1. This is valid.
    
    # Can we do better? What if we take all multiples of 3 AND add some multiples of 5
    # that also share a factor with all multiples of 3? That's hard.
    
    # Alternative: use a maximum clique solver on a reduced graph.
    # The candidate set has ~85+51-15+17-... let's count.
    
    # Multiples of 3: floor(255/3) = 85
    # Multiples of 5: floor(255/5) = 51
    # Multiples of 17: floor(255/17) = 15
    # Inclusion-exclusion for union:
    # |A ∪ B ∪ C| = 85 + 51 + 15 - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|
    # A∩B: multiples of 15: floor(255/15) = 17
    # A∩C: multiples of 51: floor(255/51) = 5
    # B∩C: multiples of 85: floor(255/85) = 3
    # A∩B∩C: multiples of 255: 1
    # Union = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127
    
    # So 127 candidates. A maximum clique on 127 vertices is feasible with good pruning.
    
    # Let's build the adjacency structure using bitsets for speed.
    n = len(candidates)
    
    # Build adjacency: adj[i] is a bitmask of j such that candidates[i] and candidates[j] are compatible.
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if factors_map[candidates[i]] & factors_map[candidates[j]]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Maximum clique via branch and bound with greedy coloring upper bound.
    # This is the classic algorithm.
    
    best_clique = [0] * n  # bitmask of best clique found
    best_size = 0
    
    # Greedy coloring for upper bound
    def color_sort(vertices):
        """Greedy coloring, returns list of (vertex, color) sorted by color."""
        colors = {}
        for v in vertices:
            # Find smallest color not used by neighbors of v in the subgraph
            used = set()
            for u in vertices:
                if u != v and (adj[v] >> u) & 1:
                    if u in colors:
                        used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        # Sort by color
        return sorted(colors.items(), key=lambda x: x[1])
    
    def expand(clique_mask, candidates_mask):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if bin(clique_mask).count('1') > best_size:
                best_size = bin(clique_mask).count('1')
                best_clique = clique_mask
            return
        
        # Get list of candidate vertices
        verts = [i for i in range(n) if (candidates_mask >> i) & 1]
        
        if not verts:
            if bin(clique_mask).count('1') > best_size:
                best_size = bin(clique_mask).count('1')
                best_clique = clique_mask
            return
        
        # Greedy coloring upper bound
        colored = color_sort(verts)
        
        # Process in reverse order (highest color first)
        for i in range(len(colored) - 1, -1, -1):
            v, c = colored[i]
            
            # Upper bound: current clique size + c
            if bin(clique_mask).count('1') + c <= best_size:
                break
            
            # Try adding v
            new_clique = clique_mask | (1 << v)
            # New candidates: those in candidates_mask that are adjacent to v
            new_cands = candidates_mask & adj[v]
            # Remove v from candidates
            new_cands &= ~(1 << v)
            
            expand(new_clique, new_cands)
            
            # Remove v from candidates for subsequent iterations
            candidates_mask &= ~(1 << v)
    
    # Start with empty clique and all candidates
    all_mask = (1 << n) - 1
    expand(0, all_mask)
    
    # Extract the best clique
    result = [candidates[i] for i in range(n) if (best_clique >> i) & 1]
    
    # Ensure 255 is in the result (it should be, but let's verify)
    if 255 not in result:
        # If 255 wasn't in the best clique, force it and re-solve
        # For now, just add it and check validity
        result.append(255)
    
    result.sort()
    
    # Verify the result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # Invalid, fall back to multiples of 3
                return list(range(3, 256, 3))
    
    return result

# EVOLVE-BLOCK-END
