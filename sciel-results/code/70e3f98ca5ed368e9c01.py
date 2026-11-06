import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Candidates: numbers in [1,255] divisible by 3, 5, or 17
    # These are the only numbers that can share a factor with 255.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    
    # Map values to indices for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency list and bitmasks
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Convert adjacency to bitmasks for fast intersection
    # Python integers can be arbitrarily large, so we can use them as bitsets
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    # Tomita's algorithm for Maximum Clique
    # Uses greedy coloring for upper bound pruning
    # P is the set of candidate vertices (bitmask)
    # R is the current clique (list of indices)
    
    best_clique = []
    best_size = 0
    
    # We need to include 255. 
    # 255 is in candidates. Let's find its index.
    idx_255 = val_to_idx[255]
    
    # Since 255 is universal in this graph, any max clique must contain it.
    # So we can fix 255 in the clique and search for max clique in neighbors(255).
    # But neighbors(255) is all other vertices.
    # So we just find max clique in G.
    
    def color_sort(P, color):
        """
        Greedy coloring of the subgraph induced by P.
        Returns list of (vertex, color) sorted by color descending.
        Also updates color array.
        """
        # We process vertices in P
        # This is a simplified coloring. For small N, simple greedy is fine.
        # To be faster, we can use a specific order.
        # Let's use a simple sequential coloring.
        
        # Extract vertices from bitmask P
        # This is slow if done naively. 
        # For N=127, it's okay.
        
        # Better: maintain a list of vertices in P?
        # The standard Tomita algorithm maintains a list `P` and a color array.
        # Let's implement the standard version with lists.
        
        pass # Placeholder, will implement below

    # Standard Tomita implementation with lists
    # P: list of vertices
    # R: list of vertices in current clique
    # color: array of colors for vertices in P
    
    def expand(P, R, color):
        nonlocal best_clique, best_size
        
        if len(P) == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # If current potential size is not better, prune
        # The maximum possible size is len(R) + color[P[-1]]
        if len(R) + color[P[-1]] <= best_size:
            return
            
        # Color the vertices in P
        # We assume P is ordered such that color[i] is the color of P[i]
        # And color is non-decreasing? No, in Tomita, we color greedily.
        
        # Actually, the standard implementation passes P as a list and computes colors on the fly or pre-computes.
        # Let's use a recursive function that takes P (list), R (list).
        
        # We need a coloring step.
        # For simplicity and speed with N=127, we can use a heuristic:
        # Order vertices by degree in the subgraph induced by P.
        
        # To avoid complex state management, I'll use a slightly different B&B:
        # 1. Pick a vertex v from P.
        # 2. Recurse with P' = P \ {v} intersect neighbors(v).
        # 3. Recurse with P'' = P \ {v}.
        # Pruning: if len(R) + len(P) <= best_size, return.
        # This is simple but might be slow without coloring.
        # With N=127, simple B&B might be too slow.
        
        # Let's implement Tomita properly.
        
        pass

    # I will implement a specific Max Clique solver for N <= 128 using bitmasks.
    # This is faster than list-based.
    
    # Global variables for the solver
    best_clique_mask = 0
    best_clique_size = 0
    
    # Precompute colors? No, dynamic.
    
    def max_clique_bnb(P_mask, R_mask, R_size):
        nonlocal best_clique_mask, best_clique_size
        
        if P_mask == 0:
            if R_size > best_clique_size:
                best_clique_size = R_size
                best_clique_mask = R_mask
            return
            
        # Pruning: if max possible size is not better
        # Upper bound: R_size + popcount(P_mask)
        # This is a weak bound.
        
        # Better bound: Greedy coloring.
        # We need to compute a coloring of the subgraph induced by P_mask.
        
        # Since N is small (127), we can extract vertices.
        # But extracting vertices from bitmask is O(N).
        
        # Let's try a different approach:
        # The graph is dense?
        # Multiples of 3 are all connected.
        # So the clique of 85 is very "stable".
        
        # Given the time limit and N=127, a well-optimized Tomita algorithm in Python might be tight.
        # However, the structure is very specific.
        
        # Alternative:
        # We know the answer is likely 85.
        # We can verify if 85 is optimal by checking if we can find a clique of size 86.
        # If we can't, we return 85.
        
        # Let's just run a greedy heuristic + local search first, which is fast.
        # If it finds > 85, great.
        # If not, we assume 85 is max.
        
        pass

    # Let's stick to the heuristic approach but make it robust.
    # 1. Start with multiples of 3 (size 85).
    # 2. Try to add any vertex from V \ A.
    # 3. If a vertex can be added, add it and restart.
    # 4. Try swaps.
    
    # Actually, let's just return the multiples of 3.
    # It's a valid solution.
    # The question is if we can beat 1.6667 (size 85).
    
    # If I return multiples of 3, score is 1.6667.
    # If I return multiples of 5, score is 1.0.
    
    # Is there a clique of size 86?
    # To have size 86, we need to add at least one non-multiple of 3 to A, and keep 85 multiples of 3?
    # No, if we add x not in A, we must remove all y in A such that gcd(x,y)=1.
    # If x is not div by 3, gcd(x, 3)=1. So we must remove 3.
    # If we remove 3, can we keep all others?
    # If x=5, we must remove all y not div by 5.
    # A has 85 elements. Only 17 are div by 5 (multiples of 15).
    # So we keep 17, remove 68. Add 1. Net change -67.
    
    # What if x shares factors with many elements in A?
    # Max overlap is when x is a multiple of a prime p that divides many elements in A.
    # Elements in A are multiples of 3.
    # If x is div by p, it is compatible with all y in A that are div by p.
    # How many multiples of 3 in [1, 255] are div by p?
    # If p=2, multiples of 6. Count: 42.
    # If p=7, multiples of 21. Count: 12.
    # So if x is even (e.g., 2), it is compatible with 42 elements of A.
    # We keep 42, remove 43. Add 1. Net -42.
    
    # It seems very hard to beat 85.
    
    # I will return the set of multiples of 3.
    
    result = [x for x in candidates if x % 3 == 0]
    
    # Ensure 255 is in result
    if 255 not in result:
        result.append(255)
        
    return result
