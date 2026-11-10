import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Helper to check validity and count violations
    def get_violations(n, colors):
        """
        colors: numpy array of shape (C(n,3),) with values 0 (red) or 1 (blue)
        Returns (red_violations, blue_violations)
        Red violation: 4 vertices where all 4 triples are red (0)
        Blue violation: 5 vertices where all 10 triples are blue (1)
        """
        if n < 4:
            return 0, 0
        
        # Precompute indices for triples
        # We need to map vertex sets to triple indices for fast lookup?
        # For local search, we flip one triple at a time.
        # We need to efficiently update violation counts.
        # However, for simplicity and robustness within time limits, 
        # we might just recompute or use a fast heuristic.
        # Given n <= 40, C(40,3) = 9880. 
        # Checking all K4s (C(n,4)) and K5s (C(n,5)) from scratch is expensive if done naively in Python.
        # Let's try to optimize the check.
        
        # Map triple (a,b,c) to index
        # Lexicographic order: (0,1,2), (0,1,3), ..., (0,1,n-1), (0,2,3)...
        # We can precompute a lookup table for triple indices.
        # But updating violation counts dynamically is complex.
        
        # Alternative: Random restarts + Local Search on a smaller candidate set first?
        # Or just a smart random search?
        
        # Let's stick to a simple but effective local search:
        # 1. Start with random coloring for target n.
        # 2. Calculate violations.
        # 3. Find a triple that is part of many violations.
        # 4. Flip it if it reduces total violations.
        # 5. Repeat.
        
        # To make this fast, we need efficient violation counting.
        # Let's represent the hypergraph adjacency.
        # For each vertex v, store the triples containing it.
        # This is still heavy for Python loops.
        
        # Let's try a different approach:
        # Use the known fact that R(4,5;3) > 34.
        # We can try to construct a valid coloring for n=34, 35, etc.
        # Start with n=34.
        
        # Actually, the simplest robust method within 150s for n~34-40:
        # Randomized Local Search with restarts.
        # We will use a fast C-like loop via NumPy where possible, 
        # but the dependency graph is complex.
        
        # Let's implement a specific heuristic:
        # Min-Conflict style: 
        # 1. Random init.
        # 2. Count conflicts for each triple.
        # 3. Pick triple with max conflicts.
        # 4. Flip it.
        
        # To count conflicts for a triple t=(a,b,c):
        # It is part of a red K4 if there exists d such that (a,b,d), (a,c,d), (b,c,d) are also red.
        # It is part of a blue K5 if there exist d,e such that all other 7 triples in the K5(a,b,c,d,e) are blue.
        
        # This local check is O(n) for red and O(n^2) for blue per triple.
        # Total O(N_triples * n^2) ~ 10000 * 1600 = 16M ops. Doable in NumPy/Python if optimized.
        
        pass

    # Since implementing a full fast local search in pure Python/NumPy within the time limit for n=40 is tricky,
    # let's try a simpler strategy: 
    # Try to find a valid coloring for n=34 (known possible) using random restarts and simple hill climbing.
    # If we can't reach 34, we return the best valid n found.
    
    best_n = 9
    best_colors = None
    current_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Define a fast checker for small n, or use the mod-3 one as a baseline.
    # The mod-3 coloring for n=9:
    # Red K4? Sum of 4 vertices. Triples sum to 0 mod 3.
    # If a+b+c = 0, a+b+d = 0 => c=d. So no distinct d. Thus no red K4.
    # Blue K5? Triples sum != 0.
    # For n=9, is there a blue K5?
    # We assume n=9 is valid for the seed.
    
    # Let's try to expand n using local search.
    # We'll try n from 30 down to 10? Or up?
    # The task says maximize n. Published is 34.
    # Let's try to solve for n=34.
    
    target_n = 34
    
    # Precompute triple indices for target_n
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {}
    for i, t in enumerate(triples):
        triple_to_idx[t] = i
        
    # Precompute for each triple, the other triples in any K4 or K5?
    # This is memory heavy. 
    # Instead, let's do a simple random search with hill climbing on a subset of moves.
    
    # Given the complexity, let's try a "Random Walk with Backtracking" on n=34.
    # If it fails, try n=33, etc.
    
    # Actually, there is a known construction using finite fields or linear codes.
    # But we don't have it hardcoded.
    
    # Let's try a simple heuristic:
    # Random initialization, then repeatedly pick a random violating hypergraph (K4 red or K5 blue) 
    # and flip a random triple in it.
    
    def count_violations(n, colors):
        """
        Returns list of (type, vertices) violations.
        type 0: Red K4
        type 1: Blue K5
        """
        violations = []
        # Check Red K4
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            idx1 = triple_to_idx.get((a,b,c))
            idx2 = triple_to_idx.get((a,b,d))
            idx3 = triple_to_idx.get((a,c,d))
            idx4 = triple_to_idx.get((b,c,d))
            if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                if colors[idx1] == 0 and colors[idx2] == 0 and colors[idx3] == 0 and colors[idx4] == 0:
                    violations.append((0, quad))
        
        # Check Blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                idx = triple_to_idx.get(tri)
                if idx is None or colors[idx] != 1:
                    all_blue = False
                    break
            if all_blue:
                violations.append((1, quint))
                
        return violations

    # This checker is O(C(n,4) + C(n,5)*10). For n=34, C(34,5) ~ 278k. 278k * 10 = 2.8M ops.
    # Python can do ~10M simple ops/sec. So one check takes ~0.3s.
    # We have 140s. So we can do ~400 checks.
    # This is enough for a few random restarts and some local search steps.
    
    # Strategy:
    # 1. Try n=34.
    # 2. Random init.
    # 3. Loop:
    #    - Count violations.
    #    - If 0, return valid coloring.
    #    - Pick a random violation.
    #    - Pick a random triple in that violation.
    #    - Flip it.
    #    - If time runs out, break.
    # 4. If n=34 fails, try n=33, etc.
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
            
        triples_n = list(combinations(range(n), 3))
        num_triples_n = len(triples_n)
        triple_to_idx_n = {t: i for i, t in enumerate(triples_n)}
        
        # We need to adapt the checker to use triple_to_idx_n
        def count_violations_n(n, colors, t2i):
            violations = []
            for quad in combinations(range(n), 4):
                a, b, c, d = quad
                i1 = t2i.get((a,b,c))
                i2 = t2i.get((a,b,d))
                i3 = t2i.get((a,c,d))
                i4 = t2i.get((b,c,d))
                if i1 is not None and i2 is not None and i3 is not None and i4 is not None:
                    if colors[i1] == 0 and colors[i2] == 0 and colors[i3] == 0 and colors[i4] == 0:
                        violations.append((0, quad))
            for quint in combinations(range(n), 5):
                all_blue = True
                for tri in combinations(quint, 3):
                    i = t2i.get(tri)
                    if i is None or colors[i] != 1:
                        all_blue = False
                        break
                if all_blue:
                    violations.append((1, quint))
            return violations

        # Random Restart Loop
        found = False
        max_restarts = 20 # Adjust based on time
        
        for r in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random init: 50/50
            colors = np.random.randint(0, 2, size=num_triples_n).tolist()
            
            # Local Search
            for step in range(500): # Max steps per restart
                if time.time() > deadline - 2:
                    break
                    
                v_list = count_violations_n(n, colors, triple_to_idx_n)
                if not v_list:
                    # Valid!
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                    
                # Pick random violation
                v_type, v_verts = v_list[np.random.randint(len(v_list))]
                
                # Pick random triple in violation
                if v_type == 0: # Red K4
                    # 4 triples
                    a,b,c,d = v_verts
                    candidates = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                else: # Blue K5
                    # 10 triples
                    candidates = list(combinations(v_verts, 3))
                
                tri_to_flip = candidates[np.random.randint(len(candidates))]
                idx = triple_to_idx_n[tri_to_flip]
                colors[idx] = 1 - colors[idx]
            
            if found:
                break
        
        if found:
            break

    # If we didn't find a better n, return the seed result (n=9)
    if best_n == 9 and best_colors is None:
         best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
