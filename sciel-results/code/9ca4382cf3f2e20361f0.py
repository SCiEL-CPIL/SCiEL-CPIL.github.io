# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden for the difference check
    # Condition: (a-b) not in F and (b-a) not in F.
    # Let d = (a-b) % m. Then (b-a) % m = (-d) % m.
    # So we need d not in F and (-d)%m not in F.
    # This is equivalent to d not in (F union -F).
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
        
    # Remove 0 just in case, though it shouldn't be in F
    forbidden_sym.discard(0)
    
    # Precompute conflict mask for faster checks
    # conflict[i] is a bitmask of elements that conflict with i
    # Actually, let's just precompute a boolean matrix or a list of sets for speed.
    # Since m=205, a 205x205 boolean matrix is small.
    # conflict[i][j] = True if (i-j) % m is in forbidden_sym
    conflict = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden_sym:
                conflict[i][j] = True
            else:
                conflict[i][j] = False
                # Note: conflict is symmetric in the sense that if i conflicts with j, 
                # j conflicts with i because forbidden_sym is symmetric.
                # Let's verify: d in forbidden_sym implies (-d)%m in forbidden_sym.
                # So conflict[i][j] == conflict[j][i].

    def is_valid_addition(S_set, x):
        """Check if x can be added to set S_set using precomputed conflicts"""
        for s in S_set:
            if conflict[x][s]:
                return False
        return True

    def greedy_construct(order):
        """Construct a set greedily based on the given order of candidates"""
        S = []
        S_set = set()
        for x in order:
            if is_valid_addition(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def local_search(S, S_set):
        """
        Local search with Add and Remove+Add moves.
        Uses precomputed conflicts for speed.
        Returns improved set.
        """
        improved = True
        while improved:
            improved = False
            # Phase 1: Try to add any possible element
            # Iterate through all candidates not in S
            for x in range(m):
                if x not in S_set:
                    if is_valid_addition(S_set, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
            
            # Phase 2: If no additions, try removing one element and refilling
            if not improved:
                # Try removing each element
                for u in list(S):
                    # Copy current state to allow safe trial
                    S_copy = list(S)
                    S_set_copy = set(S_set)
                    
                    # Remove u
                    S_copy.remove(u)
                    S_set_copy.remove(u)
                    
                    # Refill greedily: try to add any x not in S_set_copy
                    # We iterate through all candidates to maximize fill
                    # Since m is small, this is cheap
                    for x in range(m):
                        if x not in S_set_copy:
                            if is_valid_addition(S_set_copy, x):
                                S_copy.append(x)
                                S_set_copy.add(x)
                    
                    # If the new set is strictly larger, accept it
                    if len(S_copy) > len(S):
                        S = S_copy
                        S_set = S_set_copy
                        improved = True
                        
        return S, S_set

    best_set = []
    best_set_set = set()
    
    # Deadline guard: 630s limit, keep safety margin
    deadline = time.time() + 610.0
    
    # Randomized Greedy + Local Search with restarts
    # We use multiple random orderings to escape local optima
    
    # Optimization: Pre-shuffled lists can be reused or generated quickly.
    # Also, we can try different random seeds or just rely on time.
    
    for i in range(10000): # Upper bound on restarts, but limited by time
        if time.time() > deadline:
            break
        
        # Create a random order for greedy construction
        order = list(range(m))
        random.shuffle(order)
        
        # 1. Greedy construction
        S, S_set = greedy_construct(order)
        
        # 2. Local search to improve
        S, S_set = local_search(S, S_set)
        
        # 3. Update best if better
        if len(S) > len(best_set):
            best_set = S
            best_set_set = S_set
            
            # Early stopping heuristic? 
            # Theoretical max is unknown but likely small.
            # If we find a very large set, we might stop, but let's run until deadline.
            
    # Fallback: if no set found (should not happen), return [0]
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
