import itertools
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Greedy Heuristic Construction
    # We build the word character by character.
    # State: current tail (last n-1 chars), set of covered permutations.
    
    # Initialize with a random or fixed permutation to break symmetry
    # Fixed is better for determinism
    start_perm = tuple(range(1, n+1))
    word = list(start_perm)
    covered = {start_perm}
    tail = tuple(word[1:]) # Last n-1 chars
    
    total_perms = int(1)
    for i in range(2, n+1):
        total_perms *= i
        
    # Pre-compute all permutations for fast lookup? 
    # No, just check membership in 'covered'.
    # But we need to know if a candidate perm is already covered.
    
    # To speed up, we can pre-generate all perms and map them to IDs?
    # Or just use the tuple itself in the set.
    
    # Heuristic:
    # 1. Count how many times each perm has been seen? No, just covered/uncov.
    # 2. For each next char c in 1..n:
    #    a. Check if c is in tail. If yes, skip (invalid perm).
    #    b. new_tail = tail[1:] + (c,)
    #    c. new_perm = tail + (c,)
    #    d. gain = 0 if new_perm in covered else 1
    #    e. If gain == 1, it's a "good" move.
    #    f. Among good moves, pick one. Tie-breaker: 
    #       - Prefer moves that lead to a tail with more "potential" (e.g., more uncovered perms reachable in 1 step).
    #       - Or just random among good moves to avoid loops?
    #       - Or prefer moves that keep the "distribution" of tail symbols balanced?
    
    # Let's use a simple tie-breaker: 
    # If multiple moves have gain=1, pick the one that minimizes the "regret" of the next step?
    # Or just pick the first one.
    
    # To avoid infinite loops or getting stuck, we can add a penalty if we've been in a state before?
    # State is (tail, frozenset(covered)). Too big.
    # Just (tail) is not enough because covered changes.
    
    # For n=7, greedy usually works.
    
    # Optimization: Pre-compute valid next chars for each tail?
    # tail is a tuple of n-1 distinct ints.
    # Valid next chars are those not in tail.
    
    while len(covered) < total_perms:
        best_move = None
        best_score = -1
        candidates = []
        
        for c in range(1, n+1):
            if c in tail:
                continue
            
            new_perm = tail + (c,)
            gain = 0
            if new_perm not in covered:
                gain = 1
            
            # Heuristic score
            # We prefer gain=1.
            # If gain=0, we must take it if no gain=1 available? 
            # Actually, if no gain=1 is available, we are stuck in a loop of covered perms.
            # But in a superpermutation, we can always find a new one until all are covered?
            # Not necessarily, we might enter a cycle of covered perms.
            # However, the graph is strongly connected.
            
            # Let's score:
            # Primary: gain (1 is better than 0)
            # Secondary: Lookahead 1 step.
            #   If we take c, new_tail = tail[1:] + (c,)
            #   How many uncovered perms are reachable from new_tail in 1 step?
            #   Reachable perms are new_tail + (x,) for x not in new_tail.
            #   Count how many of those are NOT in covered.
            
            nt = tail[1:] + (c,)
            potential = 0
            for x in range(1, n+1):
                if x in nt:
                    continue
                next_perm = nt + (x,)
                if next_perm not in covered:
                    potential += 1
            
            score = gain * 100 + potential
            
            if score > best_score:
                best_score = score
                best_move = c
        
        if best_move is None:
            # Should not happen
            break
            
        word.append(best_move)
        new_perm = tail + (best_move,)
        covered.add(new_perm)
        tail = tail[1:] + (best_move,)
        
    return "".join(map(str, word))
