import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    N = 180
    
    # Precompute conflicts: a -> list of (b, c) such that {a, b, c} is forbidden
    # a is the smallest element.
    conflicts = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find divisors of a^2
        # u * v = a^2
        # b = a + u, c = a + v
        # b, c must be > a, distinct, <= N
        # u != v
        # To ensure b < c, we need u < v.
        # u < sqrt(a^2) = a.
        for u in range(1, a):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b > N or c > N:
                    continue
                if b == c:
                    continue
                # b < c is guaranteed by u < v
                conflicts[a].append((b, c))
                
    # Helper to check if y can be added to S
    def can_add(y, S):
        S_set = S
        for b, c in conflicts[y]:
            if b in S_set and c in S_set:
                return False
        return True

    # Initialization: Start with the "safe" large set
    # We proved no triples exist with min element >= 90.
    S = set(range(90, N + 1))
    
    # Greedy Addition of small numbers
    # Sort small numbers by number of conflicts? Or just 1..89?
    # Smaller numbers have more potential conflicts (more divisors of a^2 potentially fitting).
    # Actually, let's just try all.
    for a in range(1, 90):
        if can_add(a, S):
            S.add(a)
            
    # Also try to add any large numbers we might have missed? No, we started with all.
    # But what if removing a large number allows adding two small ones?
    
    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    # We will use a randomized kick strategy.
    # To avoid infinite loops or wasting time, we track time.
    
    # Precompute list of all numbers for random choice
    all_numbers = list(range(1, N + 1))
    
    while time.time() < deadline:
        current_size = len(S)
        
        # Phase 1: Add any free elements
        improved = False
        # Iterate through numbers not in S
        # To make it faster, we can just loop 1..180
        for y in all_numbers:
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
        
        if improved:
            continue
        
        # Phase 2: Kick
        # If we couldn't add anything, we need to remove something to free up space.
        # Choose a random element to remove.
        if not S:
            break
            
        # Heuristic: Pick an element that is in many conflicts?
        # For simplicity, random.
        x = random.choice(list(S))
        S.remove(x)
        
        # After removing x, try to add any free elements
        kick_improved = False
        for y in all_numbers:
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    kick_improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                        
        # If we didn't improve size, we might revert to best_S to stay in a good basin
        # Or we might keep the current S if size is same, to explore.
        # If size decreased, revert.
        if len(S) < best_size:
            S = set(best_S)
        # If len(S) == best_size, we keep S (might be different structure)
        # If len(S) > best_size, already updated best_S and S is new best.

    return list(best_S)
