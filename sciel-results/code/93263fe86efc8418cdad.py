import random
from itertools import combinations
from math import gcd
import time

N = 128

def generate_bad_triples():
    """
    Generate all triples (a, b, c) with 1 <= a < b < c <= N
    such that gcd(a,b) == gcd(b,c) == gcd(a,c).
    """
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad_triples.append((a, b, c))
    return bad_triples

# Precompute
BAD_TRIPLES = generate_bad_triples()

# Precompute for each x, the list of pairs (y, z) such that (x, y, z) is a bad triple
# We store pairs as sorted tuples to ensure uniqueness if needed, but here we just need membership check.
# conflicts[x] = list of (y, z) where {x, y, z} is a bad triple.
CONFLICTS = {i: [] for i in range(1, N + 1)}
for a, b, c in BAD_TRIPLES:
    CONFLICTS[a].append((b, c))
    CONFLICTS[b].append((a, c))
    CONFLICTS[c].append((a, b))

def can_add(x, current_set):
    """
    Check if x can be added to current_set without creating a bad triple.
    """
    for u, v in CONFLICTS[x]:
        if u in current_set and v in current_set:
            return False
    return True

def greedy_remove(deadline):
    """
    Start with all numbers, remove elements to resolve conflicts.
    """
    S = set(range(1, N + 1))
    
    # Count how many bad triples each element is part of in the current S
    # Initially, this is the total number of bad triples containing that element.
    counts = {i: 0 for i in range(1, N + 1)}
    for a, b, c in BAD_TRIPLES:
        counts[a] += 1
        counts[b] += 1
        counts[c] += 1
    
    # We need to efficiently find a bad triple in S.
    # We can just iterate through BAD_TRIPLES.
    # To speed up removal, we maintain the counts.
    
    removed = set()
    
    # We iterate until no bad triples remain in S
    # This is a while loop that scans BAD_TRIPLES
    # Optimization: We can break early if we find a triple.
    
    while time.time() < deadline:
        # Find a bad triple in S
        found_triple = None
        for a, b, c in BAD_TRIPLES:
            if a in S and b in S and c in S:
                found_triple = (a, b, c)
                break
        
        if found_triple is None:
            break
            
        a, b, c = found_triple
        
        # Pick the element with the highest count among a, b, c
        # Tie-breaking: random to allow different paths
        candidates = [a, b, c]
        random.shuffle(candidates)
        # Sort by count descending, then random
        candidates.sort(key=lambda x: -counts[x])
        
        # Pick the first one (highest count)
        x_to_remove = candidates[0]
        
        S.remove(x_to_remove)
        
        # Update counts for other elements in triples involving x_to_remove
        # We need to decrement counts for y, z in (x_to_remove, y, z) if y, z are in S
        for y, z in CONFLICTS[x_to_remove]:
            if y in S and z in S:
                counts[y] -= 1
                counts[z] -= 1
            # If y or z was already removed, the triple is no longer "active" in S, 
            # so we don't need to decrement counts for the remaining one?
            # Wait. counts[y] represents the number of bad triples in S containing y.
            # If (x, y, z) was a bad triple in S, and we remove x, then (x, y, z) is no longer in S.
            # So counts[y] should decrease by 1 IF z was also in S.
            # If z was not in S, the triple (x, y, z) was not fully in S to begin with, 
            # so it wasn't counted in counts[y] as an "active" conflict?
            # My initial counts were for ALL bad triples.
            # I need to be careful.
            
    # The logic above for updating counts is slightly flawed if we don't track "active" status precisely.
    # Simpler approach: 
    # Just recompute counts? No, too slow.
    # Let's stick to the simple "Scan and Remove" without complex count updates first, 
    # as N=128 is small. 
    # Scanning 350k triples * 70 removals = 24M checks. 
    # Each check is 3 set lookups. 72M lookups. 
    # Python set lookup is ~50ns. 72M * 50ns = 3.6 seconds. 
    # This is acceptable.
    
    return S

def improve(S, deadline):
    """
    Try to add back elements.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
                improved = True
                # Restart the loop or continue?
                # Continuing is fine, but restarting might find more.
                # Let's just break and restart the while loop to re-shuffle/reevaluate.
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = set()
    
    # Run multiple iterations of Greedy Remove + Improve
    # Different random seeds for tie-breaking in greedy_remove
    
    num_trials = 20
    
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        # Reset random seed for reproducibility of trial? No, let it vary.
        # But we want diversity.
        
        S = greedy_remove(deadline)
        
        # Improve
        S = improve(S, deadline)
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
        # Optional: Local search with swaps?
        # Given the time, Greedy Remove + Add Back is likely sufficient.
        # We can also try starting from a different initial state, e.g. only even numbers.
        
    # Try starting from specific subsets
    # 1. All odds
    S_odds = set(x for x in range(1, N + 1) if x % 2 != 0)
    # Remove bad triples from odds
    # Re-use greedy_remove logic but on subset?
    # The current greedy_remove starts from ALL.
    # We can modify it to accept an initial set.
    
    # Let's create a more flexible greedy_remove
    
    def greedy_remove_from(initial_set, deadline):
        S = set(initial_set)
        # We don't need counts if we just scan.
        while time.time() < deadline:
            found_triple = None
            for a, b, c in BAD_TRIPLES:
                if a in S and b in S and c in S:
                    found_triple = (a, b, c)
                    break
            
            if found_triple is None:
                break
                
            a, b, c = found_triple
            # Randomly pick one to remove?
            # Or pick the one with most conflicts in S?
            # Let's pick randomly from the triple to explore different solutions.
            x_to_remove = random.choice([a, b, c])
            S.remove(x_to_remove)
            
        return S

    # Trial with All Odds
    if time.time() < deadline:
        S_odds = greedy_remove_from(S_odds, deadline)
        S_odds = improve(S_odds, deadline)
        if len(S_odds) > len(best_set):
            best_set = S_odds.copy()

    # Trial with All Evens
    if time.time() < deadline:
        S_evens = set(x for x in range(1, N + 1) if x % 2 == 0)
        S_evens = greedy_remove_from(S_evens, deadline)
        S_evens = improve(S_evens, deadline)
        if len(S_evens) > len(best_set):
            best_set = S_evens.copy()

    # Trial with Multiples of 3
    if time.time() < deadline:
        S_mult3 = set(x for x in range(1, N + 1) if x % 3 == 0)
        S_mult3 = greedy_remove_from(S_mult3, deadline)
        S_mult3 = improve(S_mult3, deadline)
        if len(S_mult3) > len(best_set):
            best_set = S_mult3.copy()

    return list(best_set)
