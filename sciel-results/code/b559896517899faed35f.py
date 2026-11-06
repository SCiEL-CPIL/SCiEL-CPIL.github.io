"""
Strategy: Conflict-Graph Based Local Search (Max-Independent-Set heuristic)
Instead of single-bit flips with full validity checks, we use a "remove-and-refill" 
macro-move. We identify elements in the current set that have the highest "conflict degree"
(i.e., they prevent the most other numbers from being added). We remove one such element,
then greedily refill the set by adding any number that doesn't conflict with the current set.
This allows exploring larger neighborhoods and escaping local optima more effectively than SA.
"""
import time
import random
import numpy as np

N = 512

# Precompute conflict information
# For each number x in 1..N, store the list of pairs (y, z) such that 
# {x, y, z, w} forms a 4-AP. Actually, it's easier to store for each x, 
# the set of numbers that CANNOT coexist with x if two others are present.
# Let's precompute for each x, the list of "forbidden pairs" (u, v) such that 
# if u and v are in the set, x cannot be added.
# A 4-AP is a, a+d, a+2d, a+3d.
# If we consider x, it can be the 1st, 2nd, 3rd, or 4th term.
# 
# Let's precompute for each x (0-indexed), a list of pairs (i, j) such that 
# if S contains i and j, then x cannot be in S (because it would complete a 4-AP).
# This is equivalent to finding all 4-APs containing x and removing x to get the other 3.
# If x is in a 4-AP {a, b, c, d}, then:
# - If x=a, then (b,c) and (b,d) and (c,d) are "conflict pairs"? 
#   No, if b and c are present, adding a creates {a,b,c,?} - wait, we need 4 terms.
#   If b and c are present, and we add a, we have {a,b,c}. We need the 4th term d.
#   So the conflict is: if {b, c, d} are present, x=a cannot be added.
#   This is a triple, not a pair. 
#
# Let's stick to the bitset approach for speed but optimize the "can_add" check.
# Precompute for each x, the list of "forbidden triples" (u, v, w) such that 
# if u, v, w are in the set, x cannot be added.
# There are at most 3 * (N/3) ~ 500 such triples for each x.
# Checking "can_add(x)" involves iterating through these triples and checking 
# if all three are in the current set (using bit operations).
# This is O(500) per check. With N=512, a full greedy pass is 512*500 = 256k ops.
# A macro-move (remove 1, refill) involves ~500 checks.
# We can do ~10,000 macro-moves in 60s if each is ~1ms.
# 10,000 * 500 * 500 ops? No, the check is O(1) bit ops if we use Python ints as bitsets.
# Python ints are arbitrary precision, so (bits >> i) & 1 is fast.

# Precompute forbidden triples for each x
# forbidden[x] = list of (u, v, w) such that {x, u, v, w} is a 4-AP
# Indices 0..N-1 correspond to numbers 1..N

def precompute_forbidden():
    forbidden = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: a, b, c, e
            # If we have b, c, e, we cannot add a
            forbidden[a - 1].append((b - 1, c - 1, e - 1))
            # If we have a, c, e, we cannot add b
            forbidden[b - 1].append((a - 1, c - 1, e - 1))
            # If we have a, b, e, we cannot add c
            forbidden[c - 1].append((a - 1, b - 1, e - 1))
            # If we have a, b, c, we cannot add e
            forbidden[e - 1].append((a - 1, b - 1, c - 1))
    return forbidden

FORBIDDEN = precompute_forbidden()

def can_add(bits, x):
    """
    Check if x (0-indexed) can be added to the set represented by bits.
    Returns True if no forbidden triple is fully present in bits.
    """
    for u, v, w in FORBIDDEN[x]:
        if (bits >> u) & 1 and (bits >> v) & 1 and (bits >> w) & 1:
            return False
    return True

def count_conflicts(bits, x):
    """
    Count how many forbidden triples for x are fully present in bits.
    This represents the "degree" of conflict: how many 4-APs would be created 
    if we were to add x (assuming x is not already in the set).
    Wait, if x IS in the set, this counts how many 4-APs x is part of.
    We want to remove elements that are part of many 4-APs? 
    No, we want to remove elements that *block* many other elements.
    Actually, if an element x is in the set, it blocks any y such that 
    there exist u,v in the set such that {x,y,u,v} is a 4-AP.
    This is complex to count directly.
    
    Alternative heuristic: 
    1. Start with a greedy set.
    2. Calculate for each element x in the set, the number of "near-miss" 4-APs 
       it is involved in. 
    3. Or simpler: Just pick a random element to remove? 
       The "remove-and-refill" works best if we remove an element that is "low value" 
       or "high conflict". 
       
    Let's use a simpler structural move: 
    1. Greedy initialization.
    2. Loop:
       - Pick a random element r from the current set.
       - Remove r.
       - Try to add r back? No.
       - Try to add ANY element not in the set, in random order, that fits.
       - This is just "remove one, refill greedily".
       
    To improve, we should remove elements that are "expensive".
    An element x is expensive if removing it allows many new elements to be added.
    We can estimate this by: 
    For each x in set, count how many y NOT in set would become valid if x were removed.
    This is O(N^2) per step, too slow.
    
    Let's stick to random removal + greedy refill. The randomness explores the space.
    """
    pass

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Greedy Initialization
    # We will use multiple random greedy starts and keep the best.
    best_bits = 0
    best_size = 0
    
    # Greedy helper
    def greedy_init():
        bits = 0
        # Random order
        order = list(range(N))
        random.shuffle(order)
        for x in order:
            if can_add(bits, x):
                bits |= (1 << x)
        return bits

    # Run a few greedy inits to find a good starting point
    for _ in range(20):
        b = greedy_init()
        s = bin(b).count('1')
        if s > best_size:
            best_size = s
            best_bits = b
            
    current_bits = best_bits
    current_size = best_size
    
    # 2. Local Search: Remove-and-Refill
    # We will iterate:
    # a. Pick a random element from the current set.
    # b. Remove it.
    # c. Greedily add any element (in random order) that can be added.
    # d. If the new size is larger, accept. If smaller, reject (or accept with prob).
    
    # To make it faster, we can pre-generate a random order for the refill phase.
    # Or just shuffle a list of all elements not in the set? 
    # Shuffling 512 elements is fast.
    
    # Optimization: 
    # Instead of full greedy refill (which might be slow if many checks fail),
    # we can just try to add the removed element back? No, that's trivial.
    # We want to find a DIFFERENT set.
    
    # Let's use a "kick" strategy:
    # Remove 1 element. Then scan all elements NOT in the set (in random order) 
    # and add them if possible.
    
    # Precompute list of indices
    indices = list(range(N))
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a random element to remove
        # We should bias towards elements that are "less useful".
        # For now, random.
        if current_size == 0:
            break
            
        # Get list of set elements
        set_elements = [i for i in range(N) if (current_bits >> i) & 1]
        if not set_elements:
            break
            
        remove_idx = random.choice(set_elements)
        
        # Remove it
        old_bits = current_bits
        current_bits &= ~(1 << remove_idx)
        
        # Refill: try to add elements not in the set
        # We scan in a random order. 
        # To avoid O(N) shuffle every time, we can use a fixed random permutation 
        # and just iterate? No, the "not in set" condition changes.
        # Shuffling 512 ints is ~0.01ms. It's fine.
        
        candidates = [i for i in range(N) if not (current_bits >> i) & 1]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add(current_bits, c):
                current_bits |= (1 << c)
                
        new_size = bin(current_bits).count('1')
        
        # Acceptance criterion
        # If new_size > current_size, accept.
        # If new_size == current_size, accept with prob 0.5 (to explore).
        # If new_size < current_size, reject (or accept with low prob to escape local optima).
        # Since we want to maximize, let's be greedy but allow some exploration.
        
        if new_size > current_size:
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_bits = current_bits
        elif new_size == current_size:
            # Accept to move in the landscape
            if random.random() < 0.5:
                current_size = new_size
            else:
                current_bits = old_bits
        else:
            # Reject
            current_bits = old_bits
            
        # Periodically check if we are stuck? 
        # If we haven't improved best_size in 1000 steps, we could restart?
        # For now, let it run.
        
    # Convert best_bits to list
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result

# EVOLVE-BLOCK-END
