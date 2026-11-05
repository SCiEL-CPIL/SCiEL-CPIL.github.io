import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # 2. Identify Bad Triples and Conflicts
    # conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            # It is a bad triple
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))

    # 3. Greedy Construction
    # Sort by degree (number of conflicts)
    degrees = [len(conflicts[v]) for v in range(N + 1)]
    # We want to pick vertices with lowest degree first
    candidates = sorted([v for v in numbers], key=lambda x: degrees[x])
    
    S = set()
    active_pairs = set() # Set of frozensets of pairs in S
    
    def can_add(v):
        # Check if conflicts[v] intersects active_pairs
        if not conflicts[v]:
            return True
        # Optimization: if active_pairs is empty, True
        if not active_pairs:
            return True
        # Check intersection
        for pair in conflicts[v]:
            if pair in active_pairs:
                return False
        return True

    for v in candidates:
        if time.time() > deadline:
            break
        if can_add(v):
            S.add(v)
            # Update active_pairs
            for u in S:
                if u != v:
                    active_pairs.add(frozenset((u, v)))

    # 4. Local Search
    # We will try to improve the set S.
    # Strategy: 
    # 1. Try to add any remaining vertex.
    # 2. If no add possible, try swaps.
    
    remaining = set(numbers) - S
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add
        # Sort remaining by degree for better chance?
        for v in sorted(remaining, key=lambda x: degrees[x]):
            if time.time() > deadline:
                break
            if can_add(v):
                S.add(v)
                remaining.remove(v)
                for u in S:
                    if u != v:
                        active_pairs.add(frozenset((u, v)))
                improved = True
                break # Restart loop to re-evaluate adds/swaps
        
        if improved:
            continue
            
        # Phase 2: Try swaps
        # For each u in S, try to replace with v in remaining
        # We iterate through u and v.
        # To make it faster, we can just pick the first valid swap.
        
        # Optimization: Pre-calculate potential?
        # Just iterate.
        found_swap = False
        for u in list(S):
            if time.time() > deadline:
                break
            # Temporarily remove u to check swaps?
            # It's easier to check validity of S - {u} + {v}
            
            # To check if v can be added to S - {u}:
            # The active pairs in S - {u} are active_pairs minus pairs involving u.
            # Let's just simulate.
            
            # Pairs involving u in active_pairs:
            # {u, w} for w in S, w != u
            
            # We can construct the set of "blocked pairs" for v in the context of S - {u}
            # v is valid if no pair in (S - {u}) is in conflicts[v].
            
            # Pairs in S - {u} are all pairs in active_pairs except those containing u.
            
            # Efficient check:
            # Does conflicts[v] intersect (active_pairs \ {pairs with u})?
            # This is equivalent to:
            # Does conflicts[v] intersect active_pairs?
            # If yes, does the intersection consist ONLY of pairs containing u?
            # If yes, then removing u clears the conflict, so v can be added.
            
            for v in remaining:
                # Check if v can be added to S \ {u}
                
                # Get conflicts of v
                conf_v = conflicts[v]
                if not conf_v:
                    # v has no conflicts, can always be added (if u removed, S is valid)
                    # Actually, if v has no conflicts, it's always compatible.
                    # So swap is always valid?
                    # Yes, if v has no bad triples, it doesn't conflict with anything.
                    # So S \ {u} + {v} is valid.
                    # Perform swap.
                    S.remove(u)
                    S.add(v)
                    remaining.remove(v)
                    remaining.add(u)
                    
                    # Update active_pairs
                    # Remove pairs involving u
                    pairs_to_remove = set()
                    for w in S: # w is in new S
                        if w != u: # u is not in S anymore
                            pass # We need to remove pairs (u, w) that were in active_pairs
                    # Actually, let's just rebuild active_pairs or update carefully.
                    # Rebuilding is O(|S|^2) = ~3600 ops. Acceptable.
                    active_pairs = set()
                    for a, b in combinations(S, 2):
                        active_pairs.add(frozenset((a, b)))
                    
                    found_swap = True
                    improved = True
                    break
                    
                # If v has conflicts
                # Check intersection with active_pairs
                # We need to know if ALL conflicting pairs in active_pairs contain u.
                
                # Iterate conflicts of v
                valid_swap = True
                for pair in conf_v:
                    if pair in active_pairs:
                        # This pair is active in S.
                        # If it contains u, it will be removed when we remove u.
                        # If it does NOT contain u, it remains, so swap is invalid.
                        if u not in pair:
                            valid_swap = False
                            break
                
                if valid_swap:
                    # Perform swap
                    S.remove(u)
                    S.add(v)
                    remaining.remove(v)
                    remaining.add(u)
                    
                    # Rebuild active_pairs
                    active_pairs = set()
                    for a, b in combinations(S, 2):
                        active_pairs.add(frozenset((a, b)))
                    
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break

    return list(S)
