import time
import random

def construct():
    N = 256
    odds = [i for i in range(1, N+1) if i % 2 == 1]
    evens = [i for i in range(1, N+1) if i % 2 == 0]
    
    # Precompute conflict graph
    # conflict[a][b] is True if a*b % (a+b) == 0
    # We only need to check pairs (a,b) where a!=b
    # Since N=256, we can use a set of tuples or a 2D array.
    # 256x256 bool array is small.
    conflict = [[False]*N for _ in range(N)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            if (a*b) % (a+b) == 0:
                conflict[a-1][b-1] = True
                conflict[b-1][a-1] = True
                
    def get_neighbors(x, S_set):
        # Return set of neighbors of x that are in S_set
        # x is 1-indexed in logic, 0-indexed in array
        idx = x - 1
        nbs = set()
        for y in S_set:
            if conflict[idx][y-1]:
                nbs.add(y)
        return nbs

    # Initial set: all odds
    S = set(odds)
    
    # List of evens to consider
    E_avail = list(evens)
    
    # Greedy: 
    # 1. Add all evens with 0 conflicts with current S (which is Odds initially)
    # 2. Then add evens with 1 conflict, removing the odd
    # 3. Check for internal conflicts among added evens
    
    added_evens = set()
    
    # First pass: Cost 0
    # An even e has cost 0 if it has no neighbors in S (Odds)
    # And no neighbors in added_evens
    
    # We iterate until no more can be added
    changed = True
    while changed:
        changed = False
        # Sort by number of conflicts with current S
        # To make it deterministic, we can sort by value too
        E_avail_sorted = sorted(E_avail, key=lambda e: (len(get_neighbors(e, S)), e))
        
        for e in E_avail_sorted:
            if e in added_evens:
                continue
                
            nbs_in_S = get_neighbors(e, S)
            
            # If e conflicts with any already added even, we can't add it
            # (Because added evens are in S)
            # get_neighbors(e, S) includes odds and added evens.
            # We need to distinguish.
            
            # Actually, S contains Odds \ RemovedOdds + AddedEvens.
            # If nbs_in_S is not empty, we have conflicts.
            
            # If conflict is with an added even, we CANNOT add e (unless we remove the even, which is complex).
            # So we only add e if all its neighbors in S are Odds (which we can remove).
            
            conflicting_odds = set()
            conflicting_evens = set()
            for nb in nbs_in_S:
                if nb in odds: # nb is odd
                    conflicting_odds.add(nb)
                else: # nb is even (added)
                    conflicting_evens.add(nb)
            
            if conflicting_evens:
                # Cannot add e because it conflicts with an existing even
                continue
            
            # If no conflicting evens, we can add e by removing conflicting_odds
            # Gain = 1 - len(conflicting_odds)
            # We accept if Gain >= 0 (i.e., len(conflicting_odds) <= 1)
            # If len > 1, we skip for now (greedy limitation)
            
            if len(conflicting_odds) <= 1:
                # Add e
                S.add(e)
                added_evens.add(e)
                E_avail.remove(e)
                
                # Remove conflicting odds
                for o in conflicting_odds:
                    S.remove(o)
                
                changed = True
                # Restart sorting? Or just continue. 
                # Removing odds might make other evens have lower cost.
                # Breaking and restarting the while loop is safer.
                break
                
    return list(S)
