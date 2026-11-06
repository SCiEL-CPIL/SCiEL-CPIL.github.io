"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _precompute_conflicts(N=180):
    """Precompute all forbidden triples (a,b,c) with a<b<c and 1/a=1/b+1/c.
    Also build adjacency: for each element x, which pairs (i,j) form a triple with x."""
    triples = []
    triple_index = {}  # maps element -> list of (index, other1, other2)
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b and a != c:  # a < b < c, all distinct
                    idx = len(triples)
                    triples.append((a, b, c))
                    # For each element in the triple, record which other two elements pair with it
                    if a not in triple_index:
                        triple_index[a] = []
                    if b not in triple_index:
                        triple_index[b] = []
                    if c not in triple_index:
                        triple_index[c] = []
                    triple_index[a].append((idx, b, c))
                    triple_index[b].append((idx, a, c))
                    triple_index[c].append((idx, a, b))
    
    return triples, triple_index

def _check_valid(triples, S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 180
    triples, triple_index = _precompute_conflicts(N)
    
    # Start with the known good base: [91, 180]
    S = list(range(91, N + 1))
    S_set = set(S)
    
    # Precompute for each candidate x (1..90) the list of triples involving x
    # and for each such triple, which two other elements are needed
    candidate_triples = defaultdict(list)
    for x in range(1, 91):
        if x in triple_index:
            for (idx, o1, o2) in triple_index[x]:
                candidate_triples[x].append((idx, o1, o2))
    
    candidates = list(range(1, 91))
    
    # Greedy: repeatedly add the candidate that creates the fewest new conflicts
    # A candidate x has a conflict if there exists a triple (x, y, z) where y,z are both in S
    while time.time() < deadline:
        best_candidate = None
        best_conflict_count = float('inf')
        
        for x in candidates:
            if x in S_set:
                continue
            
            # Count how many triples involving x have both other elements in S
            conflict_count = 0
            for (idx, o1, o2) in candidate_triples[x]:
                if o1 in S_set and o2 in S_set:
                    conflict_count += 1
            
            if conflict_count < best_conflict_count:
                best_conflict_count = conflict_count
                best_candidate = x
        
        if best_candidate is None or best_conflict_count > 0:
            break
        
        S.append(best_candidate)
        S_set.add(best_candidate)
    
    # Local search: try removing elements to enable adding more
    # For each element in S, try removing it and see if we can add more elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(S):
            if time.time() > deadline - 5:
                break
            # Try removing x
            S_set.remove(x)
            S.remove(x)
            
            # Try to greedily add elements
            added = []
            while time.time() < deadline:
                best_candidate = None
                best_conflict_count = float('inf')
                
                for y in candidates:
                    if y in S_set:
                        continue
                    conflict_count = 0
                    for (idx, o1, o2) in candidate_triples[y]:
                        if o1 in S_set and o2 in S_set:
                            conflict_count += 1
                    if conflict_count < best_conflict_count:
                        best_conflict_count = conflict_count
                        best_candidate = y
                
                if best_candidate is None or best_conflict_count > 0:
                    break
                
                S.append(best_candidate)
                S_set.add(best_candidate)
                added.append(best_candidate)
            
            # If we added more than we removed, keep the change
            if len(added) > 1:
                improved = True
            else:
                # Revert
                for y in added:
                    S.remove(y)
                    S_set.remove(y)
                S.append(x)
                S_set.add(x)
    
    return sorted(S)
# EVOLVE-BLOCK-END
