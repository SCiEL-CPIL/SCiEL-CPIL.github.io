import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50 # 50s safety margin
    
    N = 180
    S = set(range(91, 181))
    
    # Generate all triples (a,b,c) with a<b<c <= N
    # 1/a = 1/b + 1/c  => a = bc / (b+c)
    # b+c divides bc
    # Let g = gcd(b,c). b=gx, c=gy. gcd(x,y)=1.
    # a = gxy / (x+y). x+y divides g.
    # g = k(x+y). a=kxy, b=kx(x+y), c=ky(x+y).
    
    triples = []
    # x < y, gcd(x,y)=1
    # b,c <= 180
    # kx(x+y) <= 180
    
    # Max x,y?
    # If x=1, y=2, b=3k <= 180 => k<=60.
    # If x=1, y=100, b=101k <= 180 => k=1.
    
    for x in range(1, 100):
        for y in range(x+1, 100):
            if __import__('math').gcd(x, y) != 1:
                continue
            # b = kx(x+y) <= 180
            # c = ky(x+y) <= 180
            # Since y > x, c > b. So c <= 180 is the tighter bound.
            # k <= 180 / (y(x+y))
            max_k = 180 // (y * (x + y))
            for k in range(1, max_k + 1):
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a < b < c <= N:
                    triples.append((a, b, c))
    
    # Identify Blocked vertices (a in 1..90)
    # Blocked if there exists (a,b,c) in triples with b,c in S
    blocked = set()
    conflicting_pairs = set()
    internal_triples = []
    
    for a, b, c in triples:
        if a > 90:
            continue # a is in S, but we know S has no internal triples. 
                     # And b,c > a > 90 so b,c in S. 
                     # We verified S has no triples. So these don't affect T.
        
        # a is in 1..90
        if b in S and c in S:
            blocked.add(a)
        elif b in S and c not in S:
            # c must be < 91? No, c > b. If b in S (>=91), c > 91 so c in S.
            # So this case is impossible.
            pass
        elif c in S and b not in S:
            # b < 91, c >= 91.
            # This is a conflict between pair (a,b) in T and c in S.
            if a <= 90 and b <= 90:
                conflicting_pairs.add((a, b))
        else:
            # b,c not in S. So b,c <= 90.
            # Internal triple in 1..90
            internal_triples.append((a, b, c))
            
    # Candidates for T
    candidates = [i for i in range(1, 91) if i not in blocked]
    
    # Greedy construction of T
    # We want to pick max subset of candidates such that:
    # 1. No (a,b) in conflicting_pairs is fully in T.
    # 2. No (a,b,c) in internal_triples is fully in T.
    
    # Heuristic: Sort by "constraint weight" or just index.
    # Let's try simple local search.
    # Start with all candidates.
    T = set(candidates)
    
    # Remove conflicts
    # We will iterate and remove elements that cause conflicts.
    # To be efficient, we can check conflicts for each element.
    
    # Precompute for each a in candidates:
    # - Conflicting partners in T (from conflicting_pairs)
    # - Internal triples involving a
    
    # Since N is small, we can just do a few passes of removal.
    changed = True
    while changed:
        changed = False
        # Find a conflict
        # Check internal triples
        conflict_found = None
        for a, b, c in internal_triples:
            if a in T and b in T and c in T:
                conflict_found = (a, b, c, 'internal')
                break
        if not conflict_found:
            # Check pairs
            for a, b in conflicting_pairs:
                if a in T and b in T:
                    conflict_found = (a, b, 0, 'pair')
                    break
                    
        if conflict_found:
            # Remove one element
            # Strategy: Remove the one with the most conflicts (highest degree)
            # Or just remove the largest one?
            # Let's remove the one that appears in the most remaining conflicts.
            # For simplicity, remove the largest element in the conflict.
            # Or random?
            # Let's try removing the element with the highest index to keep smaller numbers 
            # (smaller numbers might be in fewer conflicts? No, smaller a has more triples).
            # Actually, larger a has fewer triples.
            # So removing larger a is safer for future?
            # Let's just remove the max element in the conflict set.
            
            if conflict_found[3] == 'internal':
                a, b, c, _ = conflict_found
                # Remove max(a,b,c)
                rem = max(a,b,c)
            else:
                a, b, _, _ = conflict_found
                rem = max(a,b)
                
            T.remove(rem)
            changed = True
            
    return list(T | S)
