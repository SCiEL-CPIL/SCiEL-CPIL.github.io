# EVOLVE-BLOCK-START
from math import gcd

def construct():
    N = 128
    # Precompute LCMs
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
            
    # Precompute all bad triples
    bad_triples = []
    # Also precompute triples_by_element for faster validity check
    triples_by_element = {i: [] for i in range(1, N+1)}
    for a in range(1, N):
        for b in range(a+1, N):
            L_ab = lcms[a-1][b-1]
            for c in range(b+1, N):
                if lcms[a-1][c-1] == L_ab and lcms[b-1][c-1] == L_ab:
                    bad_triples.append((a, b, c))
                    triples_by_element[a].append((a, b, c))
                    triples_by_element[b].append((a, b, c))
                    triples_by_element[c].append((a, b, c))
                    
    # Calculate degree of each element (number of bad triples it belongs to)
    degrees = {i: len(triples_by_element[i]) for i in range(1, N+1)}
    
    # Helper: Check if adding x to S is valid
    def can_add(S, x):
        # Check all triples containing x
        for t in triples_by_element[x]:
            # t contains x. Check if other two are in S
            # t is (a,b,c). One is x.
            # We can just check if the set of other two is subset of S
            others = [y for y in t if y != x]
            if others[0] in S and others[1] in S:
                return False
        return True

    # Helper: Check if S is valid (should always be true in our construction, but good for safety)
    def is_valid(S):
        for t in bad_triples:
            if t[0] in S and t[1] in S and t[2] in S:
                return False
        return True

    # Phase 1: Greedy Build
    # Sort elements by degree ascending (easy to add first)
    elements = sorted(range(1, N+1), key=lambda x: degrees[x])
    S = set()
    
    for x in elements:
        if can_add(S, x):
            S.add(x)
            
    # Phase 2: Local Search (Shake)
    # Try to improve by removing one element and re-adding greedily
    # We do this for a fixed number of iterations or until no improvement
    best_S = set(S)
    best_size = len(S)
    
    # Limit iterations to ensure time safety
    # 128 elements, check 128 removals, each re-adds ~128 elements. 128*128*128 is ~2M ops. Fast.
    # Let's do a few rounds.
    for _ in range(5): # 5 rounds of shake
        improved = False
        # Try removing each element in S
        # To make it deterministic, iterate through sorted(S)
        for y in sorted(list(S)):
            S_temp = S - {y}
            # Try to add elements not in S
            # Order: maybe by degree ascending again? Or just natural order?
            # Let's stick to degree ascending for consistency
            for x in elements:
                if x in S_temp:
                    continue
                if can_add(S_temp, x):
                    S_temp.add(x)
            
            if len(S_temp) > best_size:
                best_S = S_temp
                best_size = len(S_temp)
                S = S_temp
                improved = True
                # Break out of y loop, restart with new S
                break
        
        if not improved:
            break
            
    return sorted(best_S)
# EVOLVE-BLOCK-END
