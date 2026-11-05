import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 511
    candidates = list(range(N, 0, -1))
    weights = {a: 1.0 / (512 - a) for a in candidates}
    
    def get_score(S):
        return sum(weights[a] for a in S)
    
    def is_coprime_to_all(a, S_set):
        for b in S_set:
            if gcd(a, b) != 1:
                return False
        return True
        
    # 1. Greedy Initialization
    S = []
    S_set = set()
    for a in candidates:
        if is_coprime_to_all(a, S_set):
            S.append(a)
            S_set.add(a)
            
    current_score = get_score(S)
    
    # 2. Local Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Iterate through a copy of S
        for u in list(S):
            # Remove u
            S_temp = [x for x in S if x != u]
            S_temp_set = set(S_temp)
            
            # Greedy refill
            # We start with S_temp and add candidates
            # To be efficient, we can just iterate candidates and add if valid
            # But we must ensure we don't add u again (it's not in S_temp, but might be added back if it's the best?)
            # If we add u back, the score is the same (plus whatever else we added).
            # Actually, if we add u back, we are in the same state, but maybe with extra elements?
            # No, if u was blocking x, and we remove u, we add x.
            # If we then add u back, we must remove x.
            # So we should NOT add u back in the refill step, or we handle it carefully.
            # Simplest: just refill with candidates != u?
            # No, if u is very valuable, maybe we should keep it?
            # But we are testing if removing u allows a BETTER set.
            # So we should see if we can get a better set WITHOUT u.
            # If the best set without u is still worse than with u, we keep u.
            # If it's better, we swap.
            
            new_S = list(S_temp)
            new_S_set = set(S_temp)
            
            for a in candidates:
                if a in new_S_set:
                    continue
                if is_coprime_to_all(a, new_S_set):
                    new_S.append(a)
                    new_S_set.add(a)
                    
            new_score = get_score(new_S)
            
            if new_score > current_score + 1e-12:
                S = new_S
                S_set = new_S_set
                current_score = new_score
                improved = True
                
    return S
