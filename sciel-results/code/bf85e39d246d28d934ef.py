import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    def is_valid(s):
        # Check if 200 is a subset sum of s
        possible = {0}
        for x in s:
            new = {p + x for p in possible if p + x <= 200}
            possible |= new
            if 200 in possible:
                return False
        return True
    
    def find_witness(S):
        # Find a subset of S summing to 200 using DP with path tracking
        target = 200
        S_list = sorted(S)
        
        # dp[s] = (prev_sum, index) to track path
        dp = {0: (-1, -1)}
        
        for i, x in enumerate(S_list):
            new_dp = {}
            for s in list(dp.keys()):
                if s + x <= target:
                    if s + x not in new_dp:
                        new_dp[s + x] = (s, i)
            dp.update(new_dp)
            if target in dp:
                break
        
        if target not in dp:
            return None
        
        # Reconstruct witness
        witness = set()
        s = target
        while s > 0:
            prev_s, idx = dp[s]
            witness.add(S_list[idx])
            s = prev_s
        
        return witness
    
    def local_search(S, dl):
        # Try to add elements back, then try swaps
        changed = True
        while changed:
            changed = False
            if time.time() > dl:
                break
            
            candidates = sorted([x for x in range(1, 101) if x not in S])
            
            # Try additions
            for x in candidates:
                if time.time() > dl:
                    break
                if is_valid(S | {x}):
                    S.add(x)
                    changed = True
            
            # Try swaps: remove one from S, add one from candidates
            if time.time() > dl:
                break
            S_list = sorted(S)
            for x_remove in S_list:
                if time.time() > dl:
                    break
                S_trial = S - {x_remove}
                candidates_now = sorted([x for x in range(1, 101) if x not in S_trial and x != x_remove])
                for x_add in candidates_now:
                    if time.time() > dl:
                        break
                    if is_valid(S_trial | {x_add}):
                        S = S_trial | {x_add}
                        changed = True
                        break
        
        return S
    
    def greedy_remove(S, strategy='smallest'):
        # Remove elements until valid
        while not is_valid(S):
            if time.time() > deadline:
                break
            witness = find_witness(S)
            if not witness:
                break
            if strategy == 'smallest':
                x_to_remove = min(witness)
            elif strategy == 'largest':
                x_to_remove = max(witness)
            else:
                x_to_remove = random.choice(list(witness))
            S.remove(x_to_remove)
        return S
    
    best_set = None
    best_size = 0
    
    # Try multiple strategies
    strategies = ['smallest', 'largest', 'random']
    num_restarts = 2
    
    for strategy in strategies:
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            if strategy == 'random':
                random.seed(restart * 1000 + 42)
            
            S = set(range(1, 101))
            S = greedy_remove(S, strategy)
            S = local_search(S, deadline)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    # Final polish
    if best_set and time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            old_size = len(best_set)
            best_set = local_search(best_set, deadline - 2)
            if len(best_set) == old_size:
                break
    
    if best_set is None:
        # Fallback: return the known valid set {66,...,100}
        best_set = set(range(66, 101))
    
    return list(best_set)
