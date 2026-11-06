import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    def check_and_add(S, sums_set, x):
        k = len(S)
        if k == 0:
            sums_set.add(3*x)
            return True

        new_sums = set()
        new_sums.add(3*x)
        
        # (x,x,s)
        for s in S:
            new_sums.add(2*x + s)
            
        # (x,s,t)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                new_sums.add(x + si + S[j])
        
        if new_sums & sums_set:
            return False
        
        expected = 1 + k + k*(k+1)//2
        if len(new_sums) != expected:
            return False
            
        sums_set.update(new_sums)
        return True

    def greedy_search(candidates):
        S = []
        sums_set = set()
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if check_and_add(S, sums_set, x):
                S.append(x)
        return S

    # Primary strategy: Greedy from 1
    S1 = greedy_search(range(1, 4097))
    if len(S1) > len(best_set):
        best_set = S1

    # Secondary strategy: Randomized restarts with slight variations
    # We try to find if a different initial choice leads to a larger set.
    # Since greedy from 1 is very strong, we try "greedy with random skipping"
    # where we randomly skip a small number of early candidates to see if it helps.
    
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        # Create a candidate list that is mostly 1..4096 but with some random skips
        # Or just run greedy from a random start offset?
        # No, start offset > 1 is usually worse.
        
        # Let's try a "noisy" greedy:
        # At each step, we consider the smallest valid x.
        # With prob 0.1, we skip it and try the next one?
        # This is hard to implement efficiently in a single pass.
        
        # Instead, let's try different random subsets of the "first" 1000 numbers.
        # If we can find a better core, we can extend it.
        
        # Actually, let's just run the greedy again? No.
        
        # Let's try to optimize the search by using a list of candidates
        # and randomly shuffling the "tail" of the candidate list?
        # No, order matters for greedy.
        
        # Let's try: Greedy, but if we get stuck (no valid x in next 100),
        # jump? No, greedy doesn't get stuck, it just stops.
        
        # Let's try a different heuristic:
        # Pick x randomly from a small range [1, 10] for the first few elements,
        # then greedy for the rest.
        
        S = []
        sums_set = set()
        
        # Randomize the first 3-5 elements
        first_k = random.randint(3, 5)
        candidates_first = random.sample(range(1, 20), first_k)
        candidates_first.sort()
        
        # Try to add these first
        for x in candidates_first:
            if check_and_add(S, sums_set, x):
                S.append(x)
                
        # Then greedy from 1 upwards, skipping those already added
        # We need to iterate 1..4096
        added_set = set(S)
        for x in range(1, 4097):
            if time.time() > deadline - 2:
                break
            if x in added_set:
                continue
            if check_and_add(S, sums_set, x):
                S.append(x)
                added_set.add(x)
                
        if len(S) > len(best_set):
            best_set = S

    return best_set
