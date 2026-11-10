import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    N = 512
    max_val = N - 1
    
    # Precompute weights
    weights = [0.0] * (max_val + 1)
    for i in range(1, max_val + 1):
        weights[i] = 1.0 / (N - i)
        
    # Precompute conflict sets
    # conflict[i] contains j such that gcd(i, j) != 1
    conflict = [set() for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i + 1, max_val + 1):
            if math.gcd(i, j) != 1:
                conflict[i].add(j)
                conflict[j].add(i)
                
    def score_set(S):
        return sum(weights[a] for a in S)
        
    def greedy_construct():
        S = set()
        # Process from largest to smallest to prioritize high weights
        for a in range(max_val, 0, -1):
            # Check if a is coprime to all in S
            ok = True
            for b in S:
                if a in conflict[b]:
                    ok = False
                    break
            if ok:
                S.add(a)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_a = -1
            best_b = -1
            
            # 1. Try to add a number (size increase)
            # 2. Try to swap a number (size constant, weight increase)
            
            for b in range(1, max_val + 1):
                if b in S:
                    continue
                
                # Find conflicts of b in S
                conflicts_in_S = []
                for x in S:
                    if b in conflict[x]:
                        conflicts_in_S.append(x)
                        
                if len(conflicts_in_S) == 0:
                    # Can add b
                    gain = weights[b]
                    if gain > best_gain:
                        best_gain = gain
                        best_a = -1 # Add
                        best_b = b
                elif len(conflicts_in_S) == 1:
                    # Can swap conflicts_in_S[0] with b
                    a = conflicts_in_S[0]
                    gain = weights[b] - weights[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
                        best_b = b
                        
            if best_b != -1:
                if best_a != -1:
                    S.remove(best_a)
                S.add(best_b)
                improved = True
                # Restart the while loop to recalculate conflicts
                # This is simple and robust for N=511
                
        return S

    best_S = set()
    best_score = 0.0
    
    # Run greedy + local search
    S = greedy_construct()
    S = local_search(S)
    s = score_set(S)
    if s > best_score:
        best_score = s
        best_S = S
        
    # If time allows, try a few randomized restarts
    # Randomized by shuffling the order of candidates slightly
    # or by starting with a random valid set?
    # Greedy is strong, so maybe just perturb the greedy result?
    
    while time.time() < deadline:
        # Perturb the best solution
        S_curr = set(best_S)
        # Remove 1-2 random elements
        if len(S_curr) > 2:
            items = list(S_curr)
            random.shuffle(items)
            S_curr.remove(items[0])
            if len(S_curr) > 1 and random.random() < 0.5:
                S_curr.remove(items[1])
        
        # Re-run local search from this state
        # Note: local_search as defined only adds/swaps. 
        # It might get stuck if the initial state is bad.
        # But since we removed elements, we can add back better ones.
        
        S_new = local_search(S_curr)
        s_new = score_set(S_new)
        if s_new > best_score:
            best_score = s_new
            best_S = S_new
            
    return list(best_S)
