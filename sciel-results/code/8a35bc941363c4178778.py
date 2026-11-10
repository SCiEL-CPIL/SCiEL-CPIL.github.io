"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    # Precompute all pairs coprime status
    N = 512
    # coprime[i][j] = 1 if gcd(i,j)==1 else 0, for i,j in 1..511
    # We'll use a bitset approach for speed
    
    # Precompute gcd matrix as a list of bytearrays for speed
    coprime = [[1]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            if gcd(i, j) != 1:
                coprime[i][j] = 0
                coprime[j][i] = 0
    
    # Score for each element a: 1/(512 - a)
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Sort candidates by score descending (higher score = closer to 511)
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: scores[x], reverse=True)
    
    def is_coprime_with_all(a, chosen_set):
        """Check if a is coprime with all elements in chosen_set (a set of ints)."""
        for b in chosen_set:
            if coprime[a][b] == 0:
                return False
        return True
    
    def compute_score(chosen):
        """Compute the score of a chosen set."""
        return sum(scores[a] for a in chosen)
    
    # Greedy initialization with lookahead
    def greedy_lookahead(candidates, lookahead=3):
        """Greedy with small lookahead: at each step, try the top 'lookahead' candidates
        and pick the one that leads to the best final greedy result."""
        chosen = []
        chosen_set = set()
        remaining = list(candidates)
        
        while remaining:
            best_choice = None
            best_future_score = -1
            
            # Try top min(lookahead, len(remaining)) candidates
            trial_count = min(lookahead, len(remaining))
            for i in range(trial_count):
                c = remaining[i]
                if is_coprime_with_all(c, chosen_set):
                    # Simulate adding c and continue greedy
                    new_chosen = chosen + [c]
                    new_set = chosen_set | {c}
                    new_remaining = [x for x in remaining if x != c]
                    
                    # Continue greedily from here
                    while new_remaining:
                        found = False
                        for j in range(len(new_remaining)):
                            c2 = new_remaining[j]
                            if is_coprime_with_all(c2, new_set):
                                new_chosen.append(c2)
                                new_set.add(c2)
                                new_remaining.pop(j)
                                found = True
                                break
                        if not found:
                            break
                    
                    fs = compute_score(new_chosen)
                    if fs > best_future_score:
                        best_future_score = fs
                        best_choice = i
            
            if best_choice is None:
                break
            
            c = remaining.pop(best_choice)
            chosen.append(c)
            chosen_set.add(c)
        
        return chosen
    
    # Local search: 2-swap (remove one, add one)
    def local_search_2swap(chosen, max_iters=5000):
        """Improve via 2-swap: remove one element, add another coprime element."""
        chosen_set = set(chosen)
        current_score = compute_score(chosen)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > DEADLINE - 5:
                break
            
            # Try removing each element
            for idx in range(len(chosen)):
                a = chosen[idx]
                new_set = chosen_set - {a}
                base_score = current_score - scores[a]
                
                # Try adding each candidate not in new_set
                for c in candidates:
                    if c in new_set:
                        continue
                    if is_coprime_with_all(c, new_set):
                        new_score = base_score + scores[c]
                        if new_score > current_score + 1e-12:
                            # Accept the swap
                            chosen[idx] = c
                            chosen_set = new_set | {c}
                            current_score = new_score
                            improved = True
                            break  # Restart scan after improvement
        
        return chosen, current_score
    
    # Simulated annealing
    def simulated_annealing(chosen, start_temp=0.01, end_temp=0.0001, max_iters=10000):
        """SA with 2-swap moves."""
        import random
        chosen_set = set(chosen)
        current_score = compute_score(chosen)
        best_chosen = list(chosen)
        best_score = current_score
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max(max_iters, 1))
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 3:
                break
            
            temp *= cooling
            
            # Random move: remove one, add another
            if len(chosen) < 2:
                break
            
            idx = random.randint(0, len(chosen) - 1)
            a = chosen[idx]
            new_set = chosen_set - {a}
            
            c = random.choice(candidates)
            if c in new_set:
                continue
            
            if not is_coprime_with_all(c, new_set):
                continue
            
            delta = scores[c] - scores[a]
            
            if delta > 0 or random.random() < __import__('math').exp(delta / max(temp, 1e-10)):
                # Accept
                chosen[idx] = c
                chosen_set = new_set | {c}
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = list(chosen)
        
        return best_chosen, best_score
    
    # Multi-start search
    best_chosen = []
    best_score = 0.0
    
    # Start 1: Pure greedy
    chosen1 = greedy_lookahead(candidates, lookahead=1)
    s1 = compute_score(chosen1)
    if s1 > best_score:
        best_score = s1
        best_chosen = list(chosen1)
    
    if time.time() > DEADLINE - 10:
        return best_chosen
    
    # Local search on start 1
    chosen1, s1 = local_search_2swap(chosen1, max_iters=3000)
    if s1 > best_score:
        best_score = s1
        best_chosen = list(chosen1)
    
    if time.time() > DEADLINE - 10:
        return best_chosen
    
    # Start 2: Greedy with lookahead 3
    chosen2 = greedy_lookahead(candidates, lookahead=3)
    s2 = compute_score(chosen2)
    if s2 > best_score:
        best_score = s2
        best_chosen = list(chosen2)
    
    if time.time() > DEADLINE - 10:
        return best_chosen
    
    # Local search on start 2
    chosen2, s2 = local_search_2swap(chosen2, max_iters=3000)
    if s2 > best_score:
        best_score = s2
        best_chosen = list(chosen2)
    
    if time.time() > DEADLINE - 10:
        return best_chosen
    
    # SA from best
    chosen_sa, s_sa = simulated_annealing(best_chosen, start_temp=0.005, end_temp=0.00005, max_iters=8000)
    if s_sa > best_score:
        best_score = s_sa
        best_chosen = list(chosen_sa)
    
    if time.time() > DEADLINE - 5:
        return best_chosen
    
    # Final local search polish
    chosen_final, s_final = local_search_2swap(best_chosen, max_iters=2000)
    if s_final > best_score:
        best_score = s_final
        best_chosen = list(chosen_final)
    
    return best_chosen
# EVOLVE-BLOCK-END
