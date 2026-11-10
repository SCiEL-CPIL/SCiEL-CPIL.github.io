import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    N = 511
    # Precompute prime factors
    prime_factors = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        if not prime_factors[i]:
            for j in range(i, N+1, i):
                prime_factors[j].append(i)
    
    weights = [0.0] * (N+1)
    for a in range(1, N+1):
        weights[a] = 1.0 / (512.0 - a)
        
    def get_used_primes(s):
        used = set()
        for x in s:
            for p in prime_factors[x]:
                used.add(p)
        return used

    def greedy_fill(used_primes):
        chosen = []
        for a in range(N, 0, -1):
            if all(p not in used_primes for p in prime_factors[a]):
                chosen.append(a)
                for p in prime_factors[a]:
                    used_primes.add(p)
        return chosen, used_primes

    # Initial Greedy
    used = set()
    current_set, used = greedy_fill(used)
    current_score = sum(weights[a] for a in current_set)
    
    best_set = current_set[:]
    best_score = current_score
    
    # Local Search
    # Phase 1: 1-removal
    # Phase 2: 2-removal
    
    while time.time() < deadline:
        improved = False
        
        # 1-removal
        # Sort by weight desc to try removing "expensive" ones first?
        # Or maybe removing "cheap" ones is safer?
        # Let's just iterate all.
        for a in current_set:
            if time.time() > deadline:
                break
            
            new_set = [x for x in current_set if x != a]
            new_used = get_used_primes(new_set)
            
            candidate_set, candidate_used = greedy_fill(new_used)
            candidate_score = sum(weights[x] for x in candidate_set)
            
            if candidate_score > current_score + 1e-12:
                current_set = candidate_set
                current_score = candidate_score
                used = candidate_used
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                break # Restart loop with new set
        
        if improved:
            continue
            
        # 2-removal
        # Try removing pairs
        # Limit pairs to avoid O(N^2) explosion if N is large, but N is small.
        # However, current_set size is small (~50-100).
        # Pairs ~ 5000.
        # Each check is ~12k ops. 5000 * 12k = 60M ops. ~1s.
        # Acceptable.
        
        # To speed up, only try pairs where the sum of weights of removed elements
        # is less than the potential gain? Hard to estimate.
        # Just try all pairs.
        
        # Optimization: Only try pairs (a, b) where a and b are "conflicting" or "blocking"?
        # No, just try all.
        
        for i in range(len(current_set)):
            a = current_set[i]
            for j in range(i+1, len(current_set)):
                b = current_set[j]
                if time.time() > deadline:
                    break
                
                new_set = [x for x in current_set if x != a and x != b]
                new_used = get_used_primes(new_set)
                
                candidate_set, candidate_used = greedy_fill(new_used)
                candidate_score = sum(weights[x] for x in candidate_set)
                
                if candidate_score > current_score + 1e-12:
                    current_set = candidate_set
                    current_score = candidate_score
                    used = candidate_used
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    break
            if improved:
                break
        
        if not improved:
            # Stagnation. Try random restart or perturbation.
            # Perturbation: Randomly remove 3-5 elements and refill.
            if time.time() > deadline - 5.0:
                break
                
            k = 3 + int(time.time() * 10) % 3 # 3, 4, or 5
            if len(current_set) > k:
                import random
                to_remove = random.sample(current_set, k)
                new_set = [x for x in current_set if x not in to_remove]
                new_used = get_used_primes(new_set)
                candidate_set, candidate_used = greedy_fill(new_used)
                candidate_score = sum(weights[x] for x in candidate_set)
                
                # Accept if better, or with some probability (Simulated Annealing)
                # Since we don't have a temperature schedule, let's just accept if better.
                # If not better, we might get stuck.
                # Let's accept if it's within 0.1 of best? No, just hill climb.
                # To escape, we can force accept a worse move occasionally.
                # But for safety, let's just try more random perturbations.
                
                if candidate_score > current_score:
                    current_set = candidate_set
                    current_score = candidate_score
                    used = candidate_used
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    improved = True
                else:
                    # Try a different random set
                    pass

    return best_set
