"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    primes = _primes(N + 1)
    
    best_set = []
    best_size = 0
    
    def run_greedy(strategy="min_cost", seed=None):
        if seed is not None:
            random.seed(seed)
            
        S_set = set(range(N + 1))
        # Initially all primes <= N+1 are violated
        active_primes = [p for p in primes if p <= N + 1]
        
        while active_primes:
            if time.time() > deadline:
                return S_set
            
            current_size = len(S_set)
            # Filter active primes to those <= current_size
            # Primes > current_size are automatically satisfied
            active_primes = [p for p in active_primes if p <= current_size]
            if not active_primes:
                break
            
            # Evaluate all active primes
            # We need to find the one to fix
            # Strategy: min_cost -> pick p with min (min_class_size)
            # Strategy: standard -> pick smallest p
            
            best_p = None
            best_r = None
            best_cost = float('inf')
            best_candidates = []
            
            for p in active_primes:
                # Check if p is still violated
                # Count residues
                counts = [0] * p
                # Optimization: if len(S_set) < p, it's not violated
                # But we filtered p <= current_size, so len(S_set) >= p
                # However, it's possible that some classes are already empty.
                
                # Quick check: are all residues present?
                # Instead of full count, we can just check presence.
                # But we need min count for cost.
                
                # Iterate S_set
                # To speed up, we can break early if we see all residues?
                # No, we need counts for the min.
                
                for x in S_set:
                    counts[x % p] += 1
                
                # Find min count
                min_count = min(counts)
                
                # If min_count == 0, it's satisfied (not violated)
                if min_count == 0:
                    continue
                
                # It is violated.
                # Find candidates with min_count
                candidates = [r for r, c in enumerate(counts) if c == min_count]
                
                if strategy == "min_cost":
                    if min_count < best_cost:
                        best_cost = min_count
                        best_p = p
                        best_candidates = candidates
                        # Tie break: smaller p?
                    elif min_count == best_cost:
                        if p < best_p:
                            best_p = p
                            best_candidates = candidates
                            
                elif strategy == "standard":
                    # Just pick the first one (smallest p in active list)
                    # Since active_primes is sorted, the first violated one is the smallest
                    best_p = p
                    best_candidates = candidates
                    best_cost = min_count
                    break # Only need to check the smallest one for standard greedy
                    
                elif strategy == "random_min_cost":
                    if min_count < best_cost:
                        best_cost = min_count
                        best_p = p
                        best_candidates = candidates
                    elif min_count == best_cost:
                        if p < best_p:
                            best_p = p
                            best_candidates = candidates

            if best_p is None:
                # All active primes are satisfied?
                # This shouldn't happen if we filtered correctly, 
                # but min_count==0 means satisfied.
                break
            
            # Remove the chosen class
            if strategy in ["random_min_cost", "random_standard"]:
                r = random.choice(best_candidates)
            else:
                r = best_candidates[0]
                
            S_set = {x for x in S_set if x % best_p != r}
            
            # We don't need to update active_primes explicitly for satisfaction
            # because next loop will re-evaluate.
            # But we can remove best_p from active_primes if we are sure?
            # No, removing elements might affect other primes?
            # No, removing elements only helps satisfy primes.
            # So best_p is now satisfied.
            # We can remove it from active_primes to save time?
            # Yes, because it will never be violated again.
            active_primes = [p for p in active_primes if p != best_p]
            
        return S_set

    # Run standard greedy
    S1 = run_greedy("standard")
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
        
    # Run min-cost greedy
    if time.time() < deadline:
        S2 = run_greedy("min_cost")
        if len(S2) > best_size:
            best_size = len(S2)
            best_set = S2
            
    # Run random restarts
    seed = 0
    while time.time() < deadline - 5:
        S_r = run_greedy("random_min_cost", seed=seed)
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_set = S_r
        seed += 1
        
    return list(best_set)
# EVOLVE-BLOCK-END
