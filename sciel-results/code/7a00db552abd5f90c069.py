import sys
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    primes_asc = sorted(primes)
    n_primes = len(primes_asc)
    
    # We want to maximize L. Let's set a limit for the coverage array.
    # L=1000 is the target. Let's check up to 1000.
    # If we go beyond, we might find >1000, but the task says "capped at 10^5" for score, 
    # but "L >= 1000 would realize...". The score is the largest L.
    # Let's set L_limit to 1000 for the array size to keep it fast, 
    # but actually if we want to see if we can go higher, we might need more.
    # However, the problem statement implies L=1000 is the milestone.
    # Let's use L_limit = 1000.
    L_limit = 1000
    
    best_score = 0
    best_residues = [0] * n_primes
    
    # Initial greedy to get a good lower bound for pruning
    # Greedy: for each prime, pick r that maximizes coverage of the current "frontier"
    cov_init = [0] * (L_limit + 1)
    init_residues = [0] * n_primes
    for i, p in enumerate(primes_asc):
        best_r = 0
        best_gain = -1
        for r in range(p):
            gain = 0
            # Count how many x in 1..L_limit get +1 coverage where cov[x] < 2
            # Or better, count how many x where cov[x] == 1 become 2?
            # Actually, for greedy, maximizing the number of x with cov[x] < 2 that get covered is good.
            for x in range(1, L_limit + 1):
                if x % p == r:
                    if cov_init[x] < 2:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        init_residues[i] = best_r
        for x in range(1, L_limit + 1):
            if x % p == best_r:
                cov_init[x] += 1
                
    # Calculate score for init
    for x in range(1, L_limit + 1):
        if cov_init[x] < 2:
            break
    else:
        x = L_limit + 1
    init_score = x - 1
    if init_score > best_score:
        best_score = init_score
        best_residues = init_residues[:]

    # Backtracking search
    # State: index i, residues list, cov array
    # We use a stack for iterative deepening or just recursion.
    # Recursion depth 168 is safe.
    
    def backtrack(i, residues, cov):
        nonlocal best_score, best_residues
        
        if i == n_primes:
            # Calculate final score
            score = 0
            for x in range(1, L_limit + 1):
                if cov[x] >= 2:
                    score = x
                else:
                    break
            if score > best_score:
                best_score = score
                best_residues = residues[:]
            return

        p = primes_asc[i]
        remaining = n_primes - i - 1
        
        # Generate candidates for residue r
        candidates = []
        for r in range(p):
            # Check potential prefix length
            # For each x, new_cov[x] = cov[x] + (1 if x%p==r else 0)
            # Potential max cov for x is new_cov[x] + remaining
            # We need new_cov[x] + remaining >= 2 for all x in prefix.
            # So we need new_cov[x] >= 2 - remaining.
            # If remaining >= 2, 2-remaining <= 0, so condition is always true for cov>=0.
            # If remaining == 1, we need new_cov[x] >= 1.
            # If remaining == 0, we need new_cov[x] >= 2.
            
            # Let's find the smallest x where new_cov[x] + remaining < 2.
            # Let this be x_fail. The potential prefix is x_fail - 1.
            # We want to maximize x_fail.
            
            # Optimization: Only check up to current best_score + something?
            # No, we want to extend.
            
            x_fail = L_limit + 1
            # We can break early if we find a failure
            for x in range(1, L_limit + 1):
                nc = cov[x] + (1 if x % p == r else 0)
                if nc + remaining < 2:
                    x_fail = x
                    break
            
            # If x_fail - 1 < best_score, this candidate cannot beat the current best.
            # Prune it.
            if x_fail - 1 < best_score:
                continue
                
            candidates.append((x_fail, r))
            
        # Sort candidates by potential prefix length (x_fail) descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for x_fail, r in candidates:
            new_cov = cov[:]
            for x in range(1, L_limit + 1):
                if x % p == r:
                    new_cov[x] += 1
                    
            new_residues = residues + [r]
            backtrack(i + 1, new_residues, new_cov)
            
            # Optional: If we found a perfect solution (L_limit), we can stop early
            if best_score == L_limit:
                return

    # Start backtracking
    # To speed up, we can start with the greedy solution's residues for the first few primes?
    # No, just start from scratch with empty cov.
    # But we can use the greedy residues as a "warm start" if we modify backtrack to accept a prefix.
    # Simpler: Just run backtrack. The greedy score provides a good lower bound.
    
    # To avoid recursion limit issues or just to be safe, let's stick to recursion.
    # However, pure backtracking might be slow if the branching factor is high and pruning is weak.
    # The pruning `x_fail - 1 < best_score` is strong.
    
    # Let's add a time limit check inside the recursion?
    # The problem says "finish within 120s".
    # We can add a global time check.
    
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer
    
    def backtrack_timed(i, residues, cov):
        nonlocal best_score, best_residues
        if time.time() - start_time > time_limit:
            return True # Signal timeout
        
        if i == n_primes:
            score = 0
            for x in range(1, L_limit + 1):
                if cov[x] >= 2:
                    score = x
                else:
                    break
            if score > best_score:
                best_score = score
                best_residues = residues[:]
            return False

        p = primes_asc[i]
        remaining = n_primes - i - 1
        
        candidates = []
        for r in range(p):
            x_fail = L_limit + 1
            for x in range(1, L_limit + 1):
                nc = cov[x] + (1 if x % p == r else 0)
                if nc + remaining < 2:
                    x_fail = x
                    break
            
            if x_fail - 1 < best_score:
                continue
                
            candidates.append((x_fail, r))
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for x_fail, r in candidates:
            new_cov = cov[:]
            for x in range(1, L_limit + 1):
                if x % p == r:
                    new_cov[x] += 1
                    
            new_residues = residues + [r]
            timed_out = backtrack_timed(i + 1, new_residues, new_cov)
            if timed_out:
                return True
            if best_score == L_limit:
                return False

    backtrack_timed(0, [], [0] * (L_limit + 1))
    
    return best_residues
