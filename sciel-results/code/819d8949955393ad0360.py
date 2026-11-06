import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    denoms = list(range(2, 49))
    N = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # We will search in descending order of fraction value (ascending denominator).
    # This allows us to fill up the sum quickly, and the "exclude" branch (which keeps the sum lower) 
    # is less likely to be pruned immediately if we are close to 1, but more importantly,
    # taking large items first constrains the space of "what can we add" more tightly?
    # Actually, for subset sum < 1, ordering by value descending is standard.
    # However, the inspiration suggests that iterating in descending denominator order (smallest fractions first)
    # might help pruning? Let's stick to the standard effective approach:
    # Sort by value descending (1/2, 1/3, ...).
    
    # Let's verify the "suffix sum" pruning logic.
    # If we are at index i, and current sum is S.
    # The maximum possible sum we can achieve from this state is S + sum(fracs[i:])
    # If S + sum(fracs[i:]) <= best_sum, we cannot improve.
    # This is valid regardless of order.
    
    # However, the order affects how quickly we find a good best_sum and how much of the tree is pruned.
    # With descending value, we find high sums quickly.
    
    order = list(range(N))
    # fracs are already 1/2, 1/3... so order 0..N-1 is descending value.
    
    # Precompute suffix sums for pruning
    suffix_sums = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + fracs[i]
        
    # Greedy initialization: Take items in order if they fit
    best_sum = Fraction(0)
    best_idxs = []
    current_sum = Fraction(0)
    current_idxs = []
    for i in range(N):
        if current_sum + fracs[i] < 1:
            current_sum += fracs[i]
            current_idxs.append(i)
    best_sum = current_sum
    best_idxs = current_idxs[:]
    
    # Local Search to improve the greedy solution
    # This helps find a tighter bound for the DFS pruning.
    def local_search(idxs, s):
        nonlocal best_sum, best_idxs
        improved = True
        idx_set = set(idxs)
        # Limit local search time to ~5s
        ls_deadline = time.time() + 5
        
        while improved and time.time() < ls_deadline:
            improved = False
            
            # Try 1-swap
            for i in list(idx_set):
                for j in range(N):
                    if j not in idx_set:
                        new_s = s - fracs[i] + fracs[j]
                        if new_s < 1 and new_s > best_sum:
                            best_sum = new_s
                            best_idxs = [k for k in idx_set if k != i] + [j]
                            idx_set = set(best_idxs)
                            s = new_s
                            improved = True
                            break
                if improved:
                    break
            if time.time() >= ls_deadline:
                break
                
            # Try 1-add
            for j in range(N):
                if j not in idx_set and s + fracs[j] < 1:
                    candidate_s = s + fracs[j]
                    if candidate_s > best_sum:
                        best_sum = candidate_s
                        best_idxs = list(idx_set) + [j]
                        idx_set = set(best_idxs)
                        s = candidate_s
                        improved = True
                        break
            if time.time() >= ls_deadline:
                break
                
            # Try 1-remove (only if it allows a better swap later? 
            # Actually, removing reduces sum, so it won't directly improve best_sum.
            # It only helps if it enables a swap that we missed? 
            # The swap loop above covers removing i and adding j.
            # So pure removal is not useful for maximizing sum < 1.
            # We skip pure removal.
        
        return best_sum, best_idxs
    
    best_sum, best_idxs = local_search(best_idxs, best_sum)
    
    # DFS with Branch and Bound
    # We use an iterative stack to avoid recursion limits and overhead, 
    # though recursion is fine for N=47. Recursion is simpler.
    # Let's use recursion for clarity, it's fast enough.
    
    def dfs(i, s, idxs):
        nonlocal best_sum, best_idxs
        
        if time.time() > deadline:
            return
            
        # Pruning: Even if we take all remaining, we can't beat best_sum
        if s + suffix_sums[i] <= best_sum:
            return
            
        if i == N:
            if s > best_sum:
                best_sum = s
                best_idxs = idxs[:]
            return

        # Option 1: Include item i
        # Only if s + fracs[i] < 1
        if s + fracs[i] < 1:
            dfs(i + 1, s + fracs[i], idxs + [i])
            
        # Check time before exploring exclude branch
        if time.time() > deadline:
            return

        # Option 2: Exclude item i
        dfs(i + 1, s, idxs)
        
    dfs(0, Fraction(0), [])
    
    # Final check and return
    if not best_idxs:
        return [0]
        
    # Verify validity
    s_check = Fraction(0)
    for i in best_idxs:
        s_check += fracs[i]
    if s_check >= 1:
        # Fallback to greedy result if somehow invalid
        return current_idxs
        
    return best_idxs
