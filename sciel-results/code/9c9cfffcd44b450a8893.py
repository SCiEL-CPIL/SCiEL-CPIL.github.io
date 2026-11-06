import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 180
    
    # Precompute conflicts
    # triples contains (a, b, c) such that 1/a = 1/b + 1/c with a < b < c
    # We need to know for each element x, which pairs (u, v) form a triple with x.
    # Let's build a list of all triples first.
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        # b = a + d, c = a + e
        # We need b, c <= N
        # Also b != c, which is guaranteed by d != e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                # Ensure distinctness: a, b, c are distinct because d>0 => b>a, e>d => c>b?
                # c - b = e - d > 0. So a < b < c.
                triples.append((a, b, c))
                
    # Build conflict map for efficient checking
    # conflicts[x] = list of (u, v) such that {x, u, v} is a triple
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        # Triple is {a, b, c}
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
        
    # Helper to check if x can be added to S
    def can_add(x, S):
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    # Helper to perform local search on a set S
    # Returns a potentially larger set
    def local_search(S, time_limit):
        S = set(S)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Try to add any element not in S
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        changed = True
                        # Restart the loop to check again, or just continue?
                        # Adding one might block others, but might also open up? No, adding only blocks.
                        # So we can just continue to the next x.
                        # But if we add x, we don't need to re-check previous x's.
                        # However, if we add x, it might conflict with y that we haven't checked yet.
                        # So a single pass is not enough. We need to repeat until no changes.
                        # But since we only add, the set grows.
                        # We can break and restart the outer while to be safe, or just continue.
                        # Continuing is faster. If we add x, we mark changed=True.
                        # We continue checking other x's.
                        # At the end of the for loop, if changed, we loop again.
                        # This is efficient.
        return S

    best_set = set()
    
    # Attempt to solve using MILP if available
    milp_success = False
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c_obj = -np.ones(N)
        
        # Constraints: for each triple, x_a + x_b + x_c <= 2
        # A_ub is the matrix for A_ub @ x <= b_ub
        A_ub = np.zeros((len(triples), N))
        b_ub = np.full(len(triples), 2)
        
        for idx, (a, b, c) in enumerate(triples):
            A_ub[idx, a-1] = 1
            A_ub[idx, b-1] = 1
            A_ub[idx, c-1] = 1
            
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Integrality: all variables are integer
        integrality = np.ones(N)
        
        # Solve
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            bounds=bounds,
            integrality=integrality
        )
        
        if res.status == 0:
            S = set()
            for i in range(N):
                if res.x[i] > 0.5:
                    S.add(i+1)
            
            # Verify validity
            valid = True
            for a, b, c in triples:
                if a in S and b in S and c in S:
                    valid = False
                    break
            
            if valid:
                best_set = S
                milp_success = True
                # If MILP succeeds, it's optimal. We can return immediately?
                # The task says "Maximize cardinality". MILP finds the maximum.
                # So we can return.
                return sorted(S)
                
    except Exception:
        pass
        
    # If MILP failed or wasn't used, use Heuristic Search
    # Strategy: Randomized Greedy Restarts + Local Search
    
    remaining_time = deadline - time.time()
    if remaining_time < 1:
        # Not enough time for heuristic, return empty or best found (which is empty)
        # But we must return a nonempty subset.
        # Fallback to a single fast greedy if no time.
        S = set()
        for a in range(N, 0, -1):
            if can_add(a, S):
                S.add(a)
        if S:
            return sorted(S)
        return [1] # Guaranteed non-empty, valid? 1 has no conflicts with any pair from [2..N] because 1/1 = 1, 1/b+1/c < 1.
        # Actually, is {1} valid? Yes.

    # Run multiple randomized greedy restarts
    # We have 'remaining_time'. Let's divide it into chunks.
    # Each chunk: Generate random order, greedy construct, local search.
    
    # To be safe, let's do a deterministic greedy first as a baseline.
    S_base = set()
    for a in range(1, N + 1): # Ascending
        if can_add(a, S_base):
            S_base.add(a)
    S_base = local_search(S_base, deadline - 1) # Leave 1s for return
    best_set = S_base
    best_size = len(S_base)
    
    # Now try randomized restarts if time permits
    # Estimate time per restart. Local search is the expensive part.
    # Let's try to fit as many restarts as possible.
    # Or just a few high-quality ones.
    
    num_restarts = 0
    while time.time() < deadline - 2: # Leave 2s safety
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_curr = set()
        for a in order:
            if can_add(a, S_curr):
                S_curr.add(a)
                
        # Local search
        S_curr = local_search(S_curr, deadline - 1)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            
        num_restarts += 1
        
    return sorted(best_set)
