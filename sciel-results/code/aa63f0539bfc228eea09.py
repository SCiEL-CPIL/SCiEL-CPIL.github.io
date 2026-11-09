import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    
    # Known best so far
    best_sol = [8, 31, 248]
    best_max = 248
    
    def check_solution(sol):
        if not sol:
            return False
        if len(sol) > 256:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 1 or d > 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == target

    # Search for a solution with max denominator < best_max
    # Using subset sum / BFS on sums
    # We iterate M from 1 to best_max - 1
    # If we find a solution with max denom M, we return it
    
    # To speed up, we can use the known solution's denominators as a guide?
    # No, just brute force M.
    
    # Optimization: If M is small, the number of reachable sums is small.
    # As M increases, it grows.
    # We stop if time is running out.
    
    deadline = 50  # seconds
    
    for M in range(1, best_max):
        if time.time() - start_time > deadline:
            break
            
        reachable = {Fraction(0)}
        found = False
        
        # We add denominators from 1 to M
        # Order doesn't matter for reachability, but adding smaller denominators first
        # might keep the set of reachable sums smaller?
        # Actually, adding larger denominators (smaller fractions) first might be better?
        # Let's try 1 to M.
        
        for k in range(1, M + 1):
            new_sums = []
            inv_k = Fraction(1, k)
            for s in reachable:
                s_new = s + inv_k
                if s_new <= target:
                    new_sums.append(s_new)
            if new_sums:
                reachable.update(new_sums)
                if target in reachable:
                    found = True
                    break
        
        if found:
            # We found that target is reachable with denominators <= M.
            # Now we need to extract the actual denominators.
            # We can backtrack.
            # Store parent pointers?
            # Or just re-run a DFS to find the path?
            # Since M < 248, the search space is small.
            # Let's do a DFS to find the specific denominators.
            
            def dfs(current_sum, remaining_denoms, path):
                if current_sum == target:
                    return path
                if not remaining_denoms:
                    return None
                
                # Try each remaining denominator
                for i in range(len(remaining_denoms)):
                    k = remaining_denoms[i]
                    new_sum = current_sum + Fraction(1, k)
                    if new_sum <= target:
                        new_remaining = remaining_denoms[:i] + remaining_denoms[i+1:]
                        res = dfs(new_sum, new_remaining, path + [k])
                        if res:
                            return res
                return None

            # The set of available denominators is 1..M
            # But we don't need to try all subsets, just find ONE.
            # The DFS above tries all permutations/subsets.
            # It might be slow if M is large.
            # But M < 248.
            # However, the branching factor is M. Depth is small (3-5).
            # So M^5 operations? 248^5 is too big.
            
            # Better: Use the reachable set info?
            # We didn't store the path.
            # Let's restart the BFS but store parents.
            
            # Re-run BFS with parent tracking
            parent = {Fraction(0): None} # sum -> (prev_sum, k)
            reachable = {Fraction(0)}
            found_sum = None
            
            for k in range(1, M + 1):
                new_sums = []
                inv_k = Fraction(1, k)
                for s in reachable:
                    s_new = s + inv_k
                    if s_new <= target:
                        if s_new not in parent:
                            parent[s_new] = (s, k)
                            new_sums.append(s_new)
                if new_sums:
                    reachable.update(new_sums)
                    if target in reachable:
                        found_sum = target
                        break
            
            if found_sum:
                # Reconstruct path
                path = []
                curr = target
                while curr != Fraction(0):
                    prev_sum, k = parent[curr]
                    path.append(k)
                    curr = prev_sum
                
                # Verify
                if check_solution(path):
                    return path
                # If verification fails (shouldn't happen), continue search?
                # Or return best_sol.
    
    return best_sol
