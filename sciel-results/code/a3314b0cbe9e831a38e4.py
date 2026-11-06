import math
import itertools

def construct():
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # We want to find the maximum K such that we can pick K moduli and assign residues.
    # We try K from 12 down to 1.
    # If we find a solution for K, we return it.
    
    # To make backtracking efficient, we order moduli.
    # Moduli that are multiples of small primes might be more constrained?
    # Let's just use the natural order or sort by number of divisors descending.
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # Sort moduli by number of divisors descending? 
    # More divisors -> more potential gcds > 1 -> potentially easier to satisfy?
    # Or harder? 
    # Let's try sorting by value first, it's simple.
    
    def find_solution(K):
        # current_selection: list of (r, m)
        current_selection = []
        
        def backtrack(start_idx):
            if len(current_selection) == K:
                return True
            
            # Pruning: if remaining moduli < needed, return False
            if len(moduli) - start_idx < K - len(current_selection):
                return False
                
            for i in range(start_idx, len(moduli)):
                m = moduli[i]
                # Try to find a residue r for m
                found_r = False
                for r in range(m):
                    # Check compatibility with all existing
                    compatible = True
                    for r_exist, m_exist in current_selection:
                        g = math.gcd(m, m_exist)
                        if (r - r_exist) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        current_selection.append([r, m])
                        if backtrack(i + 1):
                            return True
                        current_selection.pop()
                        # Optimization: if one residue failed, others might too?
                        # Not necessarily.
                        # But we can break if we've tried enough? No.
                        found_r = True
                        break # Try next modulus? Or next residue?
                        # If we break here, we only try ONE residue per modulus in this branch.
                        # This might miss solutions.
                        # We should try all residues for a modulus before moving to next modulus?
                        # Or move to next modulus immediately if first residue fails?
                        # If first residue fails, another might work.
                        # So we should NOT break on first failure.
                        # We should iterate all r.
                        # But if we find one, we add it and recurse.
                        # If recursion fails, we pop and try next r.
                        pass # The loop structure above is slightly wrong for "try all r".
                
                # Correct structure:
                # for i ...
                #   for r ...
                #     if compatible:
                #       add
                #       if backtrack: return True
                #       remove
                #   # if no r worked, continue to next i
                
            return False
            
        # Re-implementing the inner loop properly
        def backtrack_v2(start_idx):
            if len(current_selection) == K:
                return True
            if len(moduli) - start_idx < K - len(current_selection):
                return False
                
            for i in range(start_idx, len(moduli)):
                m = moduli[i]
                for r in range(m):
                    compatible = True
                    for r_exist, m_exist in current_selection:
                        g = math.gcd(m, m_exist)
                        if (r - r_exist) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        current_selection.append([r, m])
                        if backtrack_v2(i + 1):
                            return True
                        current_selection.pop()
            return False
            
        if backtrack_v2(0):
            return current_selection
        return None

    # Try K=12, 11, ...
    # To avoid timeout, we might limit the search or add a timer.
    # But 60s is plenty for K=12 if the solution exists and search is guided.
    # If no solution for 12, it might take long.
    # We can add a node limit.
    
    node_limit = 100000
    nodes_visited = 0
    
    def find_solution_limited(K):
        nonlocal nodes_visited
        current_selection = []
        
        def backtrack(start_idx):
            nonlocal nodes_visited
            if nodes_visited > node_limit:
                return False
            nodes_visited += 1
            
            if len(current_selection) == K:
                return True
            if len(moduli) - start_idx < K - len(current_selection):
                return False
                
            for i in range(start_idx, len(moduli)):
                m = moduli[i]
                for r in range(m):
                    compatible = True
                    for r_exist, m_exist in current_selection:
                        g = math.gcd(m, m_exist)
                        if (r - r_exist) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        current_selection.append([r, m])
                        if backtrack(i + 1):
                            return True
                        current_selection.pop()
            return False
            
        if backtrack(0):
            return current_selection
        return None

    for K in range(12, 0, -1):
        nodes_visited = 0
        sol = find_solution_limited(K)
        if sol:
            return sol
            
    # Fallback to prime 11 strategy if search fails
    p = 11
    moduli_p = [p * k for k in range(1, 12) if p * k <= 128]
    result = []
    for i, m in enumerate(moduli_p):
        result.append([i % p, m])
    return result

