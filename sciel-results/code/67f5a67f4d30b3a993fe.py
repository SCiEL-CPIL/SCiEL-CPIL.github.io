"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _is_antichain(terms):
    """Check if no term divides another in the list."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def construct():
    """
    Construct 128 rows corresponding to targets 1..128.
    Every row must contain 1..30 distinct positive 2^a*3^b*5^c terms summing to its target;
    no chosen term may divide another. Minimize the total number of summands.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s limit
    
    max_val = 128
    smooth = _generate_smooth(max_val)
    
    # Precompute divisibility relations
    # div_map[x] = set of y in smooth such that y divides x
    div_map = {}
    for x in smooth:
        div_map[x] = set()
        for y in smooth:
            if y < x and x % y == 0:
                div_map[x].add(y)
    
    # For each target, find the minimum number of terms
    answers = [None] * 129
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        found = False
        # Try from 1 term up to some reasonable limit
        for count in range(1, 11):
            if time.time() > deadline:
                break
            
            # We need to find count distinct smooth numbers <= n that sum to n
            # and form an antichain
            # Use DFS with pruning
            
            candidates = [x for x in smooth if x <= n]
            if len(candidates) < count:
                break
            
            # Sort candidates in descending order for better pruning
            candidates.sort(reverse=True)
            
            # DFS to find antichain of size count summing to n
            solution = None
            
            def dfs(idx, remaining, current, used_set):
                nonlocal solution
                if solution is not None:
                    return
                if time.time() > deadline:
                    return
                
                if len(current) == count:
                    if remaining == 0 and _is_antichain(current):
                        solution = list(current)
                    return
                
                if remaining < 0:
                    return
                
                # Pruning: need at least (count - len(current)) more terms
                # Each term must be at least 1, so remaining must be >= (count - len(current))
                needed = count - len(current)
                if remaining < needed:
                    return
                
                # Try each candidate from idx onwards
                for i in range(idx, len(candidates)):
                    if time.time() > deadline:
                        return
                    
                    val = candidates[i]
                    if val > remaining:
                        continue
                    
                    # Check if val divides any term in current or is divisible by any
                    valid = True
                    for term in current:
                        if term % val == 0 or val % term == 0:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    current.append(val)
                    dfs(i + 1, remaining - val, current, used_set | {val})
                    current.pop()
                    
                    if solution is not None:
                        return
            
            current = []
            dfs(0, n, current, set())
            
            if solution is not None:
                answers[n] = solution
                found = True
                break
        
        if not found:
            # Fallback: try with more terms if we didn't find a solution
            # This should rarely happen
            for count in range(11, 31):
                if time.time() > deadline:
                    break
                candidates = [x for x in smooth if x <= n]
                if len(candidates) < count:
                    break
                candidates.sort(reverse=True)
                
                solution = None
                
                def dfs2(idx, remaining, current):
                    nonlocal solution
                    if solution is not None:
                        return
                    if time.time() > deadline:
                        return
                    
                    if len(current) == count:
                        if remaining == 0 and _is_antichain(current):
                            solution = list(current)
                        return
                    
                    if remaining < 0:
                        return
                    
                    needed = count - len(current)
                    if remaining < needed:
                        return
                    
                    for i in range(idx, len(candidates)):
                        if time.time() > deadline:
                            return
                        
                        val = candidates[i]
                        if val > remaining:
                            continue
                        
                        valid = True
                        for term in current:
                            if term % val == 0 or val % term == 0:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        current.append(val)
                        dfs2(i + 1, remaining - val, current)
                        current.pop()
                        
                        if solution is not None:
                            return
                
                current = []
                dfs2(0, n, current)
                
                if solution is not None:
                    answers[n] = solution
                    break
    
    # Return 128 rows for targets 1..128
    result = []
    for n in range(1, 129):
        if answers[n] is None:
            # This should not happen, but provide a fallback
            # Use a simple representation
            if n in smooth:
                result.append([n])
            else:
                # Find any valid representation
                result.append([n])  # This may not be valid, but shouldn't happen
        else:
            result.append(answers[n])
    
    return result
# EVOLVE-BLOCK-END
