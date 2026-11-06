import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute perfect squares up to 512 (max sum is 255+256=511)
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Diagonal constraint: 2*i must not be a perfect square
    # 2*i = s^2 => i = s^2/2, so s must be even: s=2m, i=2m^2
    diagonal_forbidden = set()
    m = 1
    while 2 * m * m <= N:
        diagonal_forbidden.add(2 * m * m)
        m += 1
    
    # Build conflict graph: for each i, which j (j != i) conflict with i
    # i and j conflict if i + j is a perfect square
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i + j in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # A vertex is "diagonally forbidden" if 2*i is a perfect square
    # We can never include such vertices
    
    # Greedy initialization: try to build a large independent set
    # Start with all non-diagonal-forbidden vertices, then resolve conflicts
    
    def is_valid_set(s):
        """Check if set s is valid (no conflicts, no diagonal issues)"""
        for a in s:
            if a in diagonal_forbidden:
                return False
            if 2 * a in squares:
                return False
        s_list = list(s)
        for idx, a in enumerate(s_list):
            for b in s_list[idx+1:]:
                if a + b in squares:
                    return False
        return True
    
    # Greedy: process vertices in some order, add if no conflict with already-added
    def greedy_order(order):
        result = set()
        for v in order:
            if v in diagonal_forbidden:
                continue
            # Check if v conflicts with any in result
            conflict = False
            for u in result:
                if u in conflicts[v]:
                    conflict = True
                    break
            if not conflict:
                result.add(v)
        return result
    
    # Initial greedy with natural order
    best_set = greedy_order(list(range(1, N + 1)))
    best_size = len(best_set)
    
    # Try different orderings for greedy
    for trial in range(50):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add/remove/swap to improve
    def local_search(s, time_limit):
        s = set(s)
        best = set(s)
        best_sz = len(s)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try adding a vertex not in s
            candidates = [v for v in range(1, N + 1) if v not in s and v not in diagonal_forbidden]
            random.shuffle(candidates)
            for v in candidates:
                # Check if v can be added
                can_add = True
                for u in s:
                    if u in conflicts[v]:
                        can_add = False
                        break
                if can_add:
                    s.add(v)
                    improved = True
                    if len(s) > best_sz:
                        best_sz = len(s)
                        best = set(s)
                    break  # restart after one addition
            
            if not improved:
                # Try removing one and adding two
                s_list = list(s)
                random.shuffle(s_list)
                for remove_v in s_list:
                    if len(s) <= 1:
                        break
                    s.remove(remove_v)
                    # Now try to add two new vertices
                    added = 0
                    candidates2 = [v for v in range(1, N + 1) if v not in s and v not in diagonal_forbidden]
                    random.shuffle(candidates2)
                    for v in candidates2:
                        if added >= 2:
                            break
                        can_add = True
                        for u in s:
                            if u in conflicts[v]:
                                can_add = False
                                break
                        if can_add:
                            s.add(v)
                            added += 1
                    if len(s) > best_sz:
                        best_sz = len(s)
                        best = set(s)
                        improved = True
                        break
                    else:
                        s.add(remove_v)  # revert
                        if len(s) > best_sz:
                            best_sz = len(s)
                            best = set(s)
            
            if not improved:
                # Try single swap
                s_list = list(s)
                random.shuffle(s_list)
                for remove_v in s_list:
                    if len(s) <= 1:
                        break
                    s.remove(remove_v)
                    candidates3 = [v for v in range(1, N + 1) if v not in s and v not in diagonal_forbidden]
                    random.shuffle(candidates3)
                    for v in candidates3:
                        can_add = True
                        for u in s:
                            if u in conflicts[v]:
                                can_add = False
                                break
                        if can_add:
                            s.add(v)
                            if len(s) > best_sz:
                                best_sz = len(s)
                                best = set(s)
                                improved = True
                                break
                            else:
                                s.remove(v)
                                s.add(remove_v)
                            break
                    else:
                        s.add(remove_v)
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Run local search
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 2:
        best_set = local_search(best_set, DEADLINE - 2)
        best_size = len(best_set)
    
    # Final verification
    result = sorted(best_set)
    
    # Double-check validity
    valid = True
    for a in result:
        if 2 * a in squares:
            valid = False
            break
    if valid:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if result[i] + result[j] in squares:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback: greedy from scratch with natural order, verified
        result = []
        for v in range(1, N + 1):
            if v in diagonal_forbidden:
                continue
            can_add = True
            for u in result:
                if v + u in squares:
                    can_add = False
                    break
            if can_add:
                result.append(v)
        # Final verify
        valid2 = True
        for i in range(len(result)):
            if 2 * result[i] in squares:
                valid2 = False
                break
        if valid2:
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    if result[i] + result[j] in squares:
                        valid2 = False
                        break
                if not valid2:
                    break
        if not valid2:
            # Single element is always valid if not diagonal-forbidden
            result = [v for v in range(1, N + 1) if v not in diagonal_forbidden][:1]
    
    return result
