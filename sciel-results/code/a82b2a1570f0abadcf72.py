"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N = 192
    K = 3
    MAX_SIZE = 48
    
    # Greedy seed: largest-first
    selected = []
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        candidate = selected + [x]
        ok = True
        for a in candidate:
            others = [v for v in candidate if v != a]
            for t in combinations(others, K):
                if sum(t) % a == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    
    if not selected:
        return list(range(N, N - 48, -1))
    
    # Local search to improve
    # Precompute: for each candidate element, check if it can be added
    def can_add(elem, sel):
        """Check if elem can be added to sel without violations."""
        # Check elem doesn't divide sum of any 3 in sel
        for t in combinations(sel, K):
            if sum(t) % elem == 0:
                return False
        # Check each existing element doesn't divide sum of (K-1 from sel) + elem
        for a in sel:
            # Need: exists 2 from sel (excluding a) such that (sum of those 2 + elem) % a == 0
            # OR elem divides... wait, we need a doesn't divide any 3-sum that includes elem
            # 3-sums including elem: elem + x + y where x,y in sel, x!=a, y!=a, x!=y
            # So check: for all pairs (x,y) from sel\{a}, (elem + x + y) % a != 0
            others_no_a = [v for v in sel if v != a]
            for x, y in combinations(others_no_a, 2):
                if (elem + x + y) % a == 0:
                    return False
        return True
    
    def count_violations_if_added(elem, sel):
        """Count how many violations would occur if elem added to sel."""
        count = 0
        # elem divides sum of 3 in sel
        for t in combinations(sel, K):
            if sum(t) % elem == 0:
                count += 1
        # Each a in sel: a divides (elem + x + y) for some x,y in sel\{a}
        for a in sel:
            others_no_a = [v for v in sel if v != a]
            for x, y in combinations(others_no_a, 2):
                if (elem + x + y) % a == 0:
                    count += 1
                    break  # one violation per a is enough
        return count
    
    # Local search: try to add elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unused element
        unused = [x for x in range(1, N + 1) if x not in selected]
        # Sort by "ease" - try elements that cause fewest violations
        candidates = []
        for elem in unused:
            if time.time() > deadline:
                break
            v = count_violations_if_added(elem, selected)
            if v == 0:
                candidates.append((0, elem))
            elif v <= 2:
                candidates.append((v, elem))
        
        candidates.sort()
        for v, elem in candidates:
            if time.time() > deadline:
                break
            if v == 0:
                selected.append(elem)
                improved = True
                break
    
    # Try swap-based local search
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping: remove one, add one or two
            # First try: remove one element, see if we can add more
            for i, rem in enumerate(list(selected)):
                if time.time() > deadline:
                    break
                sel_without = selected[:i] + selected[i+1:]
                # Try adding elements to sel_without
                unused = [x for x in range(1, N + 1) if x not in sel_without]
                added_any = False
                for elem in unused:
                    if time.time() > deadline:
                        break
                    if can_add(elem, sel_without):
                        selected = sel_without + [elem]
                        improved = True
                        added_any = True
                        break
                if not added_any:
                    # Try adding two elements
                    for e1 in unused:
                        if time.time() > deadline:
                            break
                        sel_with_e1 = sel_without + [e1]
                        for e2 in [x for x in unused if x != e1]:
                            if time.time() > deadline:
                                break
                            if can_add(e2, sel_with_e1):
                                selected = sel_with_e1 + [e2]
                                improved = True
                                added_any = True
                                break
                        if added_any:
                            break
                if improved:
                    break
    
    # Final: ensure we have at most MAX_SIZE and return
    if len(selected) > MAX_SIZE:
        selected = selected[:MAX_SIZE]
    
    return selected
# EVOLVE-BLOCK-END
