"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_bad_triples():
    """Find all triples (a,b,c) with a<b<c where lcm(a,b)=lcm(a,c)=lcm(b,c)."""
    bad = set()
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            bad.add((a, b, c))
    return bad

def _is_valid(selected, bad_triples):
    """Check if selected set has no bad triple."""
    sel_set = set(selected)
    for (a, b, c) in bad_triples:
        if a in sel_set and b in sel_set and c in sel_set:
            return False
    return True

def _greedy_initial(bad_triples, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = []
    selected_set = set()
    for x in order:
        ok = True
        for (a, b, c) in bad_triples:
            if x in (a, b, c):
                others = [y for y in (a, b, c) if y != x]
                if others[0] in selected_set and others[1] in selected_set:
                    ok = False
                    break
        if ok:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_improve(selected, bad_triples, time_limit=30):
    """Try to improve by removing one element and adding one or more others."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        for rem in list(selected_set):
            candidate = selected_set - {rem}
            for x in range(1, 129):
                if x in candidate:
                    continue
                ok = True
                for (a, b, c) in bad_triples:
                    if x in (a, b, c):
                        others = [y for y in (a, b, c) if y != x]
                        if others[0] in candidate and others[1] in candidate:
                            ok = False
                            break
                if ok:
                    new_set = candidate | {x}
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    bad_triples = _build_bad_triples()
    
    best = []
    best_size = 0
    
    sel = _greedy_initial(bad_triples, list(range(1, 129)))
    best = sel
    best_size = len(sel)
    
    if time.time() < deadline - 10:
        improved = _local_improve(best, bad_triples, time_limit=20)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy_initial(bad_triples, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        if len(sel) >= best_size - 1 and time.time() < deadline - 15:
            improved = _local_improve(sel, bad_triples, time_limit=10)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    if time.time() < deadline - 5 and len(best) > 0:
        improved = _local_improve(best, bad_triples, time_limit=30)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    if not _is_valid(best, bad_triples):
        best = _greedy_initial(bad_triples, list(range(1, 129)))
    
    return best
# EVOLVE-BLOCK-END
