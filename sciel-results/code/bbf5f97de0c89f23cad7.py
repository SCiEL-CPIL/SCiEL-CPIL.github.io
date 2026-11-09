import time
import random
import math

def _precompute(n_max=180):
    """
    Precompute forbidden triples and conflict pairs.
    For each a, find (b, c) with b < c such that 1/a = 1/b + 1/c.
    Equation: (b-a)(c-a) = a^2.
    Let d be a divisor of a^2, d < a^2/d. Then b = a + d, c = a + a^2/d.
    We require b, c in [1, n_max] and b != c (d != a^2/d => d != a).
    """
    conflicts = [[] for _ in range(n_max + 1)]
    triples = []
    
    for a in range(1, n_max + 1):
        a2 = a * a
        # Find divisors of a2
        divisors = []
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d == 0:
                divisors.append(d)
                if d != a2 // d:
                    divisors.append(a2 // d)
        
        seen_pairs = set()
        for d in divisors:
            if d == a:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if 1 <= b <= n_max and 1 <= c <= n_max and b != c:
                pair = (min(b, c), max(b, c))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    conflicts[a].append(pair)
                    triples.append((a, b, c))
    
    # Also build reverse conflicts: for each element x, list of pairs (y, z) 
    # such that {x, y, z} is a forbidden triple.
    # This is more efficient for checking if adding x creates a conflict.
    conflict_pairs = [[] for _ in range(n_max + 1)]
    for a, b, c in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    return conflict_pairs, triples

def _is_valid(s, conflict_pairs):
    """Check if set s is valid."""
    s_set = set(s)
    for a in s_set:
        for b, c in conflict_pairs[a]:
            if b in s_set and c in s_set:
                return False
    return True

def _greedy_build(conflict_pairs, order):
    """Greedy build: add elements in the given order if they don't create conflicts."""
    s = set()
    for a in order:
        # Check if adding a creates a conflict with existing pairs in s
        ok = True
        for b, c in conflict_pairs[a]:
            if b in s and c in s:
                ok = False
                break
        
        if ok:
            s.add(a)
    return s

def _tabu_search(s, conflict_pairs, n_max, deadline, rng, tabu_len=10):
    """
    Tabu Search to improve the set.
    Moves: 1-swap, 2-swap.
    Tabu list prevents immediate reversal.
    """
    s = set(s)
    tabu_list = [] # List of (element, iteration)
    current_size = len(s)
    best_size = current_size
    best_set = s.copy()
    
    # Precompute missing elements list for faster access? 
    # We will generate random candidates on the fly.
    
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 2.0:
        iter_count += 1
        improved = False
        
        # Clean up tabu list
        while tabu_list and tabu_list[0][1] < iter_count - tabu_len:
            tabu_list.pop(0)
        
        # Generate candidate moves
        # 1. Try 1-swap: remove z, add x
        # 2. Try 2-swap: remove z1, z2, add x1, x2
        
        # To avoid cycling, we prioritize moves that are NOT in tabu list.
        
        # Strategy: Randomly pick a few swap candidates and evaluate them.
        # We want to find a move that increases size or maintains size while moving to a new area.
        
        s_list = list(s)
        missing = [x for x in range(1, n_max + 1) if x not in s]
        
        if not missing:
            break
            
        # Try 1-swap
        rng.shuffle(s_list)
        rng.shuffle(missing)
        
        # Try a few random 1-swaps
        found_move = False
        for _ in range(20):
            if time.time() > deadline - 2.0: break
            if not s_list or not missing: break
            
            z = rng.choice(s_list)
            x = rng.choice(missing)
            
            if x == z: continue
            
            # Check if this move is tabu
            is_tabu = False
            for (elem, it) in tabu_list:
                if elem == z or elem == x:
                    is_tabu = True
                    break
            
            if is_tabu:
                continue
                
            candidate = set(s)
            candidate.remove(z)
            candidate.add(x)
            
            if _is_valid(candidate, conflict_pairs):
                # Accept if size increases, or if size is same (to explore)
                # For maximization, we prefer size increase.
                # If size same, we accept to avoid getting stuck.
                new_size = len(candidate)
                if new_size > current_size or (new_size == current_size and rng.random() < 0.5):
                    s = candidate
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_set = s.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    tabu_list.append((z, iter_count))
                    tabu_list.append((x, iter_count))
                    found_move = True
                    break
        
        if found_move:
            continue
            
        # Try 2-swap
        if len(s) >= 2 and len(missing) >= 2:
            # Pick 2 to remove, 2 to add
            for _ in range(10):
                if time.time() > deadline - 2.0: break
                if len(s_list) < 2 or len(missing) < 2: break
                
                z1, z2 = rng.sample(s_list, 2)
                x1, x2 = rng.sample(missing, 2)
                
                # Check tabu
                is_tabu = False
                for (elem, it) in tabu_list:
                    if elem in (z1, z2, x1, x2):
                        is_tabu = True
                        break
                
                if is_tabu:
                    continue
                
                candidate = set(s)
                candidate.remove(z1)
                candidate.remove(z2)
                candidate.add(x1)
                candidate.add(x2)
                
                if _is_valid(candidate, conflict_pairs):
                    new_size = len(candidate)
                    if new_size > current_size or (new_size == current_size and rng.random() < 0.3):
                        s = candidate
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_set = s.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        
                        tabu_list.append((z1, iter_count))
                        tabu_list.append((z2, iter_count))
                        tabu_list.append((x1, iter_count))
                        tabu_list.append((x2, iter_count))
                        found_move = True
                        break
                        
        if not found_move:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart tabu list to allow backtracking
                tabu_list = []
                no_improve_count = 0
                # Optionally, random kick: remove 2 random, add 2 random even if same size?
                # Or just break and let outer loop restart?
                # Let's try a random kick to escape local minimum
                if len(s) >= 2 and len(missing) >= 2:
                    z1, z2 = rng.sample(list(s), 2)
                    x1, x2 = rng.sample(missing, 2)
                    candidate = set(s)
                    candidate.remove(z1)
                    candidate.remove(z2)
                    candidate.add(x1)
                    candidate.add(x2)
                    if _is_valid(candidate, conflict_pairs):
                        s = candidate
                        current_size = len(candidate)
                        tabu_list = []

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n_max = 180
    conflict_pairs, triples = _precompute(n_max)
    
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Generate initial candidates using greedy with different orders
    candidates = []
    
    # Ascending
    candidates.append(_greedy_build(conflict_pairs, list(range(1, n_max + 1))))
    # Descending
    candidates.append(_greedy_build(conflict_pairs, list(range(n_max, 0, -1))))
    
    # Random orders
    for _ in range(30): # Reduced count to leave time for Tabu
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        candidates.append(_greedy_build(conflict_pairs, order))
        
    # Degree-based order
    degs = {a: len(conflict_pairs[a]) for a in range(1, n_max + 1)}
    order_asc = sorted(range(1, n_max + 1), key=lambda x: degs[x])
    order_desc = sorted(range(1, n_max + 1), key=lambda x: -degs[x])
    candidates.append(_greedy_build(conflict_pairs, order_asc))
    candidates.append(_greedy_build(conflict_pairs, order_desc))
    
    # Phase 2: Tabu Search on each candidate
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
        improved_set = _tabu_search(cand, conflict_pairs, n_max, deadline, rng)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set.copy()
            
    # Phase 3: Final Tabu Search on best set
    if time.time() < deadline - 3.0:
        best_set = _tabu_search(best_set, conflict_pairs, n_max, deadline, rng)
    
    return sorted(best_set)
