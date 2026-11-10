import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 192
    MAX_SIZE = 48
    
    def time_left():
        return deadline - time.time()
    
    # Precompute conflicts: for each a, list of triples (b,c,d) with b,c,d distinct, != a, and a | (b+c+d)
    # Store as: conflicts[a] = list of frozensets {b,c,d}
    # But that's too much memory. Instead, for checking, we'll compute on the fly.
    
    def can_add_to_set(S_set, x):
        """Check if x can be added to set S_set without violating constraints."""
        S_list = list(S_set)
        n = len(S_list)
        
        # Check: does x divide any triple sum from S_set?
        if n >= 3:
            # For each pair (b,c) in S_list, check if there's d in S_list\{b,c} with (b+c+d) % x == 0
            # i.e., d % x == (-b-c) % x
            residue_count = {}
            for v in S_list:
                r = v % x
                residue_count[r] = residue_count.get(r, 0) + 1
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    need = (-b - c) % x
                    cnt = residue_count.get(need, 0)
                    if b % x == need:
                        cnt -= 1
                    if c % x == need:
                        cnt -= 1
                    if cnt > 0:
                        return False
        
        # Check: for each a in S_set, does a divide (b+c+x) for any pair (b,c) in S_set\{a}?
        for a in S_list:
            if a == x:
                continue
            # Need: exists b,c in S_set\{a,x} such that (b+c+x) % a == 0
            # i.e., (b+c) % a == (-x) % a
            need_bc = (-x) % a
            # Count pairs (b,c) in S_set\{a} with (b+c) % a == need_bc
            # Use residue counting
            residues = {}
            for v in S_list:
                if v == a:
                    continue
                r = v % a
                residues[r] = residues.get(r, 0) + 1
            # Count pairs with sum ≡ need_bc (mod a)
            pair_count = 0
            for r in residues:
                comp = (need_bc - r) % a
                if comp < r:
                    continue  # avoid double counting
                if comp == r:
                    cnt = residues[r]
                    pair_count += cnt * (cnt - 1) // 2
                else:
                    if comp in residues:
                        pair_count += residues[r] * residues[comp]
            if pair_count > 0:
                return False
        
        return True
    
    def is_valid(S_set):
        """Verify the entire set is valid."""
        S_list = list(S_set)
        n = len(S_list)
        for idx_a, a in enumerate(S_list):
            others = [S_list[i] for i in range(n) if i != idx_a]
            m = len(others)
            if m < 3:
                continue
            residue_set = {}
            for x in others:
                r = x % a
                residue_set[r] = residue_set.get(r, 0) + 1
            for i in range(m):
                b = others[i]
                for j in range(i + 1, m):
                    c = others[j]
                    need = (-b - c) % a
                    cnt = residue_set.get(need, 0)
                    if b % a == need:
                        cnt -= 1
                    if c % a == need:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    # WalkSAT-style local search
    def walksat_search(initial_set, max_iters=50000):
        """WalkSAT-style search to maximize cardinality while maintaining validity."""
        current = set(initial_set)
        current_size = len(current)
        full_range = set(range(1, N + 1))
        best = set(current)
        best_size = current_size
        
        for it in range(max_iters):
            if time_left() < 2.0:
                break
            
            # Try to add a random element
            complement = full_range - current
            if len(complement) == 0:
                break
            
            # With some probability, try to remove and add
            if random.random() < 0.3 and current_size > 10:
                # Remove a random element, try to add a better one
                removed = random.choice(list(current))
                current.remove(removed)
                # Try to add from complement
                candidates = list(complement) + [removed]
                random.shuffle(candidates)
                added = False
                for c in candidates[:20]:
                    if c in current:
                        continue
                    if can_add_to_set(current, c):
                        current.add(c)
                        added = True
                        break
                if not added:
                    current.add(removed)  # put it back
            else:
                # Try to add a random element
                candidates = list(complement)
                random.shuffle(candidates)
                added = False
                for c in candidates[:10]:
                    if can_add_to_set(current, c):
                        current.add(c)
                        added = True
                        break
            
            if len(current) > best_size:
                best_size = len(current)
                best = set(current)
            
            # If we reached MAX_SIZE, we can stop
            if best_size >= MAX_SIZE:
                break
        
        return best
    
    def greedy_build(order):
        """Greedy construction in given order."""
        selected = []
        sel_set = set()
        for x in order:
            if time_left() < 2.0:
                break
            if len(selected) >= MAX_SIZE:
                break
            if can_add_to_set(sel_set, x):
                selected.append(x)
                sel_set.add(x)
        return selected
    
    def local_improve(S_set):
        """Try to improve by adding elements."""
        current = set(S_set)
        full_range = set(range(1, N + 1))
        improved = True
        while improved and time_left() > 2.0:
            improved = False
            complement = full_range - current
            for y in sorted(complement):
                if time_left() < 2.0:
                    break
                if len(current) >= MAX_SIZE:
                    break
                if can_add_to_set(current, y):
                    current.add(y)
                    improved = True
                    break
        return current
    
    # Strategy: multiple approaches
    best_set = set()
    best_size = 0
    
    # Approach 1: Descending greedy (large numbers first)
    s = greedy_build(list(range(N, 0, -1)))
    s = local_improve(set(s))
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # Approach 2: WalkSAT from descending greedy
    if time_left() > 5.0:
        s2 = walksat_search(set(s), max_iters=30000)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = set(s2)
    
    # Approach 3: Randomized greedy + WalkSAT
    random.seed(42)
    for trial in range(15):
        if time_left() < 5.0:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build(order)
        s = local_improve(set(s))
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        if time_left() > 5.0 and best_size < MAX_SIZE:
            s2 = walksat_search(set(s), max_iters=20000)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)
    
    # Approach 4: Focus on upper range
    if time_left() > 5.0:
        upper = list(range(N, N // 2, -1))
        lower = list(range(N // 2, 0, -1))
        s = greedy_build(upper + lower)
        s = local_improve(set(s))
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        if time_left() > 5.0 and best_size < MAX_SIZE:
            s2 = walksat_search(set(s), max_iters=20000)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)
    
    # Approach 5: Numbers with few divisors
    if time_left() > 5.0:
        def score_num(x):
            if x == 1:
                return -1000
            divisors = 0
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0:
                    divisors += 2
            return -divisors
        order4 = sorted(range(1, N + 1), key=score_num, reverse=True)
        s = greedy_build(order4)
        s = local_improve(set(s))
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        if time_left() > 5.0 and best_size < MAX_SIZE:
            s2 = walksat_search(set(s), max_iters=20000)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)
    
    # Final: try WalkSAT from best
    if time_left() > 3.0 and best_size < MAX_SIZE:
        s2 = walksat_search(best_set, max_iters=30000)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = set(s2)
    
    # Verify
    if not is_valid(best_set):
        # Fallback to a known valid set
        best_set = set(greedy_build(list(range(N, 0, -1))))
    
    return list(best_set)[:MAX_SIZE]
