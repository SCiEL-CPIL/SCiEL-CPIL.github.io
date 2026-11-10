import time
import random
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    
    def count_3aps(S):
        """Count distinct 3-term APs in set S (positive difference, counted once)."""
        S_set = set(S)
        count = 0
        for a in S:
            for c in S:
                if c > a and (c + a) % 2 == 0:
                    b = (a + c) // 2
                    if b in S_set and b > a:
                        count += 1
        return count
    
    def has_4ap(S):
        """Check if set S contains a nonconstant 4-term AP."""
        S_set = set(S)
        for a in S:
            for d in range(1, (N - a) // 3 + 1):
                if a + d in S_set and a + 2*d in S_set and a + 3*d in S_set:
                    return True
        return False
    
    def find_4aps(S):
        """Find all 4-APs in set S."""
        S_set = set(S)
        aps = []
        for a in S:
            for d in range(1, (N - a) // 3 + 1):
                if a + d in S_set and a + 2*d in S_set and a + 3*d in S_set:
                    aps.append((a, d))
        return aps
    
    def count_3aps_fast(S_set):
        """Count 3-APs given a set."""
        count = 0
        for a in S_set:
            for c in S_set:
                if c > a and (c + a) % 2 == 0:
                    b = (a + c) // 2
                    if b in S_set and b > a:
                        count += 1
        return count
    
    def has_4ap_fast(S_set):
        """Check 4-AP given a set."""
        for a in S_set:
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if a + d in S_set and a + 2*d in S_set and a + 3*d in S_set:
                    return True
        return False
    
    # Precompute for each element, which 4-APs it participates in
    # and which 3-APs it participates in
    
    def get_4aps_containing(x, S_set):
        """Get all 4-APs containing x in S_set."""
        aps = []
        # x as first term: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if x + d in S_set and x + 2*d in S_set and x + 3*d in S_set:
                aps.append((x, d))
        # x as second term: x-d, x, x+d, x+2d
        for d in range(1, x):
            if x - d in S_set and x + d in S_set and x + 2*d in S_set:
                aps.append((x - d, d))
        # x as third term: x-2d, x-d, x, x+d
        for d in range(1, x // 2 + 1):
            if x - 2*d in S_set and x - d in S_set and x + d in S_set:
                aps.append((x - 2*d, d))
        # x as fourth term: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if x - 3*d in S_set and x - 2*d in S_set and x - d in S_set:
                aps.append((x - 3*d, d))
        return aps
    
    def get_3aps_containing(x, S_set):
        """Get all 3-APs containing x in S_set."""
        aps = []
        # x as first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if x + d in S_set and x + 2*d in S_set:
                aps.append((x, d))
        # x as second term: x-d, x, x+d
        for d in range(1, x):
            if x - d in S_set and x + d in S_set:
                aps.append((x - d, d))
        # x as third term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if x - 2*d in S_set and x - d in S_set:
                aps.append((x - 2*d, d))
        return aps
    
    # GRASP + Tabu Search
    
    def greedy_construction(randomize=True):
        """Build a 4-AP-free set greedily."""
        S = set()
        candidates = list(range(1, N + 1))
        if randomize:
            random.shuffle(candidates)
        
        for x in candidates:
            # Check if adding x creates a 4-AP
            S_test = S | {x}
            if not has_4ap_fast(S_test):
                S = S_test
        
        return S
    
    def local_search(S, max_iter=5000):
        """Local search with tabu memory."""
        S_set = set(S)
        current_score = count_3aps_fast(S_set)
        best_S = set(S_set)
        best_score = current_score
        
        tabu_list = {}  # element -> iteration when it becomes non-tabu
        tabu_tenure = 20
        iteration = 0
        
        while iteration < max_iter:
            if time.time() > DEADLINE:
                break
            
            best_move = None
            best_move_score = current_score
            best_move_type = None
            best_move_elem = None
            
            # Try adding elements
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if tabu_list.get(x, 0) > iteration:
                    continue
                S_test = S_set | {x}
                if has_4ap_fast(S_test):
                    continue
                new_score = count_3aps_fast(S_test)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_type = 'add'
                    best_move_elem = x
                    best_move = (x, new_score)
            
            # Try removing elements
            for x in S_set:
                if tabu_list.get(x, 0) > iteration:
                    continue
                S_test = S_set - {x}
                new_score = count_3aps_fast(S_test)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_type = 'remove'
                    best_move_elem = x
                    best_move = (x, new_score)
            
            # Try swaps
            for x in S_set:
                if tabu_list.get(x, 0) > iteration:
                    continue
                for y in range(1, N + 1):
                    if y in S_set or y == x:
                        continue
                    if tabu_list.get(y, 0) > iteration:
                        continue
                    S_test = (S_set - {x}) | {y}
                    if has_4ap_fast(S_test):
                        continue
                    new_score = count_3aps_fast(S_test)
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move_type = 'swap'
                        best_move_elem = (x, y)
                        best_move = ((x, y), new_score)
            
            if best_move is None:
                break
            
            # Apply best move
            if best_move_type == 'add':
                S_set.add(best_move_elem)
                tabu_list[best_move_elem] = iteration + tabu_tenure
            elif best_move_type == 'remove':
                S_set.discard(best_move_elem)
                tabu_list[best_move_elem] = iteration + tabu_tenure
            elif best_move_type == 'swap':
                x, y = best_move_elem
                S_set.discard(x)
                S_set.add(y)
                tabu_list[x] = iteration + tabu_tenure
                tabu_list[y] = iteration + tabu_tenure
            
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S_set)
            
            iteration += 1
        
        return best_S, best_score
    
    # Run GRASP: multiple greedy constructions + local search
    best_overall = set()
    best_overall_score = 0
    
    # First, try a deterministic greedy
    S0 = greedy_construction(randomize=False)
    best_S, best_score = local_search(S0, max_iter=3000)
    best_overall = best_S
    best_overall_score = best_score
    
    # Try randomized greedy constructions
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        
        S_i = greedy_construction(randomize=True)
        best_S, best_score = local_search(S_i, max_iter=2000)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    # Also try starting from a structured 4-AP-free set
    # Numbers with no digit 2 in base 4 (Cantor-like)
    S_cantor = set()
    for x in range(1, N + 1):
        # Check if x in base 4 has only digits 0 and 1
        t = x
        valid = True
        while t > 0:
            if t % 4 >= 2:
                valid = False
                break
            t //= 4
        if valid:
            S_cantor.add(x)
    
    if S_cantor and not has_4ap_fast(S_cantor):
        best_S, best_score = local_search(S_cantor, max_iter=3000)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    # Try another structured set: numbers ≡ 0 or 1 mod 4, but filter out 4-APs
    S_mod = set(x for x in range(1, N + 1) if x % 4 in (0, 1))
    # This might have 4-APs, so filter
    S_mod_filtered = set()
    for x in sorted(S_mod):
        S_test = S_mod_filtered | {x}
        if not has_4ap_fast(S_test):
            S_mod_filtered = S_test
    
    if S_mod_filtered:
        best_S, best_score = local_search(S_mod_filtered, max_iter=3000)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    # Final: try a few more random restarts if time permits
    while time.time() < DEADLINE - 10:
        S_rand = greedy_construction(randomize=True)
        best_S, best_score = local_search(S_rand, max_iter=1500)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    return sorted(best_overall)
