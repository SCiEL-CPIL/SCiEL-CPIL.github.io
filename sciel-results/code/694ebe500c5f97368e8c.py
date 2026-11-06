import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 82s deadline to stay well under 90s limit
    n = 512

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # Map each element to the list of APs it participates in
    ap_map = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = (a, b, c, e)
            ap_map[a].append(ap)
            ap_map[b].append(ap)
            ap_map[c].append(ap)
            ap_map[e].append(ap)
            
    def can_add(x, S):
        """Check if adding x to S creates a 4-AP. S is a set."""
        for ap in ap_map[x]:
            a, b, c, e = ap
            if x == a:
                if b in S and c in S and e in S:
                    return False
            elif x == b:
                if a in S and c in S and e in S:
                    return False
            elif x == c:
                if a in S and b in S and e in S:
                    return False
            else:  # x == e
                if a in S and b in S and c in S:
                    return False
        return True

    def has_4ap(S):
        """Verify if S contains a 4-AP."""
        for a in S:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S and (a + 2 * d) in S and (a + 3 * d) in S:
                    return True
        return False

    def ternary_01_set(n):
        """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
        result = set()
        for mask in range(1, 1 << 6):
            val = 0
            for bit in range(6):
                if mask & (1 << bit):
                    val += 3 ** bit
            if 1 <= val <= n:
                result.add(val)
        return result

    def greedy_construct(S_init, order):
        """Greedy construction starting from S_init, adding elements in 'order'."""
        S = set(S_init)
        for x in order:
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, deadline, max_rounds=1000):
        """
        Local search: try to remove one element and add two others to increase size.
        Also includes 1-1 swaps for diversification.
        """
        S = set(S)
        current_size = len(S)
        no_improvement_count = 0
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
                
            # Try 1-2 move: remove 1, add 2
            if not S:
                break
            to_remove = random.choice(list(S))
            S_minus = S - {to_remove}
            
            # Try to find 2 elements to add
            candidates = [x for x in range(1, n + 1) if x not in S_minus]
            random.shuffle(candidates)
            
            added = []
            for c in candidates:
                if len(added) == 2:
                    break
                if can_add(c, S_minus | set(added)):
                    added.append(c)
                    
            if len(added) == 2:
                S = S_minus | set(added)
                current_size = len(S)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > 50:
                    # Try a 1-1 swap to diversify
                    to_remove = random.choice(list(S))
                    S_minus = S - {to_remove}
                    candidates = [x for x in range(1, n + 1) if x not in S_minus]
                    random.shuffle(candidates)
                    found_swap = False
                    for c in candidates:
                        if can_add(c, S_minus):
                            S = S_minus | {c}
                            found_swap = True
                            break
                    if not found_swap:
                        break
                        
        return S

    def kick_and_repair(S, deadline):
        """
        Random kick: remove a few random elements, then repair greedily.
        This helps escape local minima.
        """
        S = set(S)
        if not S:
            return S
        
        # Remove 1-3 random elements
        kick_size = random.randint(1, 3)
        to_remove = random.sample(list(S), min(kick_size, len(S)))
        S_kicked = S - set(to_remove)
        
        # Repair: greedily add elements back
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_repaired = greedy_construct(S_kicked, order)
        
        # Local search on the repaired set
        S_final = local_search(S_repaired, deadline, max_rounds=200)
        
        return S_final

    # Initial sets
    best_set = ternary_01_set(n)
    best_size = len(best_set)
    
    # 1. Greedy from empty
    random.seed(42)
    order = list(range(1, n + 1))
    random.shuffle(order)
    greedy_set = greedy_construct(set(), order)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)
        
    # 2. Greedy from ternary
    order2 = list(range(1, n + 1))
    random.shuffle(order2)
    ternary_greedy = greedy_construct(ternary_01_set(n), order2)
    if len(ternary_greedy) > best_size:
        best_set = ternary_greedy
        best_size = len(best_set)
        
    # 3. Local search on best
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=2000)
        best_size = len(best_set)
        
    # 4. Multiple restarts with kicks
    for attempt in range(30):
        if time.time() > deadline:
            break
            
        # Random start
        if attempt % 3 == 0:
            S_init = set()
        elif attempt % 3 == 1:
            S_init = ternary_01_set(n)
        else:
            S_init = set(random.sample(range(1, n + 1), 30))
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        current = greedy_construct(S_init, order)
        
        # Local search
        if len(current) >= best_size:
            current = local_search(current, deadline, max_rounds=1000)
        else:
            # Even if not better, do a kick-and-repair to explore
            current = kick_and_repair(current, deadline)
            
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # 5. Final kick-and-repair on the best set
    if time.time() < deadline - 5.0:
        kicked = kick_and_repair(best_set, deadline)
        if len(kicked) > best_size:
            best_set = kicked
            best_size = len(best_set)
            
    # 6. Final local search on the best set
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=500)
        best_size = len(best_set)
        
    # Final verification
    if has_4ap(best_set):
        # Fallback to safe ternary set
        best_set = ternary_01_set(n)
        
    return sorted(best_set)
