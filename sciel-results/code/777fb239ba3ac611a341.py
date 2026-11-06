"""Combined program: Fast conflict-based validation with 1-swap and 2-swap local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We use 85s to be safe.
    DEADLINE = start_time + 85.0
    
    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Precompute Conflict Structure
    # conflicts[x] is a list of pairs (a, b) such that {x, a, b} is a forbidden triple.
    # i.e., gcd(x,a) == gcd(x,b) == gcd(a,b)
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g_xa = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                # Check if gcd(x,b) and gcd(a,b) are also g_xa
                if gcd_table[x][b] == g_xa and gcd_table[a][b] == g_xa:
                    conflicts[x].append((a, b))

    def can_add(x, S_set):
        """
        Check if x can be added to S_set without creating a forbidden triple.
        We just need to check if any conflicting pair for x is fully contained in S_set.
        """
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def potential(S_set):
        """
        Count how many elements not in S_set can be added to S_set.
        Used to guide swaps that maintain size but improve future flexibility.
        """
        count = 0
        for x in range(1, N + 1):
            if x not in S_set:
                if can_add(x, S_set):
                    count += 1
        return count

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        """
        Improve S using Greedy Add, Potential-Guided 1-Swaps, and 2-Swaps.
        """
        S = set(S)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            
            # Phase 1: Try to add any element (Greedy Add)
            # Iterate in a random or fixed order? Fixed is deterministic, random helps escape symmetry.
            # For speed, we check all.
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        break # Restart loop to try adding more immediately
            
            if improved:
                continue
            
            # Phase 2: If no addition possible, try 1-Swap
            # Goal: Maintain size, but increase potential. Or just try all 1-swaps if size can be increased?
            # Actually, if no single element can be added, a 1-swap (remove a, add x) cannot increase size.
            # It can only maintain size. We use it to escape local optima by changing the "shape" of S.
            current_pot = potential(S)
            best_swap_1 = None
            best_pot_1 = current_pot
            
            S_list = list(S)
            # Optimization: Only try removing elements that are "critical" or just sample?
            # With N=128, |S| is maybe 20-30. 30 * 128 = 3840 checks. Very fast.
            for a in S_list:
                S_minus_a = S - {a}
                # Check if we can add any x to S_minus_a
                # We want to find an x such that potential(S_minus_a + {x}) > best_pot_1
                # But calculating potential for every x is expensive (O(N * degree)).
                # Instead, we can just try to find ANY x that makes the set "better" in terms of future adds.
                # A simpler heuristic: Try to add an x that has low conflict degree with the remaining set?
                # Let's stick to: Find x such that can_add(x, S_minus_a) is True.
                # Among those, pick one that maximizes potential.
                
                # To save time, we don't evaluate potential for EVERY x immediately.
                # We find candidate x's.
                candidates = []
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    if can_add(x, S_minus_a):
                        candidates.append(x)
                
                if not candidates:
                    continue
                    
                # Evaluate potential for candidates. Limit to top few if list is long?
                # candidates size is usually small.
                for x in candidates:
                    S_new = S_minus_a | {x}
                    pot = potential(S_new)
                    if pot > best_pot_1:
                        best_pot_1 = pot
                        best_swap_1 = (a, x)
            
            if best_swap_1:
                a, x = best_swap_1
                S.remove(a)
                S.add(x)
                improved = True
                continue

            # Phase 3: Try 2-Swap (Remove 2, Add 2)
            # This can increase size by 0 (if we remove 2 and add 2) or effectively "jump" to a better configuration.
            # Can it increase size? No, 2 out 2 in keeps size same.
            # Wait, the goal is to maximize cardinality.
            # If we are stuck in Phase 1 (no single add), and Phase 2 (no beneficial 1-swap), 
            # a 2-swap that maintains size but changes the set might allow a subsequent add.
            # So 2-swap is an escape mechanism.
            
            # To make it efficient, we only try 2-swaps if we haven't improved in a while?
            # Or just try a limited number of 2-swaps.
            
            # Generate a few random 2-swaps to try
            S_list = list(S)
            if len(S_list) < 2:
                break
                
            # Try up to 50 random 2-swaps
            found_2_swap = False
            for _ in range(50):
                if time.time() > DEADLINE:
                    break
                # Pick two distinct elements to remove
                a, b = random.sample(S_list, 2)
                S_minus_ab = S - {a, b}
                
                # Find two elements x, y to add
                # First find one addable
                candidates_1 = []
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    if can_add(x, S_minus_ab):
                        candidates_1.append(x)
                
                if not candidates_1:
                    continue
                    
                # Pick one, say x1. Then find x2 addable to S_minus_ab + {x1}
                x1 = random.choice(candidates_1)
                S_temp = S_minus_ab | {x1}
                
                candidates_2 = []
                for y in range(1, N + 1):
                    if y in S or y == x1:
                        continue
                    if can_add(y, S_temp):
                        candidates_2.append(y)
                
                if candidates_2:
                    x2 = random.choice(candidates_2)
                    S_new = S_temp | {x2}
                    
                    # Check if this new set is "better". 
                    # Since size is same, we compare potential.
                    # But calculating potential is expensive.
                    # Heuristic: If we found a valid 2-swap, accept it with some probability or just accept it to escape.
                    # Let's accept it if potential is higher, or randomly to escape.
                    # For simplicity and speed, let's just accept the first valid 2-swap found to escape the local optimum.
                    S = S_new
                    improved = True
                    found_2_swap = True
                    break

            if not found_2_swap:
                # Stuck
                break
        
        return S

    best_S = set()
    best_size = 0
    
    # Initializations
    orders = []
    orders.append(list(range(1, N + 1)))
    orders.append(list(range(N, 0, -1)))
    
    # Degree-sorted orders
    degrees = [len(conflicts[x]) for x in range(N + 1)]
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True)) # High conflict first? Or last?
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x]))
    
    # Random orders
    random.seed(42)
    for _ in range(20):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > DEADLINE - 5:
            break
            
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
            # Early termination if we hit a high score?
            # Max possible is unknown, but if we get > 30, it's likely very good.
            if best_size >= 35:
                break

    # Final cleanup: try to add anything to best_S
    if time.time() < DEADLINE - 2:
        S = best_S
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
        best_S = S

    return list(best_S)
# EVOLVE-BLOCK-END
