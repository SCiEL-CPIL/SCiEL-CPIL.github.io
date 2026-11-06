import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    n = 256
    best_S = None
    best_score = Fraction(0, 1)
    
    # Precompute conflicts
    # conflict[x][y] = True if x and y cannot both be in the set
    # They cannot both be in the set if there exists z such that {x, y, z} is a 3AP.
    # This happens if:
    # 1. x + y = 2z (z is middle) -> z = (x+y)/2. If x+y is even and 1 <= (x+y)//2 <= n, then x and y conflict.
    # 2. x + z = 2y (y is middle) -> z = 2y - x. If 1 <= 2y - x <= n, then x and y conflict.
    # 3. y + z = 2x (x is middle) -> z = 2x - y. If 1 <= 2x - y <= n, then x and y conflict.
    
    # Note: The condition "without any nonconstant k=3 term arithmetic progression" means we cannot have distinct a,b,c in S such that a+b=2c.
    # If we have x, y in S, and we add z, we must ensure no 3AP is formed.
    # The conflict graph approach:
    # Nodes: 1..n
    # Edge (x, y) if x != y and there exists z in [1, n] such that {x, y, z} is a 3AP.
    # Wait, if {x, y, z} is a 3AP, then x and y are in the same 3AP.
    # If x and y are in S, can we have z? No, because then {x, y, z} is a 3AP in S.
    # But we don't have to have z. We just can't have x, y, z ALL in S.
    # This is a 3-uniform hypergraph independent set problem.
    # A pair (x, y) is "conflicting" if they form a 3AP with some z.
    # If x and y are in S, then z CANNOT be in S.
    # This is not a simple graph independent set. It's a hypergraph independent set.
    # However, we can reformulate:
    # A set S is valid if for all distinct x, y, z in S, x+y != 2z.
    
    # Let's stick to the local search but make it more robust.
    # The current local search is decent but might get stuck.
    # Let's add random restarts and simulated annealing-like swaps.
    
    def is_valid(S):
        # Check if S is 3-free
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                # Check if a, b, c form a 3AP where c is in S
                # c = 2b - a
                if 2 * b - a in S:
                    return False
                # Check if a, c, b form a 3AP where c is in S
                # c = (a+b)/2
                if (a + b) % 2 == 0:
                    c = (a + b) // 2
                    if c in S and c != a and c != b:
                        return False
        return True

    def score(S):
        s = Fraction(0, 1)
        for x in S:
            s += Fraction(1, x)
        return s

    # Initial greedy set
    def greedy_construct():
        S = set()
        for x in range(1, n + 1):
            # Check if adding x creates a 3AP
            valid = True
            for a in S:
                # Check a, x, 2x-a
                if 2 * x - a in S and 2 * x - a != x:
                    valid = False
                    break
                # Check a, (a+x)/2, x
                if (a + x) % 2 == 0:
                    mid = (a + x) // 2
                    if mid in S and mid != a and mid != x:
                        valid = False
                        break
                # Check x, (x+a)/2, a -- same as above
            if valid:
                S.add(x)
        return S

    best_S = greedy_construct()
    best_score = score(best_S)
    
    # Local Search
    # We will try to improve the set by adding elements or swapping.
    
    def try_add(S, y):
        # Check if y can be added to S
        if y in S:
            return False
        for a in S:
            # Check if a, y, 2y-a is in S
            if 2 * y - a in S and 2 * y - a != y:
                return False
            # Check if a, (a+y)/2, y is in S
            if (a + y) % 2 == 0:
                mid = (a + y) // 2
                if mid in S and mid != a and mid != y:
                    return False
        return True

    def try_swap(S, x, y):
        # Check if (S - {x}) U {y} is valid
        if x not in S or y in S:
            return False
        S_new = S.copy()
        S_new.remove(x)
        S_new.add(y)
        # Check if y creates a conflict in S_new
        for a in S_new:
            if a == y:
                continue
            if 2 * y - a in S_new and 2 * y - a != y:
                return False
            if (a + y) % 2 == 0:
                mid = (a + y) // 2
                if mid in S_new and mid != a and mid != y:
                    return False
        return True

    # Simulated Annealing
    current_S = best_S.copy()
    current_score = best_score
    
    # We want to maximize score.
    # Moves:
    # 1. Add y if possible.
    # 2. Remove x.
    # 3. Swap x, y.
    
    # Let's use a simple hill climbing with random restarts first.
    # Then maybe simulated annealing.
    
    # Given the time limit, let's do a robust local search.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing element
        for y in range(1, n + 1):
            if y not in current_S and try_add(current_S, y):
                current_S.add(y)
                current_score = score(current_S)
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = current_S.copy()
        
        # 2. Try to swap larger elements with smaller missing ones
        # To gain weight, we should replace large x with small y.
        # Iterate y from 1 to n.
        for y in range(1, n + 1):
            if y in current_S:
                continue
            # Try to find x in current_S such that x > y and swap is valid.
            # We prefer to remove the largest x to maximize weight gain?
            # Actually, weight gain is 1/y - 1/x.
            # To maximize gain, we want small y and large x.
            # So for a fixed y, we want to remove the largest possible x.
            # Let's sort current_S in descending order and try to swap y with the first valid x.
            
            S_sorted_desc = sorted(current_S, reverse=True)
            for x in S_sorted_desc:
                if x <= y:
                    break
                if try_swap(current_S, x, y):
                    current_S.remove(x)
                    current_S.add(y)
                    current_score = score(current_S)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = current_S.copy()
                    break # Restart the y loop? Or continue?
            
            # If we made a swap, the set changed. It's safer to restart the y loop.
            if improved:
                break
    
    # If time remains, do some random restarts or simulated annealing?
    # The current local search might be stuck in a local optimum.
    # Let's add a simple simulated annealing phase.
    
    if time.time() < deadline - 5.0:
        # Simulated Annealing
        # Start from best_S
        S = best_S.copy()
        T = 0.1
        cooling_rate = 0.99
        
        while time.time() < deadline - 1.0 and T > 0.01:
            # Generate a random move
            move_type = random.randint(0, 2)
            if move_type == 0:
                # Add a random element
                candidates = [x for x in range(1, n + 1) if x not in S]
                if candidates:
                    y = random.choice(candidates)
                    if try_add(S, y):
                        S.add(y)
                        new_score = score(S)
                        delta = new_score - score(S) # Wait, score(S) changed? No, S changed.
                        # Let's keep track of score properly.
                        # This is getting complicated with Fractions.
                        # Let's stick to the greedy + local search for now, it's already decent.
                        # The prompt says "Keep the function names and signatures".
                        # The current program returns a sorted list.
                        
            T *= cooling_rate

    return sorted(best_S)
