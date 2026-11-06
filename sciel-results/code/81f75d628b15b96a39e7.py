"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # Precompute conflict contributions for each element pair (a, b)
    # conflict[a][b] = list of (c, g) such that gcd(a,b) == gcd(a,c) == gcd(b,c) == g
    # This allows O(1) update of conflict count when adding/removing elements.
    # Specifically, if we add x, the new conflicts introduced are pairs (a,b) in S
    # such that {a,b,x} is a bad triangle.
    # We can precompute for each x, the set of pairs (a,b) that form a bad triangle with x.
    
    # Bad triangle condition: gcd(a,b) == gcd(a,c) == gcd(b,c)
    # Let g = gcd(a,b). Then gcd(a,c) must be g and gcd(b,c) must be g.
    # This implies g divides c. Also c must share exactly the prime factors of g with a and b?
    # Actually, if gcd(a,b)=g, let a=g*a', b=g*b' with gcd(a',b')=1.
    # gcd(a,c) = g * gcd(a', c/g) if g|c, else gcd(g, c) which is < g (unless g|c).
    # If g|c, let c=g*c'. gcd(a,c) = g * gcd(a', c').
    # We need gcd(a,c)=g => gcd(a', c')=1.
    # Similarly gcd(b,c)=g => gcd(b', c')=1.
    # So for a fixed pair (a,b) with gcd g, the "bad" c's are multiples of g
    # such that gcd(c/g, a/g)=1 and gcd(c/g, b/g)=1.
    
    # To make SA fast, we maintain a "bad_count" for the current set.
    # When adding x, we need to check how many existing pairs (a,b) in S form a bad triangle with x.
    # Let's precompute for each x, a list of pairs (a,b) that are "bad" with x.
    # But that's 128 * (128*128/2) which is ~1M pairs. Storing them might be heavy but manageable.
    # Alternatively, for each x, we can precompute a list of (a, b) such that {a,b,x} is bad.
    # Let's try a different approach: 
    # For each x, precompute a bitmask or list of "conflict partners".
    # Actually, the condition is symmetric. 
    # Let's precompute for each x, a dictionary mapping g -> list of y such that gcd(x,y)=g.
    # Then for a candidate x, we check pairs (a,b) in S.
    # This is still O(|S|^2) per move if done naively.
    
    # Better: Precompute for each x, a set of pairs (a,b) that form a bad triangle with x.
    # There are C(128,3) ~ 340k triangles. We can iterate through all triangles and mark conflicts.
    # For each bad triangle (a,b,c), we increment a counter for each element? No.
    # We want to know: if I add x to S, how many NEW bad triangles are formed?
    # These are triangles (a,b,x) where a,b in S.
    # So for each x, we need to know the set of pairs (a,b) such that (a,b,x) is bad.
    # Let's call this P[x].
    # P[x] is a set of frozenset({a,b}) or just a bitmask of pairs?
    # 128 elements -> 128*127/2 = 8128 pairs. Fits in a 64-bit int? No. 
    # We can use a Python set of tuples (a,b) with a<b.
    
    # Precomputation of P[x]:
    # Iterate a < b < c.
    # Check if gcd(a,b)==gcd(a,c)==gcd(b,c).
    # If yes, add (a,b) to P[c], (a,c) to P[b], (b,c) to P[a].
    
    P = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    # Bad triangle
                    P[c].add((a, b))
                    P[b].add((a, c))
                    P[a].add((b, c))
    
    # Now, for a current set S (represented as a bitmask or boolean array),
    # the cost (number of bad triangles) is the sum over all x in S of 
    # (number of pairs (a,b) in P[x] such that a in S and b in S).
    # Let C[S] be the total cost.
    # When adding x, the change in cost is delta = count of (a,b) in P[x] where a,b in S.
    # When removing x, the change is -delta.
    
    # We will use a boolean array `in_set` for O(1) membership.
    # To speed up the count of (a,b) in P[x] where a,b in S:
    # We can iterate through P[x] and check in_set[a] and in_set[b].
    # Size of P[x] is at most C(127, 2) ~ 8000.
    # This is fast enough for 128 elements.
    
    # Initialize SA
    # Start with a greedy solution to have a good starting point
    def greedy_construct():
        selected = []
        # Sort by number of conflicts in full set? Or just random?
        # Let's use the previous strategy's greedy as init
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            # Check if x can be added
            # We need to check if adding x creates any bad triangle with existing pairs.
            # i.e., is there (a,b) in P[x] such that a,b in selected?
            can_add = True
            for (a, b) in P[x]:
                if a in selected and b in selected:
                    can_add = False
                    break
            if can_add:
                selected.append(x)
        return selected
    
    # We need a fast "in selected" check. Using a set is O(1) average.
    # But for the greedy init, we can use a list and a set.
    
    # Let's define the SA state
    # state: set of elements
    # cost: number of bad triangles
    
    def get_cost_delta_add(x, current_set):
        """Calculate increase in cost if x is added to current_set."""
        delta = 0
        for (a, b) in P[x]:
            if a in current_set and b in current_set:
                delta += 1
        return delta
    
    def get_cost_delta_remove(x, current_set):
        """Calculate decrease in cost if x is removed from current_set."""
        # The triangles that disappear are those involving x.
        # They are (a,b,x) where a,b in current_set.
        # This is exactly the same count as get_cost_delta_add(x, current_set - {x})?
        # No, get_cost_delta_add checks pairs in current_set.
        # If we remove x, we lose triangles (a,b,x) where a,b are in the set (excluding x).
        # So it's the count of (a,b) in P[x] such that a,b in current_set (where x is not in current_set for the check? 
        # Wait, current_set includes x in the "remove" context? 
        # Let's say current_set contains x.
        # We want to remove x. The cost decreases by the number of bad triangles involving x.
        # These are (a,b,x) with a,b in current_set\{x}.
        # So we check (a,b) in P[x] and a in current_set and b in current_set.
        # Since x is in current_set, but a,b are distinct from x (indices in P[x] are pairs of other elements),
        # this is the same calculation.
        delta = 0
        for (a, b) in P[x]:
            if a in current_set and b in current_set:
                delta += 1
        return delta

    # SA Loop
    best_set = set()
    best_cost = float('inf')
    best_size = 0
    
    # Initialize with a few greedy runs
    for _ in range(5):
        init_set = set(greedy_construct())
        # Calculate initial cost (should be 0 for valid greedy)
        cost = 0
        # Verify cost is 0
        for x in init_set:
            cost += get_cost_delta_add(x, init_set - {x})
        # Wait, get_cost_delta_add(x, S) counts pairs in S.
        # If S is valid, cost is 0.
        
        if len(init_set) > best_size:
            best_size = len(init_set)
            best_set = set(init_set)
            best_cost = cost
            
    # SA parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_set = set(best_set)
    current_cost = best_cost
    
    # We want to MAXIMIZE size.
    # Objective function: Maximize |S| - lambda * Cost?
    # Or just accept moves that increase size even if cost increases?
    # The task requires VALID set (cost=0).
    # So we must end with cost=0.
    # Strategy: 
    # 1. Allow moves that increase size, even if cost increases (up to some limit).
    # 2. Allow moves that reduce cost (repair).
    # 3. Accept worse states with probability exp(-delta_obj / T).
    
    # Let's define objective: f(S) = |S| * 100 - Cost(S).
    # We want to maximize f.
    
    def objective(S, cost):
        return len(S) * 100 - cost
    
    current_obj = objective(current_set, current_cost)
    
    num_steps = 50000
    for step in range(num_steps):
        if time.time() > deadline:
            break
            
        # Move: Add a random element not in S, or Remove a random element in S
        if random.random() < 0.5:
            # Add
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            delta_cost = get_cost_delta_add(x, current_set)
            new_cost = current_cost + delta_cost
            new_size = len(current_set) + 1
            new_obj = objective(set(), new_cost) # placeholder
            new_obj = new_size * 100 - new_cost
            delta_obj = new_obj - current_obj
            
            if delta_obj > 0 or random.random() < (2.718281828 ** (delta_obj / T)):
                current_set.add(x)
                current_cost = new_cost
                current_obj = new_obj
                
                if current_cost == 0 and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    best_cost = current_cost
        else:
            # Remove
            if not current_set:
                continue
            x = random.choice(list(current_set))
            delta_cost = -get_cost_delta_remove(x, current_set) # cost decreases
            new_cost = current_cost + delta_cost
            new_size = len(current_set) - 1
            new_obj = new_size * 100 - new_cost
            delta_obj = new_obj - current_obj
            
            if delta_obj > 0 or random.random() < (2.718281828 ** (delta_obj / T)):
                current_set.remove(x)
                current_cost = new_cost
                current_obj = new_obj
                
                if current_cost == 0 and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    best_cost = current_cost
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Reheat occasionally?
            if step % 10000 == 0:
                T = 0.5

    # Final polish: Greedy add
    for x in range(1, N + 1):
        if x not in best_set:
            if get_cost_delta_add(x, best_set) == 0:
                best_set.add(x)
    
    return list(best_set)
# EVOLVE-BLOCK-END
