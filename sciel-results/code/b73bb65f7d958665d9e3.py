import time
import random
import math

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit
    
    # Precompute coordinates
    coords = [(i // N, i % N) for i in range(NUM_POINTS)]
    
    # We will maintain:
    # S: list of point indices in the current set
    # in_set: boolean array or set of indices in S for O(1) lookup
    # dist_from: list of sets, where dist_from[i] contains squared distances from i to all other points in S
    
    # Initial state: Empty set
    S = []
    in_set = [False] * NUM_POINTS
    dist_from = [set() for _ in range(NUM_POINTS)]
    
    best_S = []
    best_size = 0
    
    # Helper functions
    def calc_d2(p, q):
        px, py = coords[p]
        qx, qy = coords[q]
        dx = px - qx
        dy = py - qy
        return dx*dx + dy*dy

    def is_valid_add(p, S, dist_from):
        """Check if p can be added to S without forming an isosceles triangle."""
        # 1. Check if p forms isosceles with any pair in S
        # For any a, b in S, d(p,a) != d(p,b) and d(p,a) != d(a,b) and d(p,b) != d(a,b)
        # Actually, the condition is:
        # No three points form an isosceles triangle.
        # If we add p, we check all triples (p, a, b) where a, b in S.
        # Isosceles if d(p,a) == d(p,b) OR d(p,a) == d(a,b) OR d(p,b) == d(a,b).
        
        # Optimization:
        # 1. d(p,a) must be distinct for all a in S. (No two points in S are equidistant from p)
        # 2. d(p,a) must not be equal to d(a,b) for any a,b in S. 
        #    This is equivalent to: for each a in S, d(p,a) is not in dist_from[a].
        
        d2s = set()
        for a in S:
            d2_pa = calc_d2(p, a)
            # Check if this distance was already seen for p
            if d2_pa in d2s:
                return False
            d2s.add(d2_pa)
            
            # Check if this distance matches any distance from a to another point in S
            # dist_from[a] contains d2(a, b) for all b in S, b != a
            if d2_pa in dist_from[a]:
                return False
                
        return True

    def do_add(p, S, in_set, dist_from):
        S.append(p)
        in_set[p] = True
        dist_from[p] = set()
        for a in S[:-1]:
            d2 = calc_d2(p, a)
            dist_from[p].add(d2)
            dist_from[a].add(d2)

    def do_remove(p, S, in_set, dist_from):
        # Remove p from S
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        
        # Update dist_from for all remaining points
        for a in S:
            d2 = calc_d2(p, a)
            dist_from[a].discard(d2)
        
        # Clear dist_from[p]
        dist_from[p] = set()

    def do_swap(p_out, p_in, S, in_set, dist_from):
        # Remove p_out
        idx = S.index(p_out)
        S.pop(idx)
        in_set[p_out] = False
        for a in S:
            d2 = calc_d2(p_out, a)
            dist_from[a].discard(d2)
        dist_from[p_out] = set()
        
        # Add p_in
        S.append(p_in)
        in_set[p_in] = True
        dist_from[p_in] = set()
        for a in S[:-1]:
            d2 = calc_d2(p_in, a)
            dist_from[p_in].add(d2)
            dist_from[a].add(d2)

    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    min_T = 0.001
    
    # We start with a greedy initialization to get a decent starting point
    # This is similar to the seed but done once quickly
    greedy_order = list(range(NUM_POINTS))
    random.shuffle(greedy_order)
    for p in greedy_order:
        if is_valid_add(p, S, dist_from):
            do_add(p, S, in_set, dist_from)
            
    best_S = S[:]
    best_size = len(S)
    
    # Main SA Loop
    while time.time() < deadline and T > min_T:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random point
            p = random.randint(0, NUM_POINTS - 1)
            if in_set[p]:
                continue
            if is_valid_add(p, S, dist_from):
                do_add(p, S, in_set, dist_from)
                new_size = len(S)
                # Always accept if size increases
                if new_size > best_size:
                    best_size = new_size
                    best_S = S[:]
            else:
                # Cannot add, try a swap? Or just skip?
                # If we can't add, we might try to swap to make room
                pass
                
        elif move_type < 0.8:
            # Try to swap a point in S with a point not in S
            if not S:
                continue
            p_out = random.choice(S)
            p_in = random.randint(0, NUM_POINTS - 1)
            if in_set[p_in]:
                continue
            
            # Check if swap is valid
            # Temporarily remove p_out to check? 
            # Or check logic: 
            # 1. Remove p_out from consideration in dist checks for p_in?
            # It's easier to just do the swap and check validity of the new set?
            # No, we need to check if p_in is valid with (S \ {p_out}).
            
            # Let's check validity of p_in against S minus p_out
            # We can simulate this by checking:
            # 1. d(p_in, a) distinct for all a in S \ {p_out}
            # 2. d(p_in, a) not in dist_from[a] \ {d(p_out, a)} for all a in S \ {p_out}
            
            # This is complex to check without modifying state.
            # Simpler: Just perform the swap, check if the resulting set is valid?
            # No, the set S is always valid by construction.
            # We need to check if p_in can be added to S' = S \ {p_out}.
            
            # Let's implement a check function for "can add p to set S excluding point x"
            # Actually, let's just do the swap and then verify the whole set?
            # Verifying the whole set is O(K^2) which is fine for K~20.
            # But we want to be efficient.
            
            # Let's stick to: Try swap. If valid, accept. If not, revert.
            # To check validity efficiently:
            # Save old dist_from state? Too expensive.
            
            # Alternative: 
            # 1. Remove p_out
            # 2. Check if p_in is valid
            # 3. If valid, add p_in. If not, re-add p_out.
            
            # We already have do_remove and is_valid_add.
            # Let's use them.
            
            # Note: do_remove modifies S, in_set, dist_from.
            # We need to be careful to revert if it fails.
            
            # Save state? 
            # S is a list. dist_from is a list of sets.
            # Copying dist_from is expensive.
            
            # Let's try a different approach for swap:
            # 1. Check if p_in is valid against S \ {p_out}.
            #    This requires checking d(p_in, a) for all a in S \ {p_out}.
            #    And checking if d(p_in, a) is in dist_from[a] excluding d(p_out, a).
            
            # Let's implement a specialized check for this.
            d2s = set()
            valid = True
            for a in S:
                if a == p_out:
                    continue
                d2 = calc_d2(p_in, a)
                if d2 in d2s:
                    valid = False
                    break
                d2s.add(d2)
                
                # Check if d2 is in dist_from[a]
                # But dist_from[a] contains d2(a, p_out) as well.
                # If d2 == d2(a, p_out), it's okay because p_out is being removed.
                # So we need to check: is d2 in dist_from[a] AND d2 != d2(a, p_out)?
                d2_out = calc_d2(p_out, a)
                if d2 in dist_from[a] and d2 != d2_out:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                do_swap(p_out, p_in, S, in_set, dist_from)
                # Size remains same. 
                # We don't update best_size here unless we found a better configuration?
                # Actually, size is same. We just moved to a new state.
                
        else:
            # Try to remove a random point (to escape local minima)
            if S:
                p = random.choice(S)
                do_remove(p, S, in_set, dist_from)
                # Size decreased. We accept this move based on temperature?
                # In standard SA, we accept worse moves with prob exp(-delta/T).
                # Here, delta = 1 (size change).
                # We already removed it. If we want to revert, we need to add it back.
                # But adding it back is always valid if it was valid before?
                # Yes, because S is a subset of the previous valid set.
                # So we can just decide to keep it or revert.
                
                # Probability of accepting a size reduction:
                prob = math.exp(-1.0 / T)
                if random.random() > prob:
                    # Revert: add p back
                    do_add(p, S, in_set, dist_from)
        
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
        
        # Cool down
        T *= alpha
        
        # Check time
        if time.time() > deadline - 5:
            break

    return [coords[i] for i in best_S]
