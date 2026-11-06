import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCDs
    gcds = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Identify bad triples and build conflict lists
    # conflicts[x] is a list of (a, b) such that {a, b, x} is a bad triple
    conflicts = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if gcds[a][b] == gcds[b][c] == gcds[c][a]:
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))

    def can_add(x, S):
        """Check if x can be added to set S (bitmask)."""
        for a, b in conflicts[x]:
            if (S >> a) & 1 and (S >> b) & 1:
                return False
        return True

    def get_blocker(x, S):
        """Return the first pair (a, b) in conflicts[x] that is in S, or None."""
        for a, b in conflicts[x]:
            if (S >> a) & 1 and (S >> b) & 1:
                return (a, b)
        return None

    def greedy_add_bitset(order, S):
        """Add as many elements from order to S (bitmask) as possible."""
        for x in order:
            if not (S >> x) & 1 and can_add(x, S):
                S |= (1 << x)
        return S

    best_size = 0
    best_mask = 0
    
    # Precompute conflict counts for sorting strategies
    conflict_counts = [len(conflicts[i]) for i in range(N + 1)]
    
    # Strategy 1: Deterministic Greedy with least conflicts first
    order_least = sorted(elements, key=lambda x: conflict_counts[x])
    S = greedy_add_bitset(order_least, 0)
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S

    # Strategy 2: Deterministic Greedy with most conflicts first
    order_most = sorted(elements, key=lambda x: conflict_counts[x], reverse=True)
    S = greedy_add_bitset(order_most, 0)
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S

    # Strategy 3: Randomized Local Search with Bitsets
    # We will run multiple iterations of: Random Greedy -> Local Search
    
    max_restarts = 200
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random order
        current_order = elements[:]
        random.shuffle(current_order)
        
        # Greedy phase
        S = greedy_add_bitset(current_order, 0)
        current_size = S.bit_count()
        
        # Local Search phase
        improved = True
        # We use a "kick" counter to force exploration if stuck
        stuck_counter = 0
        
        while improved and time.time() < deadline:
            improved = False
            stuck_counter += 1
            
            # Try to add a random element
            # Pick a random element not in S
            not_in_S = [x for x in elements if not (S >> x) & 1]
            if not not_in_S:
                break # Full set? Unlikely
            
            v = random.choice(not_in_S)
            
            if can_add(v, S):
                S |= (1 << v)
                current_size += 1
                improved = True
                stuck_counter = 0
                continue
            
            # Try to swap: remove a blocker, add v
            blocker = get_blocker(v, S)
            if blocker:
                a, b = blocker
                # Try removing a
                if can_add(v, S ^ (1 << a)):
                    S = S ^ (1 << a) | (1 << v)
                    improved = True
                    stuck_counter = 0
                    continue
                # Try removing b
                if can_add(v, S ^ (1 << b)):
                    S = S ^ (1 << b) | (1 << v)
                    improved = True
                    stuck_counter = 0
                    continue
            
            # If stuck for a while, try a 2-move kick
            # Remove 2 random elements, try to add 2 (including v)
            if stuck_counter > 50:
                # Pick 2 random elements to remove
                in_S = [x for x in elements if (S >> x) & 1]
                if len(in_S) >= 2:
                    u1, u2 = random.sample(in_S, 2)
                    S_temp = S ^ (1 << u1) ^ (1 << u2)
                    
                    # Try to add v and another random w
                    # We want to see if we can get back to size or larger
                    # Try adding v first
                    if can_add(v, S_temp):
                        S_temp |= (1 << v)
                        # Now try to add another element w to fill the gap
                        # We need to add one more to maintain size, or two to improve
                        # Let's try to add any valid w
                        # To be fast, try a few random w
                        for _ in range(10):
                            w = random.choice(not_in_S)
                            if w != v and not (S_temp >> w) & 1 and can_add(w, S_temp):
                                S_temp |= (1 << w)
                                break
                        
                        # If we added v and maybe w, check if size is better
                        new_size = S_temp.bit_count()
                        if new_size >= current_size:
                            S = S_temp
                            current_size = new_size
                            if new_size > best_size:
                                best_size = new_size
                                best_mask = S
                            improved = True
                            stuck_counter = 0
                            continue
                stuck_counter = 0 # Reset to avoid infinite kick loops without progress

        if current_size > best_size:
            best_size = current_size
            best_mask = S

    # Final check: try to add any remaining elements to the best set found
    for x in elements:
        if not (best_mask >> x) & 1 and can_add(x, best_mask):
            best_mask |= (1 << x)
            
    # Convert bitmask to list
    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    return result
