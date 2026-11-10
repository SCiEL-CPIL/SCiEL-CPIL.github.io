"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict
import heapq

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Map element to list of indices of bad triples containing it
    elem_to_triples = {i: [] for i in range(1, n + 1)}
    for idx, t in enumerate(bad_triples):
        for x in t:
            elem_to_triples[x].append(idx)
            
    # Map triple index to its elements
    # We will maintain a set of active (remaining) triple indices
    
    def can_add(x, current_set, active_triples):
        """Check if x can be added to current_set without creating a bad triple.
        A bad triple is created if there exists a triple (x, a, b) in bad_triples
        such that a in current_set and b in current_set.
        """
        for t_idx in elem_to_triples[x]:
            t = bad_triples[t_idx]
            if t_idx in active_triples:
                # Check if the other two are in current_set
                # t is (x, a, b) or (a, x, b) etc.
                # We need to check the other two elements
                others = [y for y in t if y != x]
                if others[0] in current_set and others[1] in current_set:
                    return False
        return True

    best_size = 0
    best_set = set()
    
    # Number of restarts
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Hitting Set with dynamic counts
        in_set = [True] * (n + 1)
        current_set = set(range(1, n + 1))
        
        # Active triples: initially all
        active_triples = set(range(len(bad_triples)))
        
        # Count how many active triples each element is in
        # Note: we only count triples where ALL elements are currently in the set?
        # No, the greedy algorithm removes elements. A triple is "hit" if any of its elements is removed.
        # So a triple is "remaining" if all its elements are still in the set.
        # We want to remove an element that hits the most remaining triples.
        
        # Initial counts
        counts = defaultdict(int)
        for x in range(1, n + 1):
            counts[x] = len(elem_to_triples[x]) # This is total, not active.
            # We need to be careful. Initially all are active.
            
        # Priority queue: (-count, element)
        # We need to update counts as we remove elements.
        # When we remove x, all triples containing x become inactive.
        # For each such triple t, for each other element y in t, counts[y] decreases by 1.
        
        # To avoid stale heap entries, we can use a "lazy" deletion approach or just re-heapify if necessary.
        # Given N=128, re-heapifying or scanning is cheap.
        # Let's just scan for max count each time, it's O(N) per step, and steps <= N. Total O(N^2) per restart.
        # N=128, N^2=16k. 100 restarts -> 1.6M ops. Very fast.
        
        removed = []
        
        while active_triples:
            if time.time() > deadline:
                break
            
            # Find element in current_set with max count in active_triples
            max_count = -1
            best_x = -1
            for x in range(1, n + 1):
                if not in_set[x]:
                    continue
                # Count active triples containing x
                # We can iterate elem_to_triples[x] and check if t_idx in active_triples
                # This is O(degree(x)). Sum over x is O(E) = O(3 * |T|).
                # |T| can be up to ~10k?
                # Let's optimize: maintain counts array and update it.
                pass
            
            # Let's maintain counts array properly.
            # Initialize counts
            if r == 0: # Only init once? No, restarts need fresh state.
                 pass
            
            # Actually, let's just recompute counts from active_triples for simplicity and correctness, 
            # as N is small.
            counts = defaultdict(int)
            for t_idx in active_triples:
                a, b, c = bad_triples[t_idx]
                if in_set[a]: counts[a] += 1
                if in_set[b]: counts[b] += 1
                if in_set[c]: counts[c] += 1
                
            if not counts:
                break
                
            max_count = max(counts.values())
            candidates = [x for x, c in counts.items() if c == max_count]
            
            # Tie-breaking: random
            x = random.choice(candidates)
            
            # Remove x
            in_set[x] = False
            current_set.remove(x)
            removed.append(x)
            
            # Update active_triples
            # Remove all triples containing x
            to_remove = set()
            for t_idx in elem_to_triples[x]:
                if t_idx in active_triples:
                    to_remove.add(t_idx)
            active_triples.difference_update(to_remove)
        
        # current_set is now valid.
        
        # Local Search: Add back removed elements
        # Try multiple orders
        for _ in range(5): # 5 passes of add-back
            random.shuffle(removed)
            for x in removed:
                if time.time() > deadline:
                    break
                if in_set[x]:
                    continue
                # Check if x can be added
                # We need to check against current_set
                # can_add(x, current_set, active_triples) is not quite right because active_triples changed.
                # We should just check if adding x creates any bad triple with current_set.
                # A bad triple is (x, a, b) with a,b in current_set.
                # This is equivalent to: is there any t in bad_triples containing x such that the other two are in current_set?
                ok = True
                for t_idx in elem_to_triples[x]:
                    t = bad_triples[t_idx]
                    others = [y for y in t if y != x]
                    if others[0] in current_set and others[1] in current_set:
                        ok = False
                        break
                if ok:
                    current_set.add(x)
                    in_set[x] = True
                    removed.remove(x)
                    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
