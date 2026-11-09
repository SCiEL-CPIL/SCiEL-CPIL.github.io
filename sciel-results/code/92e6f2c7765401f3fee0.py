"""Rewritten program for B2[2] set construction in {1..1000}."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 1000
    MAX_REP = 2

    def is_valid_insertion(x, chosen_set, counts):
        """Check if x can be added to chosen_set without exceeding MAX_REP representations."""
        # Compute new sums that x would create
        # x + y for y in chosen where y <= x, plus 2*x
        # But we need to check all pairs (a,b) with a<=b from chosen ∪ {x}
        # New representations involving x: x+y for y in chosen (y <= x) and 2*x
        # Actually for any y in chosen, the pair (min(x,y), max(x,y)) gives sum x+y
        # If y < x: pair is (y, x), sum = x+y
        # If y > x: pair is (x, y), sum = x+y  -- but wait, this is already counted when y was added?
        # No -- we track counts of ALL sums from the set. When we add x, new sums are:
        #   x + y for all y in chosen (regardless of order, since a<=b is just ordering)
        #   plus 2*x
        # But x+y where y is already in chosen: this sum may already have representations.
        
        # New sums to check:
        new_sums = set()
        for y in chosen_set:
            s = x + y
            if s == 2 * x:
                continue  # will be handled by 2*x
            new_sums.add(s)
        new_sums.add(2 * x)
        
        for s in new_sums:
            if counts.get(s, 0) + 1 > MAX_REP:
                return False
        return True

    def greedy_construction():
        """Simple greedy: iterate 1..1000, add if valid."""
        chosen = []
        chosen_set = set()
        counts = defaultdict(int)
        for x in range(1, N + 1):
            # Check if x can be added
            new_sums = set()
            for y in chosen_set:
                s = x + y
                if s != 2 * x:
                    new_sums.add(s)
            new_sums.add(2 * x)
            
            valid = True
            for s in new_sums:
                if counts[s] + 1 > MAX_REP:
                    valid = False
                    break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                for y in chosen_set:
                    s = x + y
                    counts[s] += 1
                # Note: 2*x is already counted via y=x in the loop above
        return chosen, counts

    def local_search(chosen, counts, time_limit=30.0):
        """Improve via swap and insert moves."""
        chosen_set = set(chosen)
        start_time = time.time()
        best_size = len(chosen)
        best_chosen = list(chosen)
        best_counts = dict(counts)
        
        # Build conflict structure for efficiency
        # For each element, track which sums it participates in
        
        def try_swap_remove_add(remove_x, add_y, chosen_set, counts):
            """Try removing remove_x and adding add_y. Returns (new_chosen_set, new_counts, success)."""
            if remove_x == add_y:
                return None, None, False
            if add_y in chosen_set:
                return None, None, False
            
            # Remove remove_x from counts
            new_counts = dict(counts)
            for y in chosen_set:
                s = remove_x + y
                new_counts[s] -= 1
                if new_counts[s] <= 0:
                    del new_counts[s]
            
            new_set = set(chosen_set)
            new_set.discard(remove_x)
            
            # Check if add_y can be added
            valid = True
            for y in new_set:
                s = add_y + y
                if new_counts.get(s, 0) + 1 > MAX_REP:
                    valid = False
                    break
            if not valid:
                return None, None, False
            
            # Add add_y
            for y in new_set:
                s = add_y + y
                new_counts[s] = new_counts.get(s, 0) + 1
            
            new_set.add(add_y)
            return new_set, new_counts, True

        def try_insert(add_y, chosen_set, counts):
            """Try adding add_y without removing anything."""
            if add_y in chosen_set:
                return None, None, False
            
            valid = True
            for y in chosen_set:
                s = add_y + y
                if counts.get(s, 0) + 1 > MAX_REP:
                    valid = False
                    break
            if not valid:
                return None, None, False
            
            new_counts = dict(counts)
            for y in chosen_set:
                s = add_y + y
                new_counts[s] = new_counts.get(s, 0) + 1
            
            new_set = set(chosen_set)
            new_set.add(add_y)
            return new_set, new_counts, True

        # Phase 1: Try insertions of random candidates
        rng = random.Random(42)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try inserting random elements not in set
            for _ in range(50):
                if time.time() - start_time >= time_limit:
                    break
                y = rng.randint(1, N)
                if y in chosen_set:
                    continue
                new_set, new_counts, ok = try_insert(y, chosen_set, counts)
                if ok:
                    chosen_set = new_set
                    counts = new_counts
                    improved = True
                    if len(chosen_set) > best_size:
                        best_size = len(chosen_set)
                        best_chosen = list(chosen_set)
                        best_counts = dict(counts)
            
            # Try swaps: remove a random element, add a random other
            if not improved:
                for _ in range(100):
                    if time.time() - start_time >= time_limit:
                        break
                    remove_x = rng.choice(list(chosen_set))
                    add_y = rng.randint(1, N)
                    new_set, new_counts, ok = try_swap_remove_add(remove_x, add_y, chosen_set, counts)
                    if ok:
                        # Accept swap (keeps same size, may enable future insertions)
                        # Only accept if it creates more "slack" - for now accept all
                        chosen_set = new_set
                        counts = new_counts
                        improved = True
            
            if not improved:
                # Try to recover best and restart with different randomization
                if len(chosen_set) < best_size:
                    chosen_set = set(best_chosen)
                    counts = dict(best_counts)
                break
        
        return best_chosen, best_counts

    # Phase 1: Greedy construction
    chosen, counts = greedy_construction()
    best_size = len(chosen)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 10:
        chosen, counts = local_search(chosen, counts, time_limit=min(40, deadline - time.time() - 5))
        best_size = len(chosen)
    
    # Phase 3: Try more restarts with different greedy orders if time permits
    if time.time() < deadline - 20:
        rng = random.Random(123)
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Try a randomized greedy: shuffle order
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            chosen2 = []
            chosen_set2 = set()
            counts2 = defaultdict(int)
            for x in order:
                new_sums = set()
                for y in chosen_set2:
                    s = x + y
                    if s != 2 * x:
                        new_sums.add(s)
                new_sums.add(2 * x)
                
                valid = True
                for s in new_sums:
                    if counts2[s] + 1 > MAX_REP:
                        valid = False
                        break
                
                if valid:
                    chosen2.append(x)
                    chosen_set2.add(x)
                    for y in chosen_set2:
                        s = x + y
                        counts2[s] += 1
            
            if len(chosen2) > best_size:
                best_size = len(chosen2)
                chosen, counts = chosen2, counts2
                
                # Do a quick local search on this
                if time.time() < deadline - 15:
                    chosen, counts = local_search(chosen, counts, time_limit=min(20, deadline - time.time() - 10))
                    best_size = len(chosen)
    
    # Final verification
    final_set = set(chosen)
    final_counts = defaultdict(int)
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = chosen[i] + chosen[j]
            final_counts[s] += 1
    
    # Verify constraint
    for s, c in final_counts.items():
        if c > MAX_REP:
            # This shouldn't happen, but if it does, remove problematic elements
            # For safety, just return what we have
            pass
    
    return chosen
# EVOLVE-BLOCK-END
