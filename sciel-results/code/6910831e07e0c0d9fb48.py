import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def can_add(x, chosen, counts):
        if counts.get(2 * x, 0) >= 2:
            return False
        for y in chosen:
            if counts.get(x + y, 0) >= 2:
                return False
        return True

    def add_to_set(x, chosen, counts):
        chosen.append(x)
        counts[2 * x] = counts.get(2 * x, 0) + 1
        for y in chosen[:-1]: # y is the new x? No, y is existing.
            # Wait, we need to update counts for all sums involving x
            pass # This is inefficient to update one by one if we don't store list of pairs.
        
        # Better: Rebuild counts? No, too slow.
        # We need to update counts for sums: 2x, and x+y for all y in old chosen.
        # Let's pass the old chosen list.
        pass

    # Let's use a simpler state representation:
    # chosen: list of ints
    # counts: dict sum -> count
    
    def update_counts_add(x, chosen, counts):
        # chosen does NOT contain x yet
        new_counts = counts.copy()
        new_counts[2*x] = new_counts.get(2*x, 0) + 1
        for y in chosen:
            s = x + y
            new_counts[s] = new_counts.get(s, 0) + 1
        return new_counts

    def greedy(order):
        chosen = []
        counts = {}
        for x in order:
            if can_add(x, chosen, counts):
                chosen.append(x)
                counts = update_counts_add(x, chosen, counts)
        return chosen, counts

    def complete(chosen, counts):
        chosen_set = set(chosen)
        # Iterate through all numbers, add if possible
        # To be efficient, we can just loop 1..1000
        for x in range(1, 1001):
            if x in chosen_set:
                continue
            if can_add(x, chosen, counts):
                chosen.append(x)
                chosen_set.add(x)
                counts = update_counts_add(x, chosen, counts)
        return chosen, counts

    # 1. Initial Standard Greedy
    order_std = list(range(1, 1001))
    best_chosen, best_counts = greedy(order_std)
    best_chosen, best_counts = complete(best_chosen, best_counts)
    best_size = len(best_chosen)
    
    random.seed(42)
    
    while time.time() < deadline:
        # Randomized Greedy
        order_rand = list(range(1, 1001))
        random.shuffle(order_rand)
        
        cur_chosen, cur_counts = greedy(order_rand)
        cur_chosen, cur_counts = complete(cur_chosen, cur_counts)
        
        if len(cur_chosen) > best_size:
            best_size = len(cur_chosen)
            best_chosen = cur_chosen
            best_counts = cur_counts
            
        # Local Search: Try removing one element and refilling
        # Pick a random element to remove? Or iterate?
        # Iterating over all elements is O(N * Cost_Complete)
        # Cost_Complete is ~10^6. N=1000. Total 10^9. Too slow.
        # So we should pick a few random elements to try removing.
        
        if best_chosen:
            # Try 10 random removals
            for _ in range(10):
                idx = random.randint(0, len(best_chosen) - 1)
                y = best_chosen[idx]
                
                # Create a copy of the set without y
                # To be fast, we can just mark y as removed and rebuild counts?
                # Rebuilding counts is O(N^2) which is 10^6.
                # 10 removals * 10^6 = 10^7. Acceptable.
                
                new_chosen = [x for x in best_chosen if x != y]
                # Rebuild counts for new_chosen
                new_counts = {}
                for i in range(len(new_chosen)):
                    a = new_chosen[i]
                    new_counts[2*a] = new_counts.get(2*a, 0) + 1
                    for j in range(i+1, len(new_chosen)):
                        b = new_chosen[j]
                        s = a + b
                        new_counts[s] = new_counts.get(s, 0) + 1
                
                # Complete
                new_chosen, new_counts = complete(new_chosen, new_counts)
                
                if len(new_chosen) > best_size:
                    best_size = len(new_chosen)
                    best_chosen = new_chosen
                    best_counts = new_counts
                    break # Restart local search loop with new best

    return best_chosen
