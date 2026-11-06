import random
import time
import sys

def construct():
    # Deadline guard: 150s limit, aim for 110s work + safety margin
    start_time = time.time()
    time_limit = 110.0
    
    # Helper to check if adding x to set S is valid
    # S is a set of ints, counts is a dict of sum -> count
    # We need to check if any new sum exceeds 2
    
    def try_add(x, S, counts):
        # Calculate new sums
        # Sums involving x: x + y for y in S, and 2*x
        # We need to check if any of these sums would exceed count 2
        # If a sum s is already at 2, and x contributes to s, it's invalid.
        
        # Optimization: 
        # For each y in S, sum s = x + y.
        # If counts[s] == 2, invalid.
        # Also for 2*x, if counts[2*x] == 2, invalid.
        
        # But wait, if multiple y's produce the same sum?
        # x+y1 = x+y2 implies y1=y2. So sums x+y are distinct for distinct y.
        # However, x+y might equal 2*x? Only if y=x, but y in S and x not in S (for insert).
        # Or x+y1 = x+y2 is impossible.
        # So we just check each new sum individually against current counts.
        
        # However, we must also ensure that the new sums don't collide with *each other*?
        # No, the constraint is on the *total* count in the final set.
        # The new sums are distinct from each other (as shown above).
        # So we just need to ensure that for each new sum s, counts[s] < 2.
        
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def get_new_counts(x, S, counts):
        new_counts = dict(counts)
        for y in S:
            s = x + y
            new_counts[s] = new_counts.get(s, 0) + 1
        s = 2 * x
        new_counts[s] = new_counts.get(s, 0) + 1
        return new_counts

    # Initial greedy construction
    # We can try a few different greedy orders to find a good starting point
    best_set = set()
    best_counts = {}
    best_size = 0
    
    # Try standard greedy
    S = set()
    counts = {}
    for x in range(1, 1001):
        if try_add(x, S, counts):
            S.add(x)
            counts = get_new_counts(x, S, counts)
    
    if len(S) > best_size:
        best_set = S
        best_counts = counts
        best_size = len(S)

    # Try a few random permutations for greedy initialization
    for _ in range(10):
        if time.time() - start_time > time_limit * 0.1:
            break
        perm = list(range(1, 1001))
        random.shuffle(perm)
        S = set()
        counts = {}
        for x in perm:
            if try_add(x, S, counts):
                S.add(x)
                counts = get_new_counts(x, S, counts)
        if len(S) > best_size:
            best_set = S
            best_counts = counts
            best_size = len(S)

    # Local Search
    # We will try to increase the size by adding elements, or swap to enable additions.
    # To speed up, we precompute "blocked" elements? 
    # An element x is blocked if adding it violates the constraint.
    # If we remove an element y, some x's might become unblocked.
    
    # Strategy:
    # 1. Identify all candidates not in S.
    # 2. Try to add a random candidate.
    # 3. If fail, pick a random element in S to remove, and a random candidate to add.
    
    candidates = list(set(range(1, 1001)) - best_set)
    
    # To make local search faster, we can track why elements are blocked.
    # But a simple random search is okay if we have time.
    
    # Let's use a more directed local search.
    # We'll maintain a list of "conflict" sums.
    
    # Actually, a better approach for B2 sets is to use the property that 
    # if we sort the set, the gaps must be large enough.
    # But let's stick to the add/swap heuristic.
    
    # To avoid slow Python loops, we can use numpy or just efficient set operations.
    # The inner loop of try_add is O(|S|). |S| is ~50. 1000 candidates * 50 ops = 50k ops.
    # This is fast.
    
    current_set = set(best_set)
    current_counts = dict(best_counts)
    current_size = len(current_set)
    
    # Pre-compute which elements are currently addable?
    # addable = [x for x in candidates if try_add(x, current_set, current_counts)]
    # This is O(N * |S|). N=1000, |S|=50. 50k ops. Very fast.
    
    def get_addable(S, counts, candidates_list):
        addable = []
        for x in candidates_list:
            if try_add(x, S, counts):
                addable.append(x)
        return addable

    # Main loop
    while time.time() - start_time < time_limit:
        addable = get_addable(current_set, current_counts, candidates)
        
        if addable:
            # Pick one randomly
            x = random.choice(addable)
            current_set.add(x)
            current_counts = get_new_counts(x, current_set, current_counts)
            current_size += 1
            candidates.remove(x)
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
                best_counts = dict(current_counts)
        else:
            # No element can be added directly.
            # Try a swap: remove y from S, add x from candidates.
            # This keeps size constant, but might open up space for another element.
            # Or, we can try to remove an element and see if *two* others can be added?
            # That would increase size by 1.
            
            # Let's try a simple swap first to escape local minima.
            if candidates and current_set:
                y = random.choice(list(current_set))
                x = random.choice(candidates)
                
                # Remove y
                # We need to update counts by subtracting contributions of y
                new_counts = dict(current_counts)
                for z in current_set:
                    if z == y: continue
                    s = y + z
                    new_counts[s] -= 1
                    if new_counts[s] == 0:
                        del new_counts[s]
                s = 2 * y
                new_counts[s] -= 1
                if new_counts[s] == 0:
                    del new_counts[s]
                
                temp_set = set(current_set)
                temp_set.remove(y)
                
                # Try to add x
                if try_add(x, temp_set, new_counts):
                    # Swap accepted
                    temp_set.add(x)
                    final_counts = get_new_counts(x, temp_set, new_counts)
                    
                    current_set = temp_set
                    current_counts = final_counts
                    # current_size remains same
                    candidates.remove(x)
                    candidates.append(y)
                    
                    # Check if this swap allows an immediate insertion
                    # We'll let the next iteration handle it via get_addable
                else:
                    # Swap rejected, do nothing
                    pass
            else:
                # No candidates or empty set, break
                break

    return sorted(best_set)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
