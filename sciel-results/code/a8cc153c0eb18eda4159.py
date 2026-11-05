import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for 120s limit, but task says 150s kill, so 115 is safe.
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    best_set = []
    best_size = 0
    
    def is_valid_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            s = x + y
            if sum_counts[s] >= 2:
                return False
        return True

    def try_add(A, in_A, sum_counts, x):
        if not is_valid_add(A, in_A, sum_counts, x):
            return False
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def check_swap_validity(A, in_A, sum_counts, y, x):
        # Check if (A \ {y}) \cup {x} is valid
        if x in in_A or y not in in_A:
            return False
        
        # Check 2x
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
            
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
        return True

    def perform_swap(A, in_A, sum_counts, y, x):
        remove_element(A, in_A, sum_counts, y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1

    def check_double_swap_validity(A, in_A, sum_counts, y1, y2, x1, x2):
        # Check if (A \ {y1, y2}) \cup {x1, x2} is valid
        if x1 in in_A or x2 in in_A or y1 not in in_A or y2 not in in_A:
            return False
        if x1 == x2:
            return False
        
        # New set elements: A' = (A \ {y1, y2}) \cup {x1, x2}
        # We need to check all sums involving x1 or x2 in the new set.
        # Sums:
        # 1. x1 + x1
        # 2. x2 + x2
        # 3. x1 + x2
        # 4. x1 + z for z in A \ {y1, y2}
        # 5. x2 + z for z in A \ {y1, y2}
        
        # Helper to check if a specific sum s is valid in the new context
        # new_count(s) = old_count(s) - (pairs involving y1 or y2 summing to s) + (pairs involving x1 or x2 summing to s)
        
        def get_old_contrib(s, y1, y2):
            # Count how many pairs in A sum to s and involve y1 or y2
            count = 0
            # Pair (y1, w) sums to s => w = s - y1
            w1 = s - y1
            if w1 in in_A and w1 != y1 and w1 != y2: # w1 must be in A, not y1 (handled by w1!=y1? no, pair is (y1,w1). If w1==y1, pair is (y1,y1). If w1==y2, pair is (y1,y2).)
                 # Wait, pairs are unordered a<=b.
                 # If w1 == y1, pair is (y1, y1).
                 # If w1 == y2, pair is (y1, y2).
                 # If w1 != y1 and w1 != y2, pair is (y1, w1).
                 # In all cases, if w1 is in A, the pair exists.
                 # But we must ensure we don't double count or count pairs that are removed.
                 # The pairs removed are those containing y1 or y2.
                 # Pairs containing y1: (y1, w) for w in A.
                 # Pairs containing y2: (y2, w) for w in A.
                 # If w=y1, (y1, y1) is counted once in y1's list.
                 # If w=y2, (y1, y2) is counted in y1's list AND y2's list.
                 # So we need to be careful.
                 
                 # Let's just iterate through the specific pairs that change.
                 pass
            # This logic is getting messy. Let's use a simpler simulation for double swap check.
            # Since |A| is small (~50), we can just calculate the new counts for the affected sums.
            return 0

        # Simpler approach for double swap:
        # Identify all sums affected: 
        # Sums involving y1 or y2 in old set.
        # Sums involving x1 or x2 in new set.
        # We only need to check if any sum in the new set exceeds 2.
        # The only sums that can change are those involving x1, x2, y1, y2.
        # Specifically, sums x1+x1, x2+x2, x1+x2, x1+z, x2+z for z in A\{y1,y2}.
        
        # Let's check these specific sums.
        
        # 1. x1 + x1
        s = 2 * x1
        # Old count of s: sum_counts[s]
        # Subtract contributions from y1, y2 in old set
        # Add contributions from x1, x2 in new set
        # Contributions to s in old set involving y1 or y2:
        #   (y1, s-y1) if s-y1 in A
        #   (y2, s-y2) if s-y2 in A
        #   Note: if s-y1 == y2, then (y1, y2) is one pair.
        #   If s-y1 == y1, then (y1, y1).
        
        # This is complex. Let's just construct the new set temporarily? No, too slow.
        # Let's stick to single swaps for now, but add "Remove and Re-add" logic.
        # Or, just accept that double swaps are hard to verify quickly and rely on single swaps + restarts.
        
        return False # Disable double swap for now to keep code simple and fast
        
    # Initialization
    A = []
    in_A = set()
    sum_counts = [0] * (max_sum + 1)
    
    # Greedy Init
    for x in range(1, N + 1):
        if try_add(A, in_A, sum_counts, x):
            pass
            
    best_set = A[:]
    best_size = len(A)
    
    # Local Search
    # We will run multiple restarts if time permits, but first optimize the current one.
    
    temperature = 1.0
    cooling_rate = 0.999
    no_improve_count = 0
    max_no_improve = 1000 # Adjust based on speed
    
    while time.time() < deadline:
        # 1. Try to add
        added = False
        # Shuffle candidates for randomness
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        if candidates:
            random.shuffle(candidates)
            for x in candidates:
                if try_add(A, in_A, sum_counts, x):
                    added = True
                    if len(A) > best_size:
                        best_size = len(A)
                        best_set = A[:]
                        no_improve_count = 0
                    break
        
        if added:
            continue
            
        # 2. Try Swap
        swapped = False
        if A and candidates:
            # Limit attempts per iteration to avoid wasting time
            for _ in range(50):
                y = random.choice(A)
                x = random.choice(candidates)
                if check_swap_validity(A, in_A, sum_counts, y, x):
                    perform_swap(A, in_A, sum_counts, y, x)
                    swapped = True
                    break
        
        if swapped:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset or restart?
                # Let's try removing an element to break deadlock
                if A:
                    # Remove most constrained
                    scores = {}
                    for y in A:
                        score = 0
                        for z in A:
                            if sum_counts[y+z] >= 2:
                                score += 1
                        if sum_counts[2*y] >= 2:
                            score += 1
                        scores[y] = score
                    max_score = max(scores.values())
                    # Pick one with max score
                    to_remove = [y for y in A if scores[y] == max_score]
                    y = random.choice(to_remove)
                    remove_element(A, in_A, sum_counts, y)
                    no_improve_count = 0
                continue
        
        # 3. If no swap, try removing an element
        if A:
            # Remove random element to escape local optimum
            y = random.choice(A)
            remove_element(A, in_A, sum_counts, y)
            no_improve_count += 1
            if no_improve_count > 5000:
                # Restart with new greedy?
                # For now, just keep going
                pass

    return best_set
