import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer for 90s limit

    # Initial best known valid set (powers of 2)
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)

    def compute_sums_bitset(elements):
        """
        Returns a list of integers (bitsets) S[0..n] where S[k] has bit s set
        if sum s is achievable by a subset of size k.
        """
        n = len(elements)
        if n == 0:
            return [1]  # S[0] = 1 (bit 0 set)
        
        # We don't strictly need to know max sum in advance if we use Python big ints
        # But we do need the array size for cardinality.
        S = [0] * (n + 1)
        S[0] = 1  # bit 0 is set
        
        for v in elements:
            # Update in reverse order to avoid using v multiple times
            for k in range(n, 0, -1):
                S[k] |= (S[k-1] << v)
        
        return S

    def count_collisions(S):
        """
        Counts the number of cross-cardinality sum collisions.
        Returns the count of bits that are set in more than one S[k].
        """
        n = len(S) - 1
        total = 0
        # Only check pairs (i, j) with i < j
        for i in range(n + 1):
            if S[i] == 0:
                continue
            for j in range(i + 1, n + 1):
                if S[j] == 0:
                    continue
                inter = S[i] & S[j]
                if inter:
                    total += bin(inter).count('1')
        return total

    def is_valid(S):
        return count_collisions(S) == 0

    # Strategy 1: Greedy Construction
    # Try to build a set by adding elements that don't cause collisions.
    
    def greedy_construct(order):
        current_set = []
        S = [1]  # S[0]
        
        for v in order:
            # Check if adding v creates a collision
            # New S' where S'[k] = S[k] | (S[k-1] << v)
            new_S = [0] * (len(current_set) + 2)
            new_S[0] = 1
            for k in range(1, len(current_set) + 1):
                new_S[k] = S[k] | (S[k-1] << v)
            
            if is_valid(new_S):
                current_set.append(v)
                S = new_S
        
        return current_set

    # Try greedy with different orders
    random.seed(42)
    
    # Order 1: Descending
    greedy_set = greedy_construct(list(range(256, 0, -1)))
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)
        
    # Order 2: Ascending
    greedy_set = greedy_construct(list(range(1, 257)))
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)

    # Order 3: Random shuffles
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        greedy_set = greedy_construct(random_order)
        if len(greedy_set) > best_size:
            best_set = greedy_set
            best_size = len(best_set)

    # Strategy 2: Conflict-Driven Removal from a Dense Set
    # Start with a large random set, remove elements that cause the most conflicts.
    
    def conflict_removal(initial_set, time_limit):
        current_set = list(initial_set)
        if not current_set:
            return current_set, 0
            
        S = compute_sums_bitset(current_set)
        collisions = count_collisions(S)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_idx = -1
            best_collisions = collisions
            best_new_set = None
            best_new_S = None
            
            n = len(current_set)
            
            for idx in range(n):
                v = current_set[idx]
                new_set = current_set[:idx] + current_set[idx+1:]
                new_S = compute_sums_bitset(new_set)
                new_collisions = count_collisions(new_S)
                
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_idx = idx
                    best_new_set = new_set
                    best_new_S = new_S
                    improved = True
                    
            if improved:
                current_set = best_new_set
                S = best_new_S
                collisions = best_collisions
            else:
                break
                
        return current_set, collisions

    # Run conflict removal on several random dense sets
    for attempt in range(15):
        if time.time() > deadline - 15:
            break
            
        # Start with a random set of size 25-45
        size = random.randint(25, 45)
        random_set = random.sample(range(1, 257), size)
        
        result_set, final_collisions = conflict_removal(random_set, deadline - 15)
        
        if final_collisions == 0:
            if len(result_set) > best_size:
                best_set = result_set
                best_size = len(best_set)

    # Strategy 3: Local Search / Expansion
    # Take the best valid set and try to add more elements.
    
    def try_expand(valid_set, time_limit):
        current_set = list(valid_set)
        S = compute_sums_bitset(current_set)
        
        current_elements = set(current_set)
        candidates = [v for v in range(1, 257) if v not in current_elements]
        
        # We will try to add candidates.
        # To be efficient, we can pre-check or just iterate.
        # Since n is small, computing new_S is fast.
        
        added_any = True
        while added_any and time.time() < time_limit:
            added_any = False
            # Shuffle candidates to avoid bias
            random.shuffle(candidates)
            
            for i, v in enumerate(candidates):
                new_S = [0] * (len(current_set) + 2)
                new_S[0] = 1
                for k in range(1, len(current_set) + 1):
                    new_S[k] = S[k] | (S[k-1] << v)
                
                if is_valid(new_S):
                    current_set.append(v)
                    S = new_S
                    candidates.pop(i)
                    added_any = True
                    break # Restart loop with new state
                    
        return current_set

    # Try expanding the best set found
    if time.time() < deadline - 5:
        expanded_set = try_expand(best_set, deadline - 5)
        if len(expanded_set) > best_size:
            best_set = expanded_set
            best_size = len(best_set)
            
    # If we still have time, try expanding from other good candidates found in strategy 2?
    # Or just run more random greedy?
    
    # Let's do a few more random greedy runs if time permits
    if time.time() < deadline - 2:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            random_order = list(range(1, 257))
            random.shuffle(random_order)
            greedy_set = greedy_construct(random_order)
            if len(greedy_set) > best_size:
                best_set = greedy_set
                best_size = len(best_set)

    # Final Check and Return
    # Ensure types are correct
    return [int(x) for x in best_set]

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Best size: {len(res)}")
    print(res)
