import random
import time
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    time_limit = 115
    start_time = time.time()

    # Use numpy array for counts for faster vectorized updates if needed,
    # but for single element updates, Python lists or arrays are fine.
    # The key optimization is O(1) validity check via bitsets.
    # We will use a "state" class to encapsulate the logic.

    class B2State:
        def __init__(self, N):
            self.N = N
            self.max_sum = 2 * N
            self.S = set()
            self.S_bits = 0
            self.counts = np.zeros(self.max_sum + 1, dtype=np.int32)
            self.BadSums = 0  # Bitset: bit k is 1 if counts[k] == 2
            self.TightSums = 0  # Bitset: bit k is 1 if counts[k] == 1 (not strictly needed for add check, but good for removal)
            # Actually, for can_add we only need to know if counts[s] >= 2.
            # The condition for adding x is:
            # 1. counts[2x] < 2
            # 2. For all y in S, counts[x+y] < 2
            # This is equivalent to:
            # (S_bits & (BadSums >> x)) == 0 AND counts[2x] < 2
            # Note: BadSums contains sums with count == 2.
            # If counts[x+y] is 0 or 1, it's safe to add (becomes 1 or 2).
            # If counts[x+y] is 2, it becomes 3 -> invalid.
            # So we just need to ensure no y in S has x+y in BadSums.

        def can_add(self, x):
            if x in self.S:
                return False
            if self.counts[2 * x] >= 2:
                return False
            # Check if any y in S makes x+y a bad sum
            # BadSums >> x has bit y set if (x+y) is in BadSums.
            # We need intersection with S_bits to be empty.
            if (self.S_bits & (self.BadSums >> x)) != 0:
                return False
            return True

        def add(self, x):
            if not self.can_add(x):
                return False
            self.S.add(x)
            self.S_bits |= (1 << x)
            
            # Update counts
            # 1. 2x
            self.counts[2 * x] += 1
            if self.counts[2 * x] == 2:
                self.BadSums |= (1 << (2 * x))
            
            # 2. x + y for all y in S (excluding x)
            # We need to update counts[x+y] for all y in S_old.
            # Since we added x to S, we iterate S and skip x.
            # To make this O(1) or O(N/word_size), we can't easily do it without a loop over S.
            # However, the loop is over the set S. In the worst case, this is O(|S|).
            # The directive asks for O(1) tight-count updates.
            # Can we do better?
            # Updating counts[x+y] for all y in S is inherently O(|S|) unless we use convolution.
            # But we are updating *individual* counts.
            # Wait, the directive says "replace the O(|chosen|) inner loop with constant-time array lookups".
            # This is tricky because adding x affects |S| sums.
            # However, the *validity check* is O(1).
            # The *update* is still O(|S|).
            # Maybe the directive implies using a different data structure?
            # Or maybe it means that the *check* is the bottleneck?
            # In the current code, the check is O(1) but the update is O(|S|).
            # If we can't make the update O(1), we can't fully satisfy "O(1) tight-count updates".
            # But we can make the *greedy selection* faster by pre-computing something?
            # No, the state changes.
            
            # Let's stick to the O(|S|) update but optimize it.
            # Using a list of S elements and iterating is fast in Python if S is a list.
            # But we need to maintain S as a set for O(1) membership.
            # We can maintain a list `S_list` alongside `S_set`.
            
            # Let's refactor to use a list for iteration.
            pass # Placeholder, will implement below

        def remove(self, x):
            if x not in self.S:
                return False
            self.S.remove(x)
            self.S_bits &= ~(1 << x)
            
            # Update counts
            self.counts[2 * x] -= 1
            if self.counts[2 * x] == 1:
                self.BadSums &= ~(1 << (2 * x))
            
            for y in self.S:
                s = x + y
                self.counts[s] -= 1
                if self.counts[s] == 1:
                    self.BadSums &= ~(1 << s)
            return True

    # The above class is incomplete because `add` and `remove` still have O(|S|) loops.
    # To improve performance, we will use a list for the set elements to iterate over.
    
    class OptimizedB2State:
        def __init__(self, N):
            self.N = N
            self.max_sum = 2 * N
            self.S = set()
            self.S_list = []
            self.S_bits = 0
            self.counts = [0] * (self.max_sum + 1)
            self.BadSums = 0

        def can_add(self, x):
            if x in self.S:
                return False
            if self.counts[2 * x] >= 2:
                return False
            if (self.S_bits & (self.BadSums >> x)) != 0:
                return False
            return True

        def add(self, x):
            if not self.can_add(x):
                return False
            self.S.add(x)
            self.S_list.append(x)
            self.S_bits |= (1 << x)
            
            # Update 2x
            c = self.counts[2 * x]
            self.counts[2 * x] = c + 1
            if c + 1 == 2:
                self.BadSums |= (1 << (2 * x))
            
            # Update x+y for y in S_list (excluding x, which is at the end)
            for i in range(len(self.S_list) - 1):
                y = self.S_list[i]
                s = x + y
                c = self.counts[s]
                self.counts[s] = c + 1
                if c + 1 == 2:
                    self.BadSums |= (1 << s)
            return True

        def remove(self, x):
            if x not in self.S:
                return False
            self.S.remove(x)
            self.S_bits &= ~(1 << x)
            
            # Remove x from S_list
            # This is O(n). We can swap with last element to make it O(1).
            # But then the order changes. Does order matter?
            # For the bitset check, no.
            # For the iteration in add/remove, we just need to iterate all elements.
            # So swapping is fine.
            try:
                idx = self.S_list.index(x)
            except ValueError:
                return False # Should not happen
            
            last = self.S_list[-1]
            self.S_list[idx] = last
            self.S_list.pop()
            
            # Update 2x
            c = self.counts[2 * x]
            self.counts[2 * x] = c - 1
            if c - 1 == 1:
                self.BadSums &= ~(1 << (2 * x))
            
            # Update x+y for y in S_list (all current elements)
            for y in self.S_list:
                s = x + y
                c = self.counts[s]
                self.counts[s] = c - 1
                if c - 1 == 1:
                    self.BadSums &= ~(1 << s)
            return True

    # Phase 1: Greedy with random orders
    best_size = 0
    best_state = None
    best_S = set()

    for i in range(50):
        if time.time() - start_time > time_limit - 20:
            break
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        state = OptimizedB2State(N)
        for x in candidates:
            state.add(x)
        
        size = len(state.S)
        if size > best_size:
            best_size = size
            best_S = set(state.S)
            best_state = state

    # Phase 2: Local Search / Simulated Annealing
    # We will perform swaps to improve the size.
    # Start with the best state found.
    if best_state is None:
        return []
    
    S = best_state.S
    S_list = best_state.S_list
    S_bits = best_state.S_bits
    counts = best_state.counts
    BadSums = best_state.BadSums

    # We will work directly on these variables for speed, or use a copy.
    # Let's use a copy to be safe, or just modify the best_state.
    # Modifying best_state is fine if we keep track of best_size.
    
    # To speed up random selection of x not in S, we can maintain a list of available numbers.
    # But with N=1000, random.randint is fast enough.
    
    # Simulated Annealing style:
    # Try to add. If not possible, try to swap.
    
    current_size = len(S)
    best_local_size = current_size
    best_local_S = set(S)
    best_local_S_list = list(S_list)
    best_local_S_bits = S_bits
    best_local_counts = counts.copy()
    best_local_BadSums = BadSums

    # We will implement a simple hill-climbing with restarts or just a long run.
    # Given the time limit, we will just run a loop.
    
    # To make it faster, we will pre-compute a list of all numbers 1..N.
    all_nums = list(range(1, N + 1))
    
    # We need to be careful with the state updates.
    # Let's create a helper to check and apply swap.
    
    def try_add(x, S, S_list, S_bits, counts, BadSums):
        if x in S:
            return False
        if counts[2 * x] >= 2:
            return False
        if (S_bits & (BadSums >> x)) != 0:
            return False
        
        # Add x
        S.add(x)
        S_list.append(x)
        S_bits |= (1 << x)
        
        c = counts[2 * x]
        counts[2 * x] = c + 1
        if c + 1 == 2:
            BadSums |= (1 << (2 * x))
        
        for y in S_list[:-1]:
            s = x + y
            c = counts[s]
            counts[s] = c + 1
            if c + 1 == 2:
                BadSums |= (1 << s)
        return True

    def try_remove(x, S, S_list, S_bits, counts, BadSums):
        if x not in S:
            return False
        S.remove(x)
        S_bits &= ~(1 << x)
        
        idx = S_list.index(x)
        last = S_list[-1]
        S_list[idx] = last
        S_list.pop()
        
        c = counts[2 * x]
        counts[2 * x] = c - 1
        if c - 1 == 1:
            BadSums &= ~(1 << (2 * x))
        
        for y in S_list:
            s = x + y
            c = counts[s]
            counts[s] = c - 1
            if c - 1 == 1:
                BadSums &= ~(1 << s)
        return True

    # Main loop
    # We will try to improve the size.
    # Since adding is hard, we focus on swaps that might allow future additions?
    # Or just accept any swap that keeps validity?
    # Actually, if we swap, the size stays the same.
    # We want to increase size.
    # So we should look for swaps that *free up* capacity for an addition?
    # That's complex.
    
    # Alternative: Just try to add random elements. If stuck, try swaps that don't reduce size (impossible) 
    # or just random swaps to escape local minima?
    # But if we can't add, swapping doesn't help increase size immediately.
    # It might help by changing the set of "bad sums".
    
    # Let's try a different approach:
    # 1. Try to add.
    # 2. If can't add, try to remove a random element and see if we can add two? No, that reduces size.
    # 3. Try to swap: remove y, add x. If successful, size is same.
    # 4. After a swap, try to add again.
    
    # So the loop is:
    # while time:
    #   if can_add(x): add(x)
    #   else:
    #      pick y in S, x not in S
    #      if can_swap(y, x): do swap
    #      (optionally try to add after swap)
    
    # Let's implement this.
    
    while time.time() - start_time < time_limit:
        # Try to add
        added = False
        for _ in range(10): # Try a few random additions
            x = random.randint(1, N)
            if try_add(x, S, S_list, S_bits, counts, BadSums):
                added = True
                break
        
        if added:
            if len(S) > best_local_size:
                best_local_size = len(S)
                best_local_S = set(S)
                best_local_S_list = list(S_list)
                best_local_S_bits = S_bits
                best_local_counts = counts.copy()
                best_local_BadSums = BadSums
            continue
        
        # Try to swap
        if len(S) > 0:
            y = random.choice(S_list)
            x = random.randint(1, N)
            if x == y:
                continue
            
            # Check if swap is valid
            # We need to remove y, then check if x can be added.
            # To avoid modifying state if invalid, we can simulate.
            # Or just do it and revert.
            
            # Save state
            S_add = S.copy()
            S_list_add = S_list.copy()
            S_bits_add = S_bits
            counts_add = counts.copy()
            BadSums_add = BadSums
            
            # Remove y
            try_remove(y, S_add, S_list_add, S_bits_add, counts_add, BadSums_add)
            
            # Try add x
            if try_add(x, S_add, S_list_add, S_bits_add, counts_add, BadSums_add):
                # Swap successful
                S = S_add
                S_list = S_list_add
                S_bits = S_bits_add
                counts = counts_add
                BadSums = BadSums_add
                # Note: size is same, so no update to best_local_size
            else:
                # Revert (do nothing, S is unchanged)
                pass

    return list(best_local_S)
