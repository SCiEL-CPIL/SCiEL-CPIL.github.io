import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    # 1. Precompute Smallest Prime Factor (SPF)
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        """Return a frozenset of prime factors of n."""
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    # Cache prime factors for sums
    # Sums are between 1+2=3 and 9999+10000=19999
    pf_cache = [None] * (MAX_VAL + 1)
    for v in range(3, MAX_VAL + 1):
        pf_cache[v] = get_prime_factors(v)

    def count_distinct_primes(nums):
        """Calculate the number of distinct prime factors across all pairwise sums."""
        prime_counts = {}
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                for p in pf_cache[s]:
                    if p in prime_counts:
                        prime_counts[p] += 1
                    else:
                        prime_counts[p] = 1
        return len(prime_counts), prime_counts

    class State:
        def __init__(self, nums):
            self.nums = nums
            self.prime_counts = {}
            self.distinct_count = 0
            self.update_counts()

        def update_counts(self):
            self.prime_counts = {}
            n = len(self.nums)
            for i in range(n):
                for j in range(i + 1, n):
                    s = self.nums[i] + self.nums[j]
                    for p in pf_cache[s]:
                        if p in self.prime_counts:
                            self.prime_counts[p] += 1
                        else:
                            self.prime_counts[p] = 1
            self.distinct_count = len(self.prime_counts)

        def try_change(self, idx, new_val):
            """
            Try changing nums[idx] to new_val.
            Returns (new_distinct_count, success)
            Modifies state in place if we keep it, but we will rollback.
            """
            if new_val == self.nums[idx]:
                return self.distinct_count, False
            if new_val < 1 or new_val > 10000:
                return self.distinct_count, False
            if new_val in self.nums:
                return self.distinct_count, False

            old_val = self.nums[idx]
            
            # Track changes
            # 1. Remove contributions of old_val
            # 2. Add contributions of new_val
            
            # We need to be careful with counts.
            # A prime p is "distinct" if count > 0.
            # When we decrement a count, if it becomes 0, distinct_count decreases by 1.
            # When we increment a count, if it was 0, distinct_count increases by 1.
            
            delta_distinct = 0
            n = len(self.nums)
            
            # Helper to apply a delta to a prime's count and track distinct change
            def apply_delta(p, delta):
                nonlocal delta_distinct
                if p not in self.prime_counts:
                    self.prime_counts[p] = 0
                old_cnt = self.prime_counts[p]
                new_cnt = old_cnt + delta
                if new_cnt < 0:
                    # This shouldn't happen if logic is correct
                    new_cnt = 0
                if old_cnt == 0 and new_cnt > 0:
                    delta_distinct += 1
                elif old_cnt > 0 and new_cnt == 0:
                    delta_distinct -= 1
                self.prime_counts[p] = new_cnt
                # Cleanup if 0 to save memory? Optional.
                if new_cnt == 0:
                    del self.prime_counts[p]

            # Remove old pairs
            for j in range(n):
                if j == idx:
                    continue
                s_old = old_val + self.nums[j]
                for p in pf_cache[s_old]:
                    apply_delta(p, -1)
                
                # Add new pairs
                s_new = new_val + self.nums[j]
                for p in pf_cache[s_new]:
                    apply_delta(p, 1)

            new_distinct = self.distinct_count + delta_distinct
            self.distinct_count = new_distinct
            self.nums[idx] = new_val
            
            return new_distinct, True

        def rollback(self, idx, old_val):
            """Revert the last change at idx."""
            # This is tricky because we don't store the previous state of prime_counts.
            # Better approach: Store a copy of prime_counts before change? 
            # Or just recalculate? 
            # Recalculating is O(N^2 * log N) which is ~ 66 * 5 = 330 ops.
            # Given 12 elements, N^2 is small.
            # Let's just recalculate if we reject the move.
            # Actually, the incremental update above modifies self.prime_counts and self.nums.
            # If we reject, we must revert both.
            # Reverting prime_counts is hard without a log.
            # So, let's use a "trial" mechanism:
            # 1. Save old_val.
            # 2. Perform incremental update.
            # 3. If accepted, keep.
            # 4. If rejected, we must restore prime_counts.
            
            # Since prime_counts is a dict, we can save a copy? 
            # Dict size is small (number of distinct primes in sums, likely < 100).
            # Copying a dict of 100 items is fast.
            pass # We will handle rollback by restoring a snapshot.

    # Improved State with Snapshot for Rollback
    class SearchState:
        def __init__(self, nums):
            self.nums = nums[:]
            self.prime_counts = {}
            self.distinct_count = 0
            self.recalculate()

        def recalculate(self):
            self.prime_counts = {}
            n = len(self.nums)
            for i in range(n):
                for j in range(i + 1, n):
                    s = self.nums[i] + self.nums[j]
                    for p in pf_cache[s]:
                        if p in self.prime_counts:
                            self.prime_counts[p] += 1
                        else:
                            self.prime_counts[p] = 1
            self.distinct_count = len(self.prime_counts)

        def try_change(self, idx, new_val):
            """
            Returns (new_distinct, snapshot) if change is valid, else (None, None)
            Modifies internal state. Caller must rollback if rejected.
            """
            if new_val == self.nums[idx] or new_val < 1 or new_val > 10000:
                return None, None
            if new_val in self.nums:
                return None, None

            old_val = self.nums[idx]
            # Snapshot
            snapshot_counts = self.prime_counts.copy()
            snapshot_distinct = self.distinct_count
            snapshot_num = old_val

            # Apply change
            n = len(self.nums)
            delta_distinct = 0
            
            def apply_delta(p, delta):
                nonlocal delta_distinct
                if p not in self.prime_counts:
                    self.prime_counts[p] = 0
                old_cnt = self.prime_counts[p]
                new_cnt = old_cnt + delta
                if new_cnt < 0: new_cnt = 0
                if old_cnt == 0 and new_cnt > 0:
                    delta_distinct += 1
                elif old_cnt > 0 and new_cnt == 0:
                    delta_distinct -= 1
                self.prime_counts[p] = new_cnt
                if new_cnt == 0:
                    del self.prime_counts[p]

            for j in range(n):
                if j == idx: continue
                s_old = old_val + self.nums[j]
                for p in pf_cache[s_old]:
                    apply_delta(p, -1)
                
                s_new = new_val + self.nums[j]
                for p in pf_cache[s_new]:
                    apply_delta(p, 1)

            self.distinct_count += delta_distinct
            self.nums[idx] = new_val
            
            return self.distinct_count, (snapshot_counts, snapshot_distinct, snapshot_num)

        def rollback(self, snapshot):
            snapshot_counts, snapshot_distinct, snapshot_num = snapshot
            self.prime_counts = snapshot_counts
            self.distinct_count = snapshot_distinct
            # Note: self.nums[idx] was changed, need to restore it.
            # We need to know which idx was changed.
            # Let's modify try_change to return idx as well or handle it outside.
            # Actually, simpler: just pass idx to rollback.

        def rollback_idx(self, idx, snapshot):
            snapshot_counts, snapshot_distinct, snapshot_num = snapshot
            self.prime_counts = snapshot_counts
            self.distinct_count = snapshot_distinct
            self.nums[idx] = snapshot_num

    best_count = float('inf')
    best_nums = None

    def update_best(nums, count):
        nonlocal best_count, best_nums
        if count < best_count:
            best_count = count
            best_nums = nums[:]

    # Initialization Strategies
    candidates = []

    # 1. Smooth Quotient Strategy
    # a_i = M * q_i
    # Try M = 30, 60, 210, 420
    for M in [30, 60, 210, 420, 840, 1260, 2520]:
        max_q = 10000 // M
        if max_q < 12: continue
        # Generate 12 random q_i
        for _ in range(50):
            qs = random.sample(range(1, max_q + 1), 12)
            nums = [q * M for q in qs]
            candidates.append(nums)
        
        # Generate 12 consecutive q_i
        for start in range(1, min(20, max_q - 11)):
            qs = list(range(start, start + 12))
            nums = [q * M for q in qs]
            candidates.append(nums)

    # 2. Random Integers
    for _ in range(200):
        candidates.append(random.sample(range(1, 10001), 12))

    # 3. "Congruent" Strategy
    # a_i = r + k_i * M
    for M in [10, 20, 50, 100, 500, 1000]:
        for r in range(1, min(M, 100)):
            max_k = (10000 - r) // M
            if max_k < 12: continue
            for _ in range(20):
                ks = random.sample(range(0, max_k + 1), 12)
                nums = [r + k * M for k in ks]
                if all(1 <= x <= 10000 for x in nums) and len(set(nums)) == 12:
                    candidates.append(nums)

    # Evaluate candidates and run local search on the best few
    # Sort candidates by initial cost
    scored_candidates = []
    for nums in candidates:
        if len(set(nums)) != 12: continue
        if any(x < 1 or x > 10000 for x in nums): continue
        state = SearchState(nums)
        scored_candidates.append((state.distinct_count, state))

    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 20 candidates for local search
    for count, state in scored_candidates[:20]:
        if time.time() > deadline - 5:
            break
            
        # Local Search: Simulated Annealing / Hill Climbing
        current_state = state
        current_count = current_state.distinct_count
        update_best(current_state.nums, current_count)
        
        T = 5.0 # Temperature
        T_min = 0.1
        cooling = 0.99
        
        while time.time() < deadline - 2 and T > T_min:
            # Pick a random index
            idx = random.randint(0, 11)
            # Pick a new value
            # Strategy: Random jump or local step
            if random.random() < 0.5:
                # Local step
                delta = random.randint(-50, 50)
                new_val = current_state.nums[idx] + delta
            else:
                # Random jump
                new_val = random.randint(1, 10000)
            
            new_count, snapshot = current_state.try_change(idx, new_val)
            
            if new_count is None:
                continue
                
            if new_count < current_count:
                # Accept
                current_count = new_count
                if current_count < best_count:
                    update_best(current_state.nums, current_count)
            else:
                # Metropolis criterion
                delta_E = new_count - current_count
                if random.random() < math.exp(-delta_E / T):
                    current_count = new_count
                else:
                    # Reject, rollback
                    current_state.rollback_idx(idx, snapshot)
            
            T *= cooling

    # Final Polish: Try to fix any issues (distinctness, bounds) - already handled
    # Try one more pass of greedy hill climbing on the best result
    if best_nums and time.time() < deadline - 1:
        state = SearchState(best_nums)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for idx in range(12):
                # Try all neighbors? Too slow.
                # Try a few random neighbors
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    new_count, snapshot = state.try_change(idx, new_val)
                    if new_count is not None and new_count < state.distinct_count:
                        state.distinct_count = new_count
                        if new_count < best_count:
                            update_best(state.nums, new_count)
                        improved = True
                    else:
                        if snapshot:
                            state.rollback_idx(idx, snapshot)

    if best_nums is None:
        return list(range(1, 13))
        
    return best_nums
